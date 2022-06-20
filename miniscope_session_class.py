 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:11:39 2020

@author: anagoncalves
"""

##VALID PERIOD - VALID STRIDES FOR ALL FOUR PAWS
import matplotlib.pyplot as plt
import matplotlib as mp
import numpy as np
import tifffile as tiff
from statsmodels import robust
from scipy import signal, stats
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.signal import wiener
import glob
import pandas as pd
import math
from scipy.optimize import curve_fit
import os
import seaborn as sns
import scipy.cluster.hierarchy as spc
from scipy.stats import skew
from scipy.stats import median_abs_deviation as mad
import pylab as pl
import scipy.io as spio
import mat73
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import mahotas
import matplotlib.patches as mp_patch
import SlopeThreshold as ST
import read_roi
import scipy.spatial.distance as spdist

 #to call class
#os.chdir('/Users/anagoncalves/Documents/PhD/Code/Miniscope pipeline/')
#import miniscope_session_class
#mscope = miniscope_session_class.miniscope_session(path)
#and then mscope.(function name)

# to check class documentation (after importing class)
#help(mscope.compute_dFF)

#to see all methods in class
#dir(mscope)
class miniscope_session:
    def __init__(self,path):
        self.path = path
        self.delim = self.path[-1]
        # pixel_to_um = 0.93 #1px is 0.93um for miniscopes v3.2
        self.pixel_to_um = 0.608 #1px is 0.608um for miniscopes v4
        self.sr = 30 #sampling rate of miniscopes
        self.my_dpi = 128 #resolution for plotting
        self.sr_loco = 330 #sampling rate of behavioral camera
        self.fsize = 20

    @staticmethod
    def z_score(A,axis_id):
        """Normalizes an array by z-scoring it"""
        if len(np.shape(A)):
            A_norm = (A-np.nanmean(A))/np.nanstd(A)
        else:
            if axis_id == 1:
                A_mean = np.repeat(np.nanmean(A,axis=axis_id).reshape(-1,1),np.shape(A)[axis_id],axis=axis_id)
                A_std = np.repeat(np.nanstd(A,axis=axis_id).reshape(-1,1),np.shape(A)[axis_id],axis=axis_id)
                A_norm = np.divide((A-A_mean),A_std)
            else:
                A_mean = np.repeat(np.nanmean(A,axis=axis_id),np.shape(A)[axis_id],axis=axis_id)
                A_std = np.repeat(np.nanstd(A,axis=axis_id),np.shape(A)[axis_id],axis=axis_id)
                A_norm = np.divide((A-A_mean),A_std)
        return A_norm
    
    @staticmethod
    def inpaint_nans(A):
        """Interpolates NaNs in numpy arrays
        Input: A (numpy array)"""
        ok = ~np.isnan(A)
        xp = ok.ravel().nonzero()[0]
        fp = A[~np.isnan(A)]
        x  = np.isnan(A).ravel().nonzero()[0]
        A[np.isnan(A)] = np.interp(x, xp, fp)
        return A

    @staticmethod
    def mov_mean(x, w):
        """Does moving average with numpy convolution function
        Inputs:
            x: data vector
            w: window length (int)"""
        return np.convolve(x, np.ones(w),'same')/w

    @staticmethod
    def fitEllipse(cont, method):
        """Fit ellipse method to ROI coordinates. Finds ellipse coordinates and find major and minor axis lengths
        Input:
            cont: array of coordinates (nx2)
            method: 1 if want to choose maximum absolute eigenvalue from data points"""
        x = cont[:, 0]
        y = cont[:, 1]
        x = x[:, None]
        y = y[:, None]
        D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        if method == 1:
            n = np.argmax(np.abs(E))
        else:
            n = np.argmax(E)
        a = V[:, n]
        # Fit ellipse
        b, c, d, f, g, a = a[1] / 2., a[2], a[3] / 2., a[4] / 2., a[5], a[0]
        num = b * b - a * c
        cx = (c * d  - b * f) / num
        cy = (a * f - b* d) / num
        angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
        up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        a = np.sqrt(abs(up / down1))
        b = np.sqrt(abs(up / down2))
        # Get path
        ell = Ellipse((cx, cy), a * 2., b * 2., angle)
        ell_coord = ell.get_verts()
        params = [cx, cy, a, b, angle]
        return params, ell_coord

    @staticmethod
    def render(poly):
        """Return polygon as grid of points inside polygon.
        Input : poly (list of lists)
        Output : output (list of lists)
        """
        xs, ys = zip(*poly)
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]
        X = maxx - minx + 1
        Y = maxy - miny + 1
        grid = np.zeros((X, Y), dtype=np.int64)
        mahotas.polygon.fill_polygon(newPoly, grid)
        return [(x + minx, y + miny) for (x, y) in zip(*np.nonzero(grid))]

    @staticmethod
    def correlation_traces(trace1,trace2):
        cross_corr = correlate(miniscope_session.z_score(trace1, 1),
                               miniscope_session.z_score(trace2, 1), mode='full',
                               method='direct')
        p_corrcoef = np.cov(trace1, trace2)[0][1] / np.sqrt(np.var(trace1) * np.var(trace2))
        return p_corrcoef

    @staticmethod
    def norm_traces(df, norm_name, axis):
        """Function to compute the norm traces.
            Inputs:
                df: dataframe containing for each column ROI/pixel raw trace
                norm_name: (str) min_max or zscore
                axis: (str) "session" for single session norm or "trial" for each trial norm"""
        df_norm = pd.DataFrame(columns=df.columns, index=df.index.to_list())
        trials = df['trial'].unique()
        if axis == 'trial':
            for col in df.columns[2:]:
                trace_alltrials = []
                for t in trials:
                    mean_value = df.loc[df['trial'] == t, col].mean(axis=0, skipna=True)
                    std_value = df.loc[df['trial'] == t, col].std(axis=0, skipna=True)
                    min_value = df.loc[df['trial'] == t, col].min(axis=0, skipna=True)
                    max_value = df.loc[df['trial'] == t, col].max(axis=0, skipna=True)
                    if norm_name == 'zscore':
                        trace_alltrials.extend((df.loc[df['trial'] == t, col] - mean_value) / std_value)
                    if norm_name == 'min_max':
                        trace_alltrials.extend((df.loc[df['trial'] == t, col] - min_value) / (max_value-min_value))
                df_norm[col] = trace_alltrials
        if axis == 'session':
            for col in df.columns[2:]:
                mean_value = df[col].mean(axis=0, skipna=True)
                std_value = df[col].std(axis=0, skipna=True)
                min_value = df[col].min(axis=0, skipna=True)
                max_value = df[col].max(axis=0, skipna=True)
                if norm_name == 'zscore':
                    df_norm[col] = (df[col] - mean_value)/std_value
                if norm_name == 'min_max':
                    df_norm[col] = (df[col] - min_value)/(max_value-min_value)
        df_norm.iloc[:,:2] = df.iloc[:,:2]
        return df_norm

    @staticmethod
    def compute_dFF(df_fiji):
        """Function to compute dF/F for each trial using as F0 the 10th percentile of the signal for that trial"""
        trials = df_fiji['trial'].unique()
        for t in trials:
            trial_length = np.shape(df_fiji.loc[df_fiji['trial'] == t])[0]
            trial_idx = df_fiji.loc[df_fiji['trial'] == t].index
            perc_arr = np.tile(np.nanpercentile(np.array(df_fiji.iloc[trial_idx,2:]), 10, axis=0), (trial_length, 1))
            with np.errstate(divide='ignore', invalid='ignore'):
                dFF_trial = np.true_divide((np.array(df_fiji.iloc[trial_idx, 2:]) - perc_arr), perc_arr)
                dFF_trial[dFF_trial == np.inf] = 0
            df_fiji.iloc[trial_idx, 2:] = dFF_trial
        return df_fiji

    @staticmethod
    def get_roi_stats(coord_cell):
        """From the ROIs coordinates get the width, height and aspect ratio
        Input:
        coord_cell (list of ROI coordinates)"""
        width_roi = []
        height_roi = []
        aspect_ratio = []
        for r in range(len(coord_cell)):
            params, ell = miniscope_session.fitEllipse(coord_cell[r], 1)
            width_roi.append(params[2])
            height_roi.append(params[3])
            aspect_ratio.append(params[3] / params[2])
        return width_roi, height_roi, aspect_ratio

    @staticmethod
    def get_roi_centroids(coord_cell):
        """From the ROIs coordinates get the centroid
        Input:
        coord_cell (list of ROI coordinates)"""
        centroid_cell = []
        for r in range(len(coord_cell)):
            centroid_cell.append(np.array([np.nanmean(coord_cell[r][:,0]), np.nanmean(coord_cell[r][:,1])]))
        return centroid_cell

    @staticmethod
    def get_roi_list(df_data):
        """Get the list of ROIs"""
        return list(df_data.columns[2:])

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
            """Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            inputs: x rotation up and down treadmill (pitch)
                    y rotation side to side (yaw)
                    z rotation left and right (roll))"""
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            x_angles = np.arctan2(t0, t1)     
            t2 = +2.0 * (w * y - z * x)
            for t in range(len(t2)):
                if t2[t] > +1.0:
                    t2[t] = +1.0
                else:
                    t2[t] = t2[t]
            for t in range(len(t2)):
                if t2[t] < -1.0:
                    t2[t] = -1.0
                else:
                    t2[t] = t2[t]
            y_angles = np.arcsin(t2)     
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            z_angles = np.arctan2(t3, t4) 
            #giving placement of miniscope: x is roll, y is pitch, z is yaw
            pitch = y_angles
            yaw = z_angles
            roll = x_angles
            return roll, pitch, yaw # in radians

    @staticmethod
    def correct_gimbal_lock(head_angles):
        """Funtion to deal with ambiguous conversion from quartenion to euler angles
        that occurs from time to time. Different quartenions can give the same euler angle.
        Correct the circularity of those values.
        Inputs:
            head_angles (dataframe)"""
        fig, ax = plt.subplots(1, 3, figsize=(10, 10), tight_layout=True)
        ax = ax.ravel()
        ax[0].plot(head_angles['pitch'], color='black')
        ax[0].set_title('Pitch')
        ax[1].plot(head_angles['roll'], color='black')
        ax[1].set_title('Roll')
        ax[2].plot(head_angles['yaw'], color='black')
        ax[2].set_title('Yaw')
        plt.suptitle('Before corrections')
        pitch_amp = np.max(head_angles['pitch']) - np.mean(head_angles['pitch'])
        roll_amp = np.max(head_angles['roll']) - np.mean(head_angles['roll'])
        yaw_amp = np.max(head_angles['yaw']) - np.mean(head_angles['yaw'])
        if pitch_amp > 2:
            plt.figure(figsize=(15, 7), tight_layout=True)
            plt.plot(head_angles['pitch'], color='black')
            plt.title('First high threshold then low')
            coord = plt.ginput(n=2, timeout=0, show_clicks=True)
            pitch_th = np.array([coord[1][1], coord[0][1]])
            idx_nan_down_p = np.where(head_angles['pitch'] < pitch_th[0])[0]
            idx_nan_up_p = np.where(head_angles['pitch'] > pitch_th[1])[0]
            head_angles.iloc[idx_nan_up_p, 1] = head_angles.iloc[idx_nan_up_p, 1] - 2 * np.pi
            head_angles.iloc[idx_nan_down_p, 1] = head_angles.iloc[idx_nan_down_p, 1] + 2 * np.pi
        else:
            pitch_th = np.array([np.min(head_angles['pitch']) - 1, np.max(head_angles['pitch']) + 1])
        if roll_amp > 2:
            plt.figure(figsize=(15, 7), tight_layout=True)
            plt.plot(head_angles['roll'], color='black')
            plt.title('First high threshold then low')
            coord = plt.ginput(n=2, timeout=0, show_clicks=True)
            roll_th = np.array([coord[1][1], coord[0][1]])
            idx_nan_down_r = np.where(head_angles['roll'] < roll_th[0])[0]
            idx_nan_up_r = np.where(head_angles['roll'] > roll_th[1])[0]
            head_angles.iloc[idx_nan_up_r, 0] = head_angles.iloc[idx_nan_up_r, 0] - np.pi
            head_angles.iloc[idx_nan_down_r, 0] = head_angles.iloc[idx_nan_down_r, 0] + np.pi
        else:
            roll_th = np.array([np.min(head_angles['roll']) - 1, np.max(head_angles['roll']) + 1])
        if yaw_amp > 2:
            plt.figure(figsize=(15, 7), tight_layout=True)
            plt.plot(head_angles['yaw'], color='black')
            plt.title('First high threshold then low')
            coord = plt.ginput(n=2, timeout=0, show_clicks=True)
            yaw_th = np.array([coord[1][1], coord[0][1]])
            idx_nan_down_y = np.where(head_angles['yaw'] < yaw_th[0])[0]
            idx_nan_up_y = np.where(head_angles['yaw'] > yaw_th[1])[0]
            head_angles.iloc[idx_nan_up_y, 2] = head_angles.iloc[idx_nan_up_y, 2] - 2 * np.pi
            head_angles.iloc[idx_nan_down_y, 2] = head_angles.iloc[idx_nan_down_y, 2] + 2 * np.pi
        else:
            yaw_th = np.array([np.min(head_angles['yaw']) - 1, np.max(head_angles['yaw']) + 1])
        fig, ax = plt.subplots(1, 3, figsize=(10, 10), tight_layout=True)
        ax = ax.ravel()
        ax[0].plot(head_angles['pitch'], color='black')
        ax[0].set_title('Pitch')
        ax[1].plot(head_angles['roll'], color='black')
        ax[1].set_title('Roll')
        ax[2].plot(head_angles['yaw'], color='black')
        ax[2].set_title('Yaw')
        plt.suptitle('After corrections')
        return head_angles

    @staticmethod
    def pca_centroids(principalComponents_3CP, trial_clean, trials, plot_data):
        """Function to compute the centroids of the PCA space for each trial
        Inputs:
            principalComponents_3CP: array of PCA output
            trial_clean: array of trial id for PCA output
            trials: arrays with trials in session
            plot_data: boolean"""
        greys = mp.cm.get_cmap('Greys', 12)
        reds = mp.cm.get_cmap('Reds', 23)
        blues = mp.cm.get_cmap('Blues', 23)
        colors_session = [greys(5), greys(7), greys(12), reds(5), reds(7), reds(9), reds(11), reds(13), reds(15),
                          reds(17), reds(19), reds(21), reds(23), blues(5), blues(7), blues(9), blues(11), blues(13),
                          blues(15), blues(17), blues(19), blues(21), blues(23)]
        centroid_trials = []  # CHANGE NO MAXMIN NORM, CENTROID POINTS SHOULD
        for t in trials:
            idx_trial = np.where(trial_clean == t)
            x_norm = (principalComponents_3CP[idx_trial, 0] - np.min(principalComponents_3CP[idx_trial, 0])) / (
                        np.max(principalComponents_3CP[idx_trial, 0]) - np.min(principalComponents_3CP[idx_trial, 0]))
            y_norm = (principalComponents_3CP[idx_trial, 1] - np.min(principalComponents_3CP[idx_trial, 1])) / (
                        np.max(principalComponents_3CP[idx_trial, 1]) - np.min(principalComponents_3CP[idx_trial, 1]))
            centroid_trials.append(np.array([np.nanmean(x_norm), np.nanmean(y_norm)]))
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
            for t in range(len(trials)):
                plt.scatter(centroid_trials[t][0], centroid_trials[t][1], s=100, color=colors_session[t])
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.legend(trials, frameon=False)
        return centroid_trials

    @staticmethod
    def distance_neurons(centroid_cell,plot_data):
        """Computes the distance between ROIs from the centroids given by suite2p
        Inputs:
            plot_data (boolean)"""
        roi_nr = np.arange(len(centroid_cell))
        distance_neurons = np.zeros((len(centroid_cell),len(centroid_cell)))
        for r1 in range(len(centroid_cell)):
            for r2 in range(len(centroid_cell)):
                distance_neurons[r1,r2] = np.linalg.norm(np.array(centroid_cell[r1])-np.array(centroid_cell[r2]))
        if plot_data:
            mask = np.triu(np.ones_like(distance_neurons, dtype=np.bool))
            fig,ax = plt.subplots()
            with sns.axes_style("white"):
                sns.heatmap(distance_neurons, mask=mask, cmap="YlGnBu",linewidth=0.5)
                ax.set_title('distance between ROIs')
                ax.set_yticklabels(roi_nr)
                ax.set_xticklabels(roi_nr)
        return distance_neurons

    def get_animal_id(self):
        animal_name = self.path.split(self.delim)[3]
        return animal_name

    def get_protocol_id(self):
        protocol_name = self.path.split(self.delim)[2]
        protocol_id = protocol_name.replace(' ','_')
        return protocol_id

    def get_trial_id(self):
        """Function to get the trials where recordings occurred (in order)"""
        if self.delim == '/':
            ops = np.load(self.path+'Suite2p/suite2p/plane0/ops.npy', allow_pickle=True)
        else:
            ops = np.load(self.path+'Suite2p\\suite2p\\plane0\\ops.npy', allow_pickle=True)
        filelist = ops[()]['filelist']
        count_t = 0
        trial_order = np.zeros(len(filelist), dtype=np.int8)
        for f in filelist:
            filename = f.split(self.delim)[-1]
            trial_name = filename.split('_')[0]
            trial_number = int(trial_name[1:])
            trial_order[count_t] = trial_number
            count_t += 1        
        return trial_order        
    
    def trial_length(self, df_extract):
        """ Get number of frames for each trial based on traces dataframe
        Input:
        df_extract: dataframe with traces and the usual structure"""
        trials = np.unique(df_extract['trial'])
        trial_length = np.zeros(len(trials))
        for count, t in enumerate(trials):
            trial_length[count] = len(df_extract.loc[df_extract['trial'] == t].index)
        return trial_length

    def get_reg_data(self):
        """Function to get the correlation map computed from suite2p
            Outputs: x_offset, y_offset, corrXY"""
        print(self.path)
        if self.delim == '/':
            ops = np.load(self.path+'Suite2p/suite2p/plane0/ops.npy', allow_pickle=True)
        else:
            ops = np.load(self.path+'Suite2p\\suite2p\\plane0\\ops.npy', allow_pickle=True)
        y_offset = ops[()]['yoff'] #y shifts in registration
        x_offset = ops[()]['xoff'] #x shifts in registration
        corrXY = ops[()]['corrXY'] #phase correlation between ref image and each frame
        return x_offset, y_offset, corrXY
    
    def corr_FOV_movement(self, th, df_dFF, corrXY):
        """Function to make nan times where FOV moved out of focus
        Input: corrXY (array)"""
        fig = plt.figure(figsize=(5,5),tight_layout=True)
        plt.plot(corrXY,color='black')
        plt.axhline(y=th,color='gray')
        if not os.path.exists(self.path + '\\images\\'):
            os.mkdir(self.path + '\\images\\')
        delim = self.path[-1]
        if delim == '/':
            plt.savefig(self.path + 'images/' + 'corr_fov_movement', dpi=self.my_dpi)
        else:
            plt.savefig(self.path + 'images\\' + 'corr_fov_movement', dpi=self.my_dpi)
        idx_to_nan = np.where(corrXY<=th)[0]
        df_dFF.iloc[idx_to_nan,2:] = np.nan
        return idx_to_nan, df_dFF

    def rois_larger_motion(self, df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi, height_roi, plot_data):
        """Function to compute how much the ROI size is larger than the shift of the FOV done during motion correction
        Inputs:
            df_extract: (dataframe) with ROIs traces
            coord_ext. (list) with ROI coordinates
            idx_to_nan: indices of frames to make nan, parts where motion was large
            x_offset: x shift of frames during motion correction
            y_offset: y shift of frames during motion correction
            width_roi: width of ROIs
            height_roi_ height of ROIs
            plot_data: boolean"""
        reg_good_frames = np.setdiff1d(np.arange(0, np.shape(df_extract)[0]), idx_to_nan)
        x_offset_clean = x_offset[reg_good_frames]
        y_offset_clean = y_offset[reg_good_frames]
        x_offset_minmax = np.abs(
            np.array([np.min(x_offset_clean), np.max(x_offset_clean)]))  # corresponds to x in coord_cell
        y_offset_minmax = np.abs(
            np.array([np.min(y_offset_clean), np.max(y_offset_clean)]))  # corresponds to y in coord_cell
        roi_number = len(df_extract.columns[2:])
        keep_rois = np.intersect1d(np.where(width_roi > x_offset_minmax[1])[0] + 1,
                                   np.where(height_roi > y_offset_minmax[1])[0] + 1, assume_unique=True)
        coord_ext_nomotion = []
        for r in keep_rois-1:
            coord_ext_nomotion.append(coord_ext[r])
        roi_list = df_extract.columns[2:]
        rois_del = np.setdiff1d(np.arange(1,len(roi_list)+1),keep_rois)
        rois_del_list = []
        for r in rois_del:
            rois_del_list.append('ROI'+str(r))
        df_extract = df_extract.drop(columns = rois_del_list)
        if plot_data:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            ax = ax.ravel()
            ax[0].scatter(np.arange(1, roi_number+1), width_roi, s=10, color='blue')
            ax[0].axhline(x_offset_minmax[0], color='black')
            ax[0].axhline(x_offset_minmax[1], color='black')
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].set_title('ROIs width and X max and min offsets', fontsize=self.fsize-4)
            ax[1].scatter(np.arange(1, roi_number+1), height_roi, s=10, color='blue')
            ax[1].axhline(y_offset_minmax[0], color='black')
            ax[1].axhline(y_offset_minmax[1], color='black')
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_title('ROIs height and Y max and min offsets', fontsize=self.fsize-4)
        return [coord_ext_nomotion, df_extract]

    def correlation_signal_motion(self, df_extract, x_offset, y_offset, trial, idx_to_nan, plot_data):
        """Function to compute the ROIs signal correlation with the shift of the FOV done during motion correction-
        It outputs the correlation plot between the traces and the FOV offsets and an example ROI trace with the shifts.
        Inputs:
            df_extract: (dataframe) with ROIs traces
            x_offset: x shift of frames during motion correction
            y_offset: y shift of frames during motion correction
            trial: (int) example trial to do this computation
            idx_to_nan: indices of frames to make nan, parts where motion was large
            plot_data: boolean"""
        data = np.transpose(np.array(df_extract.loc[df_extract['trial'] == trial].iloc[:, 2:]))
        roi_nr = len(df_extract.columns[2:])
        p_corrcoef = np.zeros((roi_nr, 2))
        p_corrcoef[:] = np.nan
        for r in range(roi_nr):
            idx_nonan = np.where(~np.isnan(data[0, :]))[0]
            p_corrcoef[r, 0] = np.cov(data[r, idx_nonan], x_offset[idx_nonan])[0][
                                   1] / np.sqrt(
                np.var(data[r, idx_nonan]) * np.var(x_offset[idx_nonan]))
            if p_corrcoef[r, 0] > 1:
                p_corrcoef[r, 0] = 1
            p_corrcoef[r, 1] = np.cov(data[r, idx_nonan], y_offset[idx_nonan])[0][
                                   1] / np.sqrt(
                np.var(data[r, idx_nonan]) * np.var(y_offset[idx_nonan]))
            if p_corrcoef[r, 1] > 1:
                p_corrcoef[r, 1] = 1
        reg_good_frames = np.setdiff1d(np.arange(0, np.shape(df_extract)[0]), idx_to_nan)
        x_offset_clean = x_offset[reg_good_frames]
        y_offset_clean = y_offset[reg_good_frames]
        x_offset_clean_norm = (x_offset_clean - np.max(x_offset_clean)) / (
                    np.max(x_offset_clean) - np.min(x_offset_clean))
        y_offset_clean_norm = (y_offset_clean - np.max(y_offset_clean)) / (
                    np.max(y_offset_clean) - np.min(y_offset_clean))
        if plot_data:
            r = np.random.randint(1, roi_nr + 1)
            df_extract_norm = self.norm_traces(df_extract, 'min_max')
            fig, ax = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
            ax = ax.ravel()
            ax[0].plot(np.array(df_extract_norm.loc[df_extract['trial'] == trial, 'ROI' + str(r)]), color='darkgrey')
            ax[0].plot(
                x_offset_clean_norm[np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, 'ROI' + str(r)].index)],
                color='blue')
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].set_title('Example ROI trace with X offset', fontsize=self.fsize-4)
            ax[1].plot(np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, 'ROI' + str(r)]), color='darkgrey')
            ax[1].plot(
                y_offset_clean_norm[np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, 'ROI' + str(r)].index)],
                color='blue')
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_title('Example ROI trace with Y offset', fontsize=16)

            fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
            ax.plot(np.arange(1, roi_nr + 1), p_corrcoef[:, 0], color='blue', marker='o',
                    label='correlation of trace with x shifts')
            ax.plot(np.arange(1, roi_nr + 1), p_corrcoef[:, 1], color='orange', marker='o',
                    label='correlation of trace with y shifts')
            ax.legend(frameon=False, fontsize=self.fsize-6)
            ax.set_title('Correlation of traces with FOV shift during motion correction', fontsize=self.fsize - 4)
            ax.set_xticks(np.arange(1, roi_nr + 1)[::10])
            ax.set_xticklabels(list(df_extract.columns[2::10]))
            ax.set_xlabel('ROI ID', fontsize=self.fsize - 6)
            ax.set_ylabel('Correlation coefficient', fontsize=self.fsize - 6)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=self.fsize - 6)
            ax.tick_params(axis='y', labelsize=self.fsize - 6)
            if self.delim == '/':
                plt.savefig(self.path + 'images/' + 'corr_trace_motionreg_shifts',
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\' + 'corr_trace_motionreg_shifts',
                            dpi=self.my_dpi)

    def get_ref_image(self):
        """Function to get the session reference image from suite2p"""
        if self.delim == '/':
            path_ops = 'Suite2p'+self.delim+'suite2p/plane0/ops.npy'
        else:
            path_ops = 'Suite2p'+self.delim+'suite2p\\plane0\\ops.npy'
        ops = np.load(self.path+path_ops,allow_pickle=True)
        ref_image = ops[()]['meanImg']
        return ref_image

    def read_extract_output(self, threshold_spatial_weights, frame_time, trials):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (EXTRACT output)
        Inputs:
            threshold_spatial_weights: float
            frame_time: list with miniscope timestamps
            trials: list of trials"""
        if self.delim == '/':
            path_extract = self.path + '/Registered video/EXTRACT/'
        if self.delim == '\\':
            path_extract = self.path + '\\Registered video\\EXTRACT\\'
        files_extract = glob.glob(path_extract + '*.mat')
        ext_rois = mat73.loadmat(files_extract[0]) #masks are the same across trials
        spatial_weights = ext_rois['spatial_weights']
        trace_ext_list = []
        for f in range(len(files_extract)):
            ext_dict = mat73.loadmat(files_extract[f])
            trace_ext_list.append(ext_dict['trace_nonneg'])
        trace_ext_arr = np.concatenate(trace_ext_list, axis=0)
        coord_cell = []
        for c in range(np.shape(spatial_weights)[2]):
            coord_cell.append(np.transpose(np.array(np.where(spatial_weights[:, :, c] > threshold_spatial_weights))))
        coord_cell_t = []
        for c in range(len(coord_cell)):
            coord_cell_switch = np.zeros(np.shape(coord_cell[c]))
            coord_cell_switch[:, 0] = coord_cell[c][:, 1] / self.pixel_to_um
            coord_cell_switch[:, 1] = coord_cell[c][:, 0] / self.pixel_to_um
            coord_cell_t.append(coord_cell_switch)
        # trace as dataframe
        roi_list = []
        for r in range(len(coord_cell)):
            roi_list.append('ROI' + str(r + 1))
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t - 1])))
            frame_time_ext.extend(frame_time[t - 1])
        data_ext1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_ext1 = pd.DataFrame(data_ext1)
        df_ext2 = pd.DataFrame(trace_ext_arr, columns=roi_list)
        df_ext = pd.concat([df_ext1, df_ext2], axis=1)
        return coord_cell_t, df_ext

    def compute_extract_rawtrace(self, coord_ext, roi_list, trials, frame_time):
        """Function to compute the raw traces from the ROI coordinates from EXTRACT.
        Input:
            coord_cell: list with ROIs coordinates
            roi_list: list with ROIs
            trials: array with all the trials in the session
            frame_time: list with frame timestamps"""
        ext_trace_all_list = []
        for c in range(len(coord_ext)):
            ext_trace_trials = []
            for t in trials:
                tiff_stack = tiff.imread(self.path + 'Registered video\\T' + str(t) + '_reg.tif')  # read tiffs
                ext_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
                for f in range(np.shape(tiff_stack)[0]):
                    ext_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                        np.round(coord_ext[c][:, 1] * self.pixel_to_um)), np.int64(
                        np.round(coord_ext[c][:, 0] * self.pixel_to_um))]) / np.shape(coord_ext[c])[0]
                ext_trace_trials.append(ext_trace_tiffmean)
            ext_trace_concat = np.hstack(ext_trace_trials)
            ext_trace_all_list.append(ext_trace_concat)
        ext_trace_arr = np.transpose(np.vstack(ext_trace_all_list))        # trace as dataframe
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t-1])))
            frame_time_ext.extend(frame_time[t - 1])
        dict_ext = {'trial': trial_ext, 'time': frame_time_ext}
        df_ext1 = pd.DataFrame(dict_ext)
        df_ext2 = pd.DataFrame(ext_trace_arr, columns=roi_list)
        df_ext_raw = pd.concat([df_ext1, df_ext2], axis=1)
        return df_ext_raw

    def get_imagej_output(self,frame_time,trials,norm):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (ImageJ output)
        Inputs:
            frame_time: list with miniscope timestamps
            trial: (arr) - trial list
            norm: boolean to do min-max normalization"""
        path_fiji = 'Registered video' + self.delim
        filename_rois = 'RoiSet.zip'
        rois = read_roi.read_roi_zip(self.path + path_fiji + filename_rois)
        rois_names = list(rois.keys())
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        for key, value in rois.items():
            x1_list.append(value['x1'])
            x2_list.append(value['x2'])
            y1_list.append(value['y1'])
            y2_list.append(value['y2'])
        rois_dict = {'x1': x1_list, 'x2': x2_list, 'y1': y1_list, 'y2': y2_list}
        rois_df = pd.DataFrame(rois_dict, columns=['x1', 'x2', 'y1', 'y2'], index=rois_names)
        coord_fiji = []
        for r in range(np.shape(rois_df)[0]):
            coord_r = np.transpose(np.vstack(
                (np.linspace(rois_df.iloc[r, 0] / self.pixel_to_um, rois_df.iloc[r, 1] / self.pixel_to_um, 100),
                 np.linspace(rois_df.iloc[r, 2] / self.pixel_to_um, rois_df.iloc[r, 3] / self.pixel_to_um, 100))))
            coord_fiji.append(coord_r)
        # get traces of rois
        roi_trace_all = []
        for t in trials:
            tiff_stack = tiff.imread(self.path + path_fiji+'\\T'+str(t)+'_reg.tif')  ##read tiffs
            roi_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0],len(coord_fiji)))
            for c in range(len(coord_fiji)):
                for f in range(np.shape(tiff_stack)[0]):
                   roi_trace_tiffmean[f,c] = np.nansum(tiff_stack[f, np.int64(
                       coord_fiji[c][:, 1] * self.pixel_to_um), np.int64(
                       coord_fiji[c][:, 0] * self.pixel_to_um)])/len(coord_fiji[c][:, 1])
            roi_trace_all.append(roi_trace_tiffmean)
        roi_trace_concat = np.vstack(roi_trace_all)
        if norm:
            # Normalize traces
            roi_trace_minmax = np.zeros(np.shape(roi_trace_concat))
            for col in range(np.shape(roi_trace_tiffmean)[1]):
                roi_trace_minmax[:, col] = (roi_trace_concat[:, col] - np.min(roi_trace_concat[:, col])) / (
                            np.max(roi_trace_concat[:, col]) - np.min(roi_trace_concat[:, col]))
            roi_trace_concat = roi_trace_minmax
        # trace as dataframe
        roi_list =  []
        for r in range(len(coord_fiji)):
            roi_list.append('ROI'+str(r+1))
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t-1])))
            frame_time_ext.extend(frame_time[t - 1])
        data_fiji1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_fiji1 = pd.DataFrame(data_fiji1)
        df_fiji2 = pd.DataFrame(roi_trace_concat, columns=roi_list)
        df_fiji = pd.concat([df_fiji1, df_fiji2], axis=1)
        return coord_fiji, df_fiji

    def roi_curation(self, ref_image, df_dFF, coord_cell, aspect_ratio, trial_curation):
        """Check each ROI spatial and temporally (for a certain trial) and choose the ones to keep.
        Enter to keep and click to discard.
        Input:
            ref_image: array with reference image
            df_dFF: dataframe with calcium trace values
            coord_cell: list with ROI coordinates
            trial_curation: (int) trial to plot"""
        rois_idx_aspectratio = list(np.where(np.array(aspect_ratio) > 1.2)[0])
        coord_ext_aspectratio = []
        for r in rois_idx_aspectratio:
            coord_ext_aspectratio.append(coord_cell[r])
        roi_idx_bad_aspectratio = np.setdiff1d(np.arange(1, len(coord_cell)), np.array(rois_idx_aspectratio))
        roi_list_bad_aspectratio = df_dFF.columns[2:][roi_idx_bad_aspectratio]
        df_dFF_aspectratio = df_dFF.drop(columns=roi_list_bad_aspectratio)
        skewness_dFF = df_dFF_aspectratio.skew(axis=0, skipna=True)
        skewness_dFF_argsort = np.argsort(np.array(skewness_dFF[2:]))
        # ROI curation (ΔF/F with the same range)
        range_dFF = [df_dFF_aspectratio.loc[df_dFF_aspectratio['trial'] == trial_curation].min(axis=0)[2:].min(),
                     df_dFF_aspectratio.loc[df_dFF_aspectratio['trial'] == trial_curation].max(axis=0)[2:].max()]
        roi_list_after_aspectratio = df_dFF_aspectratio.columns[2:]
        roi_list_sort_skewness = roi_list_after_aspectratio[skewness_dFF_argsort[::-1]]
        roi_idx_sort_skewness = skewness_dFF_argsort[::-1]
        keep_roi = []
        keep_roi_idx = []
        count_r = 0
        for r in roi_list_sort_skewness:  # check by descending order of skewness
            fig = plt.figure(figsize=(25, 15), tight_layout=True)
            gs = fig.add_gridspec(2, 3)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(np.linspace(0, 110 - 60,
                                 df_dFF_aspectratio.loc[df_dFF_aspectratio['trial'] == trial_curation].shape[0]),
                     df_dFF_aspectratio.loc[(df_dFF_aspectratio['trial'] == trial_curation), r], color='black')
            ax1.set_title('Trial ' + str(trial_curation) + ' ' + str(count_r + 1) + '/' + str(
                len(df_dFF_aspectratio.columns[2:])))
            ax1.set_ylim(range_dFF)
            ax1.set_xlabel('Time (s)')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(coord_ext_aspectratio[roi_idx_sort_skewness[count_r]][:, 0],
                        coord_ext_aspectratio[roi_idx_sort_skewness[count_r]][:, 1], s=1, color='blue', alpha=0.5)
            ax2.set_title(r, fontsize=self.fsize)
            ax2.imshow(ref_image,
                       extent=[0, np.shape(ref_image)[1] / self.pixel_to_um,
                               np.shape(ref_image)[0] / self.pixel_to_um,
                               0], cmap=plt.get_cmap('gray'))
            ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.tick_params(axis='x', labelsize=self.fsize - 4)
            ax2.tick_params(axis='y', labelsize=self.fsize - 4)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.scatter(coord_ext_aspectratio[roi_idx_sort_skewness[count_r]][:, 0],
                        coord_ext_aspectratio[roi_idx_sort_skewness[count_r]][:, 1], s=1, color='blue')
            ax3.set_title(r + ' check for discontinuities', fontsize=self.fsize)
            ax3.set_xlabel('ROI coordinates X', fontsize=self.fsize - 4)
            ax3.set_ylabel('ROI coordinates Y', fontsize=self.fsize - 4)
            ax3.tick_params(axis='x', labelsize=self.fsize - 4)
            ax3.tick_params(axis='y', labelsize=self.fsize - 4)
            bpress = plt.waitforbuttonpress()
            if bpress:
                keep_roi.append(r)
                keep_roi_idx.append(roi_idx_sort_skewness[count_r])
            plt.close('all')
            count_r += 1
        # remove bad rois
        idx_order = np.argsort([np.int64(i[3:]) for i in keep_roi])
        keep_roi_arr = np.array(keep_roi)
        roi_list_ordered = list(keep_roi_arr[idx_order])
        roi_list_ordered.insert(0, 'trial')
        roi_list_ordered.insert(0, 'time')
        df_dFF_clean = df_dFF_aspectratio[roi_list_ordered]
        coord_cell_clean = []
        for r in np.sort(keep_roi_idx):
            coord_cell_clean.append(coord_ext_aspectratio[r])
        return coord_cell_clean, df_dFF_clean

    def refine_roi_list(self, rois_names, df_dFF, df_dFF_raw, df_dFF_events, coord_cell):
        """Refine existing data structures for the following chosen ROIs
        Input:
        rois_names: list of ROIs numbers
        df_dFF
        df_dFF_raw
        df_dFF_events
        coord_cell"""
        rois_names_ordered = np.sort(rois_names)
        rois_names_ordered_complete = ['time','trial',]
        for r in rois_names_ordered:
            rois_names_ordered_complete.append('ROI'+str(r))
        df_dFF_new = df_dFF[rois_names_ordered_complete]
        df_dFF_new_raw = df_dFF_raw[rois_names_ordered_complete]
        df_dFF_events_new = df_dFF_events[rois_names_ordered_complete]
        keep_roi_idx = []
        for r in df_dFF_new.columns[2:]:
            if r in df_dFF.columns[2:]:
                keep_roi_idx.append(np.where(df_dFF.columns[2:] == r)[0][0])
        coord_cell_new = []
        for r in np.sort(keep_roi_idx):
            coord_cell_new.append(coord_cell[r])
        return df_dFF_new, df_dFF_new_raw, df_dFF_events_new, coord_cell_new, keep_roi_idx


    def isolation_distance(self, roi, trial, coord_fiji, plot_data):
        """Isolation distance metric between ROI and neuropil (Stringer, Pachitariu, 2019, Curr.Op.Neurobio.
        Computes the mahalanobis distance between ROI cluster in PCA space and 100th closest neuropil pixel
        Outputs the distance for the raw signal and the distance for the background subtracted signal
        Inputs:
            roi: (int) roi to plot
            trial: (int) trial to plot
            coord_fiji: list of ROI ccordinates"""
        tiff_stack = tiff.imread(
            self.path + 'Registered video\\T' + str(trial) + '_reg.tif')  # read tiffs
        centroid_fiji_um = self.get_roi_centroids(coord_fiji)
        centroid_fiji = np.multiply(centroid_fiji_um, self.pixel_to_um)
        cent_fiji = centroid_fiji_um[roi]
        coord_roi = np.int64(np.multiply(coord_fiji[roi - 1], self.pixel_to_um))
        xlength_fiji = np.abs(coord_fiji[roi][-1, 0] - coord_fiji[roi][0, 0])
        ylength_fiji = np.abs(coord_fiji[roi][-1, 1] - coord_fiji[roi][0, 1])
        height_fiji = np.sqrt(np.square(xlength_fiji) + np.square(ylength_fiji))
        # get F signal for all pixels in the ROI
        F_coord_roi = np.zeros((np.shape(coord_roi)[0], np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            F_coord_roi[:, f] = tiff_stack[f, coord_roi[:, 1], coord_roi[:, 0]]
        F_coord_roi_norm = F_coord_roi - np.transpose(np.tile(np.nanmean(F_coord_roi, axis=1), (np.shape(F_coord_roi)[1],1)))  # mean subtraction is essential for PCA, zscore might be bad for mahalanobnis distance
        # Find pixels of the surrounding neuropil, ordered by distance to ROI center
        dist = []
        pixel_x = []
        pixel_y = []
        for i in range(608):
            pixel_coord_i = np.where(coord_roi[:, 0] == i)[0]
            for j in range(608):
                pixel_coord_j = coord_roi[pixel_coord_i, 1]
                if j not in pixel_coord_j:  # remove pixels of the ROI
                    dist.append(np.linalg.norm(centroid_fiji[roi - 1] - np.array([i, j])))
                    pixel_x.append(i)
                    pixel_y.append(j)
        pixel_x_arr = np.array(pixel_x)
        pixel_y_arr = np.array(pixel_y)
        x_neuropil = pixel_x_arr[np.argsort(dist)[:1000]]
        y_neuropil = pixel_y_arr[np.argsort(dist)[:1000]]
        # get F signal for the nearby neuropil pixels
        F_coord_neuropil = np.zeros((len(x_neuropil), np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            F_coord_neuropil[:, f] = tiff_stack[f, y_neuropil, x_neuropil]
        F_coord_neuropil_norm = F_coord_neuropil - np.transpose(np.tile(np.nanmean(F_coord_neuropil, axis=1), (
        np.shape(F_coord_neuropil)[1],
        1)))  # mean subtraction is essential for PCA, zscore might be bad for mahalanobnis distance
        ROI_neuropil_arr = np.transpose(np.vstack((F_coord_roi_norm, F_coord_neuropil_norm)))
        # PCA of ROI and nearby neuropil pixels
        coord_id = np.zeros(np.shape(ROI_neuropil_arr)[1])
        coord_id[:np.shape(F_coord_roi)[0]] = 1
        coord_100th = np.array(pixel_x_arr[np.argsort(dist)[100]]) + np.shape(F_coord_roi)[0]
        principalComponents_2CP = PCA(n_components=2).fit_transform(np.transpose(ROI_neuropil_arr))
        # isolation distance is the mahalanobis distance of points to the cluster center,
        # get the 100th closest neuropil pixel
        centroid_roi_cluster = np.nanmean(principalComponents_2CP[:np.shape(F_coord_roi)[0], :], axis=0)
        dist_pca = np.zeros(np.shape(F_coord_roi)[0])
        for p in range(np.shape(F_coord_roi)[0]):
            dist_pca[p] = np.linalg.norm(centroid_roi_cluster - principalComponents_2CP[p, :])
        centroid_roi_cluster_idx = np.argmin(dist_pca)
        mahalanobis_dist_100thpixel = spdist.cdist(principalComponents_2CP, np.array([principalComponents_2CP[centroid_roi_cluster_idx, :]]),
                     metric='mahalanobis')[coord_100th]
        # get F signal for ROI pixels after background subtraction
        ell = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji + 15,
                         -(90 - np.degrees(np.arctan(ylength_fiji / xlength_fiji))))
        ell2 = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji + 30,
                          -(90 - np.degrees(np.arctan(ylength_fiji / xlength_fiji))))
        ellpath = ell.get_path()
        vertices = ellpath.vertices.copy()
        coord_ell_inner = ell.get_patch_transform().transform(vertices)
        ellpath2 = ell2.get_path()
        vertices2 = ellpath2.vertices.copy()
        coord_ell_outer = ell2.get_patch_transform().transform(vertices2)
        ROIinner_fill_x, ROIinner_fill_y = zip(*self.render(np.int64(coord_ell_inner * self.pixel_to_um)))
        ROIouter_fill_x, ROIouter_fill_y = zip(*self.render(np.int64(coord_ell_outer * self.pixel_to_um)))
        ROIinner_fill_coord = np.transpose(np.vstack((ROIinner_fill_x, ROIinner_fill_y)))
        ROIouter_fill_coord = np.transpose(np.vstack((ROIouter_fill_x, ROIouter_fill_y)))
        idx_overlap_outer = []
        for x in range(np.shape(ROIinner_fill_coord)[0]):
            if ROIinner_fill_coord[x, 0] in ROIouter_fill_coord[:, 0]:
                idx_overlap = np.where((ROIouter_fill_coord[:, 0] == ROIinner_fill_coord[x, 0]) & (
                        ROIouter_fill_coord[:, 1] == ROIinner_fill_coord[x, 1]))[0]
                if len(idx_overlap) > 0:
                    idx_overlap_outer.append(idx_overlap[0])
        idx_nonoverlap = np.setdiff1d(range(np.shape(ROIouter_fill_coord)[0]), idx_overlap_outer)
        ROIdonut_coord = np.transpose(
            np.vstack((ROIouter_fill_coord[idx_nonoverlap, 0], ROIouter_fill_coord[idx_nonoverlap, 1])))
        F_coord_donut = np.zeros((np.shape(ROIdonut_coord)[0], np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            F_coord_donut[:, f] = tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]]
        F_coord_bg = F_coord_roi - np.nanmean(F_coord_donut, axis=0)
        F_coord_bg_norm = F_coord_bg - np.transpose(np.tile(np.nanmean(F_coord_bg, axis=1), (np.shape(F_coord_bg)[1],1)))  # mean subtraction is essential for PCA, zscore might be bad for mahalanobnis distance
        # PCA of bg sub ROI and nearby neuropil pixels
        ROIbg_neuropil_arr = np.transpose(np.vstack((F_coord_bg_norm, F_coord_neuropil_norm)))
        principalComponents_2CP_bg = PCA(n_components=2).fit_transform(np.transpose(ROIbg_neuropil_arr))
        # isolation distance is the mahalanobis distance of points to the cluster center,
        # get the 100th closest neuropil pixel
        centroid_roi_cluster_bg = np.nanmean(principalComponents_2CP[:np.shape(F_coord_roi)[0], :], axis=0)
        dist_pca_bg = np.zeros(np.shape(F_coord_roi)[0])
        for p in range(np.shape(F_coord_roi)[0]):
            dist_pca_bg[p] = np.linalg.norm(centroid_roi_cluster_bg - principalComponents_2CP_bg[p, :])
        centroid_roi_cluster_idx_bg = np.argmin(dist_pca_bg)
        mahalanobis_dist_100thpixel_bg = spdist.cdist(principalComponents_2CP_bg, np.array([principalComponents_2CP_bg[centroid_roi_cluster_idx_bg, :]]),
                     metric='mahalanobis')[coord_100th]
        if plot_data:
            fig = plt.figure(figsize=(15, 10), tight_layout=True)
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(np.nanmean(F_coord_neuropil_norm, axis=0), color='red', label='neuropil')
            ax1.plot(np.nanmean(F_coord_roi_norm, axis=0), color='blue', label='ROI')
            ax1.plot(np.nanmean(F_coord_bg_norm, axis=0), color='green', label='ROI bg')
            ax1.legend(frameon=False)
            ax1.set_xlabel('Frames', fontsize=14)
            ax1.set_ylabel('Fluorescence', fontsize=14)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(principalComponents_2CP[:, 0], principalComponents_2CP[:, 1], s=15, c=coord_id)
            ax2.scatter(principalComponents_2CP[coord_100th, 0], principalComponents_2CP[coord_100th, 1], s=40,
                        color='red')
            ax2.set_title('PCA space on ROI and neuropil signals ' + str(np.round(mahalanobis_dist_100thpixel[0], 2)))
            ax2.set_xlabel('PC1', fontsize=14)
            ax2.set_ylabel('PC2', fontsize=14)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2 = fig.add_subplot(gs[1, 1])
            ax2.scatter(principalComponents_2CP_bg[:, 0], principalComponents_2CP_bg[:, 1], s=15, c=coord_id)
            ax2.scatter(principalComponents_2CP_bg[coord_100th, 0], principalComponents_2CP_bg[coord_100th, 1], s=40,
                        color='red')
            ax2.set_title(
                'PCA space on ROI bg and neuropil signals ' + str(np.round(mahalanobis_dist_100thpixel_bg[0], 2)))
            ax2.set_xlabel('PC1', fontsize=14)
            ax2.set_ylabel('PC2', fontsize=14)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
        return [mahalanobis_dist_100thpixel, mahalanobis_dist_100thpixel_bg]

    def get_miniscope_frame_time(self, trials, frames_dFF, version):
        """From timestamp.dat file compute time of acquisiton of each frame for all trials in a session
        Inputs:
            trials: list with trials in a session
            frames_dFF: list with the number of frames deleted at the beginning of each trial
            trial_start: list of time elapsed between miniscope start and bcam start
            version: if v3.2 or v4 because timestamps were saved differently"""
        path_split = self.path.split(self.delim)
        if version == 'v3.2':
            path_timestamps = self.delim.join(path_split[:-2]) + self.delim + 'Miniscopes' + self.delim
            columns_to_keep = ['camNum', 'frameNum', 'sysClock', 'buffer']
        if version == 'v4':
            path_timestamps = self.path
            columns_to_keep = ['Frame Number', 'Time Stamp (ms)', 'Buffer Index']
        frame_time = []
        for t in range(len(trials)):
            if version == 'v3.2':
                df = pd.read_table(path_timestamps + 'T' + str(trials[t]) + self.delim + "timestamp.dat", sep="\s+",
                                   usecols=columns_to_keep)
                sysClock = np.array(df['sysClock'])
            if version == 'v4':
                df = pd.read_table(
                    path_timestamps + 'T' + str(trials[t]) + self.delim + "Miniscope" + self.delim + "timeStamps.csv",
                    sep=",", usecols=columns_to_keep)
                sysClock = np.array(df['Time Stamp (ms)'])
            # first sysclock has to be 0
            sysClock[0] = 0  # to mark time 0
            sysClock[1:] = sysClock[1:]
            sysClock_clean = sysClock[frames_dFF[t] - 1:] / 1000  # -1 because frame index starts at 0
            frame_time.append(
                sysClock_clean - sysClock_clean[0])  # get time of each frame to align with behavior recording
        return frame_time

    def get_black_frames(self):
        """Get the number of black frames per tiff video that had to be removed.
        Frames only removed from the beginning"""
        tiflist = glob.glob(self.path + self.delim + 'Suite2p' + self.delim + '*.tif')
        tiff_boolean = 0
        if not tiflist:
            tiflist = glob.glob(self.path + self.delim + 'Suite2p' + self.delim + '*.tiff')  # get list of tifs
            tiff_boolean = 1
        trial_id = []
        for t in range(len(tiflist)):
            tifname = tiflist[t].split(self.delim)
            tifname_split = tifname[-1].split('_')
            if tiff_boolean:
                trial_id.append(int(tifname_split[0][1:]))  # get trial order in that list
            else:
                idx_dot = tifname_split[0].find('.')
                if idx_dot == -1:
                    trial_id.append(int(tifname_split[0][1:]))
                else:
                    trial_id.append(int(tifname_split[0][1:idx_dot]))
        trial_order = np.sort(trial_id)  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(tiflist)):
            tr_ind = np.where(trial_order[f] == trial_id)[0][0]
            files_ordered.append(tiflist[tr_ind])
        frames_dFF = []
        for f in files_ordered:
            tifname = f.split(self.delim)
            tifname_split = tifname[-1].split('_')
            idx_dot = tifname_split[1].find('.')
            if idx_dot < 0:
                frames_dFF.append(int(tifname_split[1][4:]))
            else:
                frames_dFF.append(int(tifname_split[1][4:idx_dot]))
        return frames_dFF

    def compute_head_angles(self, trials):
        """From the quartenion output of Miniscope compute euler angles for each trial.
        Output of the data is a dataframe with roll, yaw, pitch angles and respective timestamps.
        Inputs:
            trials: list of trials"""
        columns_to_keep = ['Time Stamp (ms)', 'qw', 'qx', 'qy', 'qz']
        roll_list = []
        pitch_list = []
        yaw_list = []
        time_list = []
        trial_list = []
        for t in range(len(trials)):
            df = pd.read_table(
                self.path + 'T' + str(trials[t]) + self.delim + "Miniscope" + self.delim + "headOrientation.csv",
                sep=",", usecols=columns_to_keep)
            timestamps = np.array(df['Time Stamp (ms)'])
            [roll_x, pitch_y, yaw_z] = miniscope_session.euler_from_quaternion(np.array(df['qx']), np.array(df['qy']),
                                                                               np.array(df['qz']), np.array(df['qw']))
            time_list.extend(np.insert(timestamps[1:] / 1000, 0, 0))
            roll_list.extend(roll_x)
            pitch_list.extend(pitch_y)
            yaw_list.extend(yaw_z)
            trial_list.extend(np.ones((len(roll_x))) * (t + 1))
        dict_ori = {'roll': roll_list, 'pitch': pitch_list, 'yaw': yaw_list, 'time': time_list, 'trial': trial_list}
        head_orientation = pd.DataFrame(dict_ori)  # create dataframe with dFF, roi id and trial id
        return head_orientation

    def compute_head_angles_quartenions(self, trials):
        """Compute the quartenion output of Miniscope.
        Output of the data is a dataframe with roll, yaw, pitch angles and respective timestamps.
        Inputs:
            trials: list of trials"""
        columns_to_keep = ['Time Stamp (ms)', 'qw', 'qx', 'qy', 'qz']
        roll_list = []
        pitch_list = []
        yaw_list = []
        time_list = []
        trial_list = []
        for t in range(len(trials)):
            df = pd.read_table(
                self.path + 'T' + str(trials[t]) + self.delim + "Miniscope" + self.delim + "headOrientation.csv",
                sep=",", usecols=columns_to_keep)
            timestamps = np.array(df['Time Stamp (ms)'])
            [roll_x, pitch_y, yaw_z] = miniscope_session.euler_from_quaternion(np.array(df['qx']), np.array(df['qy']),
                                                                               np.array(df['qz']), np.array(df['qw']))
            time_list.extend(np.insert(timestamps[1:] / 1000, 0, 0))
            roll_list.extend(roll_x)
            pitch_list.extend(pitch_y)
            yaw_list.extend(yaw_z)
            trial_list.extend(np.ones((len(roll_x))) * (t + 1))
        dict_ori = {'roll': roll_list, 'pitch': pitch_list, 'yaw': yaw_list, 'time': time_list, 'trial': trial_list}
        head_orientation = pd.DataFrame(dict_ori)  # create dataframe with dFF, roi id and trial id
        return head_orientation

    def pca_head_angles(self, head_angles, trials, plot_data):
        """Function to compute the PCA manifold of head angle data for a single animal.
        Outputs PCA space and trial id for each point in the manifold
        Inputs:
            head_angles: dataframe with head angles info
            trials: arrays with trials in session
            plot_data: boolean"""
        head_angles_df = head_angles.iloc[:, :3]
        headangles_array = np.array(head_angles_df)
        headangles_array_clean = np.delete(headangles_array, np.where(np.isnan(headangles_array))[0], axis=0)
        headangles_trial = np.array(head_angles['trial'])
        trial_clean = np.delete(headangles_trial, np.where(np.isnan(headangles_array))[0], axis=0)
        pca = PCA(n_components=3)
        principalComponents_3CP = pca.fit_transform(headangles_array_clean)
        if plot_data:
            # Number of components and variance
            fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
            plt.plot(np.arange(1, 4), np.cumsum(pca.explained_variance_ratio_), color='black')
            plt.scatter(3, np.cumsum(pca.explained_variance_ratio_)[2], color='red')
            ax.set_xlabel('PCA components', fontsize=14)
            ax.set_ylabel('Explained variance', fontsize=14)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # Plot 2d
            cmap = plt.get_cmap('viridis', len(trials))
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
            h = plt.scatter(principalComponents_3CP[:, 0], principalComponents_3CP[:, 1], s=1, c=trial_clean, cmap=cmap,
                            vmin=trials[0], vmax=trials[-1])
            ax.set_title('First 2 PCs - explained variance of ' + str(
                np.round(np.cumsum(pca.explained_variance_ratio_)[1], decimals=3)), fontsize=24)
            ax.set_xlabel('PC component 1', fontsize=20)
            ax.set_ylabel('PC component 2', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # ax.set_xlim([-1,1])
            cb = plt.colorbar(h, ticks=trials)
            cb.ax.tick_params(labelsize='large')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if plot_data:
                if not os.path.exists(self.path + 'images'):
                    os.mkdir(self.path + 'images')
                if not os.path.exists(self.path + 'images' + self.delim + 'acc' + self.delim):
                    if self.delim == '/':
                        os.mkdir(self.path + 'images/acc/')
                    else:
                        os.mkdir(self.path + 'images\\acc\\')
                if self.delim == '/':
                    plt.savefig(self.path + 'images/acc/' + 'pca_2d', dpi=400)
                else:
                    plt.savefig(self.path + 'images\\acc\\' + 'pca_2d', dpi=400)
        return principalComponents_3CP, trial_clean

    def plot_rois_ref_image(self, ref_image, coord_cell, print_plots):
        """Plot ROIs on top of reference image.
        Inputs:
            ref_image: reference image
            coord_cell: coordinates for each ROI
            print_plots (boolean)"""
        plt.figure(figsize=(10, 10), tight_layout=True)
        for r in range(len(coord_cell)):
            plt.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], s=1, alpha=0.6)
        plt.imshow(ref_image, cmap='gray',
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um, 0])
        plt.title('ROIs grouped by activity', fontsize=self.fsize)
        plt.xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if print_plots:
            if not os.path.exists(self.path + '\\images\\'):
                os.mkdir(self.path + '\\images\\')
            delim = self.path[-1]
            if delim == '/':
                plt.savefig(self.path + 'images/' + 'rois_fov', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\' + 'rois_fov', dpi=self.my_dpi)
        return

    def plot_heatmap_baseline(self, df_dFF, plot_data):
        """Plots the heatmap for all ROIs given the their traces (min-max normalized).
        Plots the first 6 trials - baseline trials or fully tied, depending on the session
        Input:
            df_dFF: dataframe with traces
            plot_data: boolean"""
        trials = np.unique(df_dFF['trial'])
        fig, ax = plt.subplots(2, 3, figsize=(25, 12), tight_layout=True)
        ax = ax.ravel()
        for t in trials[:6]:
            sns.heatmap(np.transpose(df_dFF[df_dFF['trial'] == t].iloc[:, 2:]), cmap='coolwarm',
                        ax=ax[t - 1], cbar=False)
            ax[t - 1].set_title('Trial ' + str(t), fontsize=self.fsize - 4)
            ax[t - 1].set_xticks(np.linspace(0, len(df_dFF[df_dFF['trial'] == t].iloc[:, 1]), num=15))
            ax[t - 1].set_xticklabels(list(map(str, np.round(
                np.linspace(0, df_dFF[df_dFF['trial'] == t].iloc[-1, 1], num=15), 1))),
                                      fontsize=self.fsize - 4)
            ax[t - 1].set_yticks(np.arange(0, len(df_dFF.columns[2:]), 2))
            ax[t - 1].set_yticklabels(df_dFF.columns[2::2], fontsize=self.fsize - 4)
            ax[t - 1].set_xlabel('Time (s)', fontsize=self.fsize - 4)
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path + '/images/' + 'heatmap_1st_6trials_minmax_traces_allROIs',
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\' + 'heatmap_1st_6trials_minmax_traces_allROIs',
                            dpi=self.my_dpi)
        return

    def plot_stacked_traces(self, frame_time, df_dFF, trials_plot, print_plots):
        """"Funtion to compute stacked traces for a single trial or for the transition trials in the session.
        Input:
        frame_time: list with mscope timestamps
        df_dFF: dataframe with calcium trace
        trials_plot: int or list
        print_plots: boolean"""
        if isinstance(trials_plot,np.ndarray):
            count_t = 0
            fig, ax = plt.subplots(2, 2, figsize=(25, 30), tight_layout=True)
            ax = ax.ravel()
            for t in trials_plot:
                dFF_trial = df_dFF.loc[df_dFF['trial'] == t]  # get dFF for the desired trial
                count_r = 0
                for r in df_dFF.columns[2:]:
                    ax[count_t].plot(frame_time[t - 1], dFF_trial[r] + (count_r / 2), color='black')
                    count_r += 1
                ax[count_t].set_xlabel('Time (s)', fontsize=self.fsize - 4)
                ax[count_t].set_ylabel('Calcium trace for trial ' + str(t), fontsize=self.fsize - 4)
                plt.xticks(fontsize=self.fsize - 4)
                plt.yticks(fontsize=self.fsize - 4)
                plt.setp(ax[count_t].get_yticklabels(), visible=False)
                ax[count_t].tick_params(axis='y', which='y', length=0)
                ax[count_t].spines['right'].set_visible(False)
                ax[count_t].spines['top'].set_visible(False)
                ax[count_t].spines['left'].set_visible(False)
                plt.tick_params(axis='y', labelsize=0, length=0)
                count_t += 1
        else:
            dFF_trial = df_dFF.loc[df_dFF['trial'] == trials_plot]  # get dFF for the desired trial
            fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
            count_r = 0
            for r in df_dFF.columns[2:]:
                plt.plot(frame_time[trials_plot - 1], dFF_trial[r] + (count_r / 2), color='black')
                count_r += 1
            ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
            ax.set_ylabel('Calcium trace for trial ' + str(trials_plot), fontsize=self.fsize - 4)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='y', length=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tick_params(axis='y', labelsize=0, length=0)
        if print_plots:
            if not os.path.exists(self.path + 'images'):
                os.mkdir(self.path + 'images')
            if self.delim == '/':
                plt.savefig(self.path + 'images/' + 'dFF_stacked_traces', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\' + 'dFF_stacked_traces', dpi=self.my_dpi)

    def plot_stacked_traces_singleROI(self, frame_time, df_dFF, roi_plot, session_type, trials, print_plots):
        """"Funtion to compute stacked traces for all trials in a session for a single ROI.
        Input:
        frame_time: list with mscope timestamps
        df_dFF: dataframe with calcium trace
        roi_plot: int or list
        print_plots: boolean"""
        greys = mp.cm.get_cmap('Greys', 14)
        reds = mp.cm.get_cmap('Reds', 23)
        blues = mp.cm.get_cmap('Blues', 23)
        oranges = mp.cm.get_cmap('Oranges', 23)
        purples = mp.cm.get_cmap('Purples', 23)
        if session_type=='tied':
            if len(trials) == 6:
                colors_session = [greys(4), greys(7), greys(12), oranges(7), oranges(13), oranges(23)]
            if len(trials) == 12:
                colors_session = [greys(4), greys(7), greys(12), oranges(7), oranges(13), oranges(23), purples(7), purples(13), purples(23)]
            if len(trials) == 18:
                colors_session = [greys(4), greys(6), greys(8), greys(10), greys(12), greys(14),
                                  oranges(6), oranges(10), oranges(13), oranges(16), oranges(19), oranges(23),
                                  purples(6), purples(10), purples(13), purples(16), purples(19), purples(23)]
        if session_type == 'split':
            if len(trials) == 23:
                colors_session = [greys(4), greys(7), greys(12), reds(5), reds(7), reds(9), reds(11), reds(13), reds(15),
                              reds(17), reds(19), reds(21), reds(23), blues(5), blues(7), blues(9), blues(11),
                              blues(13),
                              blues(15), blues(17), blues(19), blues(21), blues(23)]
            if len(trials) == 26:
                colors_session = [greys(4), greys(6), greys(8), greys(10), greys(12), greys(14), reds(5), reds(7), reds(9), reds(11), reds(13), reds(15),
                              reds(17), reds(19), reds(21), reds(23), blues(5), blues(7), blues(9), blues(11),
                              blues(13),
                              blues(15), blues(17), blues(19), blues(21), blues(23)]
        fig, ax = plt.subplots(figsize=(15, 30),  tight_layout=True)
        count_t = 0
        for t in trials:
            dFF_trial = df_dFF.loc[df_dFF['trial'] == t,'ROI'+str(roi_plot)]  # get dFF for the desired trial
            ax.plot(frame_time[t - 1], dFF_trial + count_t, color=colors_session[count_t])
            ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
            ax.set_ylabel('Calcium trace for ROI ' + str(roi_plot), fontsize=self.fsize - 4)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='y', length=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tick_params(axis='y', labelsize=0, length=0)
            count_t += 1
        if print_plots:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'dFF_stacked_traces', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'dFF_stacked_traces', dpi=self.my_dpi)
        return

    def compute_roi_clustering(self, df_dFF, centroid_cell, distance_neurons, trial, th_cluster, colormap_cluster, plot_data):
        """Function to get colors of ROIs according to its cluster id
        Input:
        data_arr: dataframe with calcium trace
        centroid_cell: list with ROIs centroids
        distance_neurons: matrix with distance from the first ROI
        trial: (int) trial to compute roi clustering
        th_cluster: float, threshold for clustering
        colormap: (str) with colormap name
        plot_data: boolean"""
        data = np.transpose(np.array(df_dFF.loc[df_dFF['trial'] == trial].iloc[:, 2:]))
        xaxis_crosscorr = np.linspace(-len(data[0, :]), len(data[0, :]), len(data[0, :]) * 2 - 1)
        roi_nr = np.shape(data)[0]
        roi_list = np.arange(1, np.shape(data)[0] + 1)
        p_corrcoef = np.zeros((roi_nr, roi_nr))
        p_corrcoef[:] = np.nan
        for roi1 in roi_list:
            for roi2 in roi_list:
                idx_nonan = np.where(~np.isnan(data[0, :]))[0]
                cross_corr = correlate(self.z_score(data[roi1 - 1, idx_nonan], 1),
                                       self.z_score(data[roi2 - 1, idx_nonan], 1), mode='full',
                                       method='direct')
                p_corrcoef[roi1 - 1, roi2 - 1] = np.cov(data[roi1 - 1, idx_nonan], data[roi2 - 1, idx_nonan])[0][
                                                     1] / np.sqrt(
                    np.var(data[roi1 - 1, idx_nonan]) * np.var(data[roi2 - 1, idx_nonan]))
                if p_corrcoef[roi1 - 1, roi2 - 1] > 1:
                    p_corrcoef[roi1 - 1, roi2 - 1] = 1
        pdist = spc.distance.pdist(p_corrcoef)  # distance between correlation matrix
        linkage = spc.linkage(pdist, method='complete')  # hierarchical clustering
        idx = spc.fcluster(linkage, th_cluster * pdist.max(), 'distance')  # clustering of linkage output
        nr_clusters = np.unique(idx)
        cmap = plt.get_cmap(colormap_cluster)
        colors = [cmap(i) for i in np.linspace(0, 1, int(np.floor(len(nr_clusters) + 1)))]
        if plot_data:
            # Plot clustering dendrogram and ordered correlation matrix (ordered by ROI distance)
            fig, ax = plt.subplots(figsize=(20, 7), tight_layout=True)
            spc.dendrogram(linkage)
            plt.axhline(y=th_cluster * pdist.max(), color='black')
            ax.set_title('Dendrogram for hierarchical clustering', fontsize=24)
            ax.set_xlabel('ROI #', fontsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=14)
            if not os.path.exists(self.path + '\\images\\cluster\\'):
                os.mkdir(self.path + '\\images\\cluster\\')
            if self.delim == '/':
                plt.savefig(self.path + '/images/cluster/' + 'dendrogram', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\cluster\\' + 'dendrogram', dpi=self.my_dpi)
            furthest_neuron = np.argmax(np.array(centroid_cell)[:, 0])
            neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
            p_corrcoef_ordered = p_corrcoef[neuron_order, :][:, neuron_order]
            fig, ax = plt.subplots()
            with sns.axes_style("white"):
                sns.heatmap(p_corrcoef_ordered, cmap="YlGnBu", linewidth=0.5)
                ax.set_title('correlation matrix ordered by distance between ROIs')
            if not os.path.exists(self.path + '\\images\\cluster\\'):
                os.mkdir(self.path + '\\images\\cluster\\')
            if self.delim == '/':
                plt.savefig(self.path + 'images/cluster/' + 'pcorrcoef_ordered', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\cluster\\' + 'pcorrcoef_ordered', dpi=self.my_dpi)
        return colors, idx

    def plot_roi_clustering_spatial(self, ref_image, colors, idx, coord_cell, print_plots):
        """Plot ROIs on top of reference image color coded by the result of the hierarchical clustering
        Inputs:
            ref_image: reference image
            colors: colors for each cluster
            idx: to which cluster each ROI belongs to
            coord_cell: coordinates for each ROI
            print_plots (boolean)"""
        plt.figure(figsize=(10, 10), tight_layout=True)
        for r in range(len(coord_cell)):
            plt.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], color=colors[idx[r] - 1], s=1, alpha=0.6)
        plt.imshow(ref_image, cmap='gray',
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um, 0])
        plt.title('ROIs grouped by activity', fontsize=self.fsize)
        plt.xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if print_plots:
            if not os.path.exists(self.path + '\\images\\cluster\\'):
                os.mkdir(self.path + '\\images\\cluster\\')
            delim = self.path[-1]
            if delim == '/':
                plt.savefig(self.path + 'images/cluster/' + 'roi_clustering_fov', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\cluster\\' + 'roi_clustering_fov', dpi=self.my_dpi)
        return

    def plot_roi_clustering_temporal(self, df_dFF, frame_time, centroid_cell, distance_neurons, trial_plot, colors, idx, print_plots):
        """Plot ROIs on top of reference image color coded by the result of the hierarchical clustering
        Ordered by distance between ROIs.
        Inputs:
            df_dFF: dataframe with calcium trace
            frame_time: list of mscope timestamps
            trial_plot: (int) trial to plot
            colors: colors for each cluster
            idx: to which cluster each ROI belongs to
            print_plots (boolean)"""
        df_dFF_norm = self.norm_traces(df_dFF,'min_max')
        furthest_neuron = np.argmax(np.array(centroid_cell)[:, 0])
        neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
        roi_list = df_dFF_norm.columns[2:]
        roi_list_ordered = roi_list[neuron_order]
        dFF_trial = df_dFF_norm.loc[df_dFF_norm['trial'] == trial_plot]  # get dFF for the desired trial
        fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
        count_r = 0
        for r in roi_list_ordered:
            plt.plot(frame_time[trial_plot - 1], dFF_trial[r] + count_r, color=colors[idx[count_r] - 1])
            count_r += 1
        ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', which='y', length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=0, length=0)
        if print_plots:
            if not os.path.exists(self.path + '\\images\\cluster\\'):
                os.mkdir(self.path + '\\images\\cluster\\')
            delim = self.path[-1]
            if delim == '/':
                plt.savefig(self.path + 'images/cluster/' + 'roi_clustering_trace', dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\cluster\\' + 'roi_clustering_trace', dpi=self.my_dpi)
        return

    def compute_bg_roi_fiji(self, coord_cell, trials, frame_time, df_dFF, coeff_sub):
        """Function to compute a donut background around a determined FIJI ROI and compute its background subtracted signal.
        Input:
            coord_cell: list with ROIs coordinates
            trials: array with all the trials in the session
            frame_time: list with all the trial timestamps from the mscope
            df_dFF: dataframe with calcium traces
            coeff_sub: (float 0-1) coefficient for background subtraction"""
        height_fiji = []
        xlength_fiji = []
        ylength_fiji = []
        for r in range(len(coord_cell)):
            x_length = np.abs(coord_cell[r][-1, 0] - coord_cell[r][0, 0])
            y_length = np.abs(coord_cell[r][-1, 1] - coord_cell[r][0, 1])
            xlength_fiji.append(x_length)
            ylength_fiji.append(y_length)
            height_fiji.append(np.sqrt(np.square(x_length) + np.square(y_length)))
        donut_trace_all_list = []
        for rfiji in range(len(coord_cell)):
            cent_fiji = np.nanmean(coord_cell[rfiji], axis=0)
            ell = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji] + 15,
                             -(90 - np.degrees(np.arctan(ylength_fiji[rfiji] / xlength_fiji[rfiji]))))
            ell2 = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji] + 30,
                              -(90 - np.degrees(np.arctan(ylength_fiji[rfiji] / xlength_fiji[rfiji]))))
            ellpath = ell.get_path()
            vertices = ellpath.vertices.copy()
            coord_ell_inner = ell.get_patch_transform().transform(vertices)
            ellpath2 = ell2.get_path()
            vertices2 = ellpath2.vertices.copy()
            coord_ell_outer = ell2.get_patch_transform().transform(vertices2)
            ROIinner_fill_x, ROIinner_fill_y = zip(*self.render(np.int64(coord_ell_inner * self.pixel_to_um)))
            ROIouter_fill_x, ROIouter_fill_y = zip(*self.render(np.int64(coord_ell_outer * self.pixel_to_um)))
            ROIinner_fill_coord = np.transpose(np.vstack((ROIinner_fill_x, ROIinner_fill_y)))
            ROIouter_fill_coord = np.transpose(np.vstack((ROIouter_fill_x, ROIouter_fill_y)))
            idx_overlap_outer = []
            for x in range(np.shape(ROIinner_fill_coord)[0]):
                if ROIinner_fill_coord[x, 0] in ROIouter_fill_coord[:, 0]:
                    idx_overlap = np.where((ROIouter_fill_coord[:, 0] == ROIinner_fill_coord[x, 0]) & (
                            ROIouter_fill_coord[:, 1] == ROIinner_fill_coord[x, 1]))[0]
                    if len(idx_overlap) > 0:
                        idx_overlap_outer.append(idx_overlap[0])
            idx_nonoverlap = np.setdiff1d(range(np.shape(ROIouter_fill_coord)[0]), idx_overlap_outer)
            ROIdonut_coord = np.transpose(
                np.vstack((ROIouter_fill_coord[idx_nonoverlap, 0], ROIouter_fill_coord[idx_nonoverlap, 1])))
            donut_trace_trials = []
            for t in trials:
                tiff_stack = tiff.imread(self.path + 'Registered video\\T' + str(t) + '_reg.tif')  # read tiffs
                donut_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
                for f in range(np.shape(tiff_stack)[0]):
                    donut_trace_tiffmean[f] = np.nansum(tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]])/np.shape(ROIdonut_coord)[0]
                donut_trace_trials.append(donut_trace_tiffmean)
            donut_trace_concat = np.hstack(donut_trace_trials)
            donut_trace_all_list.append(donut_trace_concat)
        donut_trace_arr = np.transpose(np.vstack(donut_trace_all_list))
        roi_trace_bgsub_arr = np.array(df_dFF.iloc[:, 2:]) - (coeff_sub * donut_trace_arr)
        idx_neg = np.where(roi_trace_bgsub_arr < 0)
        roi_trace_bgsub_arr[idx_neg[0], idx_neg[1]] = 0
        # trace as dataframe
        roi_list = []
        for r in range(len(coord_cell)):
            roi_list.append('ROI'+str(r+1))
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t-1])))
            frame_time_ext.extend(frame_time[t - 1])
        data_fiji1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_fiji1 = pd.DataFrame(data_fiji1)
        df_fiji2 = pd.DataFrame(roi_trace_bgsub_arr, columns=roi_list)
        df_fiji = pd.concat([df_fiji1, df_fiji2], axis=1)
        return [df_fiji, roi_trace_bgsub_arr]

    def save_processed_files(self, df_extract, trials, df_events_extract, df_extract_rawtrace, coord_ext, th, amp_arr, idx_to_nan):
        """Saves calcium traces, ROI coordinates, trial number and motion frames.
        Saves them under path/processed files
        Inputs:
            df_extract: dataframe with calcium trace from EXTRACT
            trials: array with list of recorded trials
            df_events_extract: dataframe with the events using JR method on calcium trace from EXTRACT
            coord_ext: ROI coordinates from EXTRACT
            th: threshold to discard frames for poor correlation with ref image
            amp_arr: array with the amplitudes used for event detection for each ROI and trial
            idx_to_nan: indices of frames to dismiss while processing (too much motion)"""
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        if self.delim == '/':
            df_extract.to_csv(self.path + '/processed files/' + 'df_extract.csv', sep=',', index = False)
            df_events_extract.to_csv(self.path + '/processed files/' + 'df_events_extract.csv', sep=',', index=False)
            df_extract_rawtrace.to_csv(self.path + '/processed files/' + 'df_extract_raw.csv', sep=',', index=False)
            np.save(self.path + '/processed files/' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '/processed files/' + 'trials.npy', trials)
            np.save(self.path + '/processed files/' + 'reg_th.npy', th)
            np.save(self.path + '/processed files/' + 'amplitude_events.npy', amp_arr)
            np.save(self.path + '/processed files/' + 'frames_to_exclude.npy', idx_to_nan)
        else:
            df_extract.to_csv(self.path + '\\processed files\\' + 'df_extract.csv', sep=',', index = False)
            df_events_extract.to_csv(self.path + '\\processed files\\' + 'df_events_extract.csv', sep=',', index=False)
            df_extract_rawtrace.to_csv(self.path + '\\processed files\\' + 'df_extract_raw.csv', sep=',', index=False)
            np.save(self.path + '\\processed files\\' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '\\processed files\\' + 'trials.npy', trials)
            np.save(self.path + '\\processed files\\' + 'reg_th.npy', th)
            np.save(self.path + '\\processed files\\' + 'amplitude_events.npy', amp_arr)
            np.save(self.path + '\\processed files\\' + 'frames_to_exclude.npy', idx_to_nan)
        return

    def load_processed_files(self):
        """Loads processed files that were saved under path/processed files"""
        if self.delim == '/':
            df_extract = pd.read_csv(self.path + '/processed files/' + 'df_extract.csv')
            df_events_extract = pd.read_csv(self.path + '/processed files/' + 'df_events_extract.csv')
            df_extract_rawtrace = pd.read_csv(self.path + '/processed files/' + 'df_extract_raw.csv')
            coord_ext = np.load(self.path + '/processed files/' + 'coord_ext.npy', allow_pickle=True)
            trials = np.load(self.path + '/processed files/' + 'trials.npy')
            reg_th = np.load(self.path + '/processed files/' + 'reg_th.npy')
            amp_arr = np.load(self.path + '/processed files/' + 'amplitude_events.npy')
            reg_bad_frames = np.load(self.path + '/processed files/' + 'frames_to_exclude.npy')
        else:
            df_extract = pd.read_csv(self.path + '\\processed files\\' + 'df_extract.csv')
            df_events_extract = pd.read_csv(self.path + '\\processed files\\' + 'df_events_extract.csv')
            df_extract_rawtrace = pd.read_csv(self.path + '\\processed files\\' + 'df_extract_raw.csv')
            coord_ext = np.load(self.path + '\\processed files\\' + 'coord_ext.npy', allow_pickle=True)
            trials = np.load(self.path + '\\processed files\\' + 'trials.npy')
            reg_th = np.load(self.path + '\\processed files\\' + 'reg_th.npy')
            amp_arr = np.load(self.path + '\\processed files\\' + 'amplitude_events.npy')
            reg_bad_frames = np.load(self.path + '\\processed files\\' + 'frames_to_exclude.npy')
        return df_extract, df_events_extract, df_extract_rawtrace, trials, coord_ext, reg_th, amp_arr, reg_bad_frames

    def save_processed_files_ext_fiji(self, df_fiji, df_trace_bgsub, df_extract, df_events_all, df_events_unsync, trials, coord_fiji, coord_ext, th, idx_to_nan):
        """Saves calcium traces, ROI coordinates, trial number and motion frames.
        Saves them under path/processed files
        Inputs:
            df_dFF: dataframe with calcium trace from ImageJ
            df_trace_bgsub: dataframe with calcium trace from ImageJ
            df_extract: dataframe with calcium trace from EXTRACT
            df_events_all: dataframe with the events using JR method on Fiji raw calcium trace
            df_events_unsync: dataframe with the events using JR method on Fiji bgsub calcium trace
            trials: array with list of recorded trials
            coord_fiji: ROI coordinates from ImageJ
            coord_ext: ROI coordinates from EXTRACT
            idx_to_nan: indices of frames to dismiss while processing (too much motion)
            th: threshold to discard frames for poor correlation with ref image"""
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        if self.delim == '/':
            df_fiji.to_csv(self.path + '/processed files/' + 'df_fiji.csv', sep=',', index = False)
            df_trace_bgsub.to_csv(self.path + '/processed files/' + 'df_fiji_bgsub.csv', sep=',', index = False)
            df_extract.to_csv(self.path + '/processed files/' + 'df_extract.csv', sep=',', index = False)
            df_events_all.to_csv(self.path + '/processed files/' + 'df_events_all.csv', sep=',', index=False)
            df_events_unsync.to_csv(self.path + '/processed files/' + 'df_events_unsync.csv', sep=',', index=False)
            np.save(self.path + '/processed files/' + 'coord_fiji.npy', coord_fiji, allow_pickle = True)
            np.save(self.path + '/processed files/' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '/processed files/' + 'trials.npy', trials)
            np.save(self.path + '/processed files/' + 'reg_th.npy', th)
            np.save(self.path + '/processed files/' + 'frames_to_exclude.npy', idx_to_nan)
        else:
            df_fiji.to_csv(self.path + '\\processed files\\' + 'df_fiji.csv', sep=',', index = False)
            df_trace_bgsub.to_csv(self.path + '\\processed files\\' + 'df_fiji_bgsub.csv', sep=',', index = False)
            df_extract.to_csv(self.path + '\\processed files\\' + 'df_extract.csv', sep=',', index = False)
            df_events_all.to_csv(self.path + '\\processed files\\' + 'df_events_all.csv', sep=',', index=False)
            df_events_unsync.to_csv(self.path + '\\processed files\\' + 'df_events_unsync.csv', sep=',', index=False)
            np.save(self.path + '\\processed files\\' + 'coord_fiji.npy', coord_fiji, allow_pickle = True)
            np.save(self.path + '\\processed files\\' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '\\processed files\\' + 'trials.npy', trials)
            np.save(self.path + '\\processed files\\' + 'reg_th.npy', th)
            np.save(self.path + '\\processed files\\' + 'frames_to_exclude.npy', idx_to_nan)
        return

    def load_processed_files_ext_fiji(self):
        """Loads processed files that were saved under path/processed files"""
        if self.delim == '/':
            df_fiji = pd.read_csv(self.path+'/processed files/'+'df_fiji.csv')
            df_fiji_bgsub = pd.read_csv(self.path + '/processed files/' + 'df_fiji_bgsub.csv')
            df_extract = pd.read_csv(self.path + '/processed files/' + 'df_extract.csv')
            coord_fiji = np.load(self.path+'/processed files/'+'coord_fiji.npy', allow_pickle = True)
            coord_ext = np.load(self.path + '/processed files/' + 'coord_ext.npy', allow_pickle = True)
            trials = np.load(self.path+'/processed files/'+'trials.npy')
            reg_th = np.load(self.path+'/processed files/'+'reg_th.npy')
            reg_bad_frames = np.load(self.path+'/processed files/'+'frames_to_exclude.npy')
        else:
            df_fiji = pd.read_csv(self.path+'\\processed files\\'+'df_fiji.csv')
            df_fiji_bgsub = pd.read_csv(self.path+'\\processed files\\'+'df_fiji_bgsub.csv')
            df_extract = pd.read_csv(self.path+'\\processed files\\'+'df_extract.csv')
            coord_fiji = np.load(self.path+'\\processed files\\'+'coord_fiji.npy', allow_pickle = True)
            coord_ext = np.load(self.path + '\\processed files\\' + 'coord_ext.npy', allow_pickle = True)
            trials = np.load(self.path+'\\processed files\\'+'trials.npy')
            reg_th = np.load(self.path+'\\processed files\\'+'reg_th.npy')
            reg_bad_frames = np.load(self.path+'\\processed files\\'+'frames_to_exclude.npy')
        return df_fiji, df_fiji_bgsub, df_extract, trials, coord_fiji, coord_ext, reg_th, reg_bad_frames

    def get_events(self, df_dFF, timeT, amp_vec, csv_name):
        """"Function to get the calcium event using Jorge's derivative method.
        Amplitude of envelope is determined as the median absolute deviation
        Inputs:
        df_dFF: dataframe with calcium trace values after deltaF/F computation and z-scoring
        timeT: time thershold (within how many frames does the event happen)
        amp_vec: vector with ROIs amplitudes for event detection; if empty it recorded the amplitudes used for each trial and ROI
        csv_name: (str) filename of df_events create; if empty doesn't save file"""
        roi_trace = np.array(df_dFF.iloc[:,2:])
        roi_list = list(df_dFF.columns[2:])
        trial_ext = list(df_dFF['trial'])
        trials = np.unique(trial_ext)
        frame_time_ext = list(df_dFF['time'])
        data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_dFF1 = pd.DataFrame(data_dFF1)
        df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
        df_events = pd.concat([df_dFF1, df_dFF2], axis=1)
        roi_list = df_dFF.columns[2:]
        if len(amp_vec) == 0:
            amp_arr = np.zeros((len(trials),len(roi_list)))
            count_r = 0
            for r in roi_list:
                count_t = 0
                for t in trials:
                    data = np.array(df_dFF.loc[df_dFF['trial'] == t, r])
                    amp = robust.mad(data)
                    events_mat = np.zeros(len(data))
                    if (np.nanmax(data) - np.nanmin(data)) <= 1:  # if trace is event probability or min-max normed
                        # the high number of 0 values are a problem for derivative estimation
                        #data = data * 60
                        amp = np.nanpercentile(data, 95)
                    [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, amp, timeT,
                                                                                           CollapSeq=True, acausal=False,
                                                                                           verbose=0, graph=None)
                    if len(JoinedPosSet_all) == 0:
                        print('No events for trial' + str(t) + ' and ' + r)
                    else:
                        events_mat[ST.event_detection_calcium_trace(data, JoinedPosSet_all, timeT)] = 1
                    df_events.loc[df_events['trial'] == t, r] = events_mat
                    amp_arr[count_t,count_r] = amp
                    count_t += 1
                count_r += 1
        else:
            count_r = 0
            for r in roi_list:
                amp = amp_vec[count_r]
                for t in trials:
                    data = np.array(df_dFF.loc[df_dFF['trial'] == t, r])
                    events_mat = np.zeros(len(data))
                    [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, amp, timeT,
                                                                                           CollapSeq=True, acausal=False,
                                                                                           verbose=0, graph=None)
                    if len(JoinedPosSet_all) == 0:
                        print('No events for trial' + str(t) + ' and ' + r)
                    else:
                        events_mat[ST.event_detection_calcium_trace(data, JoinedPosSet_all, timeT)] = 1
                    df_events.loc[df_events['trial'] == t, r] = events_mat
                count_r += 1
            amp_arr = amp_vec #to make output the same
        if len(csv_name)>0:
            if not os.path.exists(self.path + 'processed files'):
                os.mkdir(self.path + 'processed files')
            df_events.to_csv(self.path + '\\processed files\\' + csv_name + '.csv', sep=',', index=False)
        return df_events, amp_arr

    @staticmethod
    def get_events_singletrial(traces, timeT):
        """"Function to get the calcium event using Jorge's derivative method for a single trial across some ROIs.
        Amplitude of envelope is determined as the median absolute deviation
        Inputs:
        traces: numpy array with the traces (frames x roi)
        timeT: time thershold (within how many frames does the event happen)"""
        events_traces = np.zeros(np.shape(traces))
        for r in range(np.shape(traces)[1]):
            data = traces[:,r]
            amp = robust.mad(data)
            events_mat = np.zeros(len(data))
            if (np.max(data) - np.min(data)) <= 1:  # if trace is event probability or min-max normed
                # the high number of 0 values are a problem for derivative estimation
                # data = data * 60
                amp = np.percentile(data, 90)
            # diff_data = np.diff(data)
            # diff_data_nonan = diff_data[~np.isnan(diff_data)]
            # MAD = robust.mad(diff_data_nonan)
            # if MAD == 0: #when it's very flat and some peaks - IMPROVE
            #     deriv_std = np.percentile(data,99)
            # else:
            # norm_data = np.abs(diff_data_nonan - np.median(diff_data_nonan)) / MAD #this is a change, slope is done on data not on norm_data
            # deriv_mean, deriv_std = stats.norm.fit(norm_data)
            # True_std = np.sqrt((deriv_std ** 2) / 2)
            # FiltWin = 3
            # denoised_data = signal.wiener(data, mysize=FiltWin, noise=noise_std**2)
            # [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(denoised_data, True_std, timeT,
            #                                                                        CollapSeq=True, acausal=False,
            #                                                                        verbose=0, graph=None)
            [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, amp, timeT,
                                                                                   CollapSeq=True, acausal=False,
                                                                                   verbose=0, graph=None)
            if len(JoinedPosSet_all) == 0:
                print('No events for index ' + str(r))
            else:
                events_mat[ST.event_detection_calcium_trace(data, JoinedPosSet_all, timeT)] = 1
            events_traces[:, r] = events_mat
        return events_traces

    def event_differences(self, df_fiji, roi, timeT):
        """Function to compare the several thresholds for event direction (derivative-based algorithm)
        Input:
            df_fiji: (dataframe)
            roi: (int) roi to plot
            timeT: (int) interval of calcium transient in frames"""
        r = 'ROI'+str(roi)
        df_dFF_fiji = self.compute_dFF(df_fiji)
        df_dFF_fiji_norm = self.norm_traces(df_dFF_fiji, 'zscore')
        data = np.array(df_dFF_fiji_norm.loc[df_dFF_fiji_norm['trial'] == 2, r])
        # Find outliers using MAD and neutralize them:
        MAD = robust.mad(np.diff(data))
        deriv_mean, deriv_std = stats.norm.fit(np.abs(np.diff(data) - np.median(np.diff(data))) / MAD)
        # noise is gaussian, stationary and independently distributed
        # two independent normal distributions - noise and signal????
        # variance the sum of two independent gaussian distributions
        # var(z) = var(n) + var (s), so std(n) = sqrt(var(z)/2)
        # variance of the product of independent gaussian distributions
        # var(z) = (var(n)+mean(n))*(var(s)+mean(s))-(mean(n)**2mean(s)**2)
        noise_std = np.sqrt((deriv_std ** 2) / 2)  # Based on the product of two equal gaussians
        FiltWin = 5
        denoised_data = signal.wiener(data, mysize=FiltWin, noise=deriv_std ** 2)
        MAD_denoised = robust.mad(np.diff(denoised_data))
        deriv_mean_denoised, deriv_std_denoised = stats.norm.fit(
            np.abs(np.diff(denoised_data) - np.median(np.diff(denoised_data))) / MAD_denoised)
        events_raw_mad = np.zeros(len(data))
        [JoinedPosSet_all_raw_mad, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, MAD, timeT,
                                                                                       CollapSeq=True, acausal=False,
                                                                                       verbose=0, graph=None)
        if len(JoinedPosSet_all_raw_mad) == 0:
            print('No events')
        else:
            events_raw_mad[ST.event_detection_calcium_trace(data, JoinedPosSet_all_raw_mad, timeT)] = 1
        events_raw_std = np.zeros(len(data))
        [JoinedPosSet_all_raw_std, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, deriv_std, timeT,
                                                                                       CollapSeq=True, acausal=False,
                                                                                       verbose=0, graph=None)
        if len(JoinedPosSet_all_raw_std) == 0:
            print('No events')
        else:
            events_raw_std[ST.event_detection_calcium_trace(data, JoinedPosSet_all_raw_std, timeT)] = 1
        events_denoised_mad = np.zeros(len(denoised_data))
        [JoinedPosSet_all_denoised_mad, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(denoised_data, MAD_denoised,
                                                                                            timeT,
                                                                                            CollapSeq=True,
                                                                                            acausal=False,
                                                                                            verbose=0, graph=None)
        if len(JoinedPosSet_all_denoised_mad) == 0:
            print('No events')
        else:
            events_denoised_mad[
                ST.event_detection_calcium_trace(denoised_data, JoinedPosSet_all_denoised_mad, timeT)] = 1
        events_denoised_std = np.zeros(len(denoised_data))
        [JoinedPosSet_all_denoised_std, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(denoised_data,
                                                                                            deriv_std_denoised, timeT,
                                                                                            CollapSeq=True,
                                                                                            acausal=False,
                                                                                            verbose=0, graph=None)
        if len(JoinedPosSet_all_denoised_std) == 0:
            print('No events')
        else:
            events_denoised_std[
                ST.event_detection_calcium_trace(denoised_data, JoinedPosSet_all_denoised_std, timeT)] = 1
        fig, ax = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
        ax = ax.ravel()
        ax[0].plot(data, linewidth=2, color='black')
        ax[0].plot(np.repeat(np.median(data) + MAD, len(data)), linestyle='dashed', color='red')
        ax[0].plot(np.repeat(np.median(data) - MAD, len(data)), linestyle='dashed', color='red')
        ax[0].scatter(np.where(events_raw_mad)[0], data[np.where(events_raw_mad)[0]], s=20, color='red')
        ax[0].set_title('MAD diff envelope raw data')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].plot(data, linewidth=2, color='black')
        ax[1].plot(np.repeat(np.median(data) + deriv_std, len(data)), linestyle='dashed', color='red')
        ax[1].plot(np.repeat(np.median(data) - deriv_std, len(data)), linestyle='dashed', color='red')
        ax[1].scatter(np.where(events_raw_std)[0], data[np.where(events_raw_std)[0]], s=20, color='red')
        ax[1].set_title('True std envelope raw data')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[2].plot(denoised_data, linewidth=2, color='black')
        ax[2].plot(np.repeat(np.median(denoised_data) + MAD, len(denoised_data)), linestyle='dashed', color='red')
        ax[2].plot(np.repeat(np.median(denoised_data) - MAD, len(denoised_data)), linestyle='dashed', color='red')
        ax[2].scatter(np.where(events_denoised_mad)[0], denoised_data[np.where(events_denoised_mad)[0]], s=20,
                      color='red')
        ax[2].set_title('MAD diff envelope denoised data')
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[3].plot(denoised_data, linewidth=2, color='black')
        ax[3].plot(np.repeat(np.median(denoised_data) + deriv_std, len(denoised_data)), linestyle='dashed', color='red')
        ax[3].plot(np.repeat(np.median(denoised_data) - deriv_std, len(denoised_data)), linestyle='dashed', color='red')
        ax[3].scatter(np.where(events_denoised_std)[0], denoised_data[np.where(events_denoised_std)[0]], s=20,
                      color='red')
        ax[3].set_title('True std envelope denoised data')
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        fig, ax = plt.subplots(2, 1, figsize=(15, 5), tight_layout=True)
        ax = ax.ravel()
        ax[0].scatter(np.where(events_raw_mad)[0], np.ones(len(np.where(events_raw_mad)[0])), s=20, color='red')
        ax[0].scatter(np.where(events_denoised_mad)[0], np.ones(len(np.where(events_denoised_mad)[0])), s=20,
                      color='blue')
        ax[0].set_title('MAD raw data (red) denoised data (blue)')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[1].scatter(np.where(events_raw_std)[0], np.ones(len(np.where(events_raw_std)[0])), s=20, color='red')
        ax[1].scatter(np.where(events_denoised_std)[0], np.ones(len(np.where(events_denoised_std)[0])), s=20,
                      color='blue')
        ax[1].set_title('True std raw data (red) denoised data (blue)')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        return

    def compute_isi(self, df_events, csv_name):
        """Function to compute the inter-spike interval of the dataframe with spikes. Outputs a similiar dataframe
        Inputs: 
            df_events: events (dataframe)
            csv_name: (str) filename of the dataframe"""
        isi_all = []
        roi_all = []
        trial_all = []
        for r in df_events.columns[2:]:
            for t in df_events['trial'].unique():
                isi = df_events.loc[(df_events[r] > 0) & (df_events['trial'] == t), 'time'].diff()
                isi_all.extend(np.array(isi))
                trial_all.extend(np.repeat(t, len(isi)))
                roi_all.extend(np.repeat(r, len(isi)))
        dict_isi = {'isi': isi_all, 'roi': roi_all, 'trial': trial_all}
        isi_df = pd.DataFrame(dict_isi)  # create dataframe with isi, roi id and trial id
        isi_df.to_csv(self.path + '\\processed files\\' + csv_name + '.csv', sep=',', index=False)
        return isi_df

    @staticmethod
    def compute_isi_cv(isi_df, trials):
        """Function to compute coefficient of variation and coefficient of variation for adjacent spikes (Isope, JNeurosci)
        Inputs:
            isi_df: (dataframe) with isi values
            trials: list of trials"""
        isi_cv_all = []
        isi_cv2_all = []
        roi_all = []
        trial_all = []
        for t in trials:
            for r in np.unique(isi_df.roi):
                data = np.array(isi_df.loc[(isi_df['trial']==t) & (isi_df['roi']==r),'isi'])
                diff_data = np.abs(np.diff(data))
                sum_data = data[:-1]+data[1:]
                isi_cv_value = np.nanstd(data)/np.nanmean(data)
                isi_cv2_value = np.nanmean(diff_data/sum_data)
                isi_cv2_all.append(np.float64(isi_cv_value))
                isi_cv_all.append(np.float64(isi_cv2_value))
                trial_all.append(t)
                roi_all.append(r)
        dict_isi_cv = {'isi_cv': isi_cv_all, 'roi': roi_all, 'trial': trial_all}
        isi_cv_df = pd.DataFrame(dict_isi_cv)  # create dataframe with isi, roi id and trial id
        dict_isi_cv2 = {'isi_cv': isi_cv2_all, 'roi': roi_all, 'trial': trial_all}
        isi_cv2_df = pd.DataFrame(dict_isi_cv2)  # create dataframe with isi, roi id and trial id
        return isi_cv_df, isi_cv2_df

    @staticmethod
    def compute_isi_ratio(isi_df,isi_interval,trials):
        """Function to compute the ratio between two windows of the ISI histogram
        Inputs: 
            isi_df: dataframe of inter-spike intervals per roi and trial
            isi_interval: list with range of the two windows in sec
            e.g.: [[0,0.5],[0.8,1.5]]
            trials: list of trials"""
        isi_ratio = []
        roi_all = []
        trial_all = []
        for r in isi_df.roi.unique():
            for t in trials:
                isi = np.array(isi_df.loc[(isi_df.roi==r) & (isi_df.trial==t), 'isi'])
                hist, bin_edges = np.histogram(isi, bins = 30, range=(0,3))
                ratio = np.nansum(hist[np.where(bin_edges==isi_interval[1][0])[0][0]:np.where(bin_edges==isi_interval[1][1])[0][0]])/np.nansum(hist[np.where(bin_edges==isi_interval[0][0])[0][0]:np.where(bin_edges==isi_interval[0][1])[0][0]])
                isi_ratio.append(ratio)
                trial_all.append(t)
                roi_all.append(r)
        dict_isi_ratio = {'isi_ratio': isi_ratio,'roi': roi_all,'trial': trial_all}
        isi_ratio_df = pd.DataFrame(dict_isi_ratio) # create dataframe with isi, roi id and trial id
        return isi_ratio_df

    def plot_isi_single_trial(self, trial, roi, isi_df, plot_data):
        """Function to plot the ISI distribution of a single trial for a certain ROI
        Inputs:
            trial: (int) trial id
            roi: (int) ROI id
            isi_df: (dataframe) with ISI values
            plot_data: boolean"""
        binwidth = 0.2
        barWidth = 0.2
        isi_data = np.array(isi_df.loc[(isi_df['trial']==trial)&(isi_df['roi']=='ROI'+str(roi)),'isi'])
        max_isi = np.ceil(np.nanmax(isi_data))
        binedges = np.arange(0, max_isi + 0.5, binwidth)
        hist_all = np.histogram(isi_data, bins=binedges)
        hist_norm = hist_all[0] / np.sum(hist_all[0])
        r1 = binedges[:-1]
        fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
        plt.bar(r1, hist_norm, color='darkgrey', width=barWidth, edgecolor='white')
        plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
        plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
        plt.title('ISI distribution of trial '+ str(trial) + ' ROI '+str(roi), fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI'+str(roi)):
                os.mkdir(self.path + '\\images\\events\\ROI'+str(roi))
            if self.delim == '/':
                plt.savefig(self.path+'/images/events/ROI'+str(roi)+'/'+ 'isi_hist_roi_'+str(roi)+'_trial_'+str(trial), dpi=self.my_dpi)
            else:
                plt.savefig(self.path+ '\\images\\events\\ROI'+str(roi)+'\\'+ 'isi_hist_roi_'+str(roi)+'_trial_'+str(trial), dpi=self.my_dpi)
        return

    def plot_isi_session(self, roi, isi_df, animal, session_type, trials, trials_ses, plot_data):
        """Function to plot the ISI distribution across the session for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_df: (dataframe) with ISI values
            animal: (str) with animal name
            session_type: (str) split or tied
            trials: list of trials
            trials_ses: list with the transition trials
            plot_data: boolean"""
        binwidth = 0.2
        barWidth = 0.05
        if session_type == 'split':
            isi_data_baseline = np.array(isi_df.loc[(isi_df['trial'] > 0) & (
                        isi_df['trial'] < trials_ses[1]) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            isi_data_split = np.array(isi_df.loc[(isi_df['trial'] > trials_ses[0]) & (
                        isi_df['trial'] < trials_ses[3]) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            isi_data_washout = np.array(isi_df.loc[(isi_df['trial'] > trials_ses[2]) & (
                        isi_df['trial'] < len(trials) + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            max_isi = np.ceil(np.nanmax(np.concatenate([isi_data_baseline, isi_data_split, isi_data_washout])))
            binedges = np.arange(0, max_isi + 0.5, binwidth)
            hist_norm = np.zeros((len(binedges) - 1, 3))
            hist_baseline = np.histogram(isi_data_baseline, bins=binedges)
            hist_norm[:, 0] = hist_baseline[0] / np.sum(hist_baseline[0])
            hist_split = np.histogram(isi_data_split, bins=binedges)
            hist_norm[:, 1] = hist_split[0] / np.sum(hist_split[0])
            hist_washout = np.histogram(isi_data_washout, bins=binedges)
            hist_norm[:, 2] = hist_washout[0] / np.sum(hist_washout[0])
            r1 = binedges[:-1]
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
            plt.bar(r1, hist_norm[:, 0], color='darkgrey', width=barWidth, edgecolor='white', label='baseline')
            plt.bar(r2, hist_norm[:, 1], color='crimson', width=barWidth, edgecolor='white', label='split')
            plt.bar(r3, hist_norm[:, 2], color='blue', width=barWidth, edgecolor='white', label='washout')
            plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.title('ISI distribution for ROI ' + str(roi), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                    os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/' + 'isi_hist_roi_' + str(roi),dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
        if session_type == 'tied' and animal == 'MC8855':
            isi_data_sp1 = np.array(isi_df.loc[(isi_df['trial'] > 0) & (
                        isi_df['trial'] < trials_ses[0] + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            isi_data_sp2 = np.array(isi_df.loc[(isi_df['trial'] > trials_ses[0]) & (
                        isi_df['trial'] < len(trials) + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            max_isi = np.ceil(np.nanmax(np.concatenate([isi_data_sp1, isi_data_sp2])))
            binedges = np.arange(0, max_isi + 0.5, binwidth)
            hist_norm = np.zeros((len(binedges) - 1, 2))
            hist_sp1 = np.histogram(isi_data_sp1, bins=binedges)
            hist_norm[:, 0] = hist_sp1[0] / np.sum(hist_sp1[0])
            hist_sp2 = np.histogram(isi_data_sp2, bins=binedges)
            hist_norm[:, 1] = hist_sp2[0] / np.sum(hist_sp2[0])
            r1 = binedges[:-1]
            r2 = [x + barWidth for x in r1]
            fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
            plt.bar(r1, hist_norm[:, 0], color='darkgrey', width=barWidth, edgecolor='white', label='baseline speed')
            plt.bar(r2, hist_norm[:, 1], color='crimson', width=barWidth, edgecolor='white', label='fast speed')
            plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.title('ISI distribution for ROI ' + str(roi), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                    os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/' + 'isi_hist_roi_' + str(roi),dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
        if session_type == 'tied' and animal != 'MC8855':
            isi_data_sp1 = np.array(isi_df.loc[(isi_all_events['trial'] > 0) & (
                        isi_df['trial'] < trials_ses[0] + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            isi_data_sp2 = np.array(isi_df.loc[(isi_df['trial'] > trials_ses[0]) & (
                        isi_df['trial'] < trials_ses[1] + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            isi_data_sp3 = np.array(isi_df.loc[(isi_df['trial'] > trials_ses[1]) & (
                        isi_df['trial'] < len(trials) + 1) & (isi_df['roi'] == 'ROI' + str(roi)), 'isi'])
            max_isi = np.ceil(np.nanmax(np.concatenate([isi_data_sp1, isi_data_sp2, isi_data_sp3])))
            binedges = np.arange(0, max_isi + 0.5, binwidth)
            hist_norm = np.zeros((len(binedges) - 1, 3))
            hist_sp1 = np.histogram(isi_data_sp1, bins=binedges)
            hist_norm[:, 0] = hist_sp1[0] / np.sum(hist_sp1[0])
            hist_sp2 = np.histogram(isi_data_sp2, bins=binedges)
            hist_norm[:, 1] = hist_sp2[0] / np.sum(hist_sp2[0])
            hist_sp3 = np.histogram(isi_data_sp3, bins=binedges)
            hist_norm[:, 2] = hist_sp3[0] / np.sum(hist_sp3[0])
            r1 = binedges[:-1]
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
            plt.bar(r1, hist_norm[:, 0], color='darkgrey', width=barWidth, edgecolor='white', label='baseline speed')
            plt.bar(r2, hist_norm[:, 1], color='crimson', width=barWidth, edgecolor='white', label='slow speed')
            plt.bar(r3, hist_norm[:, 2], color='blue', width=barWidth, edgecolor='white', label='fast speed')
            plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.title('ISI distribution for ROI ' + str(roi), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                    os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/' + 'isi_hist_roi_' + str(roi),dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
        return

    def plot_cv_session(self, roi, isi_cv, trials, plot_name, plot_data):
        """Function to plot the ISI distribution across the session for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_cv: (array) with CV values
            trials: list of trials
            plot_name: such as 'cv' or 'cv2'
            plot_data: boolean"""
        if len(trials) == 23:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 26:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 6:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']
        if len(trials) == 18:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue', 'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        for t in trials:
            plt.bar(t - 0.5, isi_cv.loc[(isi_cv['roi']=='ROI'+str(roi))&(isi_cv['trial']==t),'isi_cv'], width=1, color=colors_bars[t - 1], edgecolor='white')
        ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
        ax.set_xticklabels(list(map(str, trials)))
        plt.xlim([0, len(trials) + 1])
        plt.ylabel('Coefficient of variation', fontsize=self.fsize)
        plt.xlabel('Trials', fontsize=self.fsize)
        plt.title('CV for ROI ' + str(roi), fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/' + plot_name + '_roi_' + str(roi), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\' + plot_name + '_roi_' + str(roi), dpi=self.my_dpi)
        return

    def plot_isi_ratio_session(self, roi, isi_ratio, range_isiratio, trials, plot_data):
        """Function to plot the ISI ratio between a certain range across the session
        for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_ratio: (dataframe) with ISI ratio values
            range_isiratio: list with the range values
            trials: list of trials
            plot_data: boolean"""
        if len(trials) == 23:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 26:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 6:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']
        if len(trials) == 18:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue',
                           'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        for t in trials:
            plt.bar(t - 0.5, np.array(isi_ratio.loc[(isi_ratio['trial'] == t) & (
                        isi_ratio['roi'] == 'ROI' + str(roi)), 'isi_ratio'])[0], width=1,
                    color=colors_bars[t - 1], edgecolor='white')
        ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
        ax.set_xticklabels(list(map(str, trials)))
        plt.xlim([0, len(trials) + 1])
        plt.ylabel('ISI ratio', fontsize=self.fsize)
        plt.xlabel('Trials', fontsize=self.fsize)
        plt.title('ISI ratio between ' + str(range_isiratio[0]) + ' ' + str(range_isiratio[1]) + ' for ROI ' + str(roi),
                  fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/' + 'isi_ratio_roi_' + str(roi), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\' + 'isi_ratio_roi_' + str(roi), dpi=self.my_dpi)
        return

    def compute_event_waveform(self, df_fiji, df_events, roi_plot, animal, session_type, trials_ses, trials, plot_data):
        """Function to compute the complex-spike waveform from the deconvolution and dFF
        Inputs:
            df_fiji: dataframe with normalized trace
            df_events: dataframe with events
            roi_plot: (int) with ROI to plot
            animal: (str) with animal name
            session_type: (str) split or tied
            trials: list of trials
            trials_ses: list with the transition trials
            plot_data: boolean"""
        trace_data = np.array(df_fiji['ROI' + str(roi_plot)])
        if session_type == 'split':
            idx_plot = [[0, trials_ses[1]], [trials_ses[0], trials_ses[2]], [trials_ses[2], len(trials)]]
            colors_plot = ['darkgrey', 'crimson', 'blue']
        if session_type == 'tied' and animal == 'MC8855':
            idx_plot = [[0, trials_ses[0] + 1], [trials_ses[0] + 1, len(trials)]]
            colors_plot = ['darkgrey', 'orange']
        if session_type == 'tied' and animal != 'MC8855':
            idx_plot = [[0, trials_ses[0] + 1], [trials_ses[0] + 1, trials_ses[1] + 1],
                        [trials_ses[1] + 1, len(trials)]]
            colors_plot = ['darkgrey', 'lightblue', 'orange']
        cs_waveforms_mean_all = np.zeros((len(colors_plot), 40))
        cs_waveforms_sem_all = np.zeros((len(colors_plot), 40))
        for i in range(len(colors_plot)):
            events = np.array(df_events.loc[(df_events['ROI' + str(roi_plot)] == 1) & (
                        df_events['trial'] > idx_plot[i][0]) & (df_events['trial'] < idx_plot[i][1])].index)
            cs_waveforms = np.zeros((len(events), 40))
            cs_waveforms[:] = np.nan  # initialize nan array
            count_s = 0
            for s in events:
                if len(trace_data[s - 10:s + 30]) == 40:
                    cs_waveforms[count_s, :] = (trace_data[s - 10:s + 30] - np.min(trace_data[s - 10:s + 30])) / (
                                np.max(trace_data[s - 10:s + 30]) - np.min(trace_data[s - 10:s + 30]))
                count_s += 1
            cs_waveforms_mean_all[i, :] = np.nanmean(cs_waveforms, axis=0)
            cs_waveforms_sem_all[i, :] = np.nanstd(cs_waveforms, axis=0) / np.sqrt(len(events))
        fig, ax = plt.subplots(1, len(colors_plot), figsize=(15, 5), tight_layout=True)
        ax = ax.ravel()
        for i in range(len(colors_plot)):
            ax[i].plot(np.round(np.linspace(-10 / 30, 30 / 30, 40), 2), cs_waveforms_mean_all[i, :],
                       color=colors_plot[i], linewidth=2)
            ax[i].fill_between(np.round(np.linspace(-10 / 30, 30 / 30, 40), 2),
                               cs_waveforms_mean_all[i, :] - cs_waveforms_sem_all[i, :],
                               cs_waveforms_mean_all[i, :] + cs_waveforms_sem_all[i, :], color=colors_plot[i],
                               alpha=0.3)
            ax[i].set_ylabel('Amplitude (a.u.)', fontsize=self.fsize - 4)
            ax[i].set_xlabel('Time (s)', fontsize=self.fsize - 4)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].tick_params(axis='both', which='major', labelsize=self.fsize - 4)
        plt.suptitle('ROI' + str(roi_plot), fontsize=self.fsize - 2)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'event_waveform_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'event_waveform_roi_' + str(roi_plot), dpi=self.my_dpi)
        return [cs_waveforms_mean_all, cs_waveforms_sem_all]
    
    def get_event_count_wholetrial(self, df_events, trials, roi_plot, plot_data):
        """Function to compute the normalized spike count (divided by the number of frames) per trial
        Inputs: 
            df_events: dataframe with events
            trials: array of recorded trials
            roi_plot: (int) of ROI to plot"""
        if len(trials) == 23:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 26:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 6:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']
        if len(trials) == 18:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue',
                           'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        for t in trials:
            event_count = df_events.loc[
                (df_events['ROI' + str(roi_plot)] == 1) & (df_events['trial'] == t)].count()
            plt.bar(t - 0.5, event_count, width=1, color = colors_bars[t - 1], edgecolor='white')
        ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
        ax.set_xticklabels(list(map(str, trials)))
        plt.xlim([0, len(trials) + 1])
        plt.ylabel('Event count', fontsize=self.fsize)
        plt.xlabel('Trials', fontsize=self.fsize)
        plt.title('Event count (whole trial) for ROI ' + str(roi_plot), fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
        return

    def get_event_count_locomotion(self, df_events, trials, bcam_time, st_strides_trials, roi_plot, plot_data):
        """Function to compute the normalized spike count (divided by the number of frames) per trial
        Inputs:
            df_events: dataframe with events
            trials: array of recorded trials
            bcam_time: behavioral camera timestamps
            st_strides_trials: list with stride onsets (trials - paws - stridesx2x5)
            roi_plot: (int) of ROI to plot
            plot_data: boolean"""
        if len(trials) == 23:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 26:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 6:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']
        if len(trials) == 18:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue',
                           'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        event_count_clean = np.zeros((len(trials)))
        for t in trials:
            bcam_trial = bcam_time[t - 1]
            events = np.array(
                df_events.loc[(df_events['trial'] == t) & (df_events['ROI' + str(roi_plot)] == 1), 'time'])
            st_on = st_strides_trials[t - 1][0][:, 0, -1]  # FR paw
            st_off = st_strides_trials[t - 1][0][:, 1, -1]  # FR paw
            time_forwardloco = []
            event_clean_list = []
            for s in range(len(st_on)):
                time_forwardloco.append(bcam_trial[int(st_off[s])] - bcam_trial[int(st_on[s])])
                if len(np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]) > 0:
                    event_clean_list.append(len(
                        np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]))
            time_forwardloco_trial = np.sum(time_forwardloco)
            event_count_clean[t - 1] = np.sum(event_clean_list) / time_forwardloco_trial
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        for t in trials:
            plt.bar(t - 0.5, event_count_clean[t - 1], width=1, color=colors_bars[t - 1], edgecolor='white')
        ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
        ax.set_xticklabels(list(map(str, trials)))
        plt.xlim([0, len(trials) + 1])
        plt.ylabel('Event count', fontsize=self.fsize)
        plt.xlabel('Trials', fontsize=self.fsize)
        plt.title('Event count (forward locomotion) for ROI ' + str(roi_plot), fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'event_count_loco_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'event_count_loco_roi_' + str(roi_plot), dpi=self.my_dpi)
        return

    def events_stride(self, df_events, st_strides_trials, sw_pts_trials, paw, roi_plot, align):
        """Align CS to stance, swing or stride period. It outputs the CS indexes for each
        time period
        Inputs:
            df_events: dataframe with events
            st_strides_trials: list of trials with matrix with stance points
            sw_pts_trials: list of trials with matrix with swing points
            paw (str): 'FR','HR','FL','HL'
            roi_plot (int): roi number
            align (str): period to align - 'stance','swing','stride'"""
        if paw == 'FR':
            p = 0  # paw of tracking
        if paw == 'HR':
            p = 1
        if paw == 'FL':
            p = 2
        if paw == 'HL':
            p = 3
        nr_strides = np.zeros(len(st_strides_trials))
        for t in np.arange(1,len(st_strides_trials)+1):
            nr_strides[t - 1] = np.shape(st_strides_trials[t - 1][0])[0]
        maximum_nrstrides = np.int64(np.max(nr_strides))
        cs_stride = np.zeros((maximum_nrstrides, len(st_strides_trials)))
        cs_stride[:] = np.nan
        for t in np.arange(1,len(st_strides_trials)+1):
            if align == 'stride':
                excursion_beg = st_strides_trials[t - 1][p][:, 0, 4] / self.sr_loco
                excursion_end = st_strides_trials[t - 1][p][:, 1, 4] / self.sr_loco
            if align == 'stance':
                excursion_beg = st_strides_trials[t - 1][p][:, 0, 4] / self.sr_loco
                excursion_end = sw_pts_trials[t - 1][p][:, 0, 4] / self.sr_loco
            if align == 'swing':
                excursion_beg = sw_pts_trials[t - 1][p][:, 0, 4] / self.sr_loco
                excursion_end = st_strides_trials[t - 1][p][:, 1, 4] / self.sr_loco
            events = np.array(
                df_events.loc[(df_events['trial'] == t) & (df_events['ROI' + str(roi_plot)] == 1), 'time'])
            for s in range(len(excursion_beg)):
                cs_idx = np.where((events >= excursion_beg[s]) & (events <= excursion_end[s]))[0]
                if len(cs_idx) > 0:
                    cs_stride[s, t - 1] = len(cs_idx)
                if len(cs_idx) == 0:
                    cs_stride[s, t - 1] = 0
        df_cs_stride = pd.DataFrame(cs_stride, columns=np.arange(1,len(st_strides_trials)+1))
        return df_cs_stride

    def event_proportion_plot(self, df_cs_stride, paw, roi_plot, plot_data):
        """Compute event probability for CSs aligned to stride period
        (swing, stance or stride)
        Inputs:
            df_cs_stride: dataframe with number of events per stride
            paw: (str) FR, FL, HR or HL
            roi_plot: (str) ROI to plot
            plot_data: boolean"""
        trials = np.array(df_cs_stride.columns)
        if len(trials) == 23:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 26:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson',
                           'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                           'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
        if len(trials) == 6:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']
        if len(trials) == 18:
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue',
                           'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        count_t = 0
        event_proportion = np.zeros(len(trials))
        for t in trials:
            plt.bar(t - 0.5, np.count_nonzero(df_cs_stride[t] > 0) / np.count_nonzero(~np.isnan(df_cs_stride[t])),
                    width=1, color=colors_bars[t - 1], edgecolor='white')
            event_proportion[count_t] = np.count_nonzero(df_cs_stride[t] > 0) / np.count_nonzero(~np.isnan(df_cs_stride[t]))
            count_t += 1
        ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
        ax.set_xticklabels(list(map(str, trials)))
        plt.xlim([0, len(trials) + 1])
        plt.ylabel('Proportion of strides with CS', fontsize=self.fsize)
        plt.xlabel('Trials', fontsize=self.fsize)
        plt.title('Proportion of ' + paw + ' strides with CS for ROI ' + str(roi_plot), fontsize=self.fsize)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'event_prob_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'event_prob_roi_' + str(roi_plot), dpi=self.my_dpi)
        return event_proportion

    def raw_signal_align_st_sw(self, df_rawtrace, st_strides_trials, sw_strides_trials, time_window, paw, roi_plot, align, session_type, trials_ses, plot_data):
        """Align raw calcium signal to stance, swing or stride period. It outputs the matrix of aligned data
        Inputs:
            df_rawtrace: dataframe with signal
            st_strides_trials: list of trials with matrix with stance points
            sw_strides_trials: list of trials with matrix with swing points
            time_window: (float) with time_window around event in s
            paw (str): 'FR','HR','FL','HL'
            roi_plot (int): roi number
            align (str): period to align - 'stance','swing'
            session_type. (str) split or tied
            trials_ses: relevant trials for the session to compute transition lines
            plot_data: boolean"""
        df_rawtrace_norm = self.norm_traces(df_rawtrace, 'min_max', 'trial')
        if align == 'stance':
            align_str = 'st'
        if align == 'swing':
            align_str = 'sw'
        if paw == 'FR':
            p = 0  # paw of tracking
        if paw == 'HR':
            p = 1
        if paw == 'FL':
            p = 2
        if paw == 'HL':
            p = 3
        event_stride_list = []
        trial_length_strides = np.zeros(len(st_strides_trials))
        for t in np.arange(1, len(st_strides_trials) + 1):
            if align == 'stance':
                align_time = st_strides_trials[t - 1][p][:, 0, -1] / self.sr_loco
            if align == 'swing':
                align_time = sw_strides_trials[t - 1][p][:, 0, -1] / self.sr_loco
            trial_length_strides[t - 1] = len(align_time)
            event_stride_arr = np.zeros((len(align_time), np.int64(time_window * self.sr * 2)))
            for count_s, s1 in enumerate(align_time):
                data_events = df_rawtrace_norm[
                    (df_rawtrace_norm['time'].between(s1 - time_window, s1 + time_window)) & (
                            df_rawtrace_norm['trial'] == t)]
                if len(data_events) == time_window * self.sr * 2:
                    event_stride_arr[count_s, :] = np.array(data_events['ROI' + str(roi_plot)])
            event_stride_list.append(event_stride_arr)
        event_stride_all = np.vstack(event_stride_list)
        trial_length_strides_cumsum = np.cumsum(trial_length_strides)
        fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
        sns.heatmap(event_stride_all)
        ax.vlines(time_window * self.sr, *ax.get_ylim(), color='white', linestyle='dashed')
        if session_type == 'split':
            ax.hlines([trial_length_strides_cumsum[trials_ses[0]], trial_length_strides_cumsum[trials_ses[2]]],
                      *ax.get_xlim(), color='white', linewidth=0.5)
        if session_type == 'tied':
            for t in trials_ses[:-1]:
                ax.hlines([trial_length_strides_cumsum[t - 1]], *ax.get_xlim(), color='white', linewidth=0.5)
        ax.set_yticks(np.arange(0, np.shape(event_stride_all)[0], 250))
        ax.set_yticklabels(list(map(str, np.arange(0, np.shape(event_stride_all)[0], 250))))
        ax.set_xticklabels(list(map(str, np.round(np.arange(-time_window, time_window, 0.02), 2))), rotation=45)
        ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax.set_ylabel('Stride number', fontsize=self.fsize - 4)
        ax.set_title(paw + ' ' + align_str + ' raw signal', fontsize=self.fsize - 4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 12)
        plt.yticks(fontsize=self.fsize - 12)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\raw signal\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\raw signal\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/raw signal/ROI' + str(roi_plot) + '/' + 'rawsignal_' + align + '_' + paw + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\raw signal\\ROI' + str(roi_plot) + '\\' + 'rawsignal_' + align + '_' + paw + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
        return event_stride_all

    def events_align_st_sw(self, df_events, st_strides_trials, sw_strides_trials, time_window, bin_size, paw, roi_plot, align, session_type, trials_ses, plot_data):
        """Align events to stance, swing or stride period. It outputs the CS indexes for each
        time period
        Inputs:
            df_events: dataframe with events
            st_strides_trials: list of trials with matrix with stance points
            sw_strides_trials: list of trials with matrix with swing points
            time_window: (float) with time_window around event in s
            bin_size: (int) number of bins to compute heatmap
            paw (str): 'FR','HR','FL','HL'
            roi_plot (int): roi number
            align (str): period to align - 'stance','swing'
            session_type. (str) split or tied
            trials_ses: relevant trials for the session to compute transition lines
            plot_data: boolean"""
        if align == 'stance':
            align_str = 'st'
        if align == 'swing':
            align_str = 'sw'
        if paw == 'FR':
            p = 0  # paw of tracking
        if paw == 'HR':
            p = 1
        if paw == 'FL':
            p = 2
        if paw == 'HL':
            p = 3
        event_stride_list = []
        trial_length_strides = np.zeros(len(st_strides_trials))
        for t in np.arange(1, len(st_strides_trials)+1):
            if align == 'stance':
                align_time = st_strides_trials[t - 1][p][:, 0, -1] / self.sr_loco
            if align == 'swing':
                align_time = sw_strides_trials[t - 1][p][:, 0, -1] / self.sr_loco
            trial_length_strides[t - 1] = len(align_time)
            event_stride_arr = np.zeros((len(align_time), bin_size))
            count_s = 0
            for s1 in align_time:
                data_events = df_events[(df_events['time'].between(s1 - time_window, s1 + time_window)) & (
                        df_events['trial'] == t)]
                event_times = np.array(data_events.loc[data_events['ROI' + str(roi_plot)] == 1, 'time'] - s1)
                if len(event_times) > 0:
                    event_stride_arr[
                        count_s, np.where(np.histogram(event_times, bins=bin_size, range=(-0.2, 0.2))[0])[0]] = 1
                count_s += 1
            event_stride_list.append(event_stride_arr)
        event_stride_all = np.vstack(event_stride_list)
        trial_length_strides_cumsum = np.cumsum(trial_length_strides)
        fig, ax = plt.subplots(2,1,figsize=(7, 15), tight_layout=True)
        ax = ax.ravel()
        sns.heatmap(event_stride_all, cmap='viridis', cbar=False, ax = ax[0])
        ax[0].vlines(bin_size / 2, *ax[0].get_ylim(), color='white', linestyle='dashed')
        if session_type == 'split':
            ax[0].hlines([trial_length_strides_cumsum[trials_ses[0]], trial_length_strides_cumsum[trials_ses[2]]],
                      *ax[0].get_xlim(), color='white', linewidth=0.5)
        if session_type == 'tied':
            for t in trials_ses[:-1]:
                ax[0].hlines([trial_length_strides_cumsum[t-1]], *ax[0].get_xlim(), color='white', linewidth=0.5)
        ax[0].set_yticks(np.arange(0, np.shape(event_stride_all)[0], 250))
        ax[0].set_yticklabels(list(map(str, np.arange(0, np.shape(event_stride_all)[0], 250))))
        ax[0].set_xticklabels(list(map(str, np.round(np.arange(-time_window, time_window, 0.02), 2))), rotation=45)
        ax[0].set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax[0].set_ylabel('Stride number', fontsize=self.fsize - 4)
        ax[0].set_title(paw + ' '+ align_str + ' events', fontsize=self.fsize - 4)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 12)
        plt.yticks(fontsize=self.fsize - 12)
        ax[1].bar(np.round(np.arange(-time_window, time_window, 0.02), 2),np.sum(event_stride_all,axis=0), width=0.01, color='gray')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi_plot)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi_plot))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi_plot) + '/' + 'event_' + align + '_' + paw + '_binsize_' + str(
                    bin_size) + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi_plot) + '\\' + 'event_' + align + '_' + paw + '_binsize_' + str(
                    bin_size) + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
        return event_stride_all

    def create_registered_tiffs(self, frame_time, trials):
        """Function to create tiffs from the registered stacks that are suite2p output. Each tiff has the same length as the trials.
        Input:
        frame_time: list with mscope timestamps
        trials: list with trials in session"""
        reg_path = self.path + '\\Suite2p\\suite2p\\plane0\\reg_tif\\'
        if not os.path.exists(self.path + '\\Registered video\\'):
            os.mkdir(self.path + 'Registered video\\')
        trial_end = np.cumsum([len(frame_time[k]) for k in range(len(frame_time))])
        trial_beg = np.insert(trial_end[:-1], 0, 0)
        tiflist = glob.glob(reg_path + '*.tif')
        tiflist_all = []
        for tifff in tiflist:
            image_stack = tiff.imread(tifff)
            tiflist_all.append(image_stack)
        tiflist_concat = np.concatenate(tiflist_all, axis=0)
        for t in trials:
            trial_frames_full = tiflist_concat[trial_beg[t - 1]:trial_end[t - 1], :, :]
            tiff.imsave(self.path + 'Registered video\\T' + str(t) + '_reg.tif', trial_frames_full, bigtiff=True)
        return

    def create_residuals_tiffs(self, df_fiji, coord_fiji):
        """Function to create the residual movies (movie without the activity of the
        detected ROIs
        Input:
        df_fiji: dataframe with the ROIs traces
        coord_fiji: list with the ROIs coordinates"""
        reg_path_tiff = self.path + 'Registered video\\'
        os.mkdir(self.path + '\\Residuals\\')
        tiflist_reg = glob.glob(reg_path_tiff + '*.tif')
        roi_list = df_fiji.columns[2:]
        for tifff in tiflist_reg:
            tiff_name = tifff.split('\\')[-1]
            image_stack = tiff.imread(tifff)
            residuals_stack = image_stack
            for c in range(len(coord_fiji)):
                coord_fiji_pixel = np.int64(coord_fiji[c] * self.pixel_to_um)
                trace = np.array(df_fiji.loc[df_fiji['trial'] == np.int64(tiff_name.split('_')[0][1:]), roi_list[c]])
                for f in range(len(trace)):
                    residuals_stack[f, coord_fiji_pixel[:, 1], coord_fiji_pixel[:, 0]] = image_stack[f, coord_fiji_pixel[:,1], coord_fiji_pixel[:,0]] - trace[f]
            plt.imshow(np.mean(residuals_stack, axis=0))
            tiff.imsave(self.path + '\\Residuals\\' + 'T' + tiff_name.split('_')[0][1:] + '_reg_residuals.tif',
                        residuals_stack, bigtiff=True)
        return

    def create_contrast_tiffs(self):
        """Function to create tiffs from the registered stacks that have contrast enhancement.
        For each pixel, check the distribution of fluorescence values and subtract its 10th percentile."""
        perc_th = 10
        reg_path_tiff = self.path + 'Registered video\\'
        tiflist_reg = glob.glob(reg_path_tiff + '*.tif')
        path_bgsub = '\\Registered video without background\\'
        if not os.path.exists(self.path + path_bgsub):
            os.mkdir(self.path + path_bgsub)
        for tifff in tiflist_reg:
            tiff_name = tifff.split('\\')[-1]
            image_stack = tiff.imread(tifff)
            # Compute background of registered tiffs
            perc_pixel = np.zeros((np.shape(image_stack)[1], np.shape(image_stack)[2]))
            for i in range(np.shape(image_stack)[1]):
                for j in range(np.shape(image_stack)[2]):
                    perc_pixel[i, j] = np.percentile(image_stack[:, i, j], perc_th)
            perc_pixel_tile = np.tile(perc_pixel, (np.shape(image_stack)[0], 1, 1))
            image_stack_bgsub = image_stack - perc_pixel_tile
            tiff.imsave(self.path + path_bgsub + 'T' + tiff_name.split('_')[0][1:] + '_reg_bgsub.tif', image_stack_bgsub, bigtiff=True)
        return

    def events_align_trajectory(self, df_events, bcam_time, final_tracks_trials, trial, roi, plot_data):
        """Function to plot average trajectories aligned to calcium event for a certain ROI and trial.
        Input:
            df_events: dataframe with events
            bcam_time: behavioral camera timestamps for all trials
            final_tracks_trials: final_tracks list for all trials
            trial: (int) trial to plot
            roi: (int) roi to plot
            plot_data: boolean"""
        time = 0.2
        data_events = np.array(
            df_events.loc[(df_events['trial'] == trial) & (df_events['ROI' + str(roi)]), 'time'])
        bcam_time_trial = bcam_time[trial - 1]
        bcam_idx_events = []
        for e in data_events:
            bcam_idx_events.append(np.argmin(np.abs(e - bcam_time_trial)))
        traj_list = []
        for i in bcam_idx_events:
            if (i > (time * 330)) and (i < (np.shape(final_tracks_trials[trial - 1])[2] - (time * 330))):
                traj_list.append(
                    np.array(final_tracks_trials[trial - 1][0, :4, np.int64(i - (time * 330)):np.int64(i + (time * 330))]))
        traj_arr = np.dstack(traj_list)
        traj_ave = np.nanmean(traj_arr, axis=2)
        traj_sem = np.nanstd(traj_arr, axis=2) / np.sqrt(np.shape(traj_arr)[2])
        colors_paws = ['red', 'magenta', 'blue', 'cyan']
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        for p in range(4):
            ax.plot(np.arange(0, np.shape(traj_ave)[1]), traj_ave[p, :], linewidth=2, color=colors_paws[p])
            ax.fill_between(np.arange(0, np.shape(traj_ave)[1]), traj_ave[p, :] - traj_sem[p, :],
                            traj_ave[p, :] + traj_sem[p, :], color=colors_paws[p], alpha=0.3)
        ax.set_xticks(np.linspace(0, np.shape(traj_ave)[1], 10))
        ax.set_xticklabels(list(map(str, np.round(np.linspace(-time, time, 10), 2))), rotation=45)
        ax.axvline(x=np.shape(traj_ave)[1] / 2, linestyle='dashed', color='black')
        ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax.set_ylabel('Paw trajectory (mm)', fontsize=self.fsize - 4)
        ax.set_title('Events ROI' + str(roi) + ' for trial ' + str(trial), fontsize=self.fsize - 4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 12)
        plt.yticks(fontsize=self.fsize - 12)
        if plot_data:
            if not os.path.exists(self.path + '\\images\\events\\ROI' + str(roi)):
                os.mkdir(self.path + '\\images\\events\\ROI' + str(roi))
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/ROI' + str(roi) + '/'  + 'event_paw_trajectory_' + '_roi_' + str(roi) + '_trial_' + str(trial),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\ROI' + str(roi) + '\\'  + 'event_paw_trajectory_' + '_roi_' + str(roi) + '_trial_' + str(trial),
                            dpi=self.my_dpi)
        return

    def get_nearest_rois_manual_roi(self, rois_df, centroid_cell, rfiji):
        """Get the nearest ROIs (from EXTRACT or others) to a certain manual
        ROI drawn with Fiji
        Input:
        rois_df: dataframe with ROIs coordinates from Fiji
        centroid_cell (list of ROI centroids from EXTRACT or others)
        rfiji: int with the id of ROI to compare
        """
        cent_fiji = [
            np.nanmean(np.arange(rois_df.iloc[rfiji, 0] / self.pixel_to_um, rois_df.iloc[rfiji, 1] / self.pixel_to_um)),
            np.nanmean(np.arange(rois_df.iloc[rfiji, 2] / self.pixel_to_um, rois_df.iloc[rfiji, 3] / self.pixel_to_um))]
        rois_ext_near_fiji = []
        for c in range(np.shape(centroid_cell)[0]):
            if (np.abs(centroid_cell[c][0] - cent_fiji[0]) < 75) and (np.abs(centroid_cell[c][1] - cent_fiji[1]) < 75):
                rois_ext_near_fiji.append(c)
        return rois_ext_near_fiji

    def plot_overlap_extract_manual_rois(self, rfiji, rext, ref_image, rois_df, coord_cell, roi_trace_minmax_bgsub,
                                         trace):
        """Plot overlap of EXTRACT (or others) with corresponding manual ROI from Fiji. Plots also calcium trace.
        Input:
        rois_df: dataframe with ROIs coordinates from Fiji
        coord_cell (list of ROI cooridnates from EXTRACT or others)
        roi_trace_minmax_bgsub: array with Fiji calcium trace normalized (0-1)
        trace: array with EXTRACT activity probabilities
        ref_image: aray with a reference image
        rfiji: int with the id of ROI Fiji to compare
        rext: int with the id of ROI EXTRACT to compare
        """
        cent_fiji = [np.nanmean(
            np.arange(rois_df.iloc[rfiji, 0] / self.pixel_to_um, rois_df.iloc[rfiji, 1] / self.pixel_to_um)),
            np.nanmean(np.arange(rois_df.iloc[rfiji, 2] / self.pixel_to_um,
                                 rois_df.iloc[rfiji, 3] / self.pixel_to_um))]
        fig = plt.figure(figsize=(25, 12), tight_layout=True)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        for r in range(np.shape(rois_df)[0]):
            ax1.plot([rois_df.iloc[r, 0] / self.pixel_to_um, rois_df.iloc[r, 1] / self.pixel_to_um],
                     [rois_df.iloc[r, 2] / self.pixel_to_um, rois_df.iloc[r, 3] / self.pixel_to_um])
        ax1.set_title('Manual ROIs', fontsize=self.fsize)
        ax1.imshow(ref_image,
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                           0], cmap=plt.get_cmap('gray'))
        ax1.set_xlim([cent_fiji[0] - 100, cent_fiji[0] + 100])
        ax1.set_ylim([cent_fiji[1] - 100, cent_fiji[1] + 100])
        ax1.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax1.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax1.tick_params(axis='x', labelsize=self.fsize - 4)
        ax1.tick_params(axis='y', labelsize=self.fsize - 4)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot([rois_df.iloc[rfiji, 0] / self.pixel_to_um, rois_df.iloc[rfiji, 1] / self.pixel_to_um],
                 [rois_df.iloc[rfiji, 2] / self.pixel_to_um, rois_df.iloc[rfiji, 3] / self.pixel_to_um],
                 color='blue')
        ax2.scatter(coord_cell[rext][:, 0], coord_cell[rext][:, 1], s=1, color='red')
        ax2.set_title('Manual ' + str(rfiji + 1) + ' and corresponding EXTRACT ROI ' + str(rext), fontsize=self.fsize)
        ax2.imshow(ref_image,
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                           0], cmap=plt.get_cmap('gray'))
        ax2.set_xlim([cent_fiji[0] - 100, cent_fiji[0] + 100])
        ax2.set_ylim([cent_fiji[1] - 100, cent_fiji[1] + 100])
        ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.tick_params(axis='x', labelsize=self.fsize - 4)
        ax2.tick_params(axis='y', labelsize=self.fsize - 4)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(np.linspace(0, np.shape(roi_trace_minmax_bgsub)[0] / self.sr, len(roi_trace_minmax_bgsub[:, rfiji])),
                 roi_trace_minmax_bgsub[:, rfiji], color='blue')
        ax3.plot(np.linspace(0, np.shape(roi_trace_minmax_bgsub)[0] / self.sr, len(roi_trace_minmax_bgsub[:, rfiji])),
                 trace[rext, :], color='red')
        ax3.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax3.set_ylabel('Amplitude of F values', fontsize=self.fsize - 4)
        ax3.tick_params(axis='x', labelsize=self.fsize - 4)
        ax3.tick_params(axis='y', labelsize=self.fsize - 4)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        return

    def compare_extract_extract_rois(self, r1, r2, coord_cell1, coord_cell2, trace1, trace2, ref_image, comparison):
        """Function to compare between two EXTRACT ROIs computed with different parameters.
        Input:
        r1: (int) roi 1
        r2: (int) roi 2
        coord_cell1: (list)
        coord_cell2: (list)
        trace1: (array)
        trace2: (array)
        comparison: (str) with the comparison name"""
        centroid_cell = np.array([np.nanmean(coord_cell1[r1][:, 0]), np.nanmean(coord_cell1[r1][:, 1])])
        fig = plt.figure(figsize=(25, 12), tight_layout=True)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(coord_cell1[r1][:, 0], coord_cell1[r1][:, 1], s=1, color='orange')
        ax1.scatter(coord_cell2[r2][:, 0], coord_cell2[r2][:, 1], s=1, color='blue', alpha=0.5)
        ax1.set_title('ROIs', fontsize=self.fsize)
        ax1.imshow(ref_image,
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                           0], cmap=plt.get_cmap('gray'))
        ax1.set_xlim([centroid_cell[0] - 200, centroid_cell[0] + 200])
        ax1.set_ylim([centroid_cell[1] - 200, centroid_cell[1] + 200])
        ax1.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax1.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax1.tick_params(axis='x', labelsize=self.fsize - 4)
        ax1.tick_params(axis='y', labelsize=self.fsize - 4)
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(np.linspace(0, np.shape(trace1)[1] / self.sr, len(trace1[r1, :])),
                 trace1[r1, :], color='orange', label='no ' + comparison)
        ax2.plot(np.linspace(0, np.shape(trace2)[1] / self.sr, len(trace2[r2, :])),
                 trace2[r2, :], color='blue', label=comparison)
        ax2.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax2.set_ylabel('Amplitude of F values', fontsize=self.fsize - 4)
        ax2.tick_params(axis='x', labelsize=self.fsize - 4)
        ax2.tick_params(axis='y', labelsize=self.fsize - 4)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        return

    def compute_bg_roi_fiji_example(self, trial, ref_image, rois_df, rfiji):
        """Function to compute a donut background around one FIJI ROI and compute its background subtracted signal.
        Plot all steps.
        Input:
            trial: int
            ref_image: array with reference image from Suite2p
            rois_df: dataframe with ROIs from FIJI
            rfiji: int"""
        tiff_stack = tiff.imread(self.path + '\\Registered video\\T' + str(trial) + '_reg.tif')  ##read tiffs
        coord_fiji = []
        height_fiji = []
        xlength_fiji = []
        ylength_fiji = []
        for r in range(np.shape(rois_df)[0]):
            coord_r = np.transpose(np.vstack(
                (np.linspace(rois_df.iloc[r, 0] / self.pixel_to_um, rois_df.iloc[r, 1] / self.pixel_to_um, 100),
                 np.linspace(rois_df.iloc[r, 2] / self.pixel_to_um, rois_df.iloc[r, 3] / self.pixel_to_um, 100))))
            x_length = np.abs(coord_r[-1, 0] - coord_r[0, 0])
            y_length = np.abs(coord_r[-1, 1] - coord_r[0, 1])
            xlength_fiji.append(x_length)
            ylength_fiji.append(y_length)
            coord_fiji.append(coord_r)
            height_fiji.append(np.sqrt(np.square(x_length) + np.square(y_length)))
        cent_fiji = np.nanmean(coord_fiji[rfiji - 1], axis=0)
        ell = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji - 1] + 15,
                         -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ell2 = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji - 1] + 30,
                          -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ellpath = ell.get_path()
        vertices = ellpath.vertices.copy()
        coord_ell_inner = ell.get_patch_transform().transform(vertices)
        ellpath2 = ell2.get_path()
        vertices2 = ellpath2.vertices.copy()
        coord_ell_outer = ell2.get_patch_transform().transform(vertices2)
        ROIinner_fill_x, ROIinner_fill_y = zip(*self.render(np.int64(coord_ell_inner * self.pixel_to_um)))
        ROIouter_fill_x, ROIouter_fill_y = zip(*self.render(np.int64(coord_ell_outer * self.pixel_to_um)))
        ROIinner_fill_coord = np.transpose(np.vstack((ROIinner_fill_x, ROIinner_fill_y)))
        ROIouter_fill_coord = np.transpose(np.vstack((ROIouter_fill_x, ROIouter_fill_y)))
        idx_overlap_outer = []
        for x in range(np.shape(ROIinner_fill_coord)[0]):
            if ROIinner_fill_coord[x, 0] in ROIouter_fill_coord[:, 0]:
                idx_overlap = np.where((ROIouter_fill_coord[:, 0] == ROIinner_fill_coord[x, 0]) & (
                        ROIouter_fill_coord[:, 1] == ROIinner_fill_coord[x, 1]))[0]
                if len(idx_overlap) > 0:
                    idx_overlap_outer.append(idx_overlap[0])
        idx_nonoverlap = np.setdiff1d(range(np.shape(ROIouter_fill_coord)[0]), idx_overlap_outer)
        ROIdonut_coord = np.transpose(
            np.vstack((ROIouter_fill_coord[idx_nonoverlap, 0], ROIouter_fill_coord[idx_nonoverlap, 1])))
        ROI_donut_trace = np.zeros(np.shape(tiff_stack)[0])
        for f in range(np.shape(tiff_stack)[0]):
            ROI_donut_trace[f] = np.nansum(tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]])
        coeff_sub = 1
        roi_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            roi_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                coord_fiji[rfiji - 1][:, 1] * self.pixel_to_um), np.int64(
                coord_fiji[rfiji - 1][:, 0] * self.pixel_to_um)])
        roi_trace_arr = roi_trace_tiffmean / len(coord_fiji[rfiji - 1][:, 1])
        donut_trace_arr = ROI_donut_trace / len(ROIdonut_coord[:, 1])
        roi_trace_bgsub_arr = roi_trace_arr - (coeff_sub * donut_trace_arr)
        idx_neg = np.where(roi_trace_bgsub_arr < 0)[0]
        roi_trace_bgsub_arr[idx_neg] = 0
        roi_trace_bgsub_minmax = (roi_trace_bgsub_arr - np.min(roi_trace_bgsub_arr)) / (
                    np.max(roi_trace_bgsub_arr) - np.min(roi_trace_bgsub_arr))
        fig = plt.figure(figsize=(30, 20), tight_layout=True)
        gs = fig.add_gridspec(4, 3)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(coord_fiji[rfiji - 1][:, 0], coord_fiji[rfiji - 1][:, 1], linewidth=2, color='blue')
        ax2.scatter(ROIdonut_coord[:, 0] / self.pixel_to_um, ROIdonut_coord[:, 1] / self.pixel_to_um, s=10,
                    color='green')
        ax2.imshow(ref_image,
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                           0], cmap=plt.get_cmap('gray'))
        ax2.set_title('Fiji ROI ' + str(rfiji - 1) + ' and respective background', fontsize=self.fsize - 4)
        ax2.set_xlim([cent_fiji[0] - 100, cent_fiji[0] + 100])
        ax2.set_ylim([cent_fiji[1] - 100, cent_fiji[1] + 100])
        ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        ax2.tick_params(axis='x', labelsize=self.fsize - 4)
        ax2.tick_params(axis='y', labelsize=self.fsize - 4)
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(np.linspace(0, np.shape(tiff_stack)[0] / self.sr, np.shape(tiff_stack)[0]), roi_trace_bgsub_minmax,
                 color='black', label='background subtracted Fiji signal')
        ax3.legend(frameon=False, fontsize=self.fsize - 8)
        ax3.set_ylabel('Signal amplitude', fontsize=self.fsize - 4)
        ax3.set_ylabel('Time (s)', fontsize=self.fsize - 4)
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.plot(np.linspace(0, np.shape(tiff_stack)[0] / self.sr, np.shape(tiff_stack)[0]), donut_trace_arr,
                 color='green', label='background signal')
        ax4.legend(frameon=False, fontsize=self.fsize - 8)
        ax4.set_ylabel('Signal amplitude', fontsize=self.fsize - 4)
        ax4.set_ylabel('Time (s)', fontsize=self.fsize - 4)
        ax4.spines['right'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        ax5 = fig.add_subplot(gs[3, :])
        ax5.plot(np.linspace(0, np.shape(tiff_stack)[0] / self.sr, np.shape(tiff_stack)[0]), roi_trace_arr,
                 color='blue', label='roi signal')
        ax5.legend(frameon=False, fontsize=self.fsize - 8)
        ax5.set_ylabel('Signal amplitude', fontsize=self.fsize - 4)
        ax5.set_ylabel('Time (s)', fontsize=self.fsize - 4)
        ax5.spines['right'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if not os.path.exists(self.path + 'EXTRACT\\Fiji ROI bgsub'):
            os.mkdir(self.path + 'EXTRACT\\Fiji ROI bgsub')
        if self.delim == '/':
            plt.savefig(self.path + 'EXTRACT/Fiji ROI bgsub/' + 'fiji_bgsub_' + str(rfiji) + '_T' + str(trial),
                        dpi=self.my_dpi)
        else:
            plt.savefig(self.path + 'EXTRACT\\Fiji ROI bgsub\\' + 'fiji_bgsub_' + str(rfiji) + '_T' + str(trial),
                        dpi=self.my_dpi)
        return

    def compute_bg_roi_fiji_extract(self, trial, ref_image, rois_df, coord_cell, trace, rfiji, rext, amp_fiji, amp_ext,
                                    plot_data):
        """Function to compute a donut background around a determined FIJI ROI and compute its background subtracted signal.
        The plot compares with a determined ROI from EXTRACT. ROIs from Fiji start at 1 and from EXTRACT start at 0
        Input:
            trial: int
            ref_image: array with reference image from Suite2p
            rois_df: dataframe with ROIs from FIJI
            coord_cell: list ROIs from EXTRACT
            trace: array signal from EXTRACT
            rfiji: int
            rext: int
            plot_data: boolean"""
        tiff_stack = tiff.imread(self.path + 'Registered video\\T' + str(trial) + '_reg.tif')  ##read tiffs
        coord_fiji = []
        height_fiji = []
        xlength_fiji = []
        ylength_fiji = []
        for r in range(np.shape(rois_df)[0]):
            coord_r = np.transpose(np.vstack(
                (np.linspace(rois_df.iloc[r, 0] / self.pixel_to_um, rois_df.iloc[r, 1] / self.pixel_to_um, 100),
                 np.linspace(rois_df.iloc[r, 2] / self.pixel_to_um, rois_df.iloc[r, 3] / self.pixel_to_um, 100))))
            x_length = np.abs(coord_r[-1, 0] - coord_r[0, 0])
            y_length = np.abs(coord_r[-1, 1] - coord_r[0, 1])
            xlength_fiji.append(x_length)
            ylength_fiji.append(y_length)
            coord_fiji.append(coord_r)
            height_fiji.append(np.sqrt(np.square(x_length) + np.square(y_length)))
        cent_fiji = np.nanmean(coord_fiji[rfiji - 1], axis=0)
        ell = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji - 1] + 15,
                         -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ell2 = mp_patch.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji - 1] + 30,
                          -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ellpath = ell.get_path()
        vertices = ellpath.vertices.copy()
        coord_ell_inner = ell.get_patch_transform().transform(vertices)
        ellpath2 = ell2.get_path()
        vertices2 = ellpath2.vertices.copy()
        coord_ell_outer = ell2.get_patch_transform().transform(vertices2)
        ROIinner_fill_x, ROIinner_fill_y = zip(*self.render(np.int64(coord_ell_inner * self.pixel_to_um)))
        ROIouter_fill_x, ROIouter_fill_y = zip(*self.render(np.int64(coord_ell_outer * self.pixel_to_um)))
        ROIinner_fill_coord = np.transpose(np.vstack((ROIinner_fill_x, ROIinner_fill_y)))
        ROIouter_fill_coord = np.transpose(np.vstack((ROIouter_fill_x, ROIouter_fill_y)))
        idx_overlap_outer = []
        for x in range(np.shape(ROIinner_fill_coord)[0]):
            if ROIinner_fill_coord[x, 0] in ROIouter_fill_coord[:, 0]:
                idx_overlap = np.where((ROIouter_fill_coord[:, 0] == ROIinner_fill_coord[x, 0]) & (
                        ROIouter_fill_coord[:, 1] == ROIinner_fill_coord[x, 1]))[0]
                if len(idx_overlap) > 0:
                    idx_overlap_outer.append(idx_overlap[0])
        idx_nonoverlap = np.setdiff1d(range(np.shape(ROIouter_fill_coord)[0]), idx_overlap_outer)
        ROIdonut_coord = np.transpose(
            np.vstack((ROIouter_fill_coord[idx_nonoverlap, 0], ROIouter_fill_coord[idx_nonoverlap, 1])))
        ROI_donut_trace = np.zeros(np.shape(tiff_stack)[0])
        for f in range(np.shape(tiff_stack)[0]):
            ROI_donut_trace[f] = np.nansum(tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]])
        coeff_sub = 1
        roi_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            roi_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                coord_fiji[rfiji - 1][:, 1] * self.pixel_to_um), np.int64(
                coord_fiji[rfiji - 1][:, 0] * self.pixel_to_um)])
        roi_trace_arr = roi_trace_tiffmean / len(coord_fiji[rfiji - 1][:, 1])
        donut_trace_arr = ROI_donut_trace / len(ROIdonut_coord[:, 1])
        roi_trace_bgsub_arr = roi_trace_arr - (coeff_sub * donut_trace_arr)
        idx_neg = np.where(roi_trace_bgsub_arr < 0)[0]
        roi_trace_bgsub_arr[idx_neg] = 0
        roi_trace_bgsub_minmax = (roi_trace_bgsub_arr - np.min(roi_trace_bgsub_arr)) / (
                np.max(roi_trace_bgsub_arr) - np.min(roi_trace_bgsub_arr))
        trace_roi = trace[rext, :]
        # compute events in fiji bgsub and extract trace
        timeT = 10
        [JoinedPosSet_fiji, JoinedNegSet_fiji, F_Values_fiji] = ST.SlopeThreshold(roi_trace_bgsub_minmax,
                                                                                  amp_fiji, timeT,
                                                                                  CollapSeq=True, acausal=False,
                                                                                  verbose=0, graph=None)
        events_fiji = ST.event_detection_calcium_trace(roi_trace_bgsub_minmax, JoinedPosSet_fiji, timeT)
        [JoinedPosSet_ext, JoinedNegSet_ext, F_Values_ext] = ST.SlopeThreshold(trace_roi, amp_ext,
                                                                               timeT, CollapSeq=True, acausal=False,
                                                                               verbose=0, graph=None)
        events_ext = ST.event_detection_calcium_trace(trace_roi, JoinedPosSet_ext, timeT)
        if plot_data:
            fig = plt.figure(figsize=(20, 10), tight_layout=True)
            gs = fig.add_gridspec(2, 4)
            ax1 = fig.add_subplot(gs[0, 0])
            tiff_stack_ave = np.mean(tiff_stack, axis=0)
            ax1.plot(coord_fiji[rfiji - 1][:, 0], coord_fiji[rfiji - 1][:, 1], linewidth=2, color='blue')
            ax1.scatter(coord_cell[rext][:, 0], coord_cell[rext][:, 1], color='red')
            ax1.imshow(ref_image,
                       extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                               0], cmap=plt.get_cmap('gray'))
            ax1.set_title('Fiji ROI ' + str(rfiji) + ' EXTRACT ROI ' + str(rext), fontsize=self.fsize)
            ax1.set_xlim([cent_fiji[0] - 100, cent_fiji[0] + 100])
            ax1.set_ylim([cent_fiji[1] - 100, cent_fiji[1] + 100])
            ax1.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax1.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax1.tick_params(axis='x', labelsize=self.fsize - 4)
            ax1.tick_params(axis='y', labelsize=self.fsize - 4)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(coord_fiji[rfiji - 1][:, 0], coord_fiji[rfiji - 1][:, 1], linewidth=2, color='blue')
            ax2.scatter(ROIdonut_coord[:, 0] / self.pixel_to_um, ROIdonut_coord[:, 1] / self.pixel_to_um, s=10,
                        color='green')
            ax2.imshow(ref_image,
                       extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                               0], cmap=plt.get_cmap('gray'))
            ax2.set_title('Fiji ROI ' + str(rfiji - 1) + ' and respective background', fontsize=self.fsize - 4)
            ax2.set_xlim([cent_fiji[0] - 100, cent_fiji[0] + 100])
            ax2.set_ylim([cent_fiji[1] - 100, cent_fiji[1] + 100])
            ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.tick_params(axis='x', labelsize=self.fsize - 4)
            ax2.tick_params(axis='y', labelsize=self.fsize - 4)
            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(np.linspace(0, np.shape(tiff_stack)[0] / self.sr, np.shape(tiff_stack)[0]), trace_roi, color='red',
                     label='EXTRACT signal')
            ax3.scatter(events_ext / self.sr, trace_roi[events_ext], s=20, color='red')
            ax3.plot(np.linspace(0, np.shape(tiff_stack)[0] / self.sr, np.shape(tiff_stack)[0]), roi_trace_bgsub_minmax,
                     color='blue', label='background subtracted Fiji signal')
            ax3.scatter(events_fiji / self.sr, roi_trace_bgsub_minmax[events_fiji], s=20, color='blue')
            ax3.legend(frameon=False, fontsize=self.fsize - 8)
            ax3.set_ylabel('Signal amplitude', fontsize=self.fsize - 4)
            ax3.set_ylabel('Time (s)', fontsize=self.fsize - 4)
            ax3.spines['right'].set_visible(False)
            ax3.spines['top'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            ax4 = fig.add_subplot(gs[0, 2:])
            ax4.scatter(events_fiji / self.sr, np.ones(len(events_fiji)), s=20, color='blue')
            ax4.scatter(events_ext / self.sr, np.ones(len(events_ext)) * 1.2, s=20, color='red')
            ax4.set_xticks(np.arange(0, np.shape(tiff_stack)[0], 250) / self.sr)
            ax4.set_xticklabels(map(str, np.round(np.arange(0, np.shape(tiff_stack)[0], 250) / self.sr, 2)),
                                rotation=45)
            ax4.set_yticks([1, 1.2])
            ax4.set_ylim([0.9, 1.3])
            ax4.set_ylabel('Event frame Fiji', fontsize=self.fsize - 4)
            ax4.set_ylabel('Event frame EXTRACT', fontsize=self.fsize - 4)
            ax4.spines['right'].set_visible(False)
            ax4.spines['top'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if not os.path.exists(self.path + 'EXTRACT\\EXTRACT comparisons'):
                os.mkdir(self.path + 'EXTRACT\\EXTRACT comparisons')
            if self.delim == '/':
                plt.savefig(self.path + 'EXTRACT/EXTRACT comparisons/' + 'fiji_bgsub_' + str(rfiji) + '_ext_' + str(
                    rext) + '_T' + str(trial), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'EXTRACT\\EXTRACT comparisons\\' + 'fiji_bgsub_' + str(rfiji) + '_ext_' + str(
                    rext) + '_T' + str(trial), dpi=self.my_dpi)
        return coord_fiji, roi_trace_bgsub_minmax, trace_roi

    def plot_events_roi_examples_bgsub(self, trial_plot, roi_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm,
                                       df_events_all, df_events_unsync, plot_data):
        """Function to plot events on top of traces with and without background subtraction for an example.
        Input:
        trial_plot: (str)
        roi_plot: (str)
        frame_time: list with mscope timestamps
        df_fiji_norm: dataframe with traces raw
        df_fiji_bgsub_norm: dataframe with traces background subtracted
        df_events_all: dataframe with all the events
        df_events_unsync: dataframe with unsynchronous the events
        plot_data: boolean"""
        fig, ax = plt.subplots(figsize=(20, 7), tight_layout=True)
        ax.plot(frame_time[trial_plot - 1],
                df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)],
                color='black')
        events_plot = np.where(df_events_all.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[0]
        for e in events_plot:
            ax.scatter(frame_time[trial_plot - 1][e],
                       np.array(df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[e], s=60,
                       color='orange')
        ax.plot(frame_time[trial_plot - 1],
                df_fiji_bgsub_norm.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)] + 5,
                color='grey')
        events_unsync_plot = \
        np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[0]
        for e in events_unsync_plot:
            ax.scatter(frame_time[trial_plot - 1][e],
                       np.array(
                           df_fiji_bgsub_norm.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[
                           e] + 5, s=20, color='red')
        if plot_data:
            if self.delim == '/':
                plt.savefig(
                    self.path + 'images/events/' + 'event_example_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                    dpi=self.my_dpi)
            else:
                plt.savefig(
                    self.path + 'images\\events\\' + 'event_example_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                    dpi=self.my_dpi)

    def plot_events_roi_trial_bgsub(self, trial_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all,
                                    df_events_unsync, plot_data):
        """Function to plot events on top of traces with and without background subtraction for all ROIs and one trial.
        Input:
        trial_plot: (str)
        roi_plot: (str)
        frame_time: list with mscope timestamps
        df_fiji_norm: dataframe with traces raw
        df_fiji_bgsub_norm: dataframe with traces background subtracted
        df_events_all: dataframe with all the events
        df_events_unsync: dataframe with unsynchronous the events
        plot_data: boolean"""
        df_fiji_trial_norm = df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot]  # get dFF for the desired trial
        df_fiji_bgsub_trial_norm = df_fiji_bgsub_norm.loc[
            df_fiji_bgsub_norm['trial'] == trial_plot]  # get dFF for the desired trial
        fig, ax = plt.subplots(1, 2, figsize=(20, 20), tight_layout=True)
        ax = ax.ravel()
        for r in range(df_fiji_trial_norm.shape[1] - 2):
            ax[0].plot(frame_time[trial_plot - 1], df_fiji_trial_norm['ROI' + str(r + 1)] + (r * 10), color='black')
            events_plot = np.where(df_events_all.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
            for e in events_plot:
                ax[0].scatter(frame_time[trial_plot - 1][e], df_fiji_trial_norm.iloc[e, r + 2] + (r * 10), s=20,
                              color='gray')
            events_unsync_plot = \
                np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
            for e in events_unsync_plot:
                ax[0].scatter(frame_time[trial_plot - 1][e], df_fiji_bgsub_trial_norm.iloc[e, r + 2] + (r * 10), s=20,
                              color='orange')
        ax[0].set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax[0].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        plt.setp(ax[0].get_yticklabels(), visible=False)
        ax[0].tick_params(axis='y', which='y', length=0)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=0, length=0)
        for r in range(df_fiji_bgsub_trial_norm.shape[1] - 2):
            ax[1].plot(frame_time[trial_plot - 1], df_fiji_bgsub_trial_norm['ROI' + str(r + 1)] + (r * 10),
                       color='black')
            events_unsync_plot = \
                np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
            for e in events_unsync_plot:
                ax[1].scatter(frame_time[trial_plot - 1][e], df_fiji_bgsub_trial_norm.iloc[e, r + 2] + (r * 10), s=20,
                              color='gray')
        ax[1].set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax[1].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        plt.setp(ax[1].get_yticklabels(), visible=False)
        ax[1].tick_params(axis='y', which='y', length=0)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=0, length=0)
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path + 'images/events/' + 'events_trial' + str(trial_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\events\\' + 'events_trial' + str(trial_plot),
                            dpi=self.my_dpi)
        return

    def plot_events_roi_trial(self, trial_plot, roi_plot, frame_time, df_dff, df_events, plot_data):
        """Function to plot events on top of traces with and without background subtraction for all ROIs and one trial.
        Input:
        trial_plot: (str)
        roi_plot: (str)
        frame_time: list with mscope timestamps
        df_dff: dataframe with traces
        df_events: dataframe with the events
        plot_data: boolean"""
        df_dff_trial = df_dff.loc[df_dff['trial'] == trial_plot, 'ROI' + str(roi_plot)]  # get dFF for the desired trial
        fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
        ax.plot(frame_time[trial_plot - 1], df_dff_trial, color='black')
        events_plot = np.where(df_events.loc[df_dff['trial'] == trial_plot, 'ROI' + str(roi_plot)])[0]
        for e in events_plot:
            ax.scatter(frame_time[trial_plot - 1][e], df_dff_trial.iloc[e], s=20,
                       color='gray')
        ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='y', which='y', length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=0, length=0)
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path + 'images/events/' + 'events_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\events\\' + 'events_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                            dpi=self.my_dpi)
        return

    def sort_HF_files_by_days(self,animal_name):
        """Function to sort mat files from injections experiment across days
        Inputs:
            animal_name: string with the animal name to sort"""
        matfiles = glob.glob(self.path+'*.mat')
        delim = self.path[-1]
        matfiles_animal = []
        days_animal = []
        for f in matfiles:
            path_split = f.split(delim)
            filename = path_split[-1][:-4]
            filename_split = filename.split('_')
            if filename_split[0] == animal_name:
                matfiles_animal.append(f)
                days_animal.append(int(filename_split[1][:-4]))
        days_ordered = np.sort(np.array(days_animal)) #reorder days
        files_ordered = [] #order mat filenames by file order
        for f in range(len(matfiles_animal)):
            tr_ind = np.where(days_ordered[f] == days_animal)[0][0]
            files_ordered.append(matfiles_animal[tr_ind])
        return files_ordered

    # def check_f0(self):
    #     """Print ROIs from suite2p (1st trial) with respective 10% percentile"""
    #     fsize = 20
    #     delim = self.path[-1]
    #     if delim == '/':
    #         path_f0 = self.path+'/F0/'
    #     else:
    #         path_f0 = self.path+'\\F0\\'
    #     if not os.path.exists(path_f0):
    #         os.mkdir(path_f0)
    #     if delim == '/':
    #         path_F = 'Suite2p/suite2p/plane0/F.npy'
    #     else:
    #         path_F = 'Suite2p\\suite2p\\plane0\\F.npy'
    #     if delim == '/':       
    #         path_iscell = 'Suite2p/suite2p/plane0/iscell.npy'
    #     else:       
    #         path_iscell = 'Suite2p\\suite2p\\plane0\\iscell.npy'    
    #     F = np.load(self.path+path_F)
    #     iscell = np.load(self.path+path_iscell)
    #     iscell_idx = np.where(iscell[:,0])[0]
    #     F_cell = F[iscell_idx,:]       
    #     #F0 using percentile for each ROI
    #     F0_roi_perc = []
    #     for c in range(len(iscell_idx)):
    #         F0_roi_perc.append(np.percentile(F_cell[c,:],10))
    #     for r in range(np.shape(F_cell)[0]):
    #         fig, ax = plt.subplots(figsize = (10,5), tight_layout=True)
    #         xaxis = np.linspace(0,60,60*self.sr)
    #         plt.plot(xaxis,F_cell[r,:60*self.sr], color = 'black', linewidth=2)
    #         plt.axhline(y=F0_roi_perc[r])
    #         ax.set_xlabel('Time (s)', fontsize = fsize-4)
    #         ax.set_ylabel('Fluorescence', fontsize = fsize-4)
    #         ax.set_title('ROI from suite2p '+str(r+1))
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['top'].set_visible(False)
    #         plt.tick_params(axis='y', labelsize=fsize-4)
    #         plt.tick_params(axis='x', labelsize=fsize-4)
    #         if delim == '/':        
    #             plt.savefig(path_f0+ 'ROI_f0_'+str(r+1), dpi=self.my_dpi)
    #         else:
    #             plt.savefig(path_f0+ 'ROI_f0_'+str(r+1), dpi=self.my_dpi)   
    #         plt.close('all')

    # def get_background_signal(self, weight, coord_cell):
    #     """ Get neuropil background signals for each cell coordinates. Low-pass filter of
    #     image with a Hamming window weight*cell diameter. Background signal is then computed
    #     for each cell (mean for all pixels of that mask)
    #         Inputs:
    #             weight: size of hamming window (weight*cell diameter, diameter comes from ops file
    #             coord_cell: list with cell coordinates"""
    #     delim = self.path[-1]
    #     tiflist = glob.glob(self.path+delim+'Suite2p'+delim+'*.tif')
    #     if not tiflist:
    #         tiflist = glob.glob(self.path+delim+'Suite2p'+delim+'*.tiff') #get list of tifs
    #     delim = self.path[-1]
    #     if delim == '/':
    #         ops = np.load(self.path+'Suite2p/suite2p/plane0/ops.npy', allow_pickle=True)
    #     else:
    #         ops = np.load(self.path+'Suite2p\\suite2p\\plane0\\ops.npy', allow_pickle=True)
    #     diam_cell = ops[()]['diameter'][0]
    #     width = weight*diam_cell
    #     plot_data = 1
    #     Fbg_all = []
    #     for t in tiflist:
    #         print('Low-pass filter of '+ t.split(delim)[-1])
    #         image_stack = tiff.imread(t) #choose right tiff    
    #         ham = np.hamming(np.shape(image_stack)[1])[:,None] # 1D hamming
    #         ham2d = np.sqrt(np.dot(ham, ham.T)) ** width # expand to 2D hamming
    #         image_lpfilt = np.zeros(np.shape(image_stack),dtype=np.uint8)
    #         for f in range(np.shape(image_stack)[0]):
    #             image2 = image_stack[f,:,:]
    #             fft_cv2 = cv2.dft(image2.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
    #             f_shifted = np.fft.fftshift(fft_cv2)
    #             magnitude_spectrum = 20*np.log(cv2.magnitude(f_shifted[:,:,0],f_shifted[:,:,1]))
    #             if plot_data and f == np.shape(image_stack)[0]-1:
    #                 plt.imshow(magnitude_spectrum, cmap = 'gray')
    #                 plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    #                 plt.show()
    #                 plt.savefig('C:\\Users\\Ana\\Desktop\\F0\\LPfilter_spectrumFFT_window'+str(weight), dpi=self.my_dpi)
    #             f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    #             f_filtered = ham2d * f_complex
    #             f_filtered_shifted = np.fft.fftshift(f_filtered)
    #             inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
    #             filtered_img = np.abs(inv_img)
    #             # filtered_img_norm = cv2.normalize(filtered_img, None, alpha=np.min(image_stack[f,:,:]), beta=np.max(image_stack[f,:,:]), norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    #             image_lpfilt[f,:,:] = filtered_img
    #         if plot_data:
    #             plt.figure()
    #             plt.imshow(np.nanmean(image_lpfilt,axis=0))
    #             plt.title('Window size '+str(weight)+' cell diameter')
    #             plt.colorbar()
    #             plt.savefig('C:\\Users\\Ana\\Desktop\\F0\\LPfilter_image_filtersize_'+str(weight), dpi=self.my_dpi)
    #             plt.close('all')
    #         Fbg = np.zeros((len(coord_cell),np.shape(image_lpfilt)[0]))
    #         for r in range(len(coord_cell)):
    #             for f in range(np.shape(image_lpfilt)[0]):
    #                 coord_x = np.int64(coord_cell[r][:,0]*self.pixel_to_um)
    #                 coord_y = np.int64(coord_cell[r][:,1]*self.pixel_to_um)
    #                 Fbg[r,f] = np.mean(image_lpfilt[f,coord_y,coord_x])
    #         Fbg_all.append(Fbg)
    #     return Fbg_all

    # def event_detection_deconvolution(self,dFF_conv,th,dt,deconvtau,frame_time):
    #     """Function to compute deconvolution of the dFF matrix or dataframe using an exponential kernel
    #     as in Yaksi and Friedrich (2006) Nature methods
    #     Inputs: 
    #         dFF_conv = dFF (array or dataframe, single trial or whole session)
    #         th: threshold in standard deviations (e.g. 1)
    #         dt: time window (sample number) (e.g 10)
    #         deconvtau: deconvolution decay time, 1/time in seconds (e.g 1/0.3)
    #         frame_time: list with the frame timestamps per trial"""
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         dFF_array = dFF_conv.to_numpy()
    #         roi_nr = len(np.unique(dFF_array[:,1]))
    #         sample_nr = int(np.shape(dFF_array)[0]/roi_nr)
    #         dFF_array_reshaped = np.reshape(dFF_array[:,0],(roi_nr,sample_nr))
    #     if type(dFF_conv) == np.ndarray:
    #         dFF_array_reshaped = dFF_conv
    #     #filter the data with deconvtau - Butterworth filter
    #     dsig = np.diff(dFF_array_reshaped, n=1, axis = 1)
    #     sig = dFF_array_reshaped/deconvtau + (np.insert(dsig, 0, np.transpose(dsig[:,0]), axis=1)/dt)
    #     # replace inf values (division by 0) with NaN
    #     sig_norm = self.z_score(sig,0)
    #     #normalization
    #     sig_norm[np.isinf(sig_norm)] = float("NaN")
    #     zsig = np.transpose(sig_norm)
    #     pp1 = np.insert(zsig[:-1,:], 0, zsig[0,:], axis=0)
    #     pp2 = np.insert(zsig[1:,:],-1,zsig[-1,:], axis = 0)
    #     spikes_tuple = np.where((zsig>=th)&(zsig-pp1>=0)&(zsig-pp2>=0))
    #     spikes = np.zeros(np.shape(dFF_array_reshaped))
    #     spikes[spikes_tuple[1],spikes_tuple[0]] = 1
    #     frame_df = []
    #     for r in range(np.shape(dFF_array_reshaped)[0]):
    #         for t in range(len(frame_time)):
    #             frame_df.extend(frame_time[t])
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         #make dataframe with spikes matrix 
    #         spikes_flat = spikes.flatten()
    #         dict_spikes = {'spikes': spikes_flat,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #         spmat = pd.DataFrame(dict_spikes) #create dataframe with dFF, roi id and trial id
    #     if type(dFF_conv) == np.ndarray:
    #         spmat = spikes
    #     return spmat

    # def event_detection_firstderivative(self,dFF_conv,th,frame_time):
    #     """Function to find events by taking the first derivative of calcium trace
    #     as in Heffley et al (2018) Nature neuroscience
    #     Inputs: 
    #         dFF_conv = dFF (array or dataframe, single trial or whole session)
    #         th: threshold in standard deviations (e.g. 2.5)
    #         frame_time: list with the frame timestamps per trial"""
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         dFF_array = dFF_conv.to_numpy()
    #         roi_nr = len(np.unique(dFF_array[:,1]))
    #         sample_nr = int(np.shape(dFF_array)[0]/roi_nr)
    #         dFF_array_reshaped = np.reshape(dFF_array[:,0],(roi_nr,sample_nr))
    #     if type(dFF_conv) == np.ndarray:
    #         dFF_array_reshaped = dFF_conv
    #     #filter the data with deconvtau - Butterworth filter
    #     dsig = np.diff(dFF_array_reshaped, n=1, axis = 1)
    #     dsig_zscore = miniscope_session.z_score(dsig,0)
    #     pp1 = np.insert(dsig_zscore[:-1,:], 0, dsig_zscore[0,:], axis=0)
    #     pp2 = np.insert(dsig_zscore[1:,:],-1,dsig_zscore[-1,:], axis = 0)
    #     spikes_tuple = np.where((dsig_zscore>=th)&(dsig_zscore-pp1>=0)&(dsig_zscore-pp2>=0))
    #     spikes_diff = np.zeros(np.shape(dFF_array_reshaped))
    #     spikes_diff[spikes_tuple[0],spikes_tuple[1]+1] = 1
    #     frame_df = []
    #     for r in range(np.shape(dFF_array_reshaped)[0]):
    #         for t in range(len(frame_time)):
    #             frame_df.extend(frame_time[t])
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         #make dataframe with spikes matrix 
    #         spikes_flat_diff = spikes_diff.flatten()
    #         dict_spikes_diff = {'spikes': spikes_flat_diff,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #         spmat = pd.DataFrame(dict_spikes_diff) #create dataframe with dFF, roi id and trial id
    #     if type(dFF_conv) == np.ndarray:
    #         spmat = spikes_diff
    #     return spmat

    # def event_detection_templatematching(self,dFF_conv,th,frame_time):
    #     """Function to find events by taking a template-matching approach
    #     as in Ozden et al (2008) J.Neurophys.
    #     Inputs: 
    #         dFF_conv = dFF (array or dataframe, single trial or whole session)
    #         th: threshold in standard deviations (e.g. 1)
    #         frame_time: list with the frame timestamps per trial"""
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         dFF_array = dFF_conv.to_numpy()
    #         roi_nr = len(np.unique(dFF_array[:,1]))
    #         sample_nr = int(np.shape(dFF_array)[0]/roi_nr)
    #         dFF_array_reshaped = np.reshape(dFF_array[:,0],(roi_nr,sample_nr))
    #     if type(dFF_conv) == np.ndarray:
    #         dFF_array_reshaped = dFF_conv
    #     spikes_temp = np.zeros(np.shape(dFF_array_reshaped))
    #     for r in range(roi_nr):
    #         peaks = np.where(dFF_array_reshaped[r,:]>np.percentile(dFF_array_reshaped[r,:],90))[0] #get peaks
    #         #remove consecutive peaks
    #         peaks_to_del = []
    #         for p in range(len(peaks)-1):
    #             if peaks[p+1]-peaks[p] == 1:
    #                 peaks_to_del.append(peaks[p+1])
    #         peaks_single = np.setdiff1d(peaks, np.array(peaks_to_del))
    #         template_array = np.zeros((len(peaks_single),4))
    #         #build template array
    #         count_p = 0
    #         for p in peaks_single:
    #             if p>0:
    #                 template_array[count_p,:] = miniscope_session.z_score(dFF_array_reshaped[r,p-1:p+3],0)
    #                 count_p += 1
    #         template_ave = np.nanmean(template_array,axis=0) #average peaks
    #         # plt.plot(template_ave)
    #         conv_template = np.convolve(miniscope_session.z_score(dFF_array_reshaped[r,:],0), template_ave, mode='same')
    #         # plt.plot(conv_template)
    #         spike_th_idx = np.where(conv_template>th)
    #         spikes_temp[r,spike_th_idx] = 1
    #     frame_df = []
    #     for r in range(np.shape(dFF_array_reshaped)[0]):
    #         for t in range(len(frame_time)):
    #             frame_df.extend(frame_time[t])
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         #make dataframe with spikes matrix 
    #         spikes_flat_temp = spikes_temp.flatten()
    #         dict_spikes_temp = {'spikes': spikes_flat_temp,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #         spmat = pd.DataFrame(dict_spikes_temp) #create dataframe with dFF, roi id and trial id
    #     if type(dFF_conv) == np.ndarray:
    #         spmat = spikes_temp
    #     return spmat

    # def event_detection_mlspike(self,dFF_conv,frame_time,pruning):
    #     """Function to find events after MLspike (ran in MATLAB) as in Deneux et al (2016)
    #     Nature communications.
    #     Inputs: 
    #         dFF_conv = dFF (array or dataframe, single trial or whole session)
    #         frame_time: list with the frame timestamps per trial
    #         pruning: boolean (to remove consecutive spikes)"""
    #     delim = self.path[-1]
    #     spikes_ml = pickle.load(open(self.path+delim+'processed files'+delim+'spikes.pkl', "rb" ))
    #     fit_ml = pickle.load(open(self.path+delim+'processed files'+delim+'fit.pkl', "rb" ))
    #     drift_ml = pickle.load(open(self.path+delim+'processed files'+delim+'drift.pkl', "rb" ))
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         dFF_array = dFF_conv.to_numpy()
    #         roi_nr = len(np.unique(dFF_array[:,1]))
    #         sample_nr = int(np.shape(dFF_array)[0]/roi_nr)
    #         dFF_array_reshaped = np.reshape(dFF_array[:,0],(roi_nr,sample_nr))
    #     if type(dFF_conv) == np.ndarray:
    #         dFF_array_reshaped = dFF_conv
    #     frame_df = []
    #     for r in range(np.shape(dFF_array_reshaped)[0]):
    #         for t in range(len(frame_time)):
    #             frame_df.extend(frame_time[t])
    #     if pruning: #because larger amplitudes have more spikes assigned
    #         spikes_ml_pruning = np.zeros(np.shape(spikes_ml))
    #         for r in range(np.shape(dFF_array_reshaped)[0]):
    #             spikes_ml_roi = np.where(spikes_ml[r,:]>0)[0]
    #             #remove consecutive spikes
    #             spikes_to_del_ml = []
    #             for p in range(len(spikes_ml_roi)-1):
    #                 if spikes_ml_roi[p+1]-spikes_ml_roi[p] == 1:
    #                     spikes_to_del_ml.append(spikes_ml_roi[p+1])
    #             spikes_single_ml = np.setdiff1d(spikes_ml_roi, np.array(spikes_to_del_ml))
    #             spikes_ml[r,spikes_ml_roi] = 1
    #             spikes_ml_pruning[r,spikes_single_ml] = 1
    #     if isinstance(dFF_conv, pd.DataFrame):  
    #         #make dataframe with spikes matrix 
    #         fit_ml_flat = fit_ml.flatten()
    #         drift_ml_flat = drift_ml.flatten()
    #         dict_fit_ml = {'fit': fit_ml_flat,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #         dict_drift_ml = {'drift': drift_ml_flat,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #         fit_ml_arr = pd.DataFrame(dict_fit_ml)
    #         drift_ml_arr = pd.DataFrame(dict_drift_ml)
    #         if pruning:
    #             spikes_flat_ml_pruning = spikes_ml_pruning.flatten()
    #             dict_spikes_ml_pruning = {'spikes': spikes_flat_ml_pruning,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #             spmat_ml = pd.DataFrame(dict_spikes_ml_pruning) #create dataframe with dFF, roi id and trial id
    #         else:
    #             spikes_flat_ml = spikes_ml.flatten()
    #             dict_spikes_ml = {'spikes': spikes_flat_ml,'roi': dFF_array[:,1],'trial': dFF_array[:,2],'time': frame_df}
    #             spmat_ml = pd.DataFrame(dict_spikes_ml) #create dataframe with dFF, roi id and trial id                
    #     if type(dFF_conv) == np.ndarray:
    #         spmat_ml = spikes_ml
    #         if pruning:
    #             spmat_ml = spikes_ml_pruning
    #     return fit_ml_arr, drift_ml_arr, spmat_ml

        
            











                
