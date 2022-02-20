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
import matplotlib.patches as mp
import SlopeThreshold as ST
import read_roi

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
    def norm_traces(df_fiji, norm_name):
        """Function to do a min-max normalization or z-scoring of the calcim trace of the dataframe
        Input:
            df_fiji: dataframe with session info and calcium traces
            nrom-name: (str) min_max or zscore"""
        if norm_name == 'minmax':
            min_fiji = np.tile(np.array(df_fiji.iloc[:, 2:].min(axis=0)), (np.shape(df_fiji)[0], 1))
            arr_fiji_norm = (np.array(df_fiji.iloc[:, 2:]) - min_fiji) / (
                        np.array(df_fiji.iloc[:, 2:].max(axis=0)) - np.array(df_fiji.iloc[:, 2:].min(axis=0)))
        if norm_name == 'zscore':
            df_fiji_mean = np.tile(np.array(df_fiji.iloc[:, 2:].mean(axis=0)), (np.shape(df_fiji)[0], 1))
            df_fiji_std = np.tile(np.array(df_fiji.iloc[:, 2:].std(axis=0, ddof=0)), (np.shape(df_fiji)[0], 1))
            arr_fiji_norm = (np.array(df_fiji.iloc[:, 2:]) - df_fiji_mean) / df_fiji_std
        roi_list = list(df_fiji.columns[2:])
        trial_ext = list(df_fiji['trial'])
        frame_time_ext = list(df_fiji['time'])
        data_fiji1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_fiji1 = pd.DataFrame(data_fiji1)
        df_fiji2 = pd.DataFrame(arr_fiji_norm, columns=roi_list)
        df_fiji_minmax = pd.concat([df_fiji1, df_fiji2], axis=1)
        return df_fiji_minmax

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
    def get_roi_list(coord_cell):
        """ Get number of ROI names for the session"""
        roi_list =  []
        for r in range(len(coord_cell)):
            roi_list.append('ROI'+str(r+1))
        return roi_list

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
    
    def trial_length(self):
        """ Get number of frames for each trial"""
        tiflist = glob.glob(self.path+self.delim+'Suite2p'+self.delim+'*.tif')
        if not tiflist:
            tiflist = glob.glob(self.path+self.delim+'Suite2p'+self.delim+'*.tiff') #get list of tifs
        trial_id = []
        for t in range(len(tiflist)):
            tifname = tiflist[t].split(self.delim)   
            tifname_split = tifname[-1].split('_')
            trial_id.append(int(tifname_split[0][1:])) #get trial order in that list    
        trial_order = np.sort(trial_id) #reorder trials
        files_ordered = [] #order tif filenames by file order
        trial_length = []
        for f in range(len(tiflist)):
            tr_ind = np.where(trial_order[f] == trial_id)[0][0]
            files_ordered.append(tiflist[tr_ind])
        #get size of trials
        for f in range(len(files_ordered)):
            image_stack = tiff.imread(files_ordered[f]) #choose right tiff
            trial_length.append(np.shape(image_stack)[0])
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
   
    def get_ref_image(self):
        """Function to get the session reference image from suite2p"""
        if self.delim == '/':
            path_ops = 'Suite2p'+self.delim+'suite2p/plane0/ops.npy'
        else:
            path_ops = 'Suite2p'+self.delim+'suite2p\\plane0\\ops.npy'
        ops = np.load(self.path+path_ops,allow_pickle=True)
        ref_image = ops[()]['meanImg']
        return ref_image
    
    def read_extract_output(self, threshold_spatial_weights, frame_time, trial):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (EXTRACT output)
        Inputs:
            threshold_spatial_weights: float
            frame_time: list with miniscope timestamps
            trial: (int) - while EXTRACT is run on a single trial"""
        if self.delim == '/':
            path_extract = self.path + '/EXTRACT/'
        if self.delim == '\\':
            path_extract = self.path + '\\EXTRACT\\'
        ext_rois = mat73.loadmat(path_extract+'extract_output.mat')
        spatial_weights = ext_rois['spatial_weights']
        trace_ext = ext_rois['trace_nonneg']
        coord_cell = []
        for c in range(np.shape(spatial_weights)[2]):
            coord_cell.append(np.transpose(np.array(np.where(spatial_weights[:, :, c]>threshold_spatial_weights))))
        coord_cell_t = []
        for c in range(len(coord_cell)):
            coord_cell_switch = np.zeros(np.shape(coord_cell[c]))
            coord_cell_switch[:,0] = coord_cell[c][:,1]/self.pixel_to_um
            coord_cell_switch[:,1] = coord_cell[c][:,0]/self.pixel_to_um
            coord_cell_t.append(coord_cell_switch)
        # trace as dataframe
        roi_list =  []
        for r in range(len(coord_cell)):
            roi_list.append('ROI'+str(r+1))
        data_ext1 = {'trial': np.repeat(2, len(frame_time[trial - 1])), 'time': frame_time[trial - 1]}
        df_ext1 = pd.DataFrame(data_ext1)
        df_ext2 = pd.DataFrame(trace_ext, columns=roi_list)
        df_ext = pd.concat([df_ext1, df_ext2], axis=1)
        return coord_cell_t, df_ext

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

    def roi_curation(self, ref_image, df_dFF, coord_cell, trial_curation):
        """Check each ROI spatial and temporally (for a certain trial) and choose the ones to keep.
        Enter to keep and click to discard.
        Input:
            ref_image: array with reference image
            df_dFF: dataframe with calcium trace values
            coord_cell: list with ROI coordinates
            trial_curation: (int) trial to plot"""
        skewness_dFF = df_dFF.skew(axis=0, skipna=True)
        skewness_dFF_argsort = np.argsort(np.array(skewness_dFF[2:]))
        # ROI curation (ΔF/F with the same range)
        range_dFF = [df_dFF.loc[df_dFF['trial'] == trial_curation].min(axis=0)[2:].min(),
                     df_dFF.loc[df_dFF['trial'] == trial_curation].max(axis=0)[2:].max()]
        keep_roi = []
        count_r = 0
        for r in skewness_dFF_argsort[::-1]:  # check by descending order of skewness
            fig = plt.figure(figsize=(25, 7), tight_layout=True)
            gs = fig.add_gridspec(1, 3)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(np.linspace(0, 110 - 60, df_dFF.loc[df_dFF['trial'] == trial_curation].shape[0]),
                     df_dFF.loc[(df_dFF['trial'] == trial_curation), 'ROI' + str(r + 1)], color='black')
            ax1.set_title('Trial ' + str(trial_curation) + ' ROI ' + str(count_r + 1) + '/' + str(
                df_dFF.loc[df_dFF['trial'] == trial_curation].shape[1]))
            ax1.set_ylim(range_dFF)
            ax1.set_xlabel('Time (s)')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], s=1, color='blue', alpha=0.5)
            ax2.set_title('ROI '+str(r+1), fontsize=self.fsize)
            ax2.imshow(ref_image,
                       extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                               0], cmap=plt.get_cmap('gray'))
            ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.tick_params(axis='x', labelsize=self.fsize - 4)
            ax2.tick_params(axis='y', labelsize=self.fsize - 4)
            ax2.set_xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.set_ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            ax2.tick_params(axis='x', labelsize=self.fsize - 4)
            ax2.tick_params(axis='y', labelsize=self.fsize - 4)
            bpress = plt.waitforbuttonpress()
            if bpress:
                keep_roi.append(r)
            plt.close('all')
            count_r += 1
        # remove bad rois
        list_bad_rois = np.setdiff1d(np.arange(0, df_dFF.shape[1] - 2), keep_roi)
        bad_roi_list = []
        for r in list_bad_rois:
            bad_roi_list.append('ROI' + str(r + 1))
        df_dFF_clean = df_dFF.drop(bad_roi_list, axis=1)
        return keep_roi, df_dFF_clean

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
                for r in range(df_dFF.shape[1] - 2):
                    ax[count_t].plot(frame_time[t - 1], dFF_trial['ROI' + str(r + 1)] + (r / 2), color='black')
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
            for r in range(df_dFF.shape[1] - 2):
                plt.plot(frame_time[trials_plot - 1], dFF_trial['ROI' + str(r + 1)] + (r / 2), color='black')
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
        furthest_neuron = np.argmax(np.array(centroid_cell)[:, 0])
        neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
        dFF_trial = df_dFF.loc[df_dFF['trial'] == trial_plot]  # get dFF for the desired trial
        fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
        count_r = 0
        for r in neuron_order:
            plt.plot(frame_time[trial_plot - 1], dFF_trial['ROI' + str(r + 1)] + (count_r / 2), color=colors[idx[r] - 1])
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
            ell = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji] + 15,
                             -(90 - np.degrees(np.arctan(ylength_fiji[rfiji] / xlength_fiji[rfiji]))))
            ell2 = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji] + 30,
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
                tiff_stack = tiff.imread(self.path + 'Registered video\\T' + str(t) + '_reg.tif')  ##read tiffs
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
        return df_fiji

    def save_processed_files(self, df_fiji, df_trace_bgsub, df_extract, trials, coord_fiji, coord_ext, th, idx_to_nan):
        """Saves calcium traces, ROI coordinates, trial number and motion frames.
        Saves them under path/processed files
        Inputs:
            df_dFF: dataframe with calcium trace from ImageJ
            df_trace_bgsub: dataframe with calcium trace from ImageJ
            df_extract: dataframe with calcium trace from EXTRACT
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
            np.save(self.path + '/processed files/' + 'coord_fiji.npy', coord_fiji, allow_pickle = True)
            np.save(self.path + '/processed files/' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '/processed files/' + 'trials.npy', trials)
            np.save(self.path + '/processed files/' + 'reg_th.npy', th)
            np.save(self.path + '/processed files/' + 'frames_to_exclude.npy', idx_to_nan)
        else:
            df_fiji.to_csv(self.path + '\\processed files\\' + 'df_fiji.csv', sep=',', index = False)
            df_trace_bgsub.to_csv(self.path + '\\processed files\\' + 'df_fiji_bgsub.csv', sep=',', index = False)
            df_extract.to_csv(self.path + '\\processed files\\' + 'df_extract.csv', sep=',', index = False)
            np.save(self.path + '\\processed files\\' + 'coord_fiji.npy', coord_fiji, allow_pickle = True)
            np.save(self.path + '\\processed files\\' + 'coord_ext.npy', coord_ext, allow_pickle = True)
            np.save(self.path + '\\processed files\\' + 'trials.npy', trials)
            np.save(self.path + '\\processed files\\' + 'reg_th.npy', th)
            np.save(self.path + '\\processed files\\' + 'frames_to_exclude.npy', idx_to_nan)
        return

    def load_processed_files(self):
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

    def get_nearest_rois_manual_roi(self,rois_df,centroid_cell,rfiji):
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
    
    def plot_overlap_extract_manual_rois(self, rfiji, rext, ref_image, rois_df, coord_cell, roi_trace_minmax_bgsub, trace):
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
        ax2.set_title('Manual ' + str(rfiji+1) + ' and corresponding EXTRACT ROI ' + str(rext), fontsize=self.fsize)
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

    def compare_extract_extract_rois(self,r1,r2,coord_cell1,coord_cell2,trace1,trace2,ref_image,comparison):
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
                 trace1[r1, :], color='orange', label='no '+comparison)
        ax2.plot(np.linspace(0, np.shape(trace2)[1] / self.sr, len(trace2[r2, :])),
                 trace2[r2, :], color='blue', label=comparison)
        ax2.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax2.set_ylabel('Amplitude of F values', fontsize=self.fsize - 4)
        ax2.tick_params(axis='x', labelsize=self.fsize - 4)
        ax2.tick_params(axis='y', labelsize=self.fsize - 4)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        return

    def compute_bg_roi_fiji_example(self,trial,ref_image,rois_df,rfiji):
        """Function to compute a donut background around one FIJI ROI and compute its background subtracted signal.
        Plot all steps.
        Input:
            trial: int
            ref_image: array with reference image from Suite2p
            rois_df: dataframe with ROIs from FIJI
            rfiji: int"""
        tiff_stack = tiff.imread(self.path + '\\Registered video\\T'+str(trial)+'_reg.tif')  ##read tiffs
        coord_fiji = []
        height_fiji = []
        xlength_fiji = []
        ylength_fiji = []
        for r in range(np.shape(rois_df)[0]):
           coord_r = np.transpose(np.vstack(
               (np.linspace(rois_df.iloc[r, 0]/self.pixel_to_um, rois_df.iloc[r, 1]/self.pixel_to_um, 100),
                np.linspace(rois_df.iloc[r, 2]/self.pixel_to_um, rois_df.iloc[r, 3]/self.pixel_to_um, 100))))
           x_length = np.abs(coord_r[-1, 0] - coord_r[0, 0])
           y_length = np.abs(coord_r[-1, 1] - coord_r[0, 1])
           xlength_fiji.append(x_length)
           ylength_fiji.append(y_length)
           coord_fiji.append(coord_r)
           height_fiji.append(np.sqrt(np.square(x_length) + np.square(y_length)))
        cent_fiji = np.nanmean(coord_fiji[rfiji - 1], axis=0)
        ell = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji - 1]+15,
                        -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ell2 = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji - 1]+30,
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
        roi_trace_arr = roi_trace_tiffmean/len(coord_fiji[rfiji - 1][:, 1])
        donut_trace_arr = ROI_donut_trace/len(ROIdonut_coord[:, 1])
        roi_trace_bgsub_arr = roi_trace_arr - (coeff_sub * donut_trace_arr)
        idx_neg = np.where(roi_trace_bgsub_arr < 0)[0]
        roi_trace_bgsub_arr[idx_neg] = 0
        roi_trace_bgsub_minmax = (roi_trace_bgsub_arr-np.min(roi_trace_bgsub_arr))/(np.max(roi_trace_bgsub_arr)-np.min(roi_trace_bgsub_arr))
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
           plt.savefig(self.path + 'EXTRACT/Fiji ROI bgsub/' + 'fiji_bgsub_'+str(rfiji)+'_T'+str(trial), dpi=self.my_dpi)
        else:
           plt.savefig(self.path + 'EXTRACT\\Fiji ROI bgsub\\' + 'fiji_bgsub_'+str(rfiji)+'_T'+str(trial), dpi=self.my_dpi)
        return

    def compute_bg_roi_fiji_extract(self,trial,ref_image,rois_df,coord_cell,trace,rfiji,rext,amp_fiji,amp_ext,plot_data):
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
        tiff_stack = tiff.imread(self.path+'Registered video\\T'+str(trial)+'_reg.tif')  ##read tiffs
        coord_fiji = []
        height_fiji = []
        xlength_fiji = []
        ylength_fiji = []
        for r in range(np.shape(rois_df)[0]):
           coord_r = np.transpose(np.vstack(
               (np.linspace(rois_df.iloc[r, 0]/self.pixel_to_um, rois_df.iloc[r, 1]/self.pixel_to_um, 100),
                np.linspace(rois_df.iloc[r, 2]/self.pixel_to_um, rois_df.iloc[r, 3]/self.pixel_to_um, 100))))
           x_length = np.abs(coord_r[-1, 0] - coord_r[0, 0])
           y_length = np.abs(coord_r[-1, 1] - coord_r[0, 1])
           xlength_fiji.append(x_length)
           ylength_fiji.append(y_length)
           coord_fiji.append(coord_r)
           height_fiji.append(np.sqrt(np.square(x_length) + np.square(y_length)))
        cent_fiji = np.nanmean(coord_fiji[rfiji - 1], axis=0)
        ell = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 30, height_fiji[rfiji - 1]+15,
                        -(90 - np.degrees(np.arctan(ylength_fiji[rfiji - 1] / xlength_fiji[rfiji - 1]))))
        ell2 = mp.Ellipse((cent_fiji[0], cent_fiji[1]), 50, height_fiji[rfiji - 1]+30,
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
           ax4.scatter(events_fiji/self.sr, np.ones(len(events_fiji)), s=20, color='blue')
           ax4.scatter(events_ext/self.sr, np.ones(len(events_ext)) * 1.2, s=20, color='red')
           ax4.set_xticks(np.arange(0, np.shape(tiff_stack)[0], 250)/self.sr)
           ax4.set_xticklabels(map(str, np.round(np.arange(0, np.shape(tiff_stack)[0], 250)/self.sr,2)), rotation=45)
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
               plt.savefig(self.path + 'EXTRACT/EXTRACT comparisons/' + 'fiji_bgsub_'+str(rfiji)+'_ext_'+str(rext)+'_T'+str(trial), dpi=self.my_dpi)
           else:
               plt.savefig(self.path + 'EXTRACT\\EXTRACT comparisons\\' + 'fiji_bgsub_'+str(rfiji)+'_ext_'+str(rext)+'_T'+str(trial), dpi=self.my_dpi)
        return coord_fiji, roi_trace_bgsub_minmax, trace_roi

    def plot_events_roi_examples(self, trial_plot, roi_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data):
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
        ax.plot(frame_time[trial_plot - 1], df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)],
                color='black')
        events_plot = np.where(df_events_all.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[0]
        for e in events_plot:
            ax.scatter(frame_time[trial_plot - 1][e],
                       np.array(df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[e], s=60,
                       color='orange')
        ax.plot(frame_time[trial_plot - 1],
                df_fiji_bgsub_norm.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)] + 5, color='grey')
        events_unsync_plot = np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[0]
        for e in events_unsync_plot:
            ax.scatter(frame_time[trial_plot - 1][e],
                       np.array(df_fiji_bgsub_norm.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(roi_plot)])[
                           e] + 5, s=20, color='red')
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path + 'images/events/' + 'event_example_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + 'images\\events\\' + 'event_example_trial' + str(trial_plot) + '_roi' + str(roi_plot),
                            dpi=self.my_dpi)

    def plot_events_roi_trial(self, trial_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data):
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
        ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
        ax[0].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
        plt.xticks(fontsize=mscope.fsize - 4)
        plt.yticks(fontsize=mscope.fsize - 4)
        plt.setp(ax[0].get_yticklabels(), visible=False)
        ax[0].tick_params(axis='y', which='y', length=0)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        plt.tick_params(axis='y', labelsize=0, length=0)
        for r in range(df_fiji_bgsub_trial_norm.shape[1] - 2):
            ax[1].plot(frame_time[trial_plot - 1], df_fiji_bgsub_trial_norm['ROI' + str(r + 1)] + (r * 10), color='black')
            events_unsync_plot = \
            np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
            for e in events_unsync_plot:
                ax[1].scatter(frame_time[trial_plot - 1][e], df_fiji_bgsub_trial_norm.iloc[e, r + 2] + (r * 10), s=20,
                              color='gray')
        ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
        ax[1].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
        plt.xticks(fontsize=mscope.fsize - 4)
        plt.yticks(fontsize=mscope.fsize - 4)
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

    def get_events(self, df_dFF, timeT, csv_name):
        """"Function to get the calcium event using Jorge's derivative method.
        Amplitude of envelope is determined as the median absolute deviation
        Inputs:
        df_dFF: dataframe with calcium trace values
        timeT: time thershold (within how many frames does the event happen)
        csv_name: (str) filename of df_events create"""
        roi_trace = np.array(df_dFF.iloc[:,2:])
        roi_list = list(df_dFF.columns[2:])
        trial_ext = list(df_dFF['trial'])
        frame_time_ext = list(df_dFF['time'])
        data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_dFF1 = pd.DataFrame(data_dFF1)
        df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
        df_events = pd.concat([df_dFF1, df_dFF2], axis=1)
        roi_list = df_dFF.columns[2:]
        for r in roi_list:
            data = np.array(df_dFF[r])
            events_mat = np.zeros(len(data))
            data_mad = mad(data, nan_policy='omit') #TODO CHANGE TO MAD OF DIFF (MULTIPLIED??)
            [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(data, data_mad, timeT,
                                                                                   CollapSeq=True, acausal=False,
                                                                                   verbose=0, graph=None)
            if len(JoinedPosSet_all) == 0:
                print('No events for trial' + str(t) + ' and ' + r)
            else:
                events_mat[ST.event_detection_calcium_trace(data, JoinedPosSet_all, timeT)] = 1
            df_events[r] = events_mat
        df_events.to_csv(self.path + '\\processed files\\' + csv_name + '.csv', sep=',', index=False)
        return df_events

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
                isi = df_events.loc[(df_events[r] == 1) & (df_events['trial'] == t), 'time'].diff()
                isi_all.extend(np.array(isi))
                trial_all.extend(np.repeat(t, len(isi)))
                roi_all.extend(np.repeat(r, len(isi)))
        dict_isi = {'isi': isi_all, 'roi': roi_all, 'trial': trial_all}
        isi_df = pd.DataFrame(dict_isi)  # create dataframe with isi, roi id and trial id
        isi_df.to_csv(self.path + '\\processed files\\' + csv_name + '.csv', sep=',', index=False)
        return isi_df

    @staticmethod
    def compute_isi_cv(isi_df):
        """Function to compute coefficient of variation and coefficient of variation for adjacent spikes (Isope, JNeurosci)
        Inputs:
            isi_df: (dataframe) with isi values"""
        trials = np.unique(isi_df['trial'])
        isi_cv = np.zeros((len(np.unique(isi_df.roi)),len(trials)))
        isi_cv2 = np.zeros((len(np.unique(isi_df.roi)),len(trials)))
        for t in trials:
            count_r = 0
            for r in np.unique(isi_df.roi):
                data = np.array(isi_df.loc[(isi_df['trial']==t) & (isi_df['roi']==r),'isi'])
                diff_data = np.abs(np.diff(data))
                sum_data = data[:-1]+data[1:]
                isi_cv[count_r,t-1] = np.nanstd(data)/np.nanmean(data)
                isi_cv2[count_r,t-1] = np.nanmean(diff_data/sum_data)
                count_r += 1
        return isi_cv, isi_cv2

    @staticmethod
    def compute_isi_ratio(isi_df,isi_interval):
        """Function to compute the ratio between two windows of the ISI histogram
        Inputs: 
            isi_df: dataframe of inter-spike intervals per roi and trial
            isi_interval: list with range of the two windows in sec
            e.g.: [[0,0.5],[0.8,1.5]]"""
        isi_ratio = []
        roi_all = []
        trial_all = []
        for r in isi_df.roi.unique():
            for t in isi_df.trial.unique():
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
        plt.legend(fontsize=self.fsize - 4, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path+'/images/events/'+ 'isi_hist_roi_'+str(roi)+'_trial_'+str(trial), dpi=self.my_dpi)
            else:
                plt.savefig(self.path+'\\images\\events\\'+ 'isi_hist_roi_'+str(roi)+'_trial_'+str(trial), dpi=self.my_dpi)
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
            plt.legend(fontsize=self.fsize - 4, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/' + 'isi_hist_roi_' + str(roi),dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
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
            plt.bar(r1, hist_norm[:, 0], color='darkgrey', width=barWidth, edgecolor='white', label='baseine speed')
            plt.bar(r2, hist_norm[:, 1], color='crimson', width=barWidth, edgecolor='white', label='fast speed')
            plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.title('ISI distribution for ROI ' + str(roi), fontsize=self.fsize)
            plt.legend(fontsize=self.fsize - 4, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
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
            plt.legend(fontsize=self.fsize - 4, frameon=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if plot_data:
                if self.delim == '/':
                    plt.savefig(self.path + '/images/events/' + 'isi_hist_roi_' + str(roi),dpi=self.my_dpi)
                else:
                    plt.savefig(self.path + '\\images\\events\\' + 'isi_hist_roi_' + str(roi), dpi=self.my_dpi)
        return

    def plot_cv_session(self, roi, isi_cv, trials, plot_data):
        """Function to plot the ISI distribution across the session for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_cv: (array) with CV values
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
            colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'darkgrey', 'lightblue', 'lightblue',
                           'lightblue', 'lightblue', 'lightblue', 'lightblue',
                           'orange', 'orange', 'orange', 'orange', 'orange', 'orange']
        fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
        for t in trials:
            plt.bar(t - 0.5, isi_cv[roi - 1, t - 1], width=1, color=colors_bars[t - 1], edgecolor='white')
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'cv_roi_' + str(roi), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'cv_roi_' + str(roi), dpi=self.my_dpi)
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'isi_ratio_roi_' + str(roi), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'isi_ratio_cv_roi_' + str(roi), dpi=self.my_dpi)
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'event_waveform_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'event_waveform_roi_' + str(roi_plot), dpi=self.my_dpi)
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
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
        for t in trials:
            plt.bar(t - 0.5, np.count_nonzero(df_cs_stride[t] > 0) / np.count_nonzero(~np.isnan(df_cs_stride[t])),
                    width=1, color=colors_bars[t - 1], edgecolor='white')
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
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'event_count_roi_' + str(roi_plot), dpi=self.my_dpi)
        return

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
        fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
        sns.heatmap(event_stride_all, cmap='viridis', cbar=False)
        ax.vlines(bin_size / 2, *ax.get_ylim(), color='white', linestyle='dashed')
        if session_type == 'split':
            ax.hlines([trial_length_strides_cumsum[trials_ses[0]], trial_length_strides_cumsum[trials_ses[2]]],
                      *ax.get_xlim(), color='white', linewidth=0.5)
        if session_type == 'tied':
            for t in trials_ses[:-1]:
                ax.hlines([trial_length_strides_cumsum[t]], *ax.get_xlim(), color='white', linewidth=0.5)
        ax.set_yticks(np.arange(0, np.shape(event_stride_all)[0], 250))
        ax.set_yticklabels(list(map(str, np.arange(0, np.shape(event_stride_all)[0], 250))))
        ax.set_xticklabels(list(map(str, np.round(np.arange(-time_window, time_window, 0.02), 2))), rotation=45)
        ax.set_xlabel('Time (s)', fontsize=self.fsize - 4)
        ax.set_ylabel('Stride number', fontsize=self.fsize - 4)
        ax.set_title(paw + ' st events', fontsize=self.fsize - 4)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xticks(fontsize=self.fsize - 12)
        plt.yticks(fontsize=self.fsize - 12)
        if plot_data:
            if self.delim == '/':
                plt.savefig(self.path + '/images/events/' + 'event_' + align + '_' + paw + '_binsize_' + str(
                    bin_size) + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
            else:
                plt.savefig(self.path + '\\images\\events\\' + 'event_' + align + '_' + paw + '_binsize_' + str(
                    bin_size) + '_window_' + str(time_window).replace('.', ',') + '_roi_' + str(roi_plot),
                            dpi=self.my_dpi)
        return event_stride_all

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

    # def compute_dFF_bgsub(self,Fbg_array):
    #     """Function to compute dFF given a path to the suite2p output folder.
    #     It doesn't take background subtraction into account.'
    #     Input: Fbg_array: array with background values for each ROI pre-ROI curation"""
    #     if self.delim == '/':
    #         path_F = 'Suite2p/suite2p/plane0/F.npy'
    #         path_iscell = 'Suite2p/suite2p/plane0/iscell.npy'
    #     else:
    #         path_F = 'Suite2p\\suite2p\\plane0\\F.npy'
    #         path_iscell = 'Suite2p\\suite2p\\plane0\\iscell.npy' 
    #     #GEt F values from Suite2p
    #     F = np.load(self.path+path_F)
    #     iscell = np.load(self.path+path_iscell)
    #     iscell_idx = np.where(iscell[:,0])[0]
    #     F_cell = F[iscell_idx,:] 
    #     #F-r*Fbg - r is ratio between blood vessel signal and background
    #     r_bg = 0.7
    #     perc_value = 10
    #     Fcell_bg = np.zeros((np.shape(F_cell)[0],np.shape(F_cell)[1]))
    #     for r in range(np.shape(F_cell)[0]):
    #         Fcell_bg[r,:] = F_cell[r,:np.shape(Fbg_array)[1]]-(r_bg*Fbg_array[r,:])
    #     F0_roi_perc_bg = []
    #     for c in range(np.shape(Fcell_bg)[0]):
    #         F0_roi_perc_bg.append(np.percentile(Fcell_bg[c,:],perc_value))
    #     #roi x time array
    #     F0_perc_bg = np.transpose(np.tile(np.array(F0_roi_perc_bg),(np.shape(Fcell_bg)[1],1)))
    #     #compute dFF - there is division by zero
    #     dFF_perc_bg = (Fcell_bg-F0_perc_bg)/F0_perc_bg
    #     dFF_perc_bg[~np.isfinite(dFF_perc_bg)] = np.nan #for errors in division by zero
    #     return dFF_perc_bg
        
    # def compute_dFF(self,method):
    #     """Function to compute dFF given a path to the suite2p output folder.
    #     It doesn't take background subtraction into account.'
    #     Input: method -  'percentile' for computing dFF with 10% percentile for each ROI (same value across time)
    #            method - 'median' for computing with median of whole frame across time"""
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
    #     if method == 'percentile':
    #         #F0 using percentile for each ROI
    #         F0_roi_perc = []
    #         for c in range(len(iscell_idx)):
    #             F0_roi_perc.append(np.percentile(F_cell[c,:],10))
    #         #roi x time array
    #         F0 = np.transpose(np.tile(np.array(F0_roi_perc),(np.shape(F_cell)[1],1)))
    #     if method == 'median':
    #         #F0 using median of whole frame
    #         tiflist = glob.glob(self.path+'*.tif') #get list of tifs
    #         tiff_boolean = 0
    #         if not tiflist:
    #             tiflist = glob.glob(self.path+'*.tiff') 
    #             tiff_boolean = 1
    #         trial_id = []
    #         for t in range(len(tiflist)):
    #             tifname = tiflist[t].split(delim) 
    #             if tiff_boolean:
    #                 tifname_split = tifname[-1].split('_')
    #                 trial_id.append(int(tifname_split[-2]))
    #             else:
    #                 trial_id.append(int(tifname[-1][1:-4])) #get trial order in that list    
    #         trial_order = np.sort(trial_id) #reorder trials
    #         files_ordered = [] #order tif filenames by file order
    #         for f in range(len(tiflist)):
    #             tr_ind = np.where(trial_order[f] == trial_id)[0][0]
    #             files_ordered.append(tiflist[tr_ind])
    #         #read tiffs to do median of whole frame
    #         F0_frame_median = []
    #         for f in range(len(files_ordered)):
    #             image_stack = tiff.imread(files_ordered[f]) #choose right tiff
    #             F0_median_trial = np.zeros(np.shape(image_stack)[0])
    #             for frame in range(np.shape(image_stack)[0]):
    #                 F0_median_trial[frame] = np.median(image_stack[frame,:,:].flatten())
    #             F0_frame_median.extend(F0_median_trial)
    #         F0 = np.tile(F0_frame_median,(np.shape(F_cell)[0],1))   
    #     #compute dFF - there is division by zero
    #     dFF = (F_cell-F0)/F0
    #     dFF[~np.isfinite(dFF)] = np.nan #for errors in division by zero
    #     return dFF

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

        
            











                
