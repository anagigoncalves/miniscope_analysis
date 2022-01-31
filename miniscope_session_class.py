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
import glob
import pandas as pd
import math
from scipy.optimize import curve_fit
import os
import seaborn as sns
import scipy.cluster.hierarchy as spc
from scipy.stats import skew
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
    def minmax_norm(A):
        """Normalizes a vector by doing min-max normalization"""
        A_norm = (A-min(A))/(max(A)-min(A))
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
    
    def corr_FOV_movement(self,th,df_dFF,corrXY):
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
        #trace as dataframe
        roi_list =  []
        for r in range(len(coord_cell)):
            roi_list.append('ROI'+str(r+1))
        data_ext1 = {'trial': np.repeat(2, len(frame_time[trial - 1])), 'time': frame_time[trial - 1]}
        df_ext1 = pd.DataFrame(data_ext1)
        df_ext2 = pd.DataFrame(trace_ext, columns=roi_list)
        df_ext = pd.concat([df_ext1, df_ext2], axis=1)
        return coord_cell_t, df_ext

    def get_imagej_output(self,frame_time,trials):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (ImageJ output)
        Inputs:
            frame_time: list with miniscope timestamps
            trial: (arr) - trial list"""
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
                       coord_fiji[c][:, 0] * self.pixel_to_um)])
            roi_trace_all.append(roi_trace_tiffmean)
        roi_trace_concat = np.vstack(roi_trace_all)
        # Normalize traces
        roi_trace_minmax = np.zeros(np.shape(roi_trace_concat))
        for col in range(np.shape(roi_trace_tiffmean)[1]):
            roi_trace_minmax[:, col] = (roi_trace_concat[:, col] - np.min(roi_trace_concat[:, col])) / (
                        np.max(roi_trace_concat[:, col]) - np.min(roi_trace_concat[:, col]))
        #trace as dataframe
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
        df_fiji2 = pd.DataFrame(roi_trace_minmax, columns=roi_list)
        df_fiji = pd.concat([df_fiji1, df_fiji2], axis=1)
        return coord_fiji, df_fiji

    def get_roi_stats(self,coord_cell):
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

    def get_roi_centroids(self,coord_cell):
        """From the ROIs coordinates get the centroid
        Input:
        coord_cell (list of ROI coordinates)"""
        centroid_cell = []
        for r in range(len(coord_cell)):
            centroid_cell.append(np.array([np.nanmean(coord_cell[r][:,0]), np.nanmean(coord_cell[r][:,1])]))
        return centroid_cell

    def get_roi_list(self, coord_cell):
        """ Get number of ROI names for the session"""
        roi_list =  []
        for r in range(len(coord_cell)):
            roi_list.append('ROI'+str(r+1))
        return roi_list

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

    def correct_gimbal_lock(self, head_angles):
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

    def pca_centroids(self, principalComponents_3CP, trial_clean, trials, plot_data):
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

    def distance_neurons(self,centroid_cell,plot_data):
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
            delim = self.path[-1]
            if delim == '/':
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
            delim = self.path[-1]
            if delim == '/':
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

    def compute_bg_roi_fiji(self, coord_cell, df_dFF, coeff_sub):
        """Function to compute a donut background around a determined FIJI ROI and compute its background subtracted signal.
        Input:
            coord_cell: list with ROIs coordinates
            df_dFF: dataframe with calcium traces
            coeff_sub: (float 0-1) coefficient for backgrond subtraction"""
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
                    donut_trace_tiffmean[f] = np.nansum(tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]]) / \
                                              np.shape(ROIdonut_coord)[0]
                donut_trace_minmax = (donut_trace_tiffmean - np.min(donut_trace_tiffmean)) / (
                        np.max(donut_trace_tiffmean) - np.min(donut_trace_tiffmean))
                donut_trace_trials.append(donut_trace_minmax)
            donut_trace_concat = np.hstack(donut_trace_trials)
            donut_trace_all_list.append(donut_trace_concat)
        roi_trace_bgsub_arr = np.array(df_dFF.iloc[:, 2:]) - (coeff_sub * donut_trace_all_list)
        idx_neg = np.where(roi_trace_bgsub_arr < 0)
        roi_trace_bgsub_arr[idx_neg[0], idx_neg[1]] = 0
        #trace as dataframe
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
        df_fiji2 = pd.DataFrame(roi_trace_bgsub_arr, columns=roi_list)
        df_fiji = pd.concat([df_fiji1, df_fiji2], axis=1)
        return df_dFF_bgsub

    def get_events(self,coord_cell,df_dFF,timeT,thrs_amp):
        """"Function to get the calcium event using Jorge's derivative method
        Inputs:
        coord_cell: list with ROI coordinates
        df_dFF: dataframe with calcium trace values
        timeT: time thershold (within how many frames does the vent happen)
        thrs_amp: amplitude threshold for the events - percentile of the trace"""
        roi_trace = np.array(df_dFF.iloc[:, 2:])
        events_mat = np.zeros(np.shape(roi_trace))
        for r in range(np.shape(roi_trace)[1]):
            amp = np.nanpercentile(roi_trace[:, r], thrs_amp)
            [JoinedPosSet_all, JoinedNegSet_all, F_Values_all] = ST.SlopeThreshold(roi_trace[:, r], amp, timeT,
                                                                                   CollapSeq=True, acausal=False,
                                                                                   verbose=0, graph=None)
            events_mat[ST.event_detection_calcium_trace(roi_trace[:, r], JoinedPosSet_all, timeT), r] = 1
        roi_list = []
        for r in range(len(coord_cell)):
            roi_list.append('ROI' + str(r + 1))
        df_events_mat = pd.DataFrame(events_mat, columns=roi_list)
        df_events = pd.concat([df_dFF.iloc[:, :2], df_events_mat], axis=1)
        return df_events

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

    def compute_isi(self,spikes):
        #TODO MAKE SELF DEPENDENT NEW SPIKES FORMAT
        """Function to compute the inter-spike interval of the dataframe with spikes. Outputs a similiar dataframe
        Inputs: 
            spmat: spikes (dataframe, single trial or whole session)"""
        isi_all = []
        roi_all = []
        trial_all = []
        for r in spikes.roi.unique():
            for t in spikes.trial.unique():
                isi = spikes.loc[(spikes.roi==r) & (spikes.trial==t) & (spikes.spikes==1), 'time'].diff()
                isi_all.extend(np.array(isi))
                trial_all.extend(np.repeat(t,len(isi)))
                roi_all.extend(np.repeat(r,len(isi)))
        dict_isi = {'isi': isi_all,'roi': roi_all,'trial': trial_all}
        isi_df = pd.DataFrame(dict_isi) #create dataframe with isi, roi id and trial id
        return isi_df

    def compute_isi_cv(self,isi_df,trials):
        #TODO MAKE SELF DEPENDENT NEW ISI_DF FORMAT
        """Function to compute coefficient of variation and coefficient of variation for adjacent spikes (Isope, JNeurosci)
        Inputs:
            isi_df: (dataframe) with isi values
            trials: (array)"""
        isi_cv = np.zeros((len(np.unique(isi_df.roi)),len(trials)))
        isi_cv2 = np.zeros((len(np.unique(isi_df.roi)),len(trials)))
        for t in trials:
            for r in np.int8(np.unique(isi_df.roi)):
                data = np.array(isi_df.loc[(isi_df['trial']==t) & (isi_df['roi']==r),'isi'])
                diff_data = np.abs(np.diff(data))
                sum_data = data[:-1]+data[1:]
                isi_cv[r-1,t-1] = np.nanstd(data)/np.nanmean(data)
                isi_cv2[r-1,t-1] = np.nanmean(diff_data/sum_data)
        return isi_cv, isi_cv2
    
    def compute_isi_ratio(self,isi_df,isi_interval):
        # TODO MAKE SELF DEPENDENT
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
        isi_ratio_df = pd.DataFrame(dict_isi_ratio) #create dataframe with isi, roi id and trial id
        return isi_ratio_df
    
    def compute_cs_waveform(self,dFF,spmat):
        #TODO MAKE SELF DEPENDENT, CHANGE TO NEW DFF FORMAT
        """Function to compute the complex-spike waveform from the deconvolution and dFF
        Inputs: 
            dFF: dFF as an array or dataframe
            spmat: spike matrix as an array or dataframe"""
        if isinstance(dFF, pd.DataFrame):  
            dFF_array = dFF.to_numpy()
            roi_nr = len(np.unique(dFF_array[:,1]))
            sample_nr = int(np.shape(dFF_array)[0]/roi_nr)
            dFF_array_reshaped = np.reshape(dFF_array[:,0],(roi_nr,sample_nr))
        if type(dFF) == np.ndarray:
            dFF_array_reshaped = dFF
        if isinstance(spmat, pd.DataFrame):  
            spmat_array = spmat.to_numpy()
            roi_nr = len(np.unique(spmat_array[:,1]))
            sample_nr = int(np.shape(spmat_array)[0]/roi_nr)
            spmat_array_reshaped = np.reshape(spmat_array[:,0],(roi_nr,sample_nr))
        if type(spmat) == np.ndarray:
            spmat_array_reshaped = spmat
        #Compute mean waveform
        cs_waveforms_all = []
        for r in range(np.shape(dFF_array_reshaped)[0]):
            spikes = np.where(spmat_array_reshaped[r,:])[0]
            cs_waveforms = np.empty((len(spikes),40))
            cs_waveforms[:] = np.nan #initialize nan array
            for s in range(len(spikes)):
                if len(dFF_array_reshaped[r,spikes[s]-10:spikes[s]+30])==40:
                    cs_waveforms[s,:] = dFF_array_reshaped[r,spikes[s]-10:spikes[s]+30]
            cs_waveforms_all.append(cs_waveforms)
        return cs_waveforms_all
    
    def compute_decay_time(self,cs_waveforms_all):
        #TODO MAKE SELF DEPENDENT, CHANGE TO NEW DFF FORMAT
        """Function to compute the decay time (in seconds) of the average CS waveform for each ROI
        Input: 
            cs_waveforms_all: list with all the CS waveforms per ROI"""
        tau = []
        for r in range(len(cs_waveforms_all)):
            cs_mean = np.nanmean(cs_waveforms_all[r],axis=0)
            #Compute tau
            cs_max = np.argmax(cs_mean)
            cs_end = cs_max+int(0.2*self.sr)
            x_exp = np.linspace((-10+cs_max)/self.sr,(-10+cs_end)/self.sr,cs_end-cs_max)
            exp_fit = curve_fit(lambda t,a,b: a*np.exp(b*t),x_exp,cs_mean[cs_max:cs_end],p0=(1,0.5))
            tau.append(abs(exp_fit[0][1]))
        return tau
    
    def compute_fwhm(self,cs_waveforms_all):
        #TODO MAKE SELF DEPENDENT, CHANGE TO NEW DFF FORMAT
        """Function to compute the FWHM (in seconds) of the average CS waveform for each ROI
        Input: 
            cs_waveforms_all: list with all the CS waveforms per ROI"""
        fwhm = []
        for r in range(len(cs_waveforms_all)):
            cs_mean = np.nanmean(cs_waveforms_all[r],axis=0)
            #Compute fwhm
            cs_max = np.argmax(cs_mean)
            cs_beg = cs_max-int(0.2*self.sr)
            cs_end = cs_max+int(0.4*self.sr)
            x_gauss = np.linspace((-10+cs_beg)/self.sr,(-10+cs_end)/self.sr,cs_end-cs_beg)
            n_gauss = len(x_gauss)                          
            mean_gauss = sum(x_gauss*cs_mean[cs_beg:cs_end])/n_gauss                   
            sigma_gauss = sum(cs_mean[cs_beg:cs_end]*(x_gauss-mean_gauss)**2)/n_gauss
            gauss_fit = curve_fit(lambda x,a,x0,sigma: a*np.exp(-(x-x0)**2/(2*sigma**2)),x_gauss,cs_mean[cs_beg:cs_end],p0=[cs_mean[cs_max],mean_gauss,sigma_gauss])
            fwhm.append(2*gauss_fit[0][2]*np.sqrt(2*math.log(2)))
        return fwhm
    
    def get_spike_count(self,spmat,trials):
        #TODO MAKE SELF DEPENDENT
        """Function to compute the normalized spike count (divided by the number of frames) per trial
        Inputs: 
            spmat: dataframe with spikes (1 for indices with spikes)
            trial_order: list of recorded trials"""
        #Get spike count per trial
        roi_list = spmat.roi.unique()
        spike_count = np.zeros((len(trials),len(roi_list)))
        trial_count = 0
        for t in trials:
            for r in roi_list:
                spike_count[trial_count,int(r-1)] = spmat.loc[(spmat['roi'] == r) & (spmat['trial']==t), 'spikes'].sum()/spmat.loc[(spmat['roi'] == r) & (spmat['trial']==t), 'spikes'].count()
            trial_count = trial_count + 1
        return spike_count
    
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
    
    def plot_spikes_dFF(self,dFF_trial,spikes,roi,time,fig_size,fsize,print_plots):
        #TODO NEW DFF FORMAT
        """Function to plot dFF traces with deconvoluted spikes on top
        Inputs: 
            dFF_trial: dataframe with dFF for the whole session
            spikes: dataframe with the spikes
            roi: roi number to plot
            time: time in sec to plot (starts at the beginning of the trial)
            fig_size: tuple with the size of the figure
            fsize: fontsize of labels and axis
            print_plots: boolean to print or not"""
        trials = dFF_trial.trial.unique()
        fig, ax = plt.subplots(int(np.ceil(len(trials)/3)),3,figsize=fig_size, tight_layout=True)
        ax = ax.ravel()
        count = 0
        for t in trials:
            dFF_trial_subset = np.array(dFF_trial.loc[(dFF_trial['trial']==t) & (dFF_trial['roi']==roi),'dFF'])
            spikes_trial_subset = np.array(spikes.loc[(spikes['trial']==t) & (spikes['roi']==roi),'spikes'])
            dFF_plot = dFF_trial_subset[0:int(time*self.sr)]
            spikes_plot = spikes_trial_subset[0:int(time*self.sr)]
            spikes_time = np.where(spikes_plot==1)[0]/self.sr
            ax[count].plot(np.linspace(0,time,len(dFF_plot)),dFF_plot[0:],color='black')
            ax[count].scatter(spikes_time,np.ones(len(spikes_time))*np.max(dFF_plot),s=10,c='red')
            ax[count].set_xlabel('Time (s)', fontsize = fsize)
            ax[count].set_ylabel('ΔF/F', fontsize = fsize)
            ax[count].set_title('Trial '+str(t), fontsize = fsize)
            ax[count].spines['right'].set_visible(False)
            ax[count].spines['top'].set_visible(False)
            count += 1
        if print_plots:
            if not os.path.exists(self.path+'images'):
                os.mkdir(self.path+'images')
            delim = self.path[-1]
            if delim == '/':
                plt.savefig(self.path+'images/'+ 'dFF_spikes_roi_'+str(roi), dpi=self.my_dpi)
            else:
                plt.savefig(self.path+'images\\'+ 'dFF_spikes_roi_'+str(roi), dpi=self.my_dpi)
                
    def save_processed_dFF(self,dFF,dFF_trial,trials,keep_rois,th,idx_to_nan):
        #TODO NEW VARS TO SAVE
        """Saves dFF by trial after manual curation, plus list of recorded trials.
        Saves them under .../Suite2p analysis/processed files
        Inputs: 
            dFF: array with dFF values for the whole session
            dFF_trial: dataframe with dFF values by trial and ROI
            trials: array with list of recorded trials
            keep_rois: list of rois to keep
            th: threshold to discard frames for poor correlation with ref image"""
        if not os.path.exists(self.path+'processed files'):
            os.mkdir(self.path+'processed files')
        if self.delim == '/':
            np.save(self.path+'/processed files/'+'dFF_session',dFF)
            dFF_trial.to_pickle(self.path+'/processed files/'+'dFF')
            np.save(self.path+'/processed files/'+'trials',trials)
            np.save(self.path+'/processed files/'+'rois',keep_rois)
            np.save(self.path+'/processed files/'+'reg_th',th)
            np.save(self.path+'/processed files/'+'frames_to_exclude',idx_to_nan)
        else:
            np.save(self.path+'\\processed files\\'+'dFF_session',dFF)
            dFF_trial.to_pickle(self.path+'\\processed files\\'+'dFF')
            np.save(self.path+'\\processed files\\'+'trials',trials)
            np.save(self.path+'\\processed files\\'+'rois',keep_rois)
            np.save(self.path+'\\processed files\\'+'reg_th',th)
            np.save(self.path+'\\processed files\\'+'frames_to_exclude',idx_to_nan)            
        return

    def load_processed_dFF(self):
        #TODO NEW VARS TO LOAD
        """Loads processed dFF by trial, plus list of recorded trials that were saved under
        .../Suite2p/processed files"""
        if self.delim == '/':
            dFF_trial = pd.read_pickle(self.path+'/processed files/'+'dFF')
            trials = np.load(self.path+'/processed files/'+'trials.npy')
            dFF = np.load(self.path+'/processed files/'+'dFF_session.npy')
            keep_rois = np.load(self.path+'/processed files/'+'rois.npy')
            reg_th = np.load(self.path+'/processed files/'+'reg_th.npy')
            reg_bad_frames = np.load(self.path+'/processed files/'+'frames_to_exclude.npy')

        else:
            dFF_trial = pd.read_pickle(self.path+'\\processed files\\'+'dFF')
            trials = np.load(self.path+'\\processed files\\'+'trials.npy')
            dFF = np.load(self.path+'\\processed files\\'+'dFF_session.npy')
            keep_rois = np.load(self.path+'\\processed files\\'+'rois.npy')  
            reg_th = np.load(self.path+'\\processed files\\'+'reg_th.npy')
            reg_bad_frames = np.load(self.path+'\\processed files\\'+'frames_to_exclude.npy')
        return dFF, dFF_trial, trials, keep_rois, reg_th, reg_bad_frames
    
    def cs_stride_event(self,spikes,st_strides_mat,sw_pts_mat,trial,paw,roi,align): #DO THIS WITH SPIKE TIME
        #TODO NEW SPIKES FORMAT
        """Align CS to stance, swing or stride period. It outputs the CS indexes for each
        time period
        Inputs:
            spikes: dataframe with spikes
            st_stride_mat: matrix with stance points of that trial
            sw_pts_mat: matrix with swing points of that trial
            trial (int): trial number
            roi (int): roi number
            paw (str): 'FR','HR','FL','HL'
            align (str): period to align - 'stance','swing','stride'"""
        if paw == 'FR':
            p = 0 #paw of tracking
        if paw == 'HR':
            p = 1
        if paw == 'FL':
            p = 2
        if paw == 'HL':
            p = 3
        spikes_trial = spikes.loc[(spikes['trial']==trial) & (spikes['roi']==roi),['spikes','time']]
        spikes_time = np.array(spikes_trial[spikes_trial['spikes'] == 1]['time'])
        if align == 'stride':
            excursion_beg = st_strides_mat[p][:,0,4]/self.sr_loco
            excursion_end = st_strides_mat[p][:,1,4]/self.sr_loco
        if align == 'stance':
            excursion_beg = st_strides_mat[p][:,0,4]/self.sr_loco
            excursion_end = sw_pts_mat[p][:,0,4]/self.sr_loco
        if align == 'swing':
            excursion_beg = sw_pts_mat[p][:,0,4]/self.sr_loco
            excursion_end = st_strides_mat[p][:,1,4]/self.sr_loco
        cs_stride = []
        for s in range(len(excursion_beg)):
            cs_idx = np.where((spikes_time>=excursion_beg[s])&(spikes_time<=excursion_end[s]))[0]
            if len(cs_idx)>0:
                cs_stride.append(spikes_time[cs_idx])
            else:
                cs_stride.append([])
        return cs_stride
    
    def cs_probability(self,cs_stride):
        """Compute CS probability for CSs aligned to stride period
        (swing, stance or stride)
        Inputs:
            cs_stride: cs list per stride period"""
        stride_count = 0
        for s in range(len(cs_stride)):
            if len(cs_stride[s])>0:
                stride_count += 1
        cs_probability = np.round(stride_count/len(cs_stride),decimals=2)
        return cs_probability

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

    # def dFF_by_trial(self,dFF):
    #     """Function to separate computed dFF into trials, by the order of recording
    #         Input: dFF - as matrix (from compute_dFF)"""
    #     delim = self.path[-1]
    #     tiflist = glob.glob(self.path+delim+'Suite2p'+delim+'*.tif')
    #     if not tiflist:
    #         tiflist = glob.glob(self.path+delim+'Suite2p'+delim+'*.tiff') #get list of tifs
    #     trial_id = []
    #     for t in range(len(tiflist)):
    #         tifname = tiflist[t].split(delim)   
    #         tifname_split = tifname[-1].split('_')
    #         trial_id.append(int(tifname_split[0][1:])) #get trial order in that list    
    #     trial_order = np.sort(trial_id) #reorder trials
    #     files_ordered = [] #order tif filenames by file order
    #     size_trial = []
    #     for f in range(len(tiflist)):
    #         tr_ind = np.where(trial_order[f] == trial_id)[0][0]
    #         files_ordered.append(tiflist[tr_ind])
    #     #get size of trials
    #     for f in range(len(files_ordered)):
    #         image_stack = tiff.imread(files_ordered[f]) #choose right tiff
    #         size_trial.append(np.shape(image_stack)[0])
    #     size_trial_vec = []
    #     for t in range(len(size_trial)):
    #         size_trial_vec.extend(np.repeat(trial_order[t],size_trial[t]))
    #     roi_list =  []
    #     for r in range(np.shape(dFF)[0]):
    #         roi_list.append('ROI'+str(r+1))
    #     dFF_trial = pd.DataFrame(np.transpose(dFF),columns=roi_list)
    #     dFF_trial['trial'] = size_trial_vec
    #     return dFF_trial

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

        
            











                
