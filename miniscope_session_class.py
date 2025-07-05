# !/usr/bin/env python3
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
import glob
import pandas as pd
import os
import seaborn as sns
import scipy.cluster.hierarchy as spc
import mat73
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
import mahotas
import matplotlib.patches as mp_patch
import SlopeThreshold as ST
import read_roi
import scipy.spatial.distance as spdist
from scipy.spatial import Delaunay
import scipy.signal as sp
from itertools import chain

# to call class
# os.chdir('/Users/anagoncalves/Documents/PhD/Code/Miniscope pipeline/')
# import miniscope_session_class
# mscope = miniscope_session_class.miniscope_session(path)
# and then mscope.(function name)

# to check class documentation (after importing class)
# help(mscope.compute_dFF)

# to see all methods in class
# dir(mscope)
class miniscope_session:
    def __init__(self, path):
        self.path = path
        self.delim = self.path[-1]
        # pixel_to_um = 0.93 #1px is 0.93um for miniscopes v3.2
        self.pixel_to_um = 0.608  # 1px is 0.608um for miniscopes v4
        self.pixel_to_mm_behavior = 1/1.955
        self.sr = 30  # sampling rate of miniscopes
        self.my_dpi = 128  # resolution for plotting
        self.sr_loco = 330  # sampling rate of behavioral camera
        self.fsize = 20
        self.trial_time = 60 #trial duration

    @staticmethod
    def z_score(A, axis_id):
        """Normalizes an array by z-scoring it"""
        if len(np.shape(A)):
            A_norm = (A - np.nanmean(A)) / np.nanstd(A)
        else:
            if axis_id == 1:
                A_mean = np.repeat(np.nanmean(A, axis=axis_id).reshape(-1, 1), np.shape(A)[axis_id], axis=axis_id)
                A_std = np.repeat(np.nanstd(A, axis=axis_id).reshape(-1, 1), np.shape(A)[axis_id], axis=axis_id)
                A_norm = np.divide((A - A_mean), A_std)
            else:
                A_mean = np.repeat(np.nanmean(A, axis=axis_id), np.shape(A)[axis_id], axis=axis_id)
                A_std = np.repeat(np.nanstd(A, axis=axis_id), np.shape(A)[axis_id], axis=axis_id)
                A_norm = np.divide((A - A_mean), A_std)
        return A_norm

    @staticmethod
    def inpaint_nans(A):
        """Interpolates NaNs in numpy arrays
        Input: A (numpy array)"""
        ok = ~np.isnan(A)
        xp = ok.ravel().nonzero()[0]
        fp = A[~np.isnan(A)]
        x = np.isnan(A).ravel().nonzero()[0]
        A[np.isnan(A)] = np.interp(x, xp, fp)
        return A

    @staticmethod
    def mov_mean(x, w):
        """Does moving average with numpy convolution function
        Inputs:
            x: data vector
            w: window length (int)"""
        return np.convolve(x, np.ones(w), 'same') / w

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
        cx = (c * d - b * f) / num
        cy = (a * f - b * d) / num
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
    def correlation_traces(trace1, trace2):
        cross_corr = correlate(miniscope_session.z_score(trace1, 1),
                               miniscope_session.z_score(trace2, 1), mode='full',
                               method='direct')
        p_corrcoef = np.cov(trace1, trace2)[0][1] / np.sqrt(np.var(trace1) * np.var(trace2))
        return p_corrcoef

    @staticmethod
    def compute_dFF(df_fiji):
        """Function to compute dF/F for each trial using as F0 the 10th percentile of the signal for that trial"""
        trials = df_fiji['trial'].unique()
        for t in trials:
            trial_length = np.shape(df_fiji.loc[df_fiji['trial'] == t])[0]
            trial_idx = df_fiji.loc[df_fiji['trial'] == t].index
            perc_arr = np.tile(np.nanpercentile(np.array(df_fiji.iloc[trial_idx, 2:]), 10, axis=0), (trial_length, 1))
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
            try:
                params, ell = miniscope_session.fitEllipse(coord_cell[r], 1)
                width_roi.append(params[2])
                height_roi.append(params[3])
                aspect_ratio.append(params[3] / params[2])
            except:
                width_roi.append(0)
                height_roi.append(0)
                aspect_ratio.append(0)
        return width_roi, height_roi, aspect_ratio

    @staticmethod
    def get_roi_centroids(coord_cell):
        """From the ROIs coordinates get the centroid
        Input:
        coord_cell (list of ROI coordinates)"""
        centroid_cell = []
        for r in range(len(coord_cell)):
            centroid_cell.append(np.array([np.nanmean(coord_cell[r][:, 0]), np.nanmean(coord_cell[r][:, 1])]))
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
        # giving placement of miniscope: x is roll, y is pitch, z is yaw
        pitch = y_angles
        yaw = z_angles
        roll = x_angles
        return roll, pitch, yaw  # in radians

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
        head_angles.iloc[:, 1] = np.unwrap(head_angles.iloc[:, 1])
        head_angles.iloc[:, 0] = np.unwrap(head_angles.iloc[:, 0])
        head_angles.iloc[:, 2] = np.unwrap(head_angles.iloc[:, 2])
        fig, ax = plt.subplots(1, 3, figsize=(10, 10), tight_layout=True)
        ax = ax.ravel()
        ax[0].plot(head_angles['pitch'], color='black')
        ax[0].set_title('Pitch')
        ax[1].plot(head_angles['roll'], color='black')
        ax[1].set_title('Roll')
        ax[2].plot(head_angles['yaw'], color='black')
        ax[2].set_title('Yaw')
        plt.suptitle('After corrections')
        head_angles.to_csv(os.path.join(self.path, 'processed files', 'head_angles_corr.csv'), sep=',', index=False)
        return head_angles

    @staticmethod
    def distance_neurons(centroid_cell, plot_data):
        """Computes the distance between ROIs from the centroids given by suite2p
        Inputs:
            plot_data (boolean)"""
        roi_nr = np.arange(len(centroid_cell))
        distance_neurons = np.zeros((len(centroid_cell), len(centroid_cell)))
        for r1 in range(len(centroid_cell)):
            for r2 in range(len(centroid_cell)):
                distance_neurons[r1, r2] = np.linalg.norm(np.array(centroid_cell[r1]) - np.array(centroid_cell[r2]))
        if plot_data:
            mask = np.triu(np.ones_like(distance_neurons, dtype=np.bool))
            fig, ax = plt.subplots()
            with sns.axes_style("white"):
                sns.heatmap(distance_neurons, mask=mask, cmap="YlGnBu", linewidth=0.5)
                ax.set_title('distance between ROIs')
                ax.set_yticklabels(roi_nr)
                ax.set_xticklabels(roi_nr)
        return distance_neurons

    @staticmethod
    def compute_events_onset(rawdata, acq_fq, detrend_bool):
        """Computes events for a vector of fluorescence. It uses JR SlopeThreshold method
        Input:
            rawdata: vector of fluorescence trace
            acq_fq: sampling rate (float)
            detrend_bool: boolean for trace detrending"""
        if detrend_bool:
            # Estimate Baseline
            Baseline, Est_Std = ST.Estim_Baseline_PosEvents(rawdata, acq_fq, dtau=0.2, bmax_tslope=3, filtcut=1,
                                                            graph=False)
            # Calculate dF/F0:
            F0 = Baseline - Est_Std * 2
            dff = (rawdata - F0) / F0
            dff = np.where(dff < 0, np.zeros_like(dff), dff)  # Remove negative dff values
        else:
            dff = rawdata
        Ev_Onset, Ev_ApproxPeak, TrueStd, IncremSet = ST.Detect_PosEvents_ROI(dff, acq_fq, rtau=0.02, graph=None)
        return [Ev_Onset, IncremSet, TrueStd]

    @staticmethod
    def compute_events_onset_405(rawdata, acq_fq, amp, detrend_bool):
        """Computes events for a vector of fluorescence. It uses JR SlopeThreshold method
        Input:
            rawdata: vector of fluorescence trace
            acq_fq: sampling rate (float)
            amp: amplitude for noise threshold
            detrend_bool: boolean for trace detrending"""
        if detrend_bool:
            # Estimate Baseline
            Baseline, Est_Std = ST.Estim_Baseline_PosEvents(rawdata, acq_fq, dtau=0.2, bmax_tslope=3, filtcut=1,
                                                            graph=False)
            # Calculate dF/F0:
            F0 = Baseline - Est_Std * 2
            dff = (rawdata - F0) / F0
            dff = np.where(dff < 0, np.zeros_like(dff), dff)  # Remove negative dff values
        else:
            dff = rawdata
        Ev_Onset, Ev_ApproxPeak, IncremSet = ST.Detect_PosEvents_ROI_Amp_Input(dff, acq_fq, 0.02, amp, graph=None)
        return [Ev_Onset, IncremSet]

    @staticmethod
    def event_detection_calcium_trace(rawdata, Ev_Onset, IncremSet, TimePntThres):
        """"From the SlopeThreshold function compute the maximums (and not the slope rises) for the positive crossings"""
        peaks = []
        if len(IncremSet) > 0:
            if type(IncremSet[0]) is tuple:
                Ev_Onset = []
                for i in range(len(IncremSet)):
                    Ev_Onset.append(IncremSet[i][0])
                    if IncremSet[i][1] + TimePntThres >= len(rawdata):
                        values_idx = np.arange(IncremSet[i][1], len(rawdata))
                    else:
                        values_idx = np.arange(IncremSet[i][1], IncremSet[i][1] + TimePntThres)
                    peak_idx = np.argmax(rawdata[values_idx])
                    peaks.append(IncremSet[i][1] + peak_idx)
        return np.array(peaks)

    def colors_session(self, animal, session_type, trials, bar_boolean):
        """Get the colors of trials for this particular session"""
        greys = mp.cm.get_cmap('Greys', 14)
        reds = mp.cm.get_cmap('Reds', 23)
        blues = mp.cm.get_cmap('Blues', 23)
        oranges = mp.cm.get_cmap('Oranges', 23)
        purples = mp.cm.get_cmap('Purples', 23)
        if bar_boolean:
            colors_session = []
            if session_type == 'tied':
                if len(trials) == 6:
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: oranges(23), 5: oranges(13), 6: oranges(7)}
                if len(trials) == 12:
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: oranges(23), 5: oranges(13), 6: oranges(7), 7: purples(23),
                                      8: purples(13), 9: purples(7)}
                if len(trials) > 17:
                    colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: oranges(23),
                                      8: oranges(19),
                                      9: oranges(16), 10: oranges(13), 11: oranges(10), 12: oranges(6), 13: purples(23), 14: purples(19),
                                      15: purples(16), 16: purples(13), 17: purples(10), 18: purples(6)}
                if len(trials) == 26:
                    colors_session = {1: greys(23), 2: greys(21), 3: greys(19), 4: greys(17), 5: greys(15), 6: greys(13),
                    7: greys(12), 8: greys(10), 9: greys(8), 10: greys(6), 11: greys(4), 12: greys(2),
                    13: purples(23), 14: purples(21), 15: purples(19), 16: purples(17), 17: purples(15), 18: purples(
                        13),
                    19: purples(11), 20: purples(9),
                    21: oranges(23), 22: oranges(19), 23: oranges(16), 24: oranges(13), 25: oranges(10), 26: oranges(
                        6)}
            if session_type == 'split':
                if len(trials) == 23 and animal == 'MC8855':
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: reds(23), 5: reds(21), 6: reds(19), 7: reds(17), 8: reds(15),
                                      9: reds(13),
                                      10: reds(11), 11: reds(9), 12: reds(7), 13: reds(5), 14: blues(23), 15: blues(21), 16: blues(19), 17: blues(17),
                                      18: blues(15), 19: blues(13),
                                      20: blues(11), 21: blues(9), 22: blues(7), 23: blues(5)}
                if len(trials) == 23 and animal == 'MC9226':
                    colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: reds(23), 8: reds(21),
                                      9: reds(19), 10: reds(17), 11: reds(15), 12: reds(13),
                                      13: reds(11), 14: reds(9), 15: reds(7), 16: reds(5), 17: blues(23), 18: blues(21), 19: blues(19), 20: blues(17),
                                      21: blues(15), 22: blues(13), 23: blues(11)}
                if len(trials) > 23:
                    colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: reds(23), 8: reds(21),
                                      9: reds(19), 10: reds(17), 11: reds(15), 12: reds(13),
                                      13: reds(11), 14: reds(9), 15: reds(7), 16: reds(5), 17: blues(23), 18: blues(21), 19: blues(19), 20: blues(17),
                                      21: blues(15), 22: blues(13),
                                      23: blues(11), 24: blues(9), 25: blues(7), 26: blues(5)}
        else:
            if session_type == 'tied':
                if len(trials) == 6:
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: oranges(23), 5: oranges(13),
                                      6: oranges(7)}
                if len(trials) == 12:
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: oranges(23), 5: oranges(13),
                                      6: oranges(7),
                                      7: purples(23),
                                      8: purples(13), 9: purples(7)}
                if len(trials) == 18:
                    colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4),
                                      7: oranges(23),
                                      8: oranges(19),
                                      9: oranges(16), 10: oranges(13), 11: oranges(10), 12: oranges(6), 13: purples(23),
                                      14: purples(19),
                                      15: purples(16), 16: purples(13), 17: purples(10), 18: purples(6)}
                if len(trials) == 26:
                    colors_session = {1: greys(23), 2: greys(21), 3: greys(19), 4: greys(17), 5: greys(15), 6: greys(13),
                    7: greys(12), 8: greys(10), 9: greys(8), 10: greys(6), 11: greys(4), 12: greys(2),
                    13: purples(23), 14: purples(21), 15: purples(19), 16: purples(17), 17: purples(15), 18: purples(
                        13),
                    19: purples(11), 20: purples(9),
                    21: oranges(23), 22: oranges(19), 23: oranges(16), 24: oranges(13), 25: oranges(10), 26: oranges(
                        6)}
            if session_type == 'split':
                if len(trials) == 23:
                    colors_session = {1: greys(12), 2: greys(7), 3: greys(4), 4: reds(23), 5: reds(21), 6: reds(19),
                                      7: reds(17),
                                      8: reds(15),
                                      9: reds(13),
                                      10: reds(11), 11: reds(9), 12: reds(7), 13: reds(5), 14: blues(23), 15: blues(21),
                                      16: blues(19), 17: blues(17),
                                      18: blues(15), 19: blues(13),
                                      20: blues(11), 21: blues(9), 22: blues(7), 23: blues(5)}
                if len(trials) == 26:
                    colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4),
                                      7: reds(23),
                                      8: reds(21),
                                      9: reds(19), 10: reds(17), 11: reds(15), 12: reds(13),
                                      13: reds(11), 14: reds(9), 15: reds(7), 16: reds(5), 17: blues(23), 18: blues(21),
                                      19: blues(19), 20: blues(17),
                                      21: blues(15), 22: blues(13),
                                      23: blues(11), 24: blues(9), 25: blues(7), 26: blues(5)}
        np.save(os.path.join(self.path, 'processed files', 'colors_session.npy'), colors_session)
        return colors_session

    @staticmethod
    def get_session_data(trials, session_type, animal, session):
        """Get the transition trials, different phases name and the different trials for each phase of the session
        Inputs:
        trials: list of trials
        session_type: (str) split or tied
        animal: (str) name animal
        session: (int) session number"""
        if session_type == 'tied' and animal == 'MC8855':
            trials_ses = np.array([[1, 3], [4, 6]])
            trials_ses_name = ['baseline speed', 'fast speed']
            cond_plot = ['baseline', 'fast']
        if session_type == 'tied' and animal == 'MC9194':
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'slow speed', 'fast speed']
            cond_plot = ['baseline', 'slow', 'fast']
        if session_type == 'tied' and animal == 'MC9308':
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'slow speed', 'fast speed']
            cond_plot = ['baseline', 'slow', 'fast']
        if session_type == 'tied' and animal == 'MC9513':
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'slow speed', 'fast speed']
            cond_plot = ['baseline', 'slow', 'fast']
        if session_type == 'tied' and animal == 'MC13419':
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'fast speed', 'slow speed']
            cond_plot = ['baseline', 'fast', 'slow']
        if session_type == 'tied' and animal == 'MC13420':
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'fast speed', 'slow speed']
            cond_plot = ['baseline', 'fast', 'slow']
        if session_type == 'tied' and animal == 'MC10221' and session == 2:
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['baseline speed', 'slow speed', 'fast speed']
            cond_plot = ['baseline', 'slow', 'fast']
        if session_type == 'tied' and animal == 'MC10221' and session == 1:
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['slow speed', 'baseline speed', 'fast speed']
            cond_plot = ['slow', 'baseline', 'fast']
        if session_type == 'tied' and animal == 'MC9226' and session == 2:
            trials_ses = np.array([[1, 12], [13, 20], [21, 26]])
            trials_ses_name = ['baseline speed', 'fast speed', 'slow speed']
            cond_plot = ['baseline', 'fast', 'slow']
        if session_type == 'tied' and animal == 'MC9226' and session == 3:
            trials_ses = np.array([[1, 6], [7, 12], [13, 18]])
            trials_ses_name = ['slow speed', 'baseline speed', 'fast speed']
            cond_plot = ['slow', 'baseline', 'fast']
        if session_type == 'split' and animal == 'MC8855':
            trials_ses = np.array([[1, 3], [4, 13], [14, 23]])
            trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
            cond_plot = ['baseline', 'split', 'washout']
            trials_baseline = np.array([1, 2, 3])
            trials_split = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
            trials_washout = np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        if session_type == 'split' and animal != 'MC8855':
            trials_ses = np.array([[1, 6], [7, 16], [17, 26]])
            trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
            cond_plot = ['baseline', 'split', 'washout']
            trials_baseline = np.array([1, 2, 3, 4, 5, 6])
            trials_split = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            trials_washout = np.array([17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
        if session_type == 'split' and animal == 'MC9226' and session == 1:
            trials_ses = np.array([[1, 6], [7, 16], [17, 23]])
            trials_baseline = np.array([1, 2, 3, 4, 5, 6])
            trials_split = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            trials_washout = np.array([17, 18, 19, 20, 21, 22, 23])
            trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
            cond_plot = ['baseline', 'split', 'washout']
        if len(trials) < 24 and session_type == 'tied':
            trials_baseline = np.arange(trials_ses[0, 0], trials_ses[0, -1]+1)
            trials_split = trials
            trials_washout = trials
        if session_type == 'tied':
            trials_baseline = trials
            trials_split = trials
            trials_washout = trials
        return trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout

    @staticmethod
    def cumulative_time(df_data, trials):
        """Computes the cumulative time in the session
        Input:
        df_data: dataframe with events or traces
        trials: list with session trials"""
        time_cumulative = np.zeros(np.shape(df_data)[0])
        time_cumulative[df_data.loc[(df_data['trial'] == 1)].index] = np.array(
            df_data.loc[(df_data['trial'] == 1), 'time'])
        for t in trials[1:]:
            idx_trial = df_data.loc[(df_data['trial'] == t)].index
            idx_trial_minus1 = np.where(trials==t)[0][0]-1
            idx_trial_minus1 = df_data.loc[(df_data['trial'] == trials[idx_trial_minus1])].index
            time_trial = np.array(df_data.loc[(df_data['trial'] == t), 'time'])
            time_cumulative[idx_trial] = time_trial + time_cumulative[idx_trial_minus1][-1]
        return time_cumulative

    def clusters_dataframe(self, df_traces, clusters_rois, detrend, save_data):
        """Create a dataframe with the averaged traces for each cluster
        Input:
        df_traces: dataframe with traces of clusters
        clusters_rois: list with ROIs for each cluster
        detrend: boolean
        save_data: boolean"""
        # dataframe with clusters
        df_trace_clusters_ave = pd.DataFrame(columns=['trial', 'time'])
        df_trace_clusters_ave['trial'] = df_traces['trial']
        df_trace_clusters_ave['time'] = df_traces['time']
        for c in range(len(clusters_rois)):
            df_dFF_mean = df_traces[clusters_rois[c]].mean(axis=1)
            df_trace_clusters_ave['cluster' + str(c + 1)] = np.array(df_dFF_mean)
        if detrend:
            df_trace_clusters_ave = self.compute_detrended_traces(df_trace_clusters_ave,[])
        df_trace_clusters_std = pd.DataFrame(columns=['trial', 'time'])
        df_trace_clusters_std['trial'] = df_traces['trial']
        df_trace_clusters_std['time'] = df_traces['time']
        for c in range(len(clusters_rois)):
            df_dFF_std = df_traces[clusters_rois[c]].std(axis=1)
            df_trace_clusters_std['cluster' + str(c + 1)] = np.array(df_dFF_std)
        if save_data:
            df_trace_clusters_ave.to_csv(os.path.join(self.path, 'processed files', 'df_trace_clusters_ave.csv'),
                                         sep=',', index=False)
            df_trace_clusters_std.to_csv(os.path.join(self.path, 'processed files', 'df_trace_clusters_std.csv'),
                                         sep=',', index=False)
        return df_trace_clusters_ave, df_trace_clusters_std

    def shuffle_events(self, df_events, nr_shuffles):
        """Shuffle events a number of times.
        Inputs:
        df_events: dataframe with events
        nr_shuffle: how mnay shuffles"""
        df_events_shuffle = pd.DataFrame()
        for c in df_events.columns[2:]:
            for i in range(nr_shuffles):
                if i == 0:
                    df_events_shuffle[c] = np.random.permutation(df_events[c])
                else:
                    df_events_shuffle[c] = np.random.permutation(df_events_shuffle[c])
        df_events_shuffle.insert(loc=0, column='trial', value=df_events['trial'])
        df_events_shuffle.insert(loc=0, column='time', value=df_events['time'])
        return df_events_shuffle

    def norm_traces(self, df, norm_name, axis):
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
                        trace_alltrials.extend(
                            self.inpaint_nans((df.loc[df['trial'] == t, col] - mean_value) / std_value))
                    if norm_name == 'min_max':
                        trace_alltrials.extend((df.loc[df['trial'] == t, col] - min_value) / (max_value - min_value))
                df_norm[col] = trace_alltrials
            df_norm.iloc[:, :2] = df.iloc[:, :2]
        if axis == 'session':
            for col in df.columns[2:]:
                mean_value = df[col].mean(axis=0, skipna=True)
                std_value = df[col].std(axis=0, skipna=True)
                min_value = df[col].min(axis=0, skipna=True)
                max_value = df[col].max(axis=0, skipna=True)
                if norm_name == 'zscore':
                    df_norm[col] = (df[col] - mean_value) / std_value
                if norm_name == 'min_max':
                    df_norm[col] = (df[col] - min_value) / (max_value - min_value)
            df_norm.iloc[:, :2] = df.iloc[:, :2]
            for col in df.columns[2:]:
                if norm_name == 'zscore':
                    trace_alltrials = []
                    for t in trials:
                        trace_alltrials.extend(self.inpaint_nans(df_norm.loc[df_norm['trial'] == t, col]))
                    df_norm[col] = trace_alltrials
        return df_norm

    def get_animal_id(self):
        animal_name = self.path.split(self.delim)[-3]
        return animal_name

    def get_protocol_id(self):
        protocol_name = self.path.split(self.delim)[2]
        protocol_id = protocol_name.replace(' ', '_')
        return protocol_id

    def get_trial_id(self):
        """Function to get the trials where recordings occurred (in order)"""
        if self.delim == '/':
            ops = np.load(self.path + 'Suite2p/suite2p/plane0/ops.npy', allow_pickle=True)
        else:
            ops = np.load(self.path + 'Suite2p\\suite2p\\plane0\\ops.npy', allow_pickle=True)
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

    @staticmethod
    def trial_length(df_extract):
        """ Get number of frames for each trial based on traces dataframe
        Input:
        df_extract: dataframe with traces and the usual structure"""
        trials = np.unique(df_extract['trial'])
        trial_length = np.zeros(len(trials))
        for count, t in enumerate(trials):
            trial_length[count] = len(df_extract.loc[df_extract['trial'] == t].index)
        return trial_length

    @staticmethod
    def cumulative_trial_length(frame_time):
        """ Get number of frames for each trial based on mscope timestamps
        Input:
        frame_time: list of mscope timestamps"""
        trial_length = []
        for count_t in range(len(frame_time)):
            trial_length.append(len(frame_time[count_t]))
        trial_length_cumsum = np.cumsum(trial_length)
        return trial_length_cumsum

    def get_s2p_parameters(self):
        """Function to get the parameters used to run suite2p
            Outputs: ops_s2p"""
        ops = np.load(os.path.join(self.path, 'Suite2p', 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True)
        ops_s2p = {'suite2p_version': ops[()]['suite2p_version'], 'tau': ops[()]['tau'], 'fs': ops[()]['fs'],
                   'aspect_ratio': ops[()]['aspect'], 'reg_on': ops[()]['do_registration'],
                   'reg_2step': ops[()]['two_step_registration'],
                   'nimg_init': ops[()]['nimg_init'], 'batch_size': ops[()]['batch_size'],
                   'maxregshift': ops[()]['maxregshift'], 'smooth_sigma_time': ops[()]['smooth_sigma_time'],
                   'smooth_sigma': ops[()]['smooth_sigma'], 'nonrigid_on': ops[()]['nonrigid'],
                   '1preg_on': ops[()]['1Preg'], 'diameter': ops[()]['diameter'], 'connected': ops[()]['connected'],
                   'max_iterations': ops[()]['max_iterations'], 'threshold_scaling': ops[()]['threshold_scaling'],
                   'max_overlap': ops[()]['max_overlap'], 'high_pass_window': ops[()]['high_pass']}
        return ops_s2p

    def get_reg_data(self):
        """Function to get the correlation map computed from suite2p
            Outputs: x_offset, y_offset, corrXY"""
        ops = np.load(os.path.join(self.path, 'Suite2p', 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True)
        y_offset = ops[()]['yoff']  # y shifts in registration
        x_offset = ops[()]['xoff']  # x shifts in registration
        corrXY = ops[()]['corrXY']  # phase correlation between ref image and each frame
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'x_offsets.npy'), x_offset)
        np.save(os.path.join(self.path, 'processed files', 'y_offsets.npy'), y_offset)
        np.save(os.path.join(self.path, 'processed files', 'corrXY_frames.npy'), corrXY)
        return x_offset, y_offset, corrXY

    def corr_FOV_movement(self, th, df_dFF, corrXY):
        """Function to make nan times where FOV moved out of focus
        Input: corrXY (array)"""
        fig = plt.figure(figsize=(5, 5), tight_layout=True)
        plt.plot(corrXY, color='black')
        plt.axhline(y=th, color='gray')
        if not os.path.exists(os.path.join(self.path, 'images')):
            os.mkdir(os.path.join(self.path, 'images'))
        plt.savefig(os.path.join(self.path, 'images', 'corr_fov_movement'), dpi=self.my_dpi)
        idx_to_nan = np.where(corrXY <= th)[0]
        df_dFF.iloc[idx_to_nan, 2:] = np.nan
        return idx_to_nan, df_dFF

    def rois_larger_motion(self, df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi, height_roi,
                           plot_data):
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
        for r in keep_rois - 1:
            coord_ext_nomotion.append(coord_ext[r])
        roi_list = df_extract.columns[2:]
        rois_del = np.setdiff1d(np.arange(1, len(roi_list) + 1), keep_rois)
        rois_del_list = []
        for r in rois_del:
            rois_del_list.append('ROI' + str(r))
        df_extract = df_extract.drop(columns=rois_del_list)
        if plot_data:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True)
            ax = ax.ravel()
            ax[0].scatter(np.arange(1, roi_number + 1), width_roi, s=10, color='blue')
            ax[0].axhline(x_offset_minmax[0], color='black')
            ax[0].axhline(x_offset_minmax[1], color='black')
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].set_title('ROIs width and X max and min offsets', fontsize=self.fsize - 4)
            ax[1].scatter(np.arange(1, roi_number + 1), height_roi, s=10, color='blue')
            ax[1].axhline(y_offset_minmax[0], color='black')
            ax[1].axhline(y_offset_minmax[1], color='black')
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_title('ROIs height and Y max and min offsets', fontsize=self.fsize - 4)
        return [coord_ext_nomotion, df_extract]

    def correlation_signal_motion(self, df_extract, x_offset, y_offset, trial, idx_to_nan, traces_type, plot_data, print_plots):
        """Function to compute the ROIs signal correlation with the shift of the FOV done during motion correction-
        It outputs the correlation plot between the traces and the FOV offsets and an example ROI trace with the shifts.
        Inputs:
            df_extract: (dataframe) with ROIs traces
            x_offset: x shift of frames during motion correction
            y_offset: y shift of frames during motion correction
            trial: (int) example trial to do this computation
            idx_to_nan: indices of frames to make nan, parts where motion was large
            traes_type: (str) raw deconv or cluster
            plot_data: boolean
            print_plots: boolean"""
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
            r = np.random.choice(df_extract.columns[2:])
            df_extract_norm = self.norm_traces(df_extract, 'min_max', 'session')
            fig, ax = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
            ax = ax.ravel()
            ax[0].plot(np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, r]), color='darkgrey')
            ax[0].plot(
                x_offset_clean_norm[
                    np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, r].index)],
                color='blue')
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].set_title('Example ROI trace with X offset', fontsize=self.fsize - 4)
            ax[1].plot(np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, r]),
                       color='darkgrey')
            ax[1].plot(
                y_offset_clean_norm[
                    np.array(df_extract_norm.loc[df_extract_norm['trial'] == trial, r].index)],
                color='blue')
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].set_title('Example ROI trace with Y offset', fontsize=16)

            fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
            ax.plot(df_extract.columns[2:], p_corrcoef[:, 0], color='blue', marker='o',
                    label='correlation of trace with x shifts')
            ax.plot(df_extract.columns[2:], p_corrcoef[:, 1], color='orange', marker='o',
                    label='correlation of trace with y shifts')
            ax.legend(frameon=False, fontsize=self.fsize - 6)
            ax.set_title('Correlation of traces with FOV shift during motion correction', fontsize=self.fsize - 4)
            ax.set_xticks(df_extract.columns[2:][::10])
            ax.set_xticklabels(list(df_extract.columns[2::10]))
            ax.set_xlabel('ROI ID', fontsize=self.fsize - 6)
            ax.set_ylabel('Correlation coefficient', fontsize=self.fsize - 6)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=self.fsize - 6)
            ax.tick_params(axis='y', labelsize=self.fsize - 6)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'corr_trace_motionreg_shifts_'+traces_type),
                            dpi=self.my_dpi)
        return p_corrcoef

    def get_ref_image(self):
        """Function to get the session reference image from suite2p"""
        path_ops = os.path.join('Suite2p', 'suite2p', 'plane0', 'ops.npy')
        ops = np.load(self.path + path_ops, allow_pickle=True)
        ref_image = ops[()]['meanImg']
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'ref_image.npy'), ref_image)
        return ref_image

    def correct_for_deleted_trials(self, trials, trial_start, strobe_nr, bcam_time, colors_session, frame_time, frames_dFF, frames_loco):
        path_extract = os.path.join(self.path, 'Registered video', 'EXTRACT')
        files_extract = glob.glob(path_extract + self.delim + '*.mat')
        trials_in = []
        for f in range(len(files_extract)):
            filename = files_extract[f][files_extract[f].rfind(self.delim):]
            trials_in.append(np.int64(filename[filename.find('T') + 1:filename.rfind('_')]))
        trials_out_idx = []
        trials_out = []
        for count_t, t in enumerate(trials):
            if t not in list(np.sort(trials_in)):
                trials_out_idx.append(count_t)
                trials_out.append(t)
                print('Deleted trials ' + str(t))
        if len(trials_out_idx)>0:
            for k in trials_out_idx:
                trials = np.delete(trials, k)
                trial_start = np.delete(trial_start, k)
                strobe_nr = np.delete(strobe_nr, k)
                bcam_time = np.delete(bcam_time, k)
                frame_time = np.delete(frame_time, k)
                frames_dFF = np.delete(frames_dFF, k)
                frames_loco = np.delete(frames_loco, k)
            for t in trials_out:
                colors_session.pop(t)
        return trials, trial_start, strobe_nr, bcam_time, colors_session, frame_time, frames_dFF, frames_loco, trials_out_idx

    def read_extract_output(self, threshold_spatial_weights, frame_time, trials):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (EXTRACT output)
        Inputs:
            threshold_spatial_weights: float
            frame_time: list with miniscope timestamps
            trials: list of trials"""
        path_extract = os.path.join(self.path, 'Registered video', 'EXTRACT')
        files_extract = glob.glob(path_extract + self.delim + '*.mat')
        ext_rois = mat73.loadmat(files_extract[0])  # masks are the same across trials
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
            idx_trial = np.where(trials == t)[0][0]
            trial_ext.extend(np.repeat(t, len(frame_time[idx_trial])))
            frame_time_ext.extend(frame_time[idx_trial])
        data_ext1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_ext1 = pd.DataFrame(data_ext1)
        df_ext2 = pd.DataFrame(trace_ext_arr, columns=roi_list)
        df_ext = pd.concat([df_ext1, df_ext2], axis=1)
        return coord_cell_t, df_ext

    def compute_extract_rawtrace(self, coord_ext, df_extract, roi_list, trials, frame_time):
        """Function to compute the raw traces from the ROI coordinates from EXTRACT.
        Input:
            coord_cell: list with ROIs coordinates
            df_extract: deconvoluted traces (just to get the size of dataframe)
            roi_list: list with ROIs
            trials: array with all the trials in the session
            frame_time: list with frame timestamps"""
        trial_length = self.trial_length(df_extract)
        ext_trace_trials = []
        for t in trials:
            idx_trial = np.where(trials == t)[0][0]
            tiff_stack = tiff.imread(
                os.path.join(self.path, 'Registered video') + self.delim + 'T' + str(t) + '_reg.tif')  # read tiffs
            ext_trace = np.zeros((int(trial_length[idx_trial]), np.shape(df_extract.iloc[:, 2:])[1]))
            for c in range(len(coord_ext)):
                ext_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
                for f in range(np.shape(tiff_stack)[0]):
                    ext_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                        np.round(coord_ext[c][:, 1] * self.pixel_to_um)), np.int64(
                        np.round(coord_ext[c][:, 0] * self.pixel_to_um))]) / np.shape(coord_ext[c])[0]
                ext_trace[:, c] = ext_trace_tiffmean
            ext_trace_trials.append(ext_trace)
        ext_trace_arr = np.transpose(np.vstack(ext_trace_trials))  # trace as dataframe
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            idx_trial = np.where(trials==t)[0][0]
            trial_ext.extend(np.repeat(t, len(frame_time[idx_trial])))
            frame_time_ext.extend(frame_time[idx_trial])
        dict_ext = {'trial': trial_ext, 'time': frame_time_ext}
        df_ext1 = pd.DataFrame(dict_ext)
        df_ext2 = pd.DataFrame(np.transpose(ext_trace_arr), columns=roi_list)
        df_ext_raw = pd.concat([df_ext1, df_ext2], axis=1)
        return df_ext_raw

    def get_imagej_output(self, frame_time, trials, norm):
        """Function to get the pixel coordinates (list of arrays) and calcium trace
         (dataframe) for each ROI giving a threshold on the spatial weights
        (ImageJ output)
        Inputs:
            frame_time: list with miniscope timestamps
            trial: (arr) - trial list
            norm: boolean to do min-max normalization"""
        path_fiji = 'Registered video'
        filename_rois = 'RoiSet.zip'
        rois = read_roi.read_roi_zip(os.path.join(self.path, path_fiji, filename_rois))
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
            tiff_stack = tiff.imread(
                os.path.join(self.path + path_fiji) + self.delim + 'T' + str(t) + '_reg.tif')  ##read tiffs
            roi_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0], len(coord_fiji)))
            for c in range(len(coord_fiji)):
                for f in range(np.shape(tiff_stack)[0]):
                    roi_trace_tiffmean[f, c] = np.nansum(tiff_stack[f, np.int64(
                        coord_fiji[c][:, 1] * self.pixel_to_um), np.int64(
                        coord_fiji[c][:, 0] * self.pixel_to_um)]) / len(coord_fiji[c][:, 1])
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
        roi_list = []
        for r in range(len(coord_fiji)):
            roi_list.append('ROI' + str(r + 1))
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t - 1])))
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
        rois_idx_aspectratio = list(np.where(np.array(aspect_ratio) > 2)[0])
        coord_ext_aspectratio = []
        for r in rois_idx_aspectratio:
            coord_ext_aspectratio.append(coord_cell[r])
        roi_idx_bad_aspectratio = np.setdiff1d(np.arange(0, len(coord_cell)), np.array(rois_idx_aspectratio))
        roi_list_bad_aspectratio = df_dFF.columns[2:][roi_idx_bad_aspectratio]
        df_dFF_aspectratio = df_dFF.drop(columns=roi_list_bad_aspectratio)
        skewness_dFF = df_dFF_aspectratio.skew(axis=0, skipna=True)
        skewness_dFF_argsort = np.argsort(np.array(skewness_dFF[2:]))
        # ROI curation (ΔF/F with the same range)
        range_dFF = [df_dFF_aspectratio.loc[df_dFF_aspectratio['trial'] == trial_curation].min(axis=0, skipna=True)[2:].min(skipna=True),
                     df_dFF_aspectratio.loc[df_dFF_aspectratio['trial'] == trial_curation].max(axis=0, skipna=True)[2:].max(skipna=True)]
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
            # ax1.set_ylim(range_dFF)
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
        rois_names_ordered_complete = ['time', 'trial', ]
        for r in rois_names_ordered:
            rois_names_ordered_complete.append('ROI' + str(r))
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
        F_coord_roi_norm = F_coord_roi - np.transpose(np.tile(np.nanmean(F_coord_roi, axis=1), (
        np.shape(F_coord_roi)[1],
        1)))  # mean subtraction is essential for PCA, zscore might be bad for mahalanobnis distance
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
        mahalanobis_dist_100thpixel = \
        spdist.cdist(principalComponents_2CP, np.array([principalComponents_2CP[centroid_roi_cluster_idx, :]]),
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
        F_coord_bg_norm = F_coord_bg - np.transpose(np.tile(np.nanmean(F_coord_bg, axis=1), (np.shape(F_coord_bg)[1],
                                                                                             1)))  # mean subtraction is essential for PCA, zscore might be bad for mahalanobnis distance
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
        mahalanobis_dist_100thpixel_bg = \
        spdist.cdist(principalComponents_2CP_bg, np.array([principalComponents_2CP_bg[centroid_roi_cluster_idx_bg, :]]),
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
            path_timestamps = os.path.join(self.delim.join(path_split[:-2]), 'Miniscopes')
            columns_to_keep = ['camNum', 'frameNum', 'sysClock', 'buffer']
        if version == 'v4':
            path_timestamps = self.path
            columns_to_keep = ['Frame Number', 'Time Stamp (ms)', 'Buffer Index']
        frame_time = []
        for t in range(len(trials)):
            if version == 'v3.2':
                df = pd.read_table(os.path.join(path_timestamps, 'T' + str(trials[t]), "timestamp.dat"), sep="\s+",
                                   usecols=columns_to_keep)
                sysClock = np.array(df['sysClock'])
            if version == 'v4':
                df = pd.read_table(
                    os.path.join(path_timestamps, 'T' + str(trials[t]), "Miniscope", "timeStamps.csv"),
                    sep=",", usecols=columns_to_keep)
                sysClock = np.array(df['Time Stamp (ms)'])
            # first sysclock has to be 0
            sysClock[0] = 0  # to mark time 0
            sysClock[1:] = sysClock[1:]
            sysClock_clean = sysClock[frames_dFF[t] - 1:] / 1000  # -1 because frame index starts at 0
            frame_time.append(
                sysClock_clean - sysClock_clean[0])  # get time of each frame to align with behavior recording
        return frame_time       

    def get_HF_trials(self, tifflist):
        """From the list of acquired tiffs get the list of trials recorded
        Inputs:
            tifflist: list of recorded tiff files"""
        trials = []
        for count_f, f in enumerate(tifflist):
            trials.append(np.int64(f[f.rfind('T')+1:f.find('.')]))
        trials_ordered = np.sort(trials)
        return trials_ordered

    def get_HF_frame_time(self, trials, tifflist):
        """From list of acquired tiffs get the timestamp of each frame
        (Assumes set 30Hz acquisition without loss of frames)
        Inputs:
            tifflist: list of recorded tiff files"""
        tifflist_ordered = []
        for count_t, t in enumerate(trials):
            for l in tifflist:
                if np.int64(l[l.rfind('T')+1:l.rfind('.')]) == t:
                    tifflist_ordered.append(l) 
        frame_time = []
        for f in tifflist_ordered:
            image_stack = tiff.imread(f)
            frame_nr = image_stack.shape[-1]
            frame_time.append(np.linspace(0, frame_nr*(1/30), num=frame_nr))
        return frame_time  

    def get_black_frames(self):
        """Get the number of black frames per tiff video that had to be removed.
        Frames only removed from the beginning"""
        tiflist = glob.glob(os.path.join(self.path, 'Suite2p', '*.tif'))
        tiff_boolean = 0
        if not tiflist:
            tiflist = glob.glob(os.path.join(self.path, 'Suite2p', '*.tiff'))  # get list of tifs
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
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'black_frames.npy'), frames_dFF)
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
                os.path.join(self.path, 'T' + str(trials[t]), "Miniscope", "headOrientation.csv"),
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
        head_orientation.to_csv(os.path.join(self.path, 'processed files', 'head_angles.csv'), sep=',', index=False)
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
                os.path.join(self.path, 'T' + str(trials[t]), "Miniscope", "headOrientation.csv"),
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
        head_orientation.to_csv(os.path.join(self.path, 'processed files', 'head_quartenions.csv'), sep=',', index=False)
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
                if not os.path.exists(os.path.join(self.path, 'images', 'acc')):
                    os.mkdir(os.path.join(self.path, 'images', 'acc'))
                plt.savefig(os.path.join(self.path, 'images', 'acc', 'pca_2d'), dpi=400)
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
            if not os.path.exists(os.path.join(self.path, 'images')):
                os.mkdir(os.path.join(self.path, 'images'))
            plt.savefig(os.path.join(self.path, 'images', 'rois_fov'), dpi=self.my_dpi)
        return

    def plot_single_roi_ref_image(self, ref_image, coord_cell, roi_plot, traces_type, roi_list, colors_cluster,
                                  idx_roi_cluster_ordered, print_plots):
        """Plot a single ROI on top of reference image.
        Inputs:
            ref_image: reference image
            coord_cell: coordinates for each ROI
            roi_plot: int
            roi_list: list with ROIs
            colors_cluster: (list) with colors for each cluster
            idx_roi_cluster_ordered. (list) with the idx for each roi organized medial-lateral
            print_plots (boolean)"""
        roi_list_new = [np.int64(s.strip('ROI')) for s in roi_list]
        roi_idx = np.where(np.array(roi_list_new) == roi_plot)[0][0]
        plt.figure(figsize=(10, 10), tight_layout=True)
        plt.scatter(coord_cell[roi_idx][:, 0], coord_cell[roi_idx][:, 1], s=1,
                    color=colors_cluster[idx_roi_cluster_ordered[roi_idx] - 1])
        plt.imshow(ref_image, cmap='gray',
                   extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um, 0])
        plt.title('ROIs grouped by activity', fontsize=self.fsize)
        plt.xlabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.ylabel('FOV in micrometers', fontsize=self.fsize - 4)
        plt.xticks(fontsize=self.fsize - 4)
        plt.yticks(fontsize=self.fsize - 4)
        if print_plots:
            if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
            if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot))):
                os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot)))
            plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot), 'roi_fov'),
                        dpi=self.my_dpi)

    def plot_heatmap_baseline(self, df_dFF, traces_type, plot_data):
        """Plots the heatmap for all ROIs given the their traces (min-max normalized).
        Plots the first 6 trials - baseline trials or fully tied, depending on the session
        Input:
            df_dFF: dataframe with traces
            traces_type: (str) raw or deconv
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
            plt.savefig(os.path.join(self.path, 'images', 'heatmap_1st_6trials_minmax_traces_allROIs_' + traces_type),
                        dpi=self.my_dpi)
        return

    def plot_stacked_traces(self, df_dFF, traces_type, trials, trials_plot, plot_data, print_plots):
        """"Funtion to compute stacked traces for a single trial or for the transition trials in the session.
        Input:
        df_dFF: dataframe with calcium trace
        traces_type: (str) raw or deconv
        trials: list of trials in the session
        trials_plot: int or list
        plot_data: boolean
        print_plots: boolean"""
        int_find = ''.join(x for x in df_dFF.columns[2] if x.isdigit())
        int_find_idx = df_dFF.columns[2].find(int_find)
        if df_dFF.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        if isinstance(trials_plot, np.ndarray):
            count_t = 0
            if plot_data:
                fig, ax = plt.subplots(2, 2, figsize=(15, 20), tight_layout=True)
                ax = ax.ravel()
                for t in trials_plot:
                    idx_trial = np.where(trials==t)[0][0]
                    dFF_trial = df_dFF.loc[df_dFF['trial'] == t]  # get dFF for the desired trial
                    count_r = 0
                    for r in df_dFF.columns[2:]:
                        ax[count_t].plot(dFF_trial['time'], dFF_trial[r] + (count_r / 2), color='black')
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
            if plot_data:
                fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
                count_r = 0
                for r in df_dFF.columns[2:]:
                    plt.plot(dFF_trial['time'], dFF_trial[r] + (count_r / 2), color='black')
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
            if df_type == 'ROI':
                plt.savefig(os.path.join(self.path, 'images', 'dFF_stacked_traces_' + traces_type), dpi=self.my_dpi)
            if df_type == 'cluster':
                plt.savefig(os.path.join(self.path, 'images', 'dFF_stacked_traces_' + traces_type + '_cluster'),
                            dpi=self.my_dpi)

    def plot_stacked_traces_singleROI(self, df_dFF, traces_type, roi_plot, trials, colors_session, line_ratio, plot_data,
                                      print_plots):
        """"Funtion to compute stacked traces for all trials in a session for a single ROI.
        Input:
        frame_time: list with mscope timestamps
        df_dFF: dataframe with calcium trace
        traces_type: (str) raw or deconv
        roi_plot: int or list
        colors_session: list
        line_ratio: integer to decrease separation between trials plotted for better viz
        plot_data: boolean
        print_plots: boolean"""
        int_find = ''.join(x for x in df_dFF.columns[2] if x.isdigit())
        int_find_idx = df_dFF.columns[2].find(int_find)
        if df_dFF.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi_plot)
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 20), tight_layout=True)
            count_t = len(trials)
            y_plot = []
            for count_c, t in enumerate(trials):
                dFF_trial = df_dFF.loc[df_dFF['trial'] == t, idx_nr]  # get dFF for the desired trial
                frame_time = df_dFF.loc[df_dFF['trial'] == t, 'time']
                ax.plot(frame_time, dFF_trial + (count_t/line_ratio), color=colors_session[t])
                y_plot.append(np.nanmean(dFF_trial + (count_t/line_ratio)))
                count_t -= 1
            ax.set_yticks(y_plot)
            ax.set_yticklabels(map(str, trials))
            ax.set_xlabel('Time (s)', fontsize=self.fsize - 2)
            ax.set_ylabel('Trials', fontsize=self.fsize - 2)
            ax.set_title('Calcium trace for ' + df_type + ' ' + str(roi_plot), fontsize=self.fsize - 2)
            ax.spines['left'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 2)
            plt.yticks(fontsize=self.fsize - 2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot),
                                             'dFF_stacked_traces_' + traces_type), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type)):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type))
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot),
                                             'Cluster' + str(roi_plot) + '_dFF_stacked_traces_' + traces_type),
                                dpi=self.my_dpi)
        return

    def compute_roi_clustering(self, df_dFF, centroid_cell, distance_neurons, trials, th_cluster, colormap_cluster,
                               plot_data, print_plots):
        """Function to get colors of ROIs according to its cluster id
        Input:
        data_arr: dataframe with calcium trace
        centroid_cell: list with ROIs centroids
        distance_neurons: matrix with distance from the first ROI
        trials: (list) trials to compute roi clustering
        th_cluster: float, threshold for clustering
        colormap: (str) with colormap name
        plot_data: boolean
        print_plots: boolean"""
        data = np.transpose(np.array(df_dFF.loc[(df_dFF['trial'] >= trials[0])&(df_dFF['trial'] >= trials[-1])].iloc[:, 2:]))
        roi_nr = np.shape(data)[0]
        roi_list = np.arange(1, np.shape(data)[0] + 1)
        p_corrcoef = np.zeros((roi_nr, roi_nr))
        p_corrcoef[:] = np.nan
        for roi1 in roi_list:
            for roi2 in roi_list:
                idx_nonan = np.where(~np.isnan(data[0, :]))[0]
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
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'cluster')):
                    os.mkdir(os.path.join(self.path, 'images', 'cluster'))
                plt.savefig(os.path.join(self.path, 'images', 'cluster', 'dendrogram'), dpi=self.my_dpi)
            furthest_neuron = np.argmax(np.array(centroid_cell)[:, 0])
            neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
            p_corrcoef_ordered = p_corrcoef[neuron_order, :][:, neuron_order]
            fig, ax = plt.subplots()
            with sns.axes_style("white"):
                sns.heatmap(p_corrcoef_ordered, cmap="YlGnBu", linewidth=0.5)
                ax.set_title('correlation matrix ordered by distance between ROIs')
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'cluster')):
                    os.mkdir(os.path.join(self.path, 'images', 'cluster'))
                plt.savefig(os.path.join(self.path, 'images', 'cluster', 'pcorrcoef_ordered'), dpi=self.my_dpi)
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'clusters_rois_colors.npy'), colors)
        return colors, idx

    def get_rois_clusters_mediolateral(self, df_traces, idx_roi_cluster, centroid_ext):
        """Get the ROIs names that belong to which cluster. Clusters organized by their mediolateral distance
        Inputs:
            df_traces: dataframe with the traces for all ROIs
            idx_roi_cluster: list of cluster if for each ROI index
            centroid_ext: list of coordinates of ROIs centroids"""
        roi_names = df_traces.columns[2:]
        clusters_id = np.unique(idx_roi_cluster)
        clusters_rois = []
        for i in clusters_id:
            single_cluster_rois = []
            for count_r, r in enumerate(roi_names):
                if idx_roi_cluster[count_r] == i:
                    single_cluster_rois.append(r)
            clusters_rois.append(single_cluster_rois)
        centroid_cluster = []
        for c in range(len(clusters_rois)):
            centroid_cluster.append(
                centroid_ext[np.where(df_traces.columns[2:] == clusters_rois[c][0])[0][0]][
                    0])  # mediolateral coordinates of a ROI in cluster
        # order it by their mediolateral distance
        centroid_cluster_order = np.argsort(centroid_cluster)
        clusters_rois_ordered = []
        for i in centroid_cluster_order:
            clusters_rois_ordered.append(clusters_rois[i])
        idx_roi_cluster_list = []
        for c in range(len(clusters_rois_ordered)):
            idx_roi_cluster_list.append([c + 1] * len(clusters_rois_ordered[c]))
        idx_roi_cluster_flat = np.array(sum(idx_roi_cluster_list, []))
        clusters_rois_ordered_flat = sum(clusters_rois_ordered, [])
        roi_clusters_ordered_flat = []
        for count_r, r in enumerate(clusters_rois_ordered_flat):
            roi_clusters_ordered_flat.append(np.int64(r[3:]))
        idx_roi_cluster_ordered = idx_roi_cluster_flat[np.argsort(roi_clusters_ordered_flat)]
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'clusters_rois.npy'), clusters_rois_ordered)
        np.save(os.path.join(self.path, 'processed files', 'clusters_rois_idx_order.npy'), idx_roi_cluster_ordered)
        return [clusters_rois_ordered, idx_roi_cluster_ordered]

    def get_global_coordinates_cluster(self, centroid_ext, animal, idx_roi_cluster_ordered):
        """Get the coordinates of the clusters based on the mean of the centroids.
        Put the coordinates in a global scale (based on histology)
        Inputs:
            idx_roi_cluster_ordered: list of cluster if for each ROI index
            animal: (str) animal name to get the right FOV global coordinates
            centroid_ext: list of coordinates of ROIs centroids"""
        centroid_cluster_mean = np.zeros((len(np.unique(idx_roi_cluster_ordered)), 2))
        for count_i, i in enumerate(np.unique(idx_roi_cluster_ordered)):
            cluster_idx = np.where(idx_roi_cluster_ordered == i)[0]
            centroid_cluster = np.zeros((len(cluster_idx), 2))
            for count_c, c in enumerate(cluster_idx):
                centroid_cluster[count_c, :] = centroid_ext[c]
            centroid_mean = np.nanmean(centroid_cluster, axis=0)
            centroid_cluster_mean[count_i, 0] = -centroid_mean[0]  # because we are in the negative area of bregma
            centroid_cluster_mean[count_i, 1] = centroid_mean[1]
        if animal == 'MC8855':
            fov_coord = np.array([-6.27, 0.53])
        if animal == 'MC9194':
            fov_coord = np.array([-6.61, 0.89])
        if animal == 'MC9226':
            fov_coord = np.array([-6.39, 1.62])
        if animal == 'MC9513':
            fov_coord = np.array([-6.98, 1.47])
        if animal == 'MC10221':
            fov_coord = np.array([-6.80, 1.75])
        fov_corner = np.array([fov_coord[0] + 0.5, fov_coord[1] - 0.5])
        centroid_cluster_dist_corner = (centroid_cluster_mean * 0.001) + fov_corner
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'cluster_coords.npy'), centroid_cluster_dist_corner)
        return centroid_cluster_dist_corner

    def plot_roi_clustering_spatial(self, ref_image, colors, idx, coord_cell, plot_data, print_plots):
        """Plot ROIs on top of reference image color coded by the result of the hierarchical clustering
        Inputs:
            ref_image: reference image
            colors: colors for each cluster
            idx: to which cluster each ROI belongs to
            coord_cell: coordinates for each ROI
            plot_data: boolean
            print_plots (boolean)"""
        if plot_data:
            plt.figure(figsize=(10, 10), tight_layout=True)
            for r in range(len(coord_cell)):
                plt.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], color=colors[idx[r] - 1], s=1, alpha=0.6)
            plt.imshow(ref_image, cmap='gray',
                       extent=[0, np.shape(ref_image)[1] / self.pixel_to_um, np.shape(ref_image)[0] / self.pixel_to_um,
                               0])
            plt.title('ROIs grouped by activity', fontsize=self.fsize)
            plt.xlabel('FOV in micrometers', fontsize=self.fsize - 4)
            plt.ylabel('FOV in micrometers', fontsize=self.fsize - 4)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'cluster')):
                    os.mkdir(os.path.join(self.path, 'images', 'cluster'))
                plt.savefig(os.path.join(self.path, 'images', 'cluster', 'roi_clustering_fov'), dpi=self.my_dpi)
        return

    def get_rois_aligned_reference_cluster(self, df_data, coord_ext, animal):
        """Get the cluster reference id for each ROI of the session. Get the ROIs with at least
        75% overlap.
        Inputs:
        df_data: dataframe with dFF data or events
        coord_ext: list of ROI coordinates
        animal: (str) animal name"""
        path_ref_clusters = os.path.join(self.path[:self.path.find('TM RAW FILES')], 'TM RAW FILES', 'reference clusters')
        coord_ext_reference_ses = np.load(os.path.join(path_ref_clusters,
                             'coord_ext_reference_ses_' + animal + '.npy'), allow_pickle=True)
        idx_roi_cluster_ordered_reference_ses = np.load(os.path.join(path_ref_clusters,
                             'clusters_rois_idx_order_reference_ses_' + animal + '.npy'), allow_pickle=True)
        overlap = 0.75
        size_frame = np.int64(608/self.pixel_to_um)
        clusters = np.unique(idx_roi_cluster_ordered_reference_ses)
        coord_ext_overlap = np.zeros(np.shape(coord_ext)[0])
        for i in range(np.shape(coord_ext)[0]): #for all the ROIs in the session
            coord_roi_int = np.int64(coord_ext[i])
            frame_arr_coord = np.ones((size_frame, size_frame))*-1
            frame_arr_coord[coord_roi_int[:, 0], coord_roi_int[:, 1]] = 1 #make matrix of 1 where roi is
            for c in clusters:
                idx_cluster = np.where(idx_roi_cluster_ordered_reference_ses == c)[0]
                rois_coordinates_cluster = np.array(list(chain.from_iterable(coord_ext_reference_ses[idx_cluster])))
                coord_cluster_int = np.int64(rois_coordinates_cluster)
                frame_arr_cluster = np.ones((size_frame, size_frame))*-1
                frame_arr_cluster[coord_cluster_int[:, 0], coord_cluster_int[:, 1]] = 1 #make matrix of 1 where reference cluster is
                #overlap of reference cluster and roi is 2
                coord_roi_overlap = len(np.where((frame_arr_coord + frame_arr_cluster) == 2)[0])/len(np.where(frame_arr_coord == 1)[0])
                if coord_roi_overlap >= overlap:
                    coord_ext_overlap[i] = c #get cluster reference id for each roi, 0 is none
        roi_list = self.get_roi_list(df_data)
        clusters_id = np.unique(coord_ext_overlap) #get overlap rois in cluster_rois variable format
        clusters_rois_overlap = []
        for c in clusters_id:
            cluster_id_idx = np.where(coord_ext_overlap == c)[0]
            cluster_rois_single = []
            for i in cluster_id_idx:
                cluster_rois_single.append(roi_list[i])
            clusters_rois_overlap.append(cluster_rois_single)
        return coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap

    def plot_single_cluster_map(self, ref_image, colors, idx, coord_cell, traces_type, plot_data, print_plots):
        """Plot ROIs on top of reference image color coded by the result of the hierarchical clustering
        Inputs:
            ref_image: reference image
            colors: colors for each cluster
            idx: to which cluster each ROI belongs to
            coord_cell: coordinates for each ROI
            traces_type: (str) raw or deconv
            plot_data: boolean
            print_plots (boolean)"""
        if plot_data:
            for c in range(len(np.unique(idx))):
                alpha_arr = np.zeros(len(idx))
                alpha_arr[np.where(idx == c + 1)[0]] = 0.6
                plt.figure(figsize=(10, 10), tight_layout=True)
                for r in range(len(coord_cell)):
                    plt.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], color=colors[idx[r] - 1], s=1,
                                alpha=alpha_arr[r])
                plt.imshow(ref_image, cmap='gray',
                           extent=[0, np.shape(ref_image)[1] / self.pixel_to_um,
                                   np.shape(ref_image)[0] / self.pixel_to_um, 0])
                plt.title('ROIs grouped by activity', fontsize=self.fsize)
                plt.xlabel('FOV in micrometers', fontsize=self.fsize - 4)
                plt.ylabel('FOV in micrometers', fontsize=self.fsize - 4)
                plt.xticks(fontsize=self.fsize - 4)
                plt.yticks(fontsize=self.fsize - 4)
                if print_plots:
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type)):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type))
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(c + 1))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(c + 1)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(c + 1),
                                             'cluster' + str(c + 1) + '_map'),
                                dpi=self.my_dpi)
        return

    def plot_roi_clustering_temporal(self, df_dFF_norm, centroid_cell, distance_neurons, trial_plot, colors,
                                     idx, plot_ratio, plot_data, print_plots):
        """Plot ROIs on top of reference image color coded by the result of the hierarchical clustering
        Ordered by distance between ROIs.
        Inputs:
            df_dFF: dataframe with calcium trace normalized
            trial_plot: (int) trial to plot
            colors: colors for each cluster
            idx: to which cluster each ROI belongs to
            plot_ratio: if 1 plot all ROIs, if 2, plot every other ROI...
            plot_data: boolean
            print_plots (boolean)"""
        furthest_neuron = np.argmax(np.array(centroid_cell)[:, 0])
        neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
        roi_list = df_dFF_norm.columns[2:]
        roi_list_ordered = roi_list[neuron_order]
        idx_ordered = idx[neuron_order]
        dFF_trial = df_dFF_norm.loc[df_dFF_norm['trial'] == trial_plot]  # get dFF for the desired trial
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
            for count_line, r in enumerate(roi_list_ordered[::plot_ratio]):
                count_r = np.where(r == roi_list_ordered)[0][0]
                plt.plot(dFF_trial.loc[dFF_trial['trial'] == trial_plot, 'time'], dFF_trial[r] + count_line, color=colors[idx_ordered[count_r] - 1])
            ax.set_xlabel('Time (s)', fontsize=self.fsize - 2)
            ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=self.fsize - 2)
            plt.xticks(fontsize=self.fsize - 2)
            plt.yticks(fontsize=self.fsize - 2)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='y', which='y', length=0)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tick_params(axis='y', labelsize=0, length=0)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'cluster')):
                    os.mkdir(os.path.join(self.path, 'images', 'cluster'))
                plt.savefig(os.path.join(self.path, 'images', 'cluster', 'roi_clustering_trace'), dpi=self.my_dpi)
        return

    def compute_clustered_traces_events_correlations(self, df_events, df_traces, clusters_rois, colors_cluster, trials, plot_data, print_plots):
        """From the traces and events dataframes computes the correlation between ROIs and saves the clustered dataframes
        Input.
            df_events: (dataframe) with the events
            df_traces: (dataframe) with the traces
            clusters_rois: (list) ROIs clusters ordered medio-lateral
            colors_cluster: (list) with the colors for each cluster
            trials (list) trials in the session
            plot_data, print_plots: boolean"""
        # Order ROIs by cluster
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        df_events_clustered = df_events[clusters_rois_flat]
        df_traces_clustered = df_traces[clusters_rois_flat]
        if plot_data:
            for trial_plot in trials:
                fig, ax = plt.subplots(1, 2, figsize=(25, 10), tight_layout=True)
                ax = ax.ravel()
                sns.heatmap(df_events_clustered.loc[
                                df_events_clustered['trial'] == trial_plot].iloc[:, 2:].corr(), ax=ax[0])
                ax[0].set_xticks(np.arange(0, len(df_events_clustered.columns[2:]), 4))
                ax[0].set_xticklabels(df_events_clustered.columns[2::4], rotation=45, fontsize=self.fsize - 8)
                ax[0].set_yticks(np.arange(0, len(df_events_clustered.columns[2:]), 4))
                ax[0].set_yticklabels(df_events_clustered.columns[2::4], rotation=45, fontsize=self.fsize - 8)
                ax[0].set_title('Events correlation by cluster', fontsize=self.fsize - 4)
                sns.heatmap(df_traces_clustered.loc[
                                df_traces_clustered['trial'] == trial_plot].iloc[:, 2:].corr(),
                            ax=ax[1])
                ax[1].set_xticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
                ax[1].set_xticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=self.fsize - 8)
                ax[1].set_yticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
                ax[1].set_yticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=self.fsize - 8)
                ax[1].set_title('Traces correlation by cluster', fontsize=self.fsize - 4)
                for c in range(len(clusters_rois)):
                    ax[0].axvline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
                    ax[1].axvline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
                    ax[0].axhline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
                    ax[1].axhline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
                plt.suptitle('Trial ' + str(trial_plot))
                if print_plots:
                    if not os.path.exists(os.path.join(self.path, 'images', 'cluster')):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster'))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', 'cluster_events_trace_trial' + str(trial_plot)), dpi=self.my_dpi)
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        df_events_clustered.to_csv(self.path + '\\processed files\\df_events_extract_rawtrace_clustered.csv', sep=',',
                                   index=False)
        df_traces_clustered.to_csv(self.path + '\\processed files\\df_extract_rawtrace_detrended_clustered.csv',
                                   sep=',', index=False)
        return df_events_clustered, df_traces_clustered

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
                    donut_trace_tiffmean[f] = np.nansum(tiff_stack[f, ROIdonut_coord[:, 1], ROIdonut_coord[:, 0]]) / \
                                              np.shape(ROIdonut_coord)[0]
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
            roi_list.append('ROI' + str(r + 1))
        trial_ext = []
        frame_time_ext = []
        for t in trials:
            trial_ext.extend(np.repeat(t, len(frame_time[t - 1])))
            frame_time_ext.extend(frame_time[t - 1])
        data_fiji1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_fiji1 = pd.DataFrame(data_fiji1)
        df_fiji2 = pd.DataFrame(roi_trace_bgsub_arr, columns=roi_list)
        df_fiji = pd.concat([df_fiji1, df_fiji2], axis=1)
        return [df_fiji, roi_trace_bgsub_arr]

    def save_processed_files(self, df_extract, trials, df_events_extract, df_extract_rawtrace,
                             df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, th, idx_to_nan):
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
        df_extract.to_csv(os.path.join(self.path, 'processed files', 'df_extract.csv'), sep=',', index=False)
        df_events_extract.to_csv(os.path.join(self.path, 'processed files', 'df_events_extract.csv'), sep=',',
                                 index=False)
        df_extract_rawtrace.to_csv(os.path.join(self.path, 'processed files', 'df_extract_raw.csv'), sep=',',
                                   index=False)
        df_extract_rawtrace_detrended.to_csv(
            os.path.join(self.path, 'processed files', 'df_extract_rawtrace_detrended.csv'), sep=',',
            index=False)
        df_events_extract_rawtrace.to_csv(os.path.join(self.path, 'processed files', 'df_events_extract_rawtrace.csv'),
                                          sep=',', index=False)
        np.save(os.path.join(self.path, 'processed files', 'coord_ext.npy'), coord_ext, allow_pickle=True)
        np.save(os.path.join(self.path, 'processed files', 'trials.npy'), trials)
        np.save(os.path.join(self.path, 'processed files', 'reg_th.npy'), th)
        np.save(os.path.join(self.path, 'processed files', 'frames_to_exclude.npy'), idx_to_nan)
        return

    def load_processed_files(self):
        """Loads processed files that were saved under path/processed files"""
        df_extract = pd.read_csv(os.path.join(self.path, 'processed files', 'df_extract.csv'))
        df_events_extract = pd.read_csv(os.path.join(self.path, 'processed files', 'df_events_extract.csv'))
        df_extract_rawtrace = pd.read_csv(os.path.join(self.path, 'processed files', 'df_extract_raw.csv'))
        df_extract_rawtrace_detrended = pd.read_csv(
            os.path.join(self.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
        df_events_extract_rawtrace = pd.read_csv(
            os.path.join(self.path, 'processed files', 'df_events_extract_rawtrace.csv'))
        coord_ext = np.load(os.path.join(self.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
        reg_th = np.load(os.path.join(self.path, 'processed files', 'reg_th.npy'))
        reg_bad_frames = np.load(os.path.join(self.path, 'processed files', 'frames_to_exclude.npy'))
        trials = np.load(os.path.join(self.path, 'processed files', 'trials.npy'))
        clusters_rois = np.load(os.path.join(self.path, 'processed files', 'clusters_rois.npy'), allow_pickle=True)
        colors_cluster = np.load(os.path.join(self.path, 'processed files', 'clusters_rois_colors.npy'), allow_pickle=True)
        idx_roi_cluster_ordered = np.load(os.path.join(self.path, 'processed files', 'clusters_rois_idx_order.npy'), allow_pickle=True)
        ref_image = np.load(os.path.join(self.path, 'processed files', 'ref_image.npy'), allow_pickle=True)
        frames_dFF = np.load(os.path.join(self.path, 'processed files', 'black_frames.npy'), allow_pickle=True)
        colors_session = np.load(os.path.join(self.path, 'processed files', 'colors_session.npy'), allow_pickle=True)
        colors_session = colors_session[()]
        return df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials, clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF

    def load_processed_files_clusters(self):
        """Loads processed files that were saved under path/processed files for clustered data"""
        df_trace_clusters_ave = pd.read_csv(os.path.join(self.path, 'processed files', 'df_trace_clusters_ave.csv'))
        df_trace_clusters_std = pd.read_csv(os.path.join(self.path, 'processed files', 'df_trace_clusters_std.csv'))
        df_events_trace_clusters = pd.read_csv(
            os.path.join(self.path, 'processed files', 'df_events_trace_clusters.csv'))
        return df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters

    def save_processed_files_ext_fiji(self, df_fiji, df_trace_bgsub, df_extract, df_events_all, df_events_unsync,
                                      trials, coord_fiji, coord_ext, th, idx_to_nan):
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
        df_fiji.to_csv(os.path.join(self.path + 'processed files' + 'df_fiji.csv'), sep=',', index=False)
        df_trace_bgsub.to_csv(os.path.join(self.path + 'processed files' + 'df_fiji_bgsub.csv'), sep=',', index=False)
        df_extract.to_csv(os.path.join(self.path + 'processed files' + 'df_extract.csv'), sep=',', index=False)
        df_events_all.to_csv(os.path.join(self.path + 'processed files' + 'df_events_all.csv'), sep=',', index=False)
        df_events_unsync.to_csv(os.path.join(self.path + 'processed files' + 'df_events_unsync.csv'), sep=',',
                                index=False)
        np.save(os.path.join(self.path + 'processed files' + 'coord_fiji.npy'), coord_fiji, allow_pickle=True)
        np.save(os.path.join(self.path + 'processed files' + 'coord_ext.npy'), coord_ext, allow_pickle=True)
        np.save(os.path.join(self.path + 'processed files' + 'trials.npy'), trials)
        np.save(os.path.join(self.path + 'processed files' + 'reg_th.npy'), th)
        np.save(os.path.join(self.path + 'processed files' + 'frames_to_exclude.npy'), idx_to_nan)
        return

    def load_processed_files_ext_fiji(self):
        """Loads processed files that were saved under path/processed files"""
        df_fiji = pd.read_csv(os.path.join(self.path, 'processed files', 'df_fiji.csv'))
        df_fiji_bgsub = pd.read_csv(os.path.join(self.path, 'processed files', 'df_fiji_bgsub.csv'))
        df_extract = pd.read_csv(os.path.join(self.path, 'processed files', 'df_extract.csv'))
        coord_fiji = np.load(os.path.join(self.path, 'processed files', 'coord_fiji.npy'), allow_pickle=True)
        coord_ext = np.load(os.path.join(self.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
        trials = np.load(os.path.join(self.path, 'processed files', 'trials.npy'))
        reg_th = np.load(os.path.join(self.path, 'processed files', 'reg_th.npy'))
        reg_bad_frames = np.load(os.path.join(self.path, 'processed files', 'frames_to_exclude.npy'))
        return df_fiji, df_fiji_bgsub, df_extract, trials, coord_fiji, coord_ext, reg_th, reg_bad_frames

    def compute_detrended_traces(self, df_dFF, csv_name):
        """"Function to get the detrended traces using Jorge's derivative method.
        Inputs:
        df_dFF: dataframe with calcium trace values after deltaF/F computation and z-scoring
        csv_name: (str) filename of df_events create; if empty doesn't save file"""
        roi_trace = np.array(df_dFF.iloc[:, 2:])
        roi_list = list(df_dFF.columns[2:])
        trial_ext = list(df_dFF['trial'])
        trials = np.unique(trial_ext)
        frame_time_ext = list(df_dFF['time'])
        data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_dFF1 = pd.DataFrame(data_dFF1)
        df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
        df_dFF_detrended = pd.concat([df_dFF1, df_dFF2], axis=1)
        roi_list = df_dFF.columns[2:]
        count_r = 0
        for r in roi_list:
            print('Processing detrended trace of ' + r)
            count_t = 0
            for t in trials:
                data = np.array(df_dFF.loc[df_dFF['trial'] == t, r])
                # Estimate Baseline
                Baseline, Est_Std = ST.Estim_Baseline_PosEvents(data, self.sr, dtau=0.2, bmax_tslope=3, filtcut=1,
                                                                graph=False)
                # Calculate dF/F0:
                F0 = Baseline - Est_Std * 2
                dff = (data - F0) / F0
                dff = np.where(dff < 0, np.zeros_like(dff), dff)  # Remove negative dff values
                df_dFF_detrended.loc[df_dFF_detrended['trial'] == t, r] = dff
                count_t += 1
            count_r += 1
        if len(csv_name) > 0:
            if not os.path.exists(self.path + 'processed files'):
                os.mkdir(self.path + 'processed files')
            df_dFF_detrended.to_csv(self.path + '\\processed files\\' + csv_name + '.csv', sep=',', index=False)
        return df_dFF_detrended

    def get_events(self, df_dFF, detrend_bool, csv_name):
        """"Function to get the calcium event using Jorge's derivative method.
        Inputs:
        df_dFF: dataframe with calcium trace values after deltaF/F computation and z-scoring
        detrend_bool: 1 if trace is raw signal
        csv_name: (str) filename of df_events create; if empty doesn't save file"""
        roi_trace = np.array(df_dFF.iloc[:, 2:])
        roi_list = list(df_dFF.columns[2:])
        trial_ext = list(df_dFF['trial'])
        trials = np.unique(trial_ext)
        frame_time_ext = list(df_dFF['time'])
        data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
        df_dFF1 = pd.DataFrame(data_dFF1)
        df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
        df_events = pd.concat([df_dFF1, df_dFF2], axis=1)
        roi_list = df_dFF.columns[2:]
        count_r = 0
        for r in roi_list:
            print('Processing events of ' + r)
            count_t = 0
            for t in trials:
                data = np.array(df_dFF.loc[df_dFF['trial'] == t, r])
                events_mat = np.zeros(len(data))
                if len(np.where(np.isnan(data))[0]) != len(data):
                    [Ev_Onset, IncremSet, TrueStd] = self.compute_events_onset(data, self.sr, detrend_bool)
                    if len(Ev_Onset) > 0:
                        events = self.event_detection_calcium_trace(data, Ev_Onset, IncremSet, 3)
                        if detrend_bool == 0:
                            events_new = []
                            for e in events:
                                if data[e] >= np.nanpercentile(data, 75):
                                    events_new.append(e)
                            events = events_new
                        events_mat[events] = 1
                    else:
                        print('No events for ' + r + ' trial ' + str(t))
                    df_events.loc[df_events['trial'] == t, r] = events_mat
                else:
                    print('No events for ' + r + ' trial ' + str(t))
                    df_events.loc[df_events['trial'] == t, r] = events_mat
                count_t += 1
            count_r += 1
        if len(csv_name) > 0:
            if not os.path.exists(os.path.join(self.path, 'processed files')):
                os.mkdir(os.path.join(self.path, 'processed files'))
            df_events.to_csv(os.path.join(self.path, 'processed files', csv_name + '.csv'), sep=',', index=False)
        return df_events

    def compute_isi(self, df_events, traces_type, csv_name):
        """Function to compute the inter-spike interval of the dataframe with spikes. Outputs a similiar dataframe
        Inputs: 
            df_events: events (dataframe)
            traces_type: (str) raw or deconv
            csv_name: (str) filename of the dataframe if you want to save it"""
        isi_all = []
        roi_all = []
        trial_all = []
        time_all = []
        for r in df_events.columns[2:]:
            for t in df_events['trial'].unique():
                isi = df_events.loc[(df_events[r] > 0) & (df_events['trial'] == t), 'time'].diff()
                isi_all.extend(np.array(isi))
                trial_all.extend(np.repeat(t, len(isi)))
                roi_all.extend(np.repeat(r, len(isi)))
                time_all.extend(df_events.loc[(df_events[r] > 0) & (df_events['trial'] == t), 'time'])
        dict_isi = {'isi': isi_all, 'roi': roi_all, 'trial': trial_all, 'time': time_all}
        isi_df = pd.DataFrame(dict_isi)  # create dataframe with isi, roi id and trial id
        if len(csv_name) > 0:
            isi_df.to_csv(self.path + '\\processed files\\' + csv_name + '_' + traces_type + '.csv', sep=',',
                          index=False)
        return isi_df

    @staticmethod
    def compute_isi_cv(isi_events, trials):
        """Function to compute coefficient of variation and coefficient of variation for adjacent spikes (Isope, JNeurosci; deSchutter 2007 PLosOne)
        Inputs:
            isi_events: (dataframe) with isi values
            trials: list of trials"""
        isi_cv2_df = pd.DataFrame()
        isi_cv2_df['roi'] = isi_events['roi']
        isi_cv2_df['trial'] = isi_events['trial']
        isi_cv2_df['time'] = isi_events['time']
        isi_cv_all = []
        roi_all = []
        trial_all = []
        for t in trials:
            for r in np.unique(isi_events.roi):
                isi_data = isi_events.loc[(isi_events['trial'] == t) & (isi_events['roi'] == r), 'isi']
                data = np.array(isi_data)
                isi_cv_value = np.nanstd(data) / np.nanmean(data)
                isi_cv_all.append(np.float64(isi_cv_value))
                isi_cv2_df.loc[(isi_events['trial'] == t) & (isi_events['roi'] == r), 'cv2'] = 2 * (
                    isi_data.diff().abs()) / isi_data.rolling(2).sum()
                trial_all.append(t)
                roi_all.append(r)
        dict_isi_cv = {'isi_cv': isi_cv_all, 'roi': roi_all, 'trial': trial_all}
        isi_cv_df = pd.DataFrame(dict_isi_cv)  # create dataframe with isi, roi id and trial id
        return isi_cv_df, isi_cv2_df

    @staticmethod
    def compute_isi_ratio(isi_df, isi_interval, trials):
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
                isi = np.array(isi_df.loc[(isi_df.roi == r) & (isi_df.trial == t), 'isi'])
                hist, bin_edges = np.histogram(isi, bins=30, range=(0, 3))
                ratio = np.nansum(hist[np.where(bin_edges == isi_interval[1][0])[0][0]:
                                       np.where(bin_edges == isi_interval[1][1])[0][0]]) / np.nansum(hist[np.where(
                    bin_edges == isi_interval[0][0])[0][0]:np.where(bin_edges == isi_interval[0][1])[0][0]])
                isi_ratio.append(ratio)
                trial_all.append(t)
                roi_all.append(r)
        dict_isi_ratio = {'isi_ratio': isi_ratio, 'roi': roi_all, 'trial': trial_all}
        isi_ratio_df = pd.DataFrame(dict_isi_ratio)  # create dataframe with isi, roi id and trial id
        return isi_ratio_df

    def plot_isi_single_trial(self, trial, roi, isi_df, traces_type, plot_data, print_plots):
        """Function to plot the ISI distribution of a single trial for a certain ROI
        Inputs:
            trial: (int) trial id
            roi: (int) ROI id
            isi_df: (dataframe) with ISI values
            traces_type: (str) raw or deconv
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in isi_df['roi'][0] if x.isdigit())
        int_find_idx = isi_df['roi'][0].find(int_find)
        if isi_df['roi'][0][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        binwidth = 0.05
        barWidth = 0.05
        isi_data = np.array(isi_df.loc[(isi_df['trial'] == trial) & (isi_df['roi'] == idx_nr), 'isi'])
        max_isi = np.ceil(np.nanmax(isi_data))
        binedges = np.arange(0, max_isi + 0.5, binwidth)
        hist_all = np.histogram(isi_data, bins=binedges)
        hist_norm = hist_all[0] / np.sum(hist_all[0])
        r1 = binedges[:-1]
        rate = np.round(1 / np.nanmedian(isi_data), 2)
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
            plt.bar(r1, hist_norm, color='darkgrey', width=barWidth, edgecolor='white')
            plt.xlabel('Inter-event interval (s)', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.ylabel('Event count', fontsize=self.fsize)  # Add xticks on the middle of the group bars
            plt.title('Event rate of ' + str(rate) + ' trial ' + str(trial) + ' ROI ' + str(roi), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'isi_hist_roi_' + str(roi) + '_trial_' + str(trial)), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'isi_hist_cluster_' + str(roi) + '_trial_' + str(trial)),
                                dpi=self.my_dpi)
        return r1, hist_norm

    def plot_isi_boxplots(self, roi, isi_events, traces_type, session_type, animal, session, trials, plot_data, print_plots):
        """Function to plot the all ISI distributions across the session for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_events: (dataframe) with ISI values
            traces_type: (str) raw or deconv
            session_type: (str) tied or split
            animal: (str) animal name
            session: (str) session number
            colors_session: (list) colors for each trial in the session
            trials: (list)
            plot_data: boolean
            print_plots: boolean"""
        [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split,
         trials_washout] = self.get_session_data(trials, session_type, animal, session)
        colors_session = self.colors_session(session_type, trials, 1)
        colors_session_boxplot = self.colors_session(session_type, trials, 0)
        int_find = ''.join(x for x in isi_events['roi'][0] if x.isdigit())
        int_find_idx = isi_events['roi'][0].find(int_find)
        if isi_events['roi'][0][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        if plot_data:
            fig, ax = plt.subplots(figsize=(20, 5), tight_layout=True)
            sns.boxplot(x='trial', y='isi', data=isi_events.loc[isi_events['roi'] == idx_nr],
                        medianprops=dict(color='white'), palette=colors_session_boxplot)
            ax.set_xlabel('Trial number', fontsize=self.fsize - 4)
            ax.set_ylabel('IEI (s)', fontsize=self.fsize - 4)
            ax.set_title('IEI distribution for ' + idx_nr, fontsize=self.fsize - 4)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='x', labelsize=self.fsize - 6)
            ax.tick_params(axis='y', labelsize=self.fsize - 6)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'isi_boxplot_roi_' + str(roi)),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'isi_boxplot_cluster_' + str(roi)),
                                dpi=self.my_dpi)
            fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
            isi_median = np.zeros(len(trials))
            for t in trials:
                isi_median[t - 1] = isi_events.loc[
                    (isi_events['roi'] == idx_nr) & (
                                isi_events['trial'] == t), 'isi'].median()
                ax.scatter(t, isi_median[t - 1], color=colors_session[t - 1])
            rectangle = plt.Rectangle((trials_baseline[-1] + 0.5, np.min(isi_median)-0.05), len(trials_split), (np.max(isi_median)-np.min(isi_median))+0.05, fc='grey', alpha=0.3, zorder=0)
            plt.gca().add_patch(rectangle)
            ax.plot(trials, isi_median, color='black')
            ax.set_xlabel('Trial', fontsize=20)
            ax.set_ylabel('ISI median', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            if df_type == 'cluster':
                ax.set_title('Cluster ' + str(roi) + ' ISI median', color='black', fontsize=16)
            if df_type == 'ROI':
                ax.set_title('ROI ' + str(roi) + ' ISI median', color='black', fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'isi_median_roi_' + str(roi)),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'isi_median_cluster_' + str(roi)),
                                dpi=self.my_dpi)
        return

    def plot_cv_session(self, roi, isi_cv, traces_type, colors_session, trials, plot_name, plot_data, print_plots):
        """Function to plot the ISI distribution across the session for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_cv: (array) with CV values
            traces_type: (str) raw or deconv
            colors_session: (list) color for the trials in the session
            trials: list of trials
            plot_name: such as 'cv' or 'cv2'
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in isi_cv['roi'][0] if x.isdigit())
        int_find_idx = isi_cv['roi'][0].find(int_find)
        if isi_cv['roi'][0][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        if plot_data:
            if plot_name == 'cv':
                fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
                for t in trials:
                    plt.bar(t - 0.5, np.array(isi_cv.loc[isi_cv['roi'] == idx_nr, 'isi_cv'])[t - 1], width=1,
                            color=colors_session[t - 1], edgecolor='white')
                ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
                ax.set_xticklabels(list(map(str, trials)))
                plt.xlim([0, len(trials) + 1])
                plt.ylabel('Coefficient of variation', fontsize=self.fsize)
                plt.xlabel('Trials', fontsize=self.fsize)
                plt.title('CV for ' + idx_nr, fontsize=self.fsize)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.xticks(fontsize=self.fsize - 4)
                plt.yticks(fontsize=self.fsize - 4)
                if print_plots:
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                    if df_type == 'ROI':
                        if not os.path.exists(
                                os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi))):
                            os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi)))
                        plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi),
                                                 plot_name + '_roi_' + str(roi)), dpi=self.my_dpi)
                        if df_type == 'cluster':
                            if not os.path.exists(
                                    os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                                os.mkdir(
                                    os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                            plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                                     plot_name + '_cluster_' + str(roi)), dpi=self.my_dpi)
            if plot_name == 'cv2':
                fig, ax = plt.subplots(figsize=(20, 5), tight_layout=True)
                sns.boxplot(x='trial', y='cv2', data=isi_cv.loc[isi_cv['roi'] == idx_nr],
                            medianprops=dict(color='white'), palette=colors_session)
                ax.set_xlabel('Trial number', fontsize=self.fsize - 4)
                ax.set_ylabel('CV2', fontsize=self.fsize - 4)
                ax.set_title('CV2 distribution for ' + idx_nr, fontsize=self.fsize - 4)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='x', labelsize=self.fsize - 6)
                ax.tick_params(axis='y', labelsize=self.fsize - 6)
                if print_plots:
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                    if df_type == 'ROI':
                        if not os.path.exists(
                                os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                            os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                        plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi),
                                                 plot_name + '_roi_' + str(roi)), dpi=self.my_dpi)
                    if df_type == 'cluster':
                        if not os.path.exists(
                                os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                            os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                        plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                                 plot_name + '_cluster_' + str(roi)), dpi=self.my_dpi)
        return

    def plot_isi_ratio_session(self, roi, isi_ratio, isi_ratio_shuffle, traces_type, colors_session, range_isiratio,
                               trials, plot_data, print_plots):
        """Function to plot the ISI ratio between a certain range across the session
        for a certain ROI
        Inputs:
            roi: (int) ROI id
            isi_ratio: (dataframe) with ISI ratio values from shuffled events
            isi_ratio: (dataframe) with ISI ratio values
            traces_type: (str) raw or deconv
            colors_session: (list) color for the trials in the session
            range_isiratio: list with the range values
            trials: list of trials
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in isi_ratio['roi'][0] if x.isdigit())
        int_find_idx = isi_ratio['roi'][0].find(int_find)
        if isi_ratio['roi'][0][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        if plot_data:
            fig, ax = plt.subplots(2, 1, figsize=(15, 10), tight_layout=True)
            ax = ax.ravel()
            for t in trials:
                ax[0].bar(t - 0.5, np.array(isi_ratio.loc[(isi_ratio['trial'] == t) & (
                        isi_ratio['roi'] == idx_nr), 'isi_ratio'])[0], width=1,
                          color=colors_session[t - 1], edgecolor='white')
            ax[0].set_xticks(np.arange(0.5, len(trials) + 0.5))
            ax[0].set_xticklabels(list(map(str, trials)))
            ax[0].set_xlim([0, len(trials) + 1])
            ax[0].set_ylabel('ISI ratio', fontsize=self.fsize)
            ax[0].set_xlabel('Trials', fontsize=self.fsize)
            ax[0].set_title(
                'ISI ratio between ' + str(range_isiratio[0]) + ' ' + str(range_isiratio[1]) + ' for ' + idx_nr,
                fontsize=self.fsize)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].tick_params(axis='both', which='major', labelsize=self.fsize - 4)
            for t in trials:
                ax[1].bar(t - 0.5, np.array(isi_ratio_shuffle.loc[(isi_ratio_shuffle['trial'] == t) & (
                        isi_ratio_shuffle['roi'] == idx_nr), 'isi_ratio'])[0], width=1,
                          color=colors_session[t - 1], edgecolor='white')
            ax[1].set_xticks(np.arange(0.5, len(trials) + 0.5))
            ax[1].set_xticklabels(list(map(str, trials)))
            ax[1].set_xlim([0, len(trials) + 1])
            ax[1].set_ylabel('ISI ratio', fontsize=self.fsize)
            ax[1].set_xlabel('Trials', fontsize=self.fsize)
            ax[1].set_title('ISI ratio shuffle events between ' + str(range_isiratio[0]) + ' ' + str(
                range_isiratio[1]) + ' for ' + idx_nr,
                            fontsize=self.fsize)
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].tick_params(axis='both', which='major', labelsize=self.fsize - 4)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'isi_ratio_roi_' + str(roi)), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'isi_ratio_cluster_' + str(roi)), dpi=self.my_dpi)
        return

    def compute_event_waveform(self, df_fiji, traces_type, df_events, roi_plot, animal, session_type, trials_ses,
                               trials, plot_data, print_plots):
        """Function to compute the complex-spike waveform from the deconvolution and dFF
        Inputs:
            df_fiji: dataframe with normalized trace
            traces_type: (str) raw or deconv
            df_events: dataframe with events
            roi_plot: (int) with ROI to plot
            animal: (str) with animal name
            session_type: (str) split or tied
            trials: list of trials
            trials_ses: list with the transition trials
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in df_fiji.columns[2] if x.isdigit())
        int_find_idx = df_fiji.columns[2].find(int_find)
        if df_fiji.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi_plot)
        trace_data = np.array(df_fiji[idx_nr])
        if session_type == 'split':
            idx_plot = [[trials_ses[0, 0], trials_ses[0, 1]], [trials_ses[1, 0], trials_ses[1, 1]], [trials_ses[2, 0], trials_ses[2, 1]]]
            colors_plot = ['darkgrey', 'crimson', 'blue']
        if session_type == 'tied' and animal == 'MC8855':
            idx_plot = [[trials_ses[0, 0], trials_ses[0, 1]], [trials_ses[1, 0], trials_ses[1, 1]]]
            colors_plot = ['darkgrey', 'orange']
        if session_type == 'tied' and animal != 'MC8855':
            idx_plot = [[trials_ses[0, 0], trials_ses[0, 1]], [trials_ses[1, 0], trials_ses[1, 1]], [trials_ses[2, 0], trials_ses[2, 1]]]
            colors_plot = ['darkgrey', 'lightblue', 'orange']
        cs_waveforms_mean_all = np.zeros((len(colors_plot), 40))
        cs_waveforms_sem_all = np.zeros((len(colors_plot), 40))
        for i in range(len(colors_plot)):
            events = np.array(df_events.loc[(df_events[idx_nr] == 1) & (
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
        if plot_data:
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
            plt.suptitle(df_type + str(roi_plot), fontsize=self.fsize - 2)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot),
                                             'event_waveform_roi_' + str(roi_plot)), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot),
                                             'event_waveform_cluster_' + str(roi_plot)),
                                dpi=self.my_dpi)
        return cs_waveforms_mean_all, cs_waveforms_sem_all

    def get_event_count_wholetrial(self, df_events, traces_type, colors_session, trials, roi_plot, plot_data,
                                   print_plots):
        """Function to compute the normalized spike count (divided by the number of frames) per trial
        Inputs: 
            df_events: dataframe with events
            traces_type: (str) raw or deconv
            colors_session: (list) colors of the session trials
            trials: array of recorded trials
            roi_plot: (int) of ROI to plot
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in df_events.columns[2] if x.isdigit())
        int_find_idx = df_events.columns[2].find(int_find)
        if df_events.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi_plot)
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
            for t in trials:
                idx_trial = np.where(trials==t)[0]
                event_count = df_events.loc[
                    (df_events[idx_nr] == 1) & (df_events['trial'] == t)].count()
                plt.bar(t - 0.5, event_count, width=1, color=colors_session[t], edgecolor='white')
            ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
            ax.set_xticklabels(list(map(str, trials)))
            plt.xlim([0, len(trials) + 1])
            plt.ylabel('Event count', fontsize=self.fsize)
            plt.xlabel('Trials', fontsize=self.fsize)
            plt.title('Event count (whole trial) for ' + df_type + ' ' + str(roi_plot), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot),
                                             'event_count_roi_' + str(roi_plot)), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot),
                                             'event_count_cluster_' + str(roi_plot)), dpi=self.my_dpi)
        return

    def get_event_count_locomotion(self, df_events, traces_type, colors_session, trials, bcam_time, st_strides_trials,
                                   roi_plot, plot_data, print_plots):
        """Function to compute the normalized spike count (divided by the number of frames) per trial
        Inputs:
            df_events: dataframe with events
            traces_type: (str) raw or deconv
            colors_session: (list) colors of the session trials
            trials: array of recorded trials
            bcam_time: behavioral camera timestamps
            st_strides_trials: list with stride onsets (trials - paws - stridesx2x5)
            roi_plot: (int) of ROI to plot
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in df_events.columns[2] if x.isdigit())
        int_find_idx = df_events.columns[2].find(int_find)
        if df_events.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi_plot)
        event_count_clean = np.zeros((len(trials)))
        for count_t, t in enumerate(trials):
            bcam_trial = bcam_time[count_t]
            events = np.array(
                df_events.loc[(df_events['trial'] == t) & (df_events[idx_nr] == 1), 'time'])
            st_on = st_strides_trials[count_t][0][:, 0, -1]  # FR paw
            st_off = st_strides_trials[count_t][0][:, 1, -1]  # FR paw
            time_forwardloco = []
            event_clean_list = []
            for s in range(len(st_on)):
                time_forwardloco.append(bcam_trial[int(st_off[s])] - bcam_trial[int(st_on[s])])
                if len(np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]) > 0:
                    event_clean_list.append(len(
                        np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]))
            time_forwardloco_trial = np.sum(time_forwardloco)
            event_count_clean[count_t] = np.sum(event_clean_list) / time_forwardloco_trial
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True)
            for t in trials:
                idx_trial = np.where(trials==t)[0]
                plt.bar(t - 0.5, event_count_clean[idx_trial], width=1, color=colors_session[t], edgecolor='white')
            ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
            ax.set_xticklabels(list(map(str, trials)))
            plt.xlim([0, len(trials) + 1])
            plt.ylabel('Event count', fontsize=self.fsize)
            plt.xlabel('Trials', fontsize=self.fsize)
            plt.title('Event count (forward locomotion) for ' + df_type + ' ' + str(roi_plot), fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=self.fsize - 4)
            plt.yticks(fontsize=self.fsize - 4)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'events', traces_type, df_type + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, df_type + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, df_type + str(roi_plot),
                                             'event_count_loco_roi_' + str(roi_plot)), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot),
                                             'event_count_loco_cluster_' + str(roi_plot)), dpi=self.my_dpi)
        return event_count_clean

    @staticmethod
    def event_swst_stride(df_events, st_strides_trials, sw_strides_trials, final_tracks_phase, bcam_time, dim, align, trials, p1, roi, time_window):
        """Get the event times aligned to swing or stance of a certain paw for one ROI/cluster.
        If align is phase you can get the calcium event behavioral phase.
        Inputs:
        df_events: (dataframe) with the events for each ROI/cluster and trials
        st_strides_trials: list with strides for all trials st to st
        sw_strides_trials: list with strides for all trials sw to sw
        final_tracks_phase: list with the paw positions transformed in phase
        bcam_time: list of timestamps of behavioral camera
        dim: (str) time or phase
        align: (str) st or sw
        trials: trial list
        p1: reference paw to get events (FR, HR, FL, HL)
        roi: (str) ROI1 or cluster1
        time_window: (float) window in seconds"""
        if align == 'st':
            data_strides = st_strides_trials
        if align == 'sw':
            data_strides = sw_strides_trials
        if p1 == 'FR':
            p1_idx = 0
        if p1 == 'HR':
            p1_idx = 1
        if p1 == 'FL':
            p1_idx = 2
        if p1 == 'HL':
            p1_idx = 3
        events_stride_trial = []
        cumulative_idx = []
        trial_id = []
        for count_t, trial in enumerate(trials):
            t = np.where(trials == trial)[0][0]
            df_trial = df_events.loc[df_events['trial'] == trial, [roi, 'time']].reset_index()
            event_times_trial = np.array(df_trial.iloc[np.where(df_trial.iloc[:, 1])[0], 2]) * 1000
            events_stride_list = []
            for s in range(np.shape(data_strides[t][p1_idx])[0]):
                event_on_time = data_strides[t][p1_idx][s, 0, 0]
                if dim == 'time':
                    event_idx_stride = np.where((event_times_trial > event_on_time - (time_window * 1000)) & (
                            event_times_trial < event_on_time + (time_window * 1000)))[0]
                if dim == 'phase':
                    st_on_time = data_strides[t][p1_idx][s, 0, 0]
                    st_off_time = data_strides[t][p1_idx][s, 1, 0]
                    event_idx_stride = np.where((event_times_trial > st_on_time) & (event_times_trial < st_off_time))[0]
                if len(event_idx_stride) > 0:
                    if dim == 'time':
                        event_stride = event_times_trial[event_idx_stride] - event_on_time
                    if dim == 'phase':
                        event_stride = [] #is a list because it can be more than 1
                        for i in event_idx_stride:
                            event_stride_bcam_time = np.argmin(np.abs(
                                (event_times_trial[i] / 1000) - bcam_time[
                                    t]))  # find closest behavioral timestamp
                            event_stride.append(final_tracks_phase[t][0, p1_idx, event_stride_bcam_time])
                    for count_i, i in enumerate(event_stride):
                        events_stride_list.append(i)
                        if count_t == 0 and s == 0:
                            cumulative_idx.append(1)
                        if count_i == 0:  # if its the first event its the next stride
                            cumulative_idx.append(cumulative_idx[-1] + 1)
                        if count_i > 0:  # if its the 2nd, 3rd event its the same stride as first event
                            cumulative_idx.append(cumulative_idx[-1])
                else:
                    event_stride = np.nan
                    events_stride_list.append(event_stride)
                    if count_t == 0 and s == 0:
                        cumulative_idx.append(1)
                    else:
                        cumulative_idx.append(cumulative_idx[-1] + 1)
            events_stride_trial.extend(events_stride_list)
            trial_id.extend(np.repeat(trial, len(events_stride_list)))
        return np.array(cumulative_idx), np.array(trial_id), np.array(events_stride_trial)

    def firing_rate_swst(self, events_stride_trial, trial_id, final_tracks_phase, trials, bins, align_dimension, paw):
        """Compute firing rate of CS around the locomotor events (can do this in phase or time).
        Inputs:
            event_stride_trial: list with the time/%phase of CS aligned to locomotor event
            trial_id: trial identification of each row of events_stride_trial
            final_tracks_phase: list of paw excursions in phase
            trials: list of trials in the session
            bins: vector of time or phase bins
            align_dimension: (str) phase or time
            paw: (str) FR, HR, FL, HL"""
        if paw == 'FR':
            p1_idx = 0
        if paw == 'HR':
            p1_idx = 1
        if paw == 'FL':
            p1_idx = 2
        if paw == 'HL':
            p1_idx = 3
        spikes_count_tr = []
        for count_t, trial in enumerate(trials):
            spikes_count = np.zeros(len(bins) - 1)
            dataset = events_stride_trial[trial_id == trial]
            for value in dataset:
                if ~np.isnan(value):
                    bin_value = np.digitize(value, bins)
                    spikes_count[bin_value - 1] += 1
            spikes_count_tr.append(spikes_count)
        firing_rate = np.zeros((len(trials), len(bins) - 1))
        spike_prob = np.zeros((len(trials), len(bins) - 1))
        for tr in range(len(trials)):
            phase_paw = final_tracks_phase[tr][0, p1_idx, :]
            frames_bin, _ = np.histogram(phase_paw[~np.isnan(phase_paw)],
                                         bins=len(bins) - 1)  # Compute time spent in each bin
            time_bin = frames_bin * (1 / self.sr_loco)
            if align_dimension == 'phase':
                firing_rate[tr] = spikes_count_tr[tr] / time_bin  # Compute firing rate
                spike_prob[tr] = (spikes_count_tr[tr] / np.sum(spikes_count_tr[tr])) / time_bin
            if align_dimension == 'time':
                time_bin = np.round(bins[1] - bins[0], 4)
                firing_rate[tr] = spikes_count_tr[tr] / time_bin  # Compute firing rate
                spike_prob[tr] = (spikes_count_tr[tr] / np.sum(spikes_count_tr[tr])) / time_bin
        return firing_rate, spike_prob

    def event_corridor_distribution(self, df_events, final_tracks_trials, bcam_time, roi, paw, trials_analysis,
                                    pixel_step, traces_type, plot_data, print_plots):
        """Plot distribution of events in relation to the position of a certain tracking feature (one of the four paws).
        Also plots if desired the feature distribution in the corridor
        Inputs:
            df_events: dataframe with events
            final_tracks_trials: list with the final_tracks for each trial
            bcam_time: list with the behavioral camera timestamps for each trial
            roi: (int) roi to plot
            paw: (str) FR, FL, HR or HL
            trials_analysis: (list) trials to pool together
            pixel_step: how many pixels to bin together for distribution plot
            traces_type: (str) raw or deconv
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in df_events.columns[2] if x.isdigit())
        int_find_idx = df_events.columns[2].find(int_find)
        if df_events.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        x_pixels = np.arange(0, 300, pixel_step)
        y_pixels = np.arange(0, 640, pixel_step)
        frame_image_bins_list = []
        frame_image_bins_feature_list = []
        for t in trials_analysis:
            if paw == 'FR':
                idx_p = 0
            if paw == 'HR':
                idx_p = 1
            if paw == 'FL':
                idx_p = 2
            if paw == 'HL':
                idx_p = 3
            feature_X = self.inpaint_nans(final_tracks_trials[t - 1][0, idx_p, :])
            feature_Y = self.inpaint_nans(final_tracks_trials[t - 1][1, idx_p, :])
            time = bcam_time[t - 1]
            events_trial = df_events.loc[df_events['trial'] == t, idx_nr]
            events_trial_timestamps = df_events.loc[df_events['trial'] == t, 'time']
            events_time = np.array(events_trial_timestamps[events_trial[events_trial == 1].index])
            feature_events_idx = np.zeros(len(events_time))
            for i, e in enumerate(events_time):
                feature_events_idx[i] = np.abs(e - time).argmin()
            feature_values_events_X = np.int64(feature_X[np.int64(feature_events_idx)])
            feature_values_events_Y = np.int64(feature_Y[np.int64(feature_events_idx)])
            # for bins
            frame_image_bins = np.zeros((len(x_pixels), len(y_pixels)))
            for e in range(len(feature_events_idx)):
                for x_count, x in enumerate(x_pixels):
                    for y_count, y in enumerate(y_pixels):
                        if feature_values_events_X[e] >= y and feature_values_events_X[e] <= y + pixel_step and \
                                feature_values_events_Y[e] >= x and feature_values_events_Y[e] <= x + pixel_step:
                            frame_image_bins[x_count, y_count] += 1
            frame_image_bins_feature = np.zeros((len(x_pixels), len(y_pixels)))
            for e in range(len(feature_X)):
                for x_count, x in enumerate(x_pixels):
                    for y_count, y in enumerate(y_pixels):
                        if feature_X[e] >= y and feature_X[e] <= y + pixel_step and feature_Y[e] >= x and feature_Y[
                            e] <= x + pixel_step:
                            frame_image_bins_feature[x_count, y_count] += 1
            frame_image_bins_list.append(frame_image_bins)
            frame_image_bins_feature_list.append(frame_image_bins_feature)
        frame_image_bins_array = np.sum(np.dstack(frame_image_bins_list), axis=2)
        frame_image_bins_feature_array = np.sum(np.dstack(frame_image_bins_feature_list), axis=2)
        frame_image_bins_norm = np.divide(frame_image_bins_array, frame_image_bins_feature_array,
                                          out=np.zeros_like(frame_image_bins_array),
                                          where=frame_image_bins_feature_array != 0)
        if plot_data:
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)  # feature over time (sanity check with video)
            data_corridor = np.divide(frame_image_bins_feature_array, np.max(frame_image_bins_feature_array))
            h1 = ax.imshow(data_corridor / np.max(data_corridor))
            ax.set_xticklabels(list(map(str, y_pixels)))
            ax.set_yticklabels(list(map(str, x_pixels)))
            plt.colorbar(h1)
            if print_plots:
                if not os.path.exists(os.path.join(self.path + 'images', 'events')):
                    os.mkdir(os.path.join(self.path + 'images', 'events'))
                plt.savefig(os.path.join(self.path + 'images', 'events',
                                         'corridor_bottom_' + paw + '_distribution'),
                            dpi=self.my_dpi)
            fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)  # events in correspondent feature space
            h2 = ax.imshow(frame_image_bins_norm / np.max(frame_image_bins_norm))
            ax.set_xticklabels(list(map(str, y_pixels)))
            ax.set_yticklabels(list(map(str, x_pixels)))
            plt.colorbar(h2)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'event_corridor_bottom_' + paw + '_distribution'),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'event_corridor_bottom_' + paw + '_distribution'),
                                dpi=self.my_dpi)
        return frame_image_bins_norm

    def param_events_plot(self, param_trials, st_strides_trials, df_events, param_name, roi, p1, p2, step_size,
                          trials_compute, trials, traces_type, plot_condition, stride_duration_trials, plot_data):
        """Compute the proportion of events for bins of a certain gait parameter for each ROI and paw combination.
        Compared with the dataset shifted by 100 frames.
        Inputs:
            param_trials: (list) gait parameter values
            st_strides_trials: (list) stride structure stance to stance
            df_events: (dataframe) events for each ROI and trial
            param_name: (str)
            roi: (int)
            p1 and p2: (str) such as FR and FL
            step_size: (int) step size of gait parameter values
            trials_compute: (list) trials to do this analysis
            trials: (list) all the trials in a session
            traces_type: (str) raw or deconv
            plot_condition: (str) e.g. baseline
            stride_duration_trials: (list) stride duration values for each stride
            plot_data: (boolean)
        bins_all, sl_p1_events_bin_all, sl_p1_events_bin_all_shuffle, t_stat, p_value
        Outputs:
            bins_all: (list) bin values for the gait parameter chosen
            sl_p1_events_bin_all: (list) distribution of event probability
            sl_p1_events_bin_all_shuffle: (list) distribution of shuffled event probability
            t_stat: test statistic value for Kolmogorov-Smirnov test
            p_value: p-value for Kolmogorov-Smirnov test"""
        nr_strides = 10
        if p1 == 'FR':
            p1_idx = 0
        if p1 == 'HR':
            p1_idx = 1
        if p1 == 'FL':
            p1_idx = 2
        if p1 == 'HL':
            p1_idx = 3
        if p2 == 'FR':
            p2_idx = 0
        if p2 == 'HR':
            p2_idx = 1
        if p2 == 'FL':
            p2_idx = 2
        if p2 == 'HL':
            p2_idx = 3
        bin_vector = np.arange(-50, 50 + step_size, step_size)
        sl_p1_events_trials = np.zeros((len(trials), len(bin_vector)))
        sl_p1_events_trials[:] = np.nan
        sl_sym_all = []
        trial_id = []
        sl_event_all = []
        sl_event_all_shuffle = []
        stride_duration_all = []
        isi_all = []
        isi_all_shuffle = []
        df_events_shuffle = pd.DataFrame()
        for c in df_events.columns[2:]:
            df_events_shuffle[c] = np.roll(df_events[c], 200)
        df_events_shuffle.insert(loc=0, column='trial', value=df_events['trial'])
        df_events_shuffle.insert(loc=0, column='time', value=df_events['time'])
        for count_t, t in enumerate(trials_compute):
            trial_index = np.where(trials == t)[0][0]
            df_events_trial = df_events.loc[df_events['trial'] == t].reset_index(drop=True)
            # shuffle events for all trials
            df_events_trial_shuffle = df_events_shuffle.loc[df_events_shuffle['trial'] == t].reset_index(drop=True)
            isi_all.extend(df_events_trial.loc[(df_events_trial['ROI' + str(roi)] > 0) & (
                    df_events_trial['trial'] == t), 'time'].diff())
            isi_all_shuffle.extend(df_events_trial_shuffle.loc[(df_events_trial_shuffle['ROI' + str(roi)] > 0) & (
                    df_events_trial_shuffle['trial'] == t), 'time'].diff())
            sl_p1 = param_trials[trial_index][p1_idx]
            sl_p2 = param_trials[trial_index][p2_idx]
            strides_p1 = st_strides_trials[trial_index][p1_idx]
            strides_p2 = st_strides_trials[trial_index][p2_idx]
            # get events between the strides
            sl_p1_events = np.zeros(np.shape(strides_p1)[0])
            sl_p1_events_shuffle = np.zeros(np.shape(strides_p1)[0])
            sl_sym = np.zeros(np.shape(strides_p1)[0])
            sl_sym[:] = np.nan
            for s in range(np.shape(strides_p1)[0]):
                events_stride = df_events_trial.loc[(df_events_trial['time'] >= strides_p1[s][0, 0] / 1000) & (
                        df_events_trial['time'] <= strides_p1[s][1, 0] / 1000), 'ROI' + str(roi)]
                events_stride_shuffle = df_events_trial_shuffle.loc[
                    (df_events_trial['time'] >= strides_p1[s][0, 0] / 1000) & (
                            df_events_trial['time'] <= strides_p1[s][1, 0] / 1000), 'ROI' + str(roi)]
                stride_contra = \
                    np.where((strides_p2[:, 0, 0] > strides_p1[s, 0, 0]) & (strides_p2[:, 0, 0] < strides_p1[s, 1, 0]))[
                        0]
                if len(stride_contra) == 1:  # one event in the stride
                    sl_sym[s] = sl_p1[s] - sl_p2[stride_contra]
                if len(stride_contra) > 1:  # more than two events in a stride
                    sl_sym[s] = sl_p1[s] - np.nanmean(sl_p2[stride_contra])
                if len(np.where(events_stride)[0]) > 0:
                    sl_p1_events[s] = len(np.where(events_stride)[0])
                    sl_p1_events_shuffle[s] = len(np.where(events_stride_shuffle)[0])
            sl_sym_all.extend(sl_sym)
            trial_id.extend(np.repeat(t, len(sl_sym)))
            sl_event_all.extend(sl_p1_events)
            sl_event_all_shuffle.extend(sl_p1_events_shuffle)
            stride_duration_all.extend(stride_duration_trials[trial_index][p1_idx])
        bins_all = np.histogram(sl_sym_all, bin_vector)[1]
        sl_event_all_array = np.array(sl_event_all)
        sl_event_all_shuffle = np.array(sl_event_all_shuffle)
        stride_duration_all_array = np.array(stride_duration_all)
        sl_p1_events_bin_all = np.zeros(len(bins_all))
        sl_p1_events_bin_all[:] = np.nan
        sl_p1_events_bin_all_shuffle = np.zeros(len(bins_all))
        sl_p1_events_bin_all_shuffle[:] = np.nan
        sl_p1_events_bin_count_shuffle = np.zeros(len(bins_all))
        sl_p1_events_bin_count_shuffle[:] = np.nan
        stride_duration_bin_all = np.zeros(len(bins_all))
        stride_duration_bin_all[:] = np.nan
        stride_duration_bin_avg = np.zeros(len(bins_all))
        stride_duration_bin_avg[:] = np.nan
        stride_duration_bin_sem = np.zeros(len(bins_all))
        stride_duration_bin_sem[:] = np.nan
        stride_nr_bin = np.zeros(len(bins_all))
        stride_nr_bin[:] = np.nan
        sl_p1_events_bin_count = np.zeros(len(bins_all))
        sl_p1_events_bin_count[:] = np.nan
        for count_b, b in enumerate(bins_all):
            if count_b == len(bins_all) - 1:
                idx_bin = np.where((sl_sym_all >= bins_all[count_b]) & (sl_sym_all < bins_all[-1]))[0]
            else:
                idx_bin = np.where((sl_sym_all >= bins_all[count_b]) & (sl_sym_all < bins_all[count_b + 1]))[0]
            if len(idx_bin) > nr_strides:
                sl_p1_events_bin_count[count_b] = np.sum(sl_event_all_array[idx_bin])
                sl_p1_events_bin_count_shuffle[count_b] = np.sum(sl_event_all_shuffle[idx_bin])
                stride_duration_bin_all[count_b] = np.cumsum(stride_duration_all_array[idx_bin])[-1]
                stride_duration_bin_avg[count_b] = np.mean(stride_duration_all_array[idx_bin])
                stride_duration_bin_sem[count_b] = np.std(stride_duration_all_array[idx_bin]) / np.sqrt(len(idx_bin))
                sl_p1_events_bin_all[count_b] = sl_p1_events_bin_count[count_b] / stride_duration_bin_all[count_b]
                sl_p1_events_bin_all_shuffle[count_b] = sl_p1_events_bin_count_shuffle[count_b] / \
                                                        stride_duration_bin_all[count_b]
                stride_nr_bin[count_b] = len(idx_bin)
        if plot_data:
            if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
            if not os.path.exists(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
            fig, ax = plt.subplots(3, 2, figsize=(30, 20), tight_layout=True)
            ax = ax.ravel()
            ax[0].bar(bins_all[~np.isnan(sl_p1_events_bin_all)], sl_p1_events_bin_all[~np.isnan(sl_p1_events_bin_all)],
                      width=step_size - (step_size * 0.1), color='black')
            ax[0].bar(bins_all[~np.isnan(sl_p1_events_bin_all_shuffle)] + step_size / 8,
                      sl_p1_events_bin_all_shuffle[~np.isnan(sl_p1_events_bin_all_shuffle)],
                      width=step_size - (step_size * 0.1), color='gray')
            ax[0].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=self.fsize - 4)
            ax[0].set_ylabel('Event prob. norm. cum. stride duration',
                             fontsize=self.fsize - 4)
            ax[0].set_title(
                'Event proportion for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
                fontsize=self.fsize - 4)
            ax[0].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[1].bar(bins_all[~np.isnan(stride_duration_bin_all)],
                      stride_duration_bin_all[~np.isnan(stride_duration_bin_all)], width=step_size - (step_size * 0.1),
                      color='black')
            ax[1].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=self.fsize - 4)
            ax[1].set_ylabel('Cumulative stride duration (ms)', fontsize=self.fsize - 4)
            ax[1].set_title(
                'Stride duration for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
                fontsize=self.fsize - 4)
            ax[1].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[2].bar(bins_all[~np.isnan(stride_nr_bin)], stride_nr_bin[~np.isnan(stride_nr_bin)],
                      width=step_size - (step_size * 0.1), color='black')
            ax[2].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=self.fsize - 4)
            ax[2].set_ylabel('Stride count', fontsize=self.fsize - 4)
            ax[2].set_title(
                'Stride count for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
                fontsize=self.fsize - 4)
            ax[2].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['top'].set_visible(False)
            ax[3].bar(bins_all[~np.isnan(sl_p1_events_bin_count)],
                      sl_p1_events_bin_count[~np.isnan(sl_p1_events_bin_count)], width=step_size - (step_size * 0.1),
                      color='black')
            ax[3].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=self.fsize - 4)
            ax[3].set_ylabel('Event count', fontsize=self.fsize - 4)
            ax[3].set_title(
                'Event count for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
                fontsize=self.fsize - 4)
            ax[3].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['top'].set_visible(False)
            ax[4].bar(bins_all[~np.isnan(stride_duration_bin_avg)],
                      stride_duration_bin_avg[~np.isnan(stride_duration_bin_avg)], width=step_size - (step_size * 0.1),
                      color='black')
            ax[4].errorbar(bins_all[~np.isnan(stride_duration_bin_avg)],
                           stride_duration_bin_avg[~np.isnan(stride_duration_bin_avg)],
                           yerr=stride_duration_bin_sem[~np.isnan(stride_duration_bin_sem)], xerr=0, fmt='.',
                           color='black')
            ax[4].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=self.fsize - 4)
            ax[4].set_ylabel('Stride duration mean + sem (ms)', fontsize=self.fsize - 4)
            ax[4].set_title(
                'Stride duration for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
                fontsize=self.fsize - 4)
            ax[4].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['top'].set_visible(False)
            binwidth = 0.05
            barWidth = 0.05
            binedges = np.arange(0, np.nanmax(isi_all) + 0.5, binwidth)
            hist_all = np.histogram(isi_all, bins=binedges)
            hist_norm = hist_all[0] / np.sum(hist_all[0])
            hist_all_shuffle = np.histogram(isi_all_shuffle, bins=binedges)
            hist_norm_shuffle = hist_all_shuffle[0] / np.sum(hist_all_shuffle[0])
            r1 = binedges[:-1]
            ax[5].bar(r1, hist_norm, color='black', width=barWidth, edgecolor='white')
            ax[5].bar(r1, hist_norm_shuffle, color='gray', width=barWidth, edgecolor='white', alpha=0.5)
            ax[5].set_xlabel('Time (s)', fontsize=self.fsize - 4)
            ax[5].set_ylabel('Event count', fontsize=self.fsize - 4)
            ax[5].set_title('Inter-event interval distribution across trials ' + plot_condition,
                            fontsize=self.fsize - 4)
            ax[5].tick_params(axis='both', which='major', labelsize=self.fsize - 6)
            ax[5].spines['right'].set_visible(False)
            ax[5].spines['top'].set_visible(False)
            plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                     'event_proportion_' + param_name + '_' + plot_condition),
                        dpi=self.my_dpi)
        return bins_all, sl_p1_events_bin_all, sl_p1_events_bin_all_shuffle

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
                    residuals_stack[f, coord_fiji_pixel[:, 1], coord_fiji_pixel[:, 0]] = image_stack[
                                                                                             f, coord_fiji_pixel[:,
                                                                                                1], coord_fiji_pixel[:,
                                                                                                    0]] - trace[f]
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
            tiff.imsave(self.path + path_bgsub + 'T' + tiff_name.split('_')[0][1:] + '_reg_bgsub.tif',
                        image_stack_bgsub, bigtiff=True)
        return

    def events_align_trajectory(self, df_events, traces_type, traj_type, bcam_time, final_tracks_trials, trial, trials, roi, plot_name, plot_data, print_plots):
        """Function to plot average trajectories aligned to calcium event for a certain ROI and trial.
        Input:
            df_events: dataframe with events
            traces_type: (str) raw or deconv
            traj_type: (str) time or phase
            bcam_time: behavioral camera timestamps for all trials
            final_tracks_trials: final_tracks list for all trials (best the one for periods only forward locomotion)
            trial: trial or trials to plot
            trials: (list) all trials in the session
            roi: (int) roi to plot
            plot_name: (str) roi, cluster or sync
            plot_data: boolean
            print_plots: boolean"""
        int_find = ''.join(x for x in df_events.columns[2] if x.isdigit())
        int_find_idx = df_events.columns[2].find(int_find)
        if df_events.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        time = 0.2
        colors_paws = ['blue', 'red', 'cyan', 'magenta']
        idx_trial = np.where(trials == trial)[0][0]
        data_events = np.array(
            df_events.loc[(df_events['trial'] == trial) & (df_events[idx_nr]), 'time'])
        bodycenter = np.nanmean(final_tracks_trials[idx_trial][0, :4, :], axis=0)
        bcam_time_trial = bcam_time[idx_trial]
        bcam_idx_events = []
        for e in data_events:
            bcam_idx_events.append(np.argmin(np.abs(e - bcam_time_trial)))
        traj_list = []
        traj_time = []
        for i in bcam_idx_events:
            if (i > (time * self.sr_loco)) and (i < (np.shape(final_tracks_trials[idx_trial])[2] - (time * self.sr_loco))):
                traj_time.append(bcam_time_trial[i])
                traj = np.array(final_tracks_trials[idx_trial][0, :4, np.int64(i - (time * self.sr_loco)):np.int64(i + (time * self.sr_loco))])
                bc_traj = bodycenter[np.int64(i - (time * self.sr_loco)):np.int64(i + (time * self.sr_loco))]
                if traj_type == 'time':
                    traj_bcsub = traj - bc_traj
                    traj_zscore = (traj_bcsub - np.nanmean(traj_bcsub)) / np.nanstd(traj_bcsub)
                    traj_list.append(traj_zscore)
                if traj_type == 'phase':
                    traj_zscore = traj
                    traj_list.append(traj_zscore)
        traj_arr = np.dstack(traj_list)
        traj_ave = np.nanmean(traj_arr, axis=2)
        traj_std = np.nanstd(traj_arr, axis=2)
        if plot_data:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
            ax = ax.ravel()
            for count_p, p in enumerate(np.array([2, 0, 3, 1])):
                ax[count_p].plot(np.arange(0, np.shape(traj_ave)[1]), traj_ave[p, :], linewidth=2,
                                 color=colors_paws[count_p])
                ax[count_p].fill_between(np.arange(0, np.shape(traj_ave)[1]), traj_ave[p, :] - traj_std[p, :],
                                         traj_ave[p, :] + traj_std[p, :], color=colors_paws[count_p], alpha=0.3)
                ax[count_p].set_xticks(np.linspace(0, np.shape(traj_ave)[1], 10))
                ax[count_p].set_xticklabels(list(map(str, np.round(np.linspace(-time, time, 10), 2))), rotation=45)
                ax[count_p].axvline(x=np.shape(traj_ave)[1] / 2, linestyle='dashed', color='black')
                ax[count_p].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax[count_p].set_ylabel('Paw trajectory (mm)', fontsize=self.fsize - 8)
                ax[count_p].spines['right'].set_visible(False)
                ax[count_p].spines['top'].set_visible(False)
                ax[count_p].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(
                            os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'event_paw_' + traj_type + '_' + plot_name + '_' + str(roi) + '_trial_' + str(trial)),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'event_paw_' + traj_type + '_' + plot_name + '_' + str(roi) + '_trial_' + str(trial)),
                                dpi=self.my_dpi)
            fig, ax = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True)
            ax = ax.ravel()
            for count_p, p in enumerate(np.array([2, 0, 3, 1])):
                ax[count_p].plot(np.arange(0, np.shape(traj_ave)[1]), traj_ave[p, :], linewidth=2,
                                 color=colors_paws[count_p])
                for s in range(np.shape(traj_arr)[2]):
                    ax[count_p].plot(np.arange(0, np.shape(traj_ave)[1]), traj_arr[p, :, s], color='black')
                ax[count_p].set_xticks(np.linspace(0, np.shape(traj_ave)[1], 10))
                ax[count_p].set_xticklabels(list(map(str, np.round(np.linspace(-time, time, 10), 2))), rotation=45)
                ax[count_p].axvline(x=np.shape(traj_ave)[1] / 2, linestyle='dashed', color='black')
                ax[count_p].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax[count_p].set_ylabel('Paw trajectory (mm)', fontsize=self.fsize - 8)
                ax[count_p].spines['right'].set_visible(False)
                ax[count_p].spines['top'].set_visible(False)
                ax[count_p].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(
                            os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
                        os.mkdir(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
                    plt.savefig(os.path.join(self.path + 'images', 'events', traces_type, 'ROI' + str(roi),
                                             'event_paw_' + traj_type + '_' + plot_name + '_' + str(roi) + '_trial_' + str(
                                                 trial) + '_all'),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi),
                                             'event_paw_' + traj_type + '_' + plot_name + '_' + str(roi) + '_trial_' + str(
                                                 trial) + '_all'), dpi=self.my_dpi)
        return traj_time, traj_arr

    def events_align_trajectory_plot_all(self, traj_cluster_trials, traj_type, cluster_plot, traces_type, event_type, trials, colors_session, plot_data, print_plots):
        """Function to plot average trajectories aligned to calcium event for a certain ROI and trial.
        Input:
            traj_cluster_trials: (list) trajectories aligned to calcium event
            traj_type: (str) time or phase
            cluster_plot: (int) cluster to plot
            traces_type: (str) raw or deconv
            event_type. (str) roi or cluster or sync
            trials: (list) all trials in the session
            colors_session: (list) colors for each trial in a session
            plot_data: boolean
            print_plots: boolean"""
        time = 0.2
        if plot_data:
            fig, ax = plt.subplots(4, len(trials), figsize=(25, 15), tight_layout=True, sharex=True, sharey=True)
            for count_p, p in enumerate(np.array([2, 0, 3, 1])):
                for t in trials:
                    ax[count_p, t-1].plot(traj_cluster_trials[t-1][p], linewidth=0.5, color=colors_session[t-1])
                    ax[count_p, t-1].set_xticks(np.linspace(0, len(traj_cluster_trials[t-1][0]), 10))
                    ax[count_p, t-1].set_xticklabels(list(map(str, np.round(np.linspace(-time, time, 10), 2))), rotation=45)
                    ax[count_p, t-1].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                    ax[count_p, t-1].spines['right'].set_visible(False)
                    ax[count_p, t-1].spines['top'].set_visible(False)
                    ax[count_p, t-1].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
                plt.suptitle('Cluster ' + str(cluster_plot) + 'paws FL, FR, HL, HR')
                if print_plots:
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'event_paw_' + traj_type + '_cluster_' + event_type), dpi=self.my_dpi)
        return

    def events_cv_trajectory(self, df_events, traj_type, bcam_time, final_tracks_trials, trial, trials, roi):
        """Function to compute coefficient of variation of trajectories aligned to calcium event for a certain ROI and trial.
        Input:
            df_events: dataframe with events
            traj_type: (str) time or phase
            bcam_time: behavioral camera timestamps for all trials
            final_tracks_trials: final_tracks list for all trials (best the one for periods only forward locomotion)
            trial: trial or trials to plot
            trials: (list) all trials in the session
            roi: (int) roi to plot"""
        int_find = ''.join(x for x in df_events.columns[2] if x.isdigit())
        int_find_idx = df_events.columns[2].find(int_find)
        if df_events.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi)
        time = 0.2
        idx_trial = np.where(trials == trial)[0][0]
        data_events = np.array(
            df_events.loc[(df_events['trial'] == trial) & (df_events[idx_nr]), 'time'])
        bodycenter = np.nanmean(final_tracks_trials[idx_trial][0, :4, :], axis=0)
        bcam_time_trial = bcam_time[idx_trial]
        bcam_idx_events = []
        for e in data_events:
            bcam_idx_events.append(np.argmin(np.abs(e - bcam_time_trial)))
        traj_list = []
        for i in bcam_idx_events:
            if (i > (time * self.sr_loco)) and (i < (np.shape(final_tracks_trials[idx_trial])[2] - (time * self.sr_loco))):
                traj = np.array(final_tracks_trials[idx_trial][0, :4, np.int64(i - (time * self.sr_loco)):np.int64(i + (time * self.sr_loco))])
                bc_traj = bodycenter[np.int64(i - (time * self.sr_loco)):np.int64(i + (time * self.sr_loco))]
                if traj_type == 'time':
                    traj_bcsub = traj - bc_traj
                    traj_zscore = (traj_bcsub - np.nanmean(traj_bcsub)) / np.nanstd(traj_bcsub)
                    traj_list.append(traj_zscore)
                if traj_type == 'phase':
                    traj_zscore = traj
                    traj_list.append(traj_zscore)
        traj_arr = np.dstack(traj_list)
        traj_ave = np.nanmean(traj_arr, axis=2)
        traj_std = np.nanstd(traj_arr, axis=2)
        traj_cv = traj_std/traj_ave
        return traj_cv

    def events_cv_trajectory_plot(self, traj_cv_cluster, traj_type, cluster_plot, traces_type, event_type, trials, colors_session, plot_data, print_plots):
        """Function to plot the coefficient of variation of trajectories aligned to calcium event for a certain ROI and trial.
        It plots coefficient of variation for the whole time around event onset and for the specific event onset time.
        Input:
            traj_cv_cluster: (list) coefficient of variation of trajectories aligned to calcium event
            traj_type: (str) time or phase
            cluster_plot: (int) cluster to plot
            traces_type: (str) raw or deconv
            event_type. (str) roi or cluster or sync
            trials: (list) all trials in the session
            colors_session: (list) colors for each trial in a session
            plot_data: boolean
            print_plots: boolean"""
        time = 0.2
        colors_paws = ['blue', 'red', 'cyan', 'magenta']
        data_cv_line_FL_FR_HL_HR = 0
        if plot_data:
            fig, ax = plt.subplots(2, 2, tight_layout=True, sharex=True)
            ax = ax.ravel()
            for t in trials:
                for count_p, p in enumerate(np.array([2, 0, 3, 1])):
                    ax[count_p].plot(traj_cv_cluster[t-1][p], color=colors_session[t-1])
            ax[0].set_title('FL', color=colors_paws[0])
            ax[1].set_title('FR', color=colors_paws[1])
            ax[2].set_title('HL', color=colors_paws[2])
            ax[3].set_title('HR', color=colors_paws[3])
            for p in range(4):
                ax[p].set_xticks(np.linspace(0, len(traj_cv_cluster[t-1][0]), 10))
                ax[p].set_xticklabels(list(map(str, np.round(np.linspace(-time, time, 10), 2))), rotation=45)
                ax[p].axvline(x=len(traj_cv_cluster[t-1][0]) / 2, linestyle='dashed', color='black')
                ax[p].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax[p].set_ylabel('CV', fontsize=self.fsize - 8)
                ax[p].spines['right'].set_visible(False)
                ax[p].spines['top'].set_visible(False)
                ax[p].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            plt.suptitle('Cluster ' + str(cluster_plot))
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'event_paw_' + traj_type + '_' + event_type + '_cv'), dpi=self.my_dpi)
            time = 66
            fig, ax = plt.subplots(2, 2, tight_layout=True, sharex=True)
            ax = ax.ravel()
            data_cv_line_FL_FR_HL_HR = []
            for count_p, p in enumerate(np.array([2, 0, 3, 1])):
                data_line = np.zeros(len(trials))
                for t in trials:
                    data_line[t-1] = traj_cv_cluster[t-1][p][time]
                    ax[count_p].scatter(t, data_line[t-1], color=colors_session[t-1])
                data_cv_line_FL_FR_HL_HR.append(data_line)
                ax[count_p].plot(trials, data_line, color='black')
                ax[count_p].set_xlabel('Trials', fontsize=self.fsize - 8)
                ax[count_p].set_ylabel('CV', fontsize=self.fsize - 8)
                ax[count_p].spines['right'].set_visible(False)
                ax[count_p].spines['top'].set_visible(False)
                ax[count_p].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            plt.suptitle('Cluster ' + str(cluster_plot) + ' for time of event peak')
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'event_paw_' + traj_type + '_' + event_type + '_cv_event_onset'), dpi=self.my_dpi)
        return data_cv_line_FL_FR_HL_HR

    def diff_paws_around_event(self, df_events, traj_type, cluster_plot, traces_type, event_type, bcam_time, final_tracks_trials, trials, colors_session,plot_data, print_plots):
        """Function to plot the difference in paw amplitude or phase of the trajectories aligned to calcium event for a certain ROI and trial.
        Input:
            df_events: (dataframe) events of ROI, clsuters or sync events of clusters
            traj_type: (str) time or phase
            cluster_plot: (int) cluster to plot
            traces_type: (str) raw or deconv
            event_type. (str) roi or cluster or sync
            bcam_time: (list) behavioral camera timestamps of each trial
            trials: (list) all trials in the session
            final_tracks_trials: (list) final tracks for each trial in time or in phase
            colors_session: (list) colors for each trial in a session
            plot_data: boolean
            print_plots: boolean"""
        traj_diff_cluster_front_ave = 0
        traj_diff_cluster_front_sem = 0
        traj_diff_cluster_hind_ave = 0
        traj_diff_cluster_hind_sem = 0
        if plot_data:
            fig, ax = plt.subplots(2, 1, figsize=(15, 10), tight_layout=True)
            ax = ax.ravel()
            traj_time_all = []
            traj_diff_cluster_front_ave = np.zeros(len(trials))
            traj_diff_cluster_hind_ave = np.zeros(len(trials))
            traj_diff_cluster_front_sem = np.zeros(len(trials))
            traj_diff_cluster_hind_sem = np.zeros(len(trials))
            for t in trials:
                [traj_time, traj_cluster] = self.events_align_trajectory(df_events, traces_type, traj_type, bcam_time, final_tracks_trials, t, trials, cluster_plot, event_type, 0, 0)
                if t == 1:
                    traj_time_all.extend(np.array(traj_time))
                else:
                    traj_time_all.extend((np.array(traj_time))+(60*(t-1)))
                traj_diff_cluster_front = traj_cluster[0][66, :]-traj_cluster[2][66, :]
                traj_diff_cluster_hind = traj_cluster[1][66, :]-traj_cluster[3][66, :]
                traj_diff_cluster_front_ave[t-1] = np.nanmean(traj_diff_cluster_front)
                traj_diff_cluster_hind_ave[t-1] = np.nanmean(traj_diff_cluster_hind)
                traj_diff_cluster_front_sem[t-1] = np.nanstd(traj_diff_cluster_front)/np.sqrt(len(traj_diff_cluster_front))
                traj_diff_cluster_hind_sem[t-1] = np.nanstd(traj_diff_cluster_hind)/np.sqrt(len(traj_diff_cluster_hind))
                time_xaxis = traj_time_all[-len(traj_time):]
                ax[0].plot(time_xaxis, traj_diff_cluster_front, color=colors_session[t - 1], zorder=0)
                ax[0].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax[0].set_ylabel('Front paw difference', fontsize=self.fsize - 8)
                ax[0].spines['right'].set_visible(False)
                ax[0].spines['top'].set_visible(False)
                ax[0].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
                ax[1].plot(time_xaxis, traj_diff_cluster_hind, color=colors_session[t-1], zorder=0)
                ax[1].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax[1].set_ylabel('Hind paw difference', fontsize=self.fsize - 8)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['top'].set_visible(False)
                ax[1].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            plt.suptitle('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction ' + event_type + ' events in ' + traj_type)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
                                         'event_paw_' + traj_type + '_diff_' + event_type), dpi=self.my_dpi)
            fig, ax = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
            ax = ax.ravel()
            for t in trials:
                ax[0].scatter(t, traj_diff_cluster_front_ave[t-1], s=30, color=colors_session[t-1])
                ax[1].scatter(t, traj_diff_cluster_hind_ave[t-1], s=30, color=colors_session[t - 1])
            ax[0].plot(trials, traj_diff_cluster_front_ave, color='black')
            ax[0].fill_between(trials, traj_diff_cluster_front_ave-traj_diff_cluster_front_sem, traj_diff_cluster_front_ave+traj_diff_cluster_front_sem, color='black', alpha=0.3)
            ax[0].set_xlabel('Time (s)', fontsize=self.fsize - 8)
            ax[0].set_ylabel('Front paw difference', fontsize=self.fsize - 8)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            ax[0].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction ' + event_type + ' events in ' + traj_type)
            ax[1].plot(trials, traj_diff_cluster_hind_ave, color='black')
            ax[1].fill_between(trials, traj_diff_cluster_hind_ave-traj_diff_cluster_hind_sem, traj_diff_cluster_hind_ave+traj_diff_cluster_hind_sem, color='black', alpha=0.3)
            ax[1].set_xlabel('Time (s)', fontsize=self.fsize - 8)
            ax[1].set_ylabel('Hind paw difference', fontsize=self.fsize - 8)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            ax[1].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction ' + event_type + ' events in ' + traj_type)
        if print_plots:
            plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
                                     'event_paw_' + traj_type + '_diff_' + event_type + '_ave'), dpi=self.my_dpi)
        return traj_diff_cluster_front_ave, traj_diff_cluster_front_sem, traj_diff_cluster_hind_ave, traj_diff_cluster_hind_sem

    def cumulative_activity_cluster(self, df_events, time_cumulative, clusters_rois, trials, colors_cluster, plot_data, print_plots):
        """Compute and plot the cumulative activity for each cluster across trials
        Inputs:
        df_events: (dataframe) with events for each cluster
        time_cumulative: (vector) with timestamps cumulative in the session
        clusters_rois: (list) with the ROIs in each cluster
        colors_cluster: colors for each cluster
        trials: (list) with the trials in the session
        plot_data: boolean
        print_plots: boolean"""
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
            for c in range(len(clusters_rois)):
                ax.scatter(time_cumulative / 60, np.cumsum(df_events['cluster' + str(c + 1)]), s=0.1, color=colors_cluster[c])
                for t in trials:
                    ax.axvline(time_cumulative[df_events.loc[df_events['trial'] == t].index[-1]] / 60, color='black', linestyle='dashed')
            ax.set_ylabel('Event cumulative count', fontsize=self.fsize)
            ax.set_xlabel('Time (min)', fontsize=self.fsize)
            ax.set_title('Cumulative event count for the clusters', fontsize=self.fsize)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=self.fsize - 4)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cumulative_event_count'), dpi=self.my_dpi)
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
        if not os.path.exists(os.path.join(self.path, 'EXTRACT', 'Fiji ROI bgsub')):
            os.mkdir(os.path.join(self.path, 'EXTRACT', 'Fiji ROI bgsub'))
        plt.savefig(
            os.path.join(self.path, 'EXTRACT', 'Fiji ROI bgsub' + 'fiji_bgsub_' + str(rfiji) + '_T' + str(trial)),
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
        [Ev_Onset_fiji, IncremSet_fiji] = ST.SlopeThreshold(roi_trace_bgsub_minmax, self.sr)
        if len(Ev_Onset_fiji) > 0:
            events_fiji = np.zeros(len(roi_trace_bgsub_minmax))
            events_fiji_list = self.event_detection_calcium_trace(roi_trace_bgsub_minmax, Ev_Onset_fiji, IncremSet_fiji,
                                                                  3)
            events_fiji[events_fiji_list] = 1
        [Ev_Onset_ext, IncremSet_ext] = ST.SlopeThreshold(trace_roi, self.sr)
        if len(Ev_Onset_fiji) > 0:
            events_ext = np.zeros(len(trace_roi))
            events_ext_list = self.event_detection_calcium_trace(trace_roi, Ev_Onset_ext, IncremSet_ext, 3)
            events_ext[events_ext_list] = 1
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
            if not os.path.exists(os.path.join(self.path, 'EXTRACT', 'EXTRACT comparisons')):
                os.mkdir(os.path.join(self.path, 'EXTRACT', 'EXTRACT comparisons'))
            plt.savefig(
                os.path.join(self.path, 'EXTRACT', 'EXTRACT comparisons', 'fiji_bgsub_' + str(rfiji) + '_ext_' + str(
                    rext) + '_T' + str(trial)), dpi=self.my_dpi)
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
            plt.savefig(
                os.path.join(self.path, 'images', 'events',
                             'event_example_trial' + str(trial_plot) + '_roi' + str(roi_plot)),
                dpi=self.my_dpi)

    def plot_events_roi_trial_bgsub(self, trial_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all,
                                    df_events_unsync, plot_data, print_plots):
        """Function to plot events on top of traces with and without background subtraction for all ROIs and one trial.
        Input:
        trial_plot: (str)
        roi_plot: (str)
        frame_time: list with mscope timestamps
        df_fiji_norm: dataframe with traces raw
        df_fiji_bgsub_norm: dataframe with traces background subtracted
        df_events_all: dataframe with all the events
        df_events_unsync: dataframe with unsynchronous the events
        plot_data, print_plots: boolean"""
        df_fiji_trial_norm = df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot]  # get dFF for the desired trial
        df_fiji_bgsub_trial_norm = df_fiji_bgsub_norm.loc[
            df_fiji_bgsub_norm['trial'] == trial_plot]  # get dFF for the desired trial
        if plot_data:
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
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'events', 'events_trial' + str(trial_plot)),
                            dpi=self.my_dpi)
        return

    def plot_events_roi_trial(self, trial_plot, roi_plot, frame_time, df_dff, traces_type, df_events, trials, plot_data, print_plots):
        """Function to plot events on top of traces with and without background subtraction for all ROIs and one trial.
        Input:
        trial_plot: (str)
        roi_plot: (str)
        frame_time: list with mscope timestamps
        df_dff: dataframe with traces
        traces_type: (str) raw or deconv
        df_events: dataframe with the events
        trials: list of trials in a session
        plot_data: boolean
        print_plots: boolean"""
        int_find = ''.join(x for x in df_dff.columns[2] if x.isdigit())
        int_find_idx = df_dff.columns[2].find(int_find)
        if df_dff.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type =  'cluster'
        idx_nr = df_type + str(roi_plot)
        df_dff_trial = df_dff.loc[df_dff['trial'] == trial_plot, idx_nr]  # get dFF for the desired trial
        if plot_data:
            fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
            idx_trial = np.where(trials==trial_plot)[0][0]
            ax.plot(frame_time[idx_trial], df_dff_trial, color='black')
            events_plot = np.where(df_events.loc[df_events['trial'] == trial_plot, idx_nr])[0]
            for e in events_plot:
                ax.scatter(frame_time[idx_trial][e], df_dff_trial.iloc[e], s=60,
                           color='orange')
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
                if df_type == 'ROI':
                    plt.savefig(os.path.join(self.path, 'images', 'events',
                                             'events_trial' + str(trial_plot) + '_roi' + str(
                                                 roi_plot) + '_' + traces_type),
                                dpi=self.my_dpi)
                if df_type == 'cluster':
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type,
                                             'events_trial' + str(trial_plot) + '_' + idx_nr + '_' + traces_type),
                                dpi=self.my_dpi)
        return

    def number_sync_events_cluster(self, df_traces, df_events, clusters_rois, trials, colors_session, traces_type,
                                   th_sync, plot_data, print_plots, save_data):
        """Compute the number of events with a synchrony higher than 0.9 and plot their count for each trial.
        Inputs:
        df_traces: dataframe with traces
        df_events: dataframe with events
        clusters_rois: list with the ROIs for each cluster
        trials: list with the trials of the session
        colors_session: (str) list with trials colors
        traces_type: (str) raw or deconv
        th_sync: float between 0 and 1 - theshold for 5 of cells active at the same time
        plot_data, print_plots, save_data. boolean"""
        df_events_sync = pd.DataFrame()
        for cluster_plot in range(len(clusters_rois)):
            time_sync_trials = []
            for t in trials:
                # data_plot = df_traces.loc[df_traces['trial'] == t, clusters_rois[cluster_plot]].reset_index(
                #     drop=True).iloc[:, 2:]
                data_event_plot = df_events.loc[df_events['trial'] == t, clusters_rois[cluster_plot]].iloc[:, 2:]
                th_sync_vec = np.floor(th_sync * len(data_event_plot.columns))
                sum_event_rois = data_event_plot.sum(axis=1)
                time_sync = np.zeros(len(sum_event_rois))
                time_sync[np.where(sum_event_rois >= th_sync_vec)[0]] = 1
                time_sync_trials.append(time_sync)
                time_sync_trials_flat = [item for trial in time_sync_trials for item in trial]
            df_events_sync['cluster'+str(cluster_plot+1)] = time_sync_trials_flat
            # fig, ax = plt.subplots(3, 1, figsize=(25, 15), tight_layout=True)
            # ax = ax.ravel()
            # count_r = 0
            # for r in data_plot.columns[2:]:
            #     ax[0].plot(data_plot[r] + (count_r / 2), color='black')
            #     ax[0].scatter(np.where(data_event_plot[r])[0],
            #                   data_plot[r].iloc[np.where(data_event_plot[r])[0]] + (count_r / 2), s=20, color='orange')
            #     count_r += 1
            # ax[0].set_xlim([0, len(sum_event_rois)])
            # sns.heatmap(np.transpose(data_event_plot), cmap='viridis', cbar=False, ax=ax[1])
            # ax[2].plot(np.arange(0, len(sum_event_rois)), time_sync, color='black')
            # ax[2].set_xlim([0, len(sum_event_rois)])
            # fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
            # tiff_stack = tiff.imread(mscope.path + '\\Registered video\\T' + str(t) + '_reg.tif')  # read tiffs
            # idx_sync = np.where(sum_event_rois >= th_sync_vec)[0]
            # ax = ax.ravel()
            # ax[0].imshow(tiff_stack[np.random.choice(idx_sync, size=1)[0], :, :])
            # ax[0].set_title('Sync events for that cluster')
            # ax[1].imshow(
            #     tiff_stack[np.random.choice(np.setdiff1d(np.where(sum_event_rois)[0], idx_sync), size=1)[0], :, :])
            # ax[1].set_title('Other period for that cluster')
            if plot_data:
                fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True, sharey=True)
                for t in trials:
                    ax.bar(t - 0.5, np.sum(time_sync_trials[t - 1]), width=1, color=colors_session[t - 1],
                              edgecolor='white')
                ax.set_xticks(np.arange(0.5, len(trials) + 0.5))
                ax.set_xticklabels(list(map(str, trials)))
                ax.set_xlim([0, len(trials) + 1])
                ax.set_ylabel('Event count', fontsize=self.fsize)
                ax.set_xlabel('Trials', fontsize=self.fsize)
                ax.set_title('Syncronized event count for cluster ' + str(cluster_plot + 1), fontsize=self.fsize)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=self.fsize - 4)
                if print_plots:
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot + 1))):
                        os.mkdir(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot + 1)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot + 1),
                                             'cluster' + str(cluster_plot + 1) + '_sync_events_count_th' + str(
                                                 th_sync).replace('.', ',') + '_' + traces_type), dpi=self.my_dpi)
        df_events_sync.insert(loc=0, column='trial', value=df_events['trial'])
        df_events_sync.insert(loc=0, column='time', value=df_events['time'])
        if save_data:
            df_events_sync.to_csv(os.path.join(self.path, 'processed files', 'df_events_sync.csv'), sep=',',
                                  index=False)
        return df_events_sync

    def sort_HF_files_by_days(self, animal_name):
        """Function to sort mat files from injections experiment across days
        Inputs:
            animal_name: string with the animal name to sort"""
        matfiles = glob.glob(self.path + '*.mat')
        matfiles_animal = []
        days_animal = []
        for f in matfiles:
            path_split = f.split(self.delim)
            filename = path_split[-1][:-4]
            filename_split = filename.split('_')
            if filename_split[0] == animal_name:
                matfiles_animal.append(f)
                days_animal.append(int(filename_split[1][:-4]))
        days_ordered = np.sort(np.array(days_animal))  # reorder days
        files_ordered = []  # order mat filenames by file order
        for f in range(len(matfiles_animal)):
            tr_ind = np.where(days_ordered[f] == days_animal)[0][0]
            files_ordered.append(matfiles_animal[tr_ind])
        return files_ordered

    def plot_sl_sym_session(self, param_sym, trials_ses, trials, session_type, colors_session, plot_data, print_plots):
        """Plot step length symmetry for all trials in a session with the colors for the session
        Inputs:
        param_sym: (list) with step length front paw symmetry values
        trials_ses: (list) with transition trials
        trials: (list) of trials in the session
        session_type: (str) tied or split
        colors_session: (list) with the colors for each trial
        plot_data, print_plots: boolean"""
        param_sym_baseline = np.nanmean(param_sym[:trials_ses[0, 1]])
        param_sym_bs = param_sym-param_sym_baseline
        if plot_data:
            fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
            if session_type == 'split':
                rectangle = plt.Rectangle((trials_ses[0, 1]+0.5, min(param_sym_bs)), 10, max(param_sym_bs)-min(param_sym_bs), fc='grey', alpha=0.3)
                ax.add_patch(rectangle)
            ax.hlines(0, 1, len(param_sym_bs), colors='grey', linestyles='--')
            ax.plot(trials, param_sym_bs, color='black')
            for count_t, t in enumerate(trials):
                idx_trial = np.where(trials==t)[0][0]
                ax.scatter(t, param_sym_bs[idx_trial], s=80, color=colors_session[t])
            ax.set_xlabel('Trials', fontsize=20)
            ax.set_ylabel('Step length symmetry', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if not os.path.exists(self.path+'images'):
                    os.mkdir(self.path+'images')
                plt.savefig(os.path.join(self.path, 'images', 'step_length_symmetry_frontpaws'), dpi=self.my_dpi)


    def response_time_population_avg(self, df_norm, time_beg, time_end, clusters_rois, cluster_transition_idx,
                                     data_type, plot_type, plot_data, print_plots):
        """Compute the heatmap for ROI calcium activity for a certain time period. It can do a random order, by distance or by cluster - depending on input data
        df_norm: dataframe with activity for all ROIs and timepoints
        time_beg: (int) vector with starting points for time
        time_end: (int) vector with ending points for time
        clusters_rois: list with the ROIs belonging to each cluster
        cluster_transition_idx: list with the ROIs at the end of each cluster
        data_type: (str) raw or events or deconv
        plot_type: (str) raw, distance or cluster
        plot_data: boolean
        print_plots: boolean"""
        trials = np.unique(df_norm['trial'])
        if plot_type == 'cluster':
            if len(clusters_rois) == 1:
                clusters_rois_flat = clusters_rois[0]
            else:
                clusters_rois_flat = np.transpose(sum(clusters_rois, []))
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        for count_t, t in enumerate(time_beg):
            data_list = []
            frames_len = []
            for count_trial, t in enumerate(trials):
                data_list.append(np.transpose(
                    df_norm.loc[df_norm['trial'] == t].iloc[time_beg[count_t] * self.sr:time_end[count_t] * self.sr,
                    2:]))
                frames_len.append(np.shape(
                    df_norm.loc[df_norm['trial'] == t].iloc[time_beg[count_t] * self.sr:time_end[count_t] * self.sr,
                    2:])[0])
            data = np.concatenate(data_list, axis=1)
            cum_frames_len = np.cumsum(frames_len)
            if plot_data:
                fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
                sns.heatmap(data, cbar='False', cmap='viridis')
                for t in trials:
                    idx_trial = np.where(trials==t)[0][0]
                    ax.vlines(cum_frames_len[idx_trial], *ax.get_ylim(), color='white', linestyle='dashed')
                ax.set_yticks(np.arange(0, len(df_norm.columns[2:]), 4))
                ax.set_yticklabels(df_norm.columns[2::4], rotation=45, fontsize=self.fsize - 12)
                if plot_type == 'cluster':
                    for c in cluster_transition_idx:
                        ax.hlines(c + 1, *ax.get_xlim(), color='white',
                                  linestyle='dashed')  # +1 puts in beginning of cluster
                    ax.set_yticks(np.arange(0, len(clusters_rois_flat[2:]), 4))
                    ax.set_yticklabels(clusters_rois_flat[2::4], rotation=45, fontsize=self.fsize - 12)
                ax.set_xlabel('Frames', fontsize=self.fsize - 4)
                if print_plots:
                    if not os.path.exists(os.path.join(self.path, 'images', plot_type)):
                        os.mkdir(os.path.join(self.path, 'images', plot_type))
                    plt.savefig(os.path.join(self.path, 'images', plot_type,
                                             'heatmap_' + str(time_beg[count_t]) + 's_' + str(
                                                 time_end[count_t]) + 's_' + plot_type + '_' + data_type), dpi=self.my_dpi)
        return

    def response_time_population_events(self, df_norm, time_beg, time_end, clusters_rois, cluster_transition_idx,
                                        plot_type, trials_baseline, trials_split, colors_session, save_plot):
        """Compute the heatmap for ROI event activity for a certain time period and its summary. It can do a random order, by distance or by cluster - depending on input data
        df_norm: dataframe with activity for all ROIs and timepoints
        time_beg: (int) vector with starting points for time
        time_end: (int) vector with ending points for time
        clusters_rois: list with the ROIs belonging to each cluster
        cluster_transition_idx: list with the ROIs at the end of each cluster
        plot_type: (str) raw, distance or cluster
        traces_type: (str) raw or events or deconv
        trials_baseline: list with trial id for the baseline
        trials_split: list with trial id for split
        colors_session: colors for the trials in the session
        save_plot: boolean"""
        trials = np.unique(df_norm['trial'])
        if plot_type == 'cluster':
            trial_avg = np.zeros((len(time_beg), len(trials), len(clusters_rois)))
        else:
            trial_avg = np.zeros((len(time_beg), len(trials)))
        for count_t, t in enumerate(time_beg):
            data_list = []
            frames_len = []
            for count_trial, t in enumerate(trials):
                data_list.append(np.transpose(
                    df_norm.loc[df_norm['trial'] == t].iloc[time_beg[count_t] * self.sr:time_end[count_t] * self.sr,
                    2:]))
                frames_len.append(np.shape(
                    df_norm.loc[df_norm['trial'] == t].iloc[time_beg[count_t] * self.sr:time_end[count_t] * self.sr,
                    2:])[0])
                if plot_type == 'cluster':
                    for c in range(len(clusters_rois)):
                        trial_avg[count_t, count_trial, c] = df_norm.loc[df_norm['trial'] == t].iloc[
                                                             time_beg[count_t] * self.sr:time_end[
                                                                                             count_t] * self.sr][
                                                                 clusters_rois[c]].sum().iloc[2:].sum() / (
                                                                         len(df_norm.columns) - 2)
                else:
                    trial_avg[count_t, count_trial] = df_norm.loc[df_norm['trial'] == t].iloc[
                                                      time_beg[count_t] * self.sr:time_end[
                                                                                      count_t] * self.sr].sum().iloc[
                                                      2:].sum() / (len(df_norm.columns) - 2)
            data = np.concatenate(data_list, axis=1)
            cum_frames_len = np.cumsum(frames_len)
            fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
            sns.heatmap(data, cbar='False', cmap='viridis')
            for t in trials:
                ax.vlines(cum_frames_len[t - 1], *ax.get_ylim(), color='white', linestyle='dashed')
            ax.set_yticklabels(df_norm.columns[2::2], rotation=45,
                               fontsize=self.fsize - 10)
            if plot_type == 'cluster':
                for c in cluster_transition_idx:
                    ax.hlines(c + 1, *ax.get_xlim(), color='white',
                              linestyle='dashed')  # +1 puts in beginning of cluster
            ax.set_xlabel('Frames', fontsize=self.fsize - 4)
            if save_plot:
                if not os.path.exists(os.path.join(self.path, 'images', plot_type)):
                    os.mkdir(os.path.join(self.path, 'images', plot_type))
                plt.savefig(os.path.join(self.path, 'images', plot_type,
                                         'heatmap_' + str(time_beg[count_t]) + 's_' + str(
                                             time_end[count_t]) + 's_' + plot_type + '_events'), dpi=self.my_dpi)
            if plot_type == 'cluster':
                colors = []
                colors.append((0.967671, 0.439703, 0.35981, 1.0))  # orange center
                colors.append((0.994738, 0.62435, 0.427397, 1.0))  # salmon
                colors.append((0.390384, 0.100379, 0.501864, 1.0))  # purple dark
                colors.append((0, 0, 0, 1.0))  # black
                colors.append((0.716387, 0.214982, 0.47529, 1.0))  # purple light
                fig, ax = plt.subplots(1, 5, figsize=(15, 5), tight_layout=True, sharey=True)
                ax = ax.ravel()
                for c in np.arange(len(clusters_rois)):
                    ax[c].axhline(y=0, linestyle='dashed', color='black')
                    ax[c].add_patch(
                        plt.Rectangle((trials_baseline[-1] + 0.5, np.min(trial_avg[count_t, :])), len(trials_split),
                                      np.max(trial_avg[count_t, :]) - np.min(trial_avg[count_t, :]),
                                      fc='lightgray', alpha=0.7, zorder=0))
                    ax[c].plot(trials, trial_avg[count_t, :, c], color=colors[c])
                    ax[c].scatter(trials, trial_avg[count_t, :, c], s=60, color=colors[c])
                    ax[c].set_ylabel(
                        'Sum of activity ' + str(time_beg[count_t]) + '-' + str(time_end[count_t]) + 's',
                        fontsize=self.fsize - 2)
                    ax[c].set_xlabel('Trials', fontsize=self.fsize - 2)
                    ax[c].spines['right'].set_visible(False)
                    ax[c].spines['top'].set_visible(False)
                    ax[c].tick_params(axis='both', which='major', labelsize=self.fsize - 2)
                    if save_plot:
                        if not os.path.exists(os.path.join(self.path, 'images', plot_type)):
                            os.mkdir(os.path.join(self.path, 'images', plot_type))
                        plt.savefig(os.path.join(self.path, 'images', plot_type,
                                                 'sum_activity_' + str(time_beg[count_t]) + 's_' + str(
                                                     time_end[count_t]) + 's_events'), dpi=self.my_dpi)
            else:
                fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
                ax.axhline(y=0, linestyle='dashed', color='black')
                ax.add_patch(
                    plt.Rectangle((trials_baseline[-1] + 0.5, np.min(trial_avg[count_t, :])), 10,
                                  np.max(trial_avg[count_t, :]) - np.min(trial_avg[count_t, :]),
                                  fc='lightgray', alpha=0.7, zorder=0))
                ax.plot(trials, trial_avg[count_t, :], color='black')
                for i in range(len(colors_session)):
                    ax.scatter(trials[i], trial_avg[count_t, i], s=60, color=colors_session[i])
                ax.set_ylabel('Sum of activity ' + str(time_beg[count_t]) + '-' + str(time_end[count_t]) + 's',
                              fontsize=self.fsize - 2)
                ax.set_xlabel('Trials', fontsize=self.fsize - 2)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='majsor', labelsize=self.fsize - 2)
                if save_plot:
                    if not os.path.exists(os.path.join(self.path, 'images', plot_type)):
                        os.mkdir(os.path.join(self.path, 'images', plot_type))
                    plt.savefig(os.path.join(self.path, 'images', plot_type,
                                             'sum_activity_' + str(time_beg[count_t]) + 's_' + str(
                                                 time_end[count_t]) + 's_events'), dpi=self.my_dpi)
            plt.close('all')
        return trial_avg

    def plot_param_continuous(self, param_all_time, param_all, trials, time_beg, time_end, colors_session, plot_data, print_plots):
        """"Plot a gait parameter continuously in time across trials for a certain time period.
        Inputs:
        param_all_time: (list) with time for each gait parameter value
        param_all: (list) gait parameter value, either sym or intralimb
        trials: (list) list of trials in a session
        time_beg: (int) time start
        time_end: (int) time end
        colors_session: (list) colors for each trial in the session
        plot_data: boolean
        print_plots: boolean"""
        if plot_data:
            fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
            time_start = np.arange(time_beg, time_end + (len(trials) * 5), 5)
            time_finish = np.arange(time_end, time_end + (len(trials) * 5), 5)
            for t in trials:
                if t == 1:
                    time_beg_t = time_beg
                else:
                    time_beg_t = (t - 1) * 60 + time_beg
                time_end_t = time_beg_t + 5
                time_idx = np.where((param_all_time >= time_beg_t) & (param_all_time <= time_end_t))[0]
                ax.plot(np.linspace(time_start[t - 1], time_finish[t - 1], len(param_all[time_idx])),
                        param_all[time_idx], color=colors_session[t - 1])
                ax.set_xlabel('Time (s)', fontsize=self.fsize - 8)
                ax.set_ylabel('Step length symmetry front paws', fontsize=self.fsize - 8)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster',
                                         'step_length_sym_front_' + str(time_beg) + 's_' + str(
                                             time_end) + 's'), dpi=self.my_dpi)
        return

    def cv2_time_period(self, df_events, plot_type, clusters_rois, trials, time_beg, time_end, colors_session, plot_data, print_plots):
        """"Plot CV2 of events of calcium traces across trials for a certain time period.
        Inputs:
        df_events: (dataframe) with the calcium events information for the ROIs or clusters
        plot_type: (str) roi or cluster
        clusters_rois: (list) list with the rois for each cluster
        trials: (list) list of trials in a session
        time_beg: (int) time start
        time_end: (int) time end
        colors_session: (list) colors for each trial in the session
        plot_data: boolean
        print_plots: boolean"""
        if plot_data:
            fig, ax = plt.subplots(len(clusters_rois), 1, figsize=(25, 12), tight_layout=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                for t in trials:
                    if plot_type == 'roi':
                        data_events = df_events.loc[df_events['trial'] == t].iloc[
                                      time_beg * self.sr:time_end * self.sr]
                        for col in clusters_rois[c]:
                            data_isi = data_events.loc[(data_events[col] > 0) & (data_events['trial'] == t), 'time'].diff()
                            if t > 1:
                                data_time = data_events.loc[
                                                (data_events[col] > 0) & (data_events['trial'] == t), 'time'] + (
                                                        5.1 * (t - 1))
                            else:
                                data_time = data_events.loc[(data_events[col] > 0) & (data_events['trial'] == t), 'time']
                            data_cv2 = 2 * (data_isi.diff().abs()) / data_isi.rolling(2).sum()
                            ax[c].plot(data_time, data_cv2, color=colors_session[t - 1])
                    if plot_type == 'cluster':
                        data_events = df_events.loc[df_events['trial'] == t].iloc[
                                      time_beg * self.sr:time_end * self.sr]
                        data_isi = data_events.loc[
                            (data_events['cluster' + str(c + 1)] > 0) & (data_events['trial'] == t), 'time'].diff()
                        if t > 1:
                            data_time = data_events.loc[(data_events['cluster' + str(c + 1)] > 0) & (
                                        data_events['trial'] == t), 'time'] + (
                                                5.1 * (t - 1))
                        else:
                            data_time = data_events.loc[
                                (data_events['cluster' + str(c + 1)] > 0) & (data_events['trial'] == t), 'time']
                        data_cv2 = 2 * (data_isi.diff().abs()) / data_isi.rolling(2).sum(skipna=True)
                        ax[c].plot(data_time, data_cv2, color=colors_session[t - 1])
                    ax[c].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                    ax[c].set_ylabel('CV2', fontsize=self.fsize - 8)
                    ax[c].spines['right'].set_visible(False)
                    ax[c].spines['top'].set_visible(False)
                    ax[c].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster',
                                         'cv2_' + plot_type + 's_' + str(time_beg) + 's_' + str(
                                             time_end) + 's_raw'), dpi=self.my_dpi)
        return

    def event_probability_time_period(self, df_events, window, clusters_rois, trials, plot_type, colors_session, time_beg, time_end, plot_data, print_plots):
        """"Plot event probability across trials for a certain time period, within a certain time window
        Inputs:
        df_events: (dataframe) with the calcium events information for the ROIs or clusters
        window: (int) time in seconds
        clusters_rois: (list) list with the rois for each cluster
        trials: (list) list of trials in a session
        plot_type: (str) roi or cluster
        colors_session: (list) colors for each trial in the session
        time_beg: (int) time start
        time_end: (int) time end
        plot_data: boolean
        print_plots: boolean"""
        time_diff = time_end - time_beg
        window_start = np.arange(0, time_diff, window) * self.sr
        window_end = np.arange(window, time_diff + window, window) * self.sr
        if plot_data:
            fig, ax = plt.subplots(len(clusters_rois), 1, figsize=(25, 12), tight_layout=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                for t in trials:
                    if plot_type == 'roi':
                        df_trial = df_events.loc[df_events['trial'] == t].reset_index(drop=True)
                        data_events = df_trial.iloc[time_beg * self.sr:time_end * self.sr]
                        for col in clusters_rois[c]:
                            data_prob = np.zeros(len(window_start))
                            data_prob[:] = np.nan
                            data_time = np.zeros(len(window_start))
                            data_time[:] = np.nan
                            for w in range(len(window_start)):
                                data_prob[w] = np.nansum(np.array(data_events[col].iloc[window_start[w]:window_end[w]])) / (window * self.sr)
                                if t > 1:
                                    data_time[w] = np.nanmean(np.array(data_events['time'].iloc[window_start[w]:window_end[w]]))-data_events['time'].iloc[0] + ((time_diff + 0.1) * (t - 1))
                                else:
                                    data_time[w] = np.nanmean(np.array(data_events['time'].iloc[window_start[w]:window_end[w]]))-data_events['time'].iloc[0]
                            ax[c].plot(data_time, data_prob, color=colors_session[t - 1])
                    if plot_type == 'cluster':
                        df_trial = df_events.loc[df_events['trial'] == t].reset_index(drop=True)
                        data_events = df_trial.iloc[time_beg * self.sr:time_end * self.sr]
                        data_prob = np.zeros(len(window_start))
                        data_prob[:] = np.nan
                        data_time = np.zeros(len(window_start))
                        data_time[:] = np.nan
                        for w in range(len(window_start)):
                            data_prob[w] = np.nansum(np.array(data_events['cluster'+str(c+1)].iloc[window_start[w]:window_end[w]]))/(window*self.sr)
                            if t > 1:
                                data_time[w] = np.nanmean(np.array(data_events['time'].iloc[window_start[w]:window_end[w]]))-data_events['time'].iloc[0] + ((time_diff+0.1) * (t - 1))
                            else:
                                data_time[w] = np.nanmean(np.array(data_events['time'].iloc[window_start[w]:window_end[w]]))-data_events['time'].iloc[0]
                        ax[c].plot(data_time, data_prob, color=colors_session[t - 1])
                    ax[c].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                    ax[c].set_ylabel('Event probability', fontsize=self.fsize - 8)
                    ax[c].spines['right'].set_visible(False)
                    ax[c].spines['top'].set_visible(False)
                    ax[c].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster',
                                         'event_prob_' + plot_type + 's_' + str(time_beg) + 's_' + str(
                                             time_end) + 's_raw'), dpi=self.my_dpi)
        return

    def rois_sync_time_period(self, df_events, clusters_rois, trials, time_beg, time_end, colors_session, plot_data, print_plots):
        """"Plot CV2 of events of calcium traces across trials for a certain time period.
        Inputs:
        df_events: (dataframe) with the calcium events information for the ROIs or clusters
        plot_type: (str) roi or cluster
        clusters_rois: (list) list with the rois for each cluster
        trials: (list) list of trials in a session
        time_beg: (int) time start
        time_end: (int) time end
        colors_session: (list) colors for each trial in the session
        plot_data: boolean
        print_plots: boolean"""
        # nr rois sync clusters
        if plot_data:
            fig, ax = plt.subplots(len(clusters_rois), 1, figsize=(25, 12))
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                for t in trials:
                    data_events = df_events.loc[df_events['trial'] == t].iloc[time_beg * self.sr:time_end * self.sr]
                    if t>1:
                        data_time = data_events['time']+(5.1*(t-1))
                    else:
                        data_time = data_events['time']
                    data_nr_sync = (data_events[clusters_rois[c]].sum(axis=1)/len(clusters_rois[c]))*100
                    ax[c].plot(data_time, data_nr_sync, color=colors_session[t - 1], zorder=0)
                    ax[c].scatter(np.nanmean(data_time), np.nanmean(data_nr_sync), color='white')
                    ax[c].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                    ax[c].set_ylabel('step length symmetry front paws', fontsize=self.fsize - 8)
                    ax[c].spines['right'].set_visible(False)
                    ax[c].spines['top'].set_visible(False)
                    ax[c].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster', 'rois_sync_0,9_clusters_' + str(time_beg) + 's_' + str(time_end) + 's_raw'), dpi=self.my_dpi)
        return

    def mean_activity_time_period(self, df_trace, plot_type, clusters_rois, trials, time_beg, time_end, colors_session, plot_data, print_plots):
        """"Plot mean activity across trials across trials for a certain time period.
        Inputs:
        df_trace: (dataframe) with the trace information for the ROIs or clusters
        plot_type: (str) roi or cluster
        clusters_rois: (list) list with the rois for each cluster
        trials: (list) list of trials in a session
        time_beg: (int) time start
        time_end: (int) time end
        colors_session: (list) colors for each trial in the session
        plot_data: boolean
        print_plots: boolean"""
        if plot_data:
            fig, ax = plt.subplots(len(clusters_rois), 1, figsize=(25, 12), tight_layout=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                for t in trials:
                    if plot_type == 'roi':
                        ax[c].scatter(np.ones(len(clusters_rois[c])) + t - 1 + np.random.rand(1, len(clusters_rois[c])),
                                      df_trace.loc[df_trace['trial'] == t, clusters_rois[c]].iloc[
                                      time_beg * self.sr:time_end * self.sr].mean(),
                                      color=colors_session[t - 1])
                    if plot_type == 'cluster':
                        ax[c].scatter(t, df_trace.loc[
                                             df_trace['trial'] == t, 'cluster' + str(c + 1)].iloc[
                                         time_beg * self.sr:time_end * self.sr].mean(), s=60,
                                      color=colors_session[t - 1])
                    ax[c].set_xlabel('Time (s)', fontsize=self.fsize - 8)
                    ax[c].set_ylabel('Mean activity (dFF)', fontsize=self.fsize - 8)
                    ax[c].spines['right'].set_visible(False)
                    ax[c].spines['top'].set_visible(False)
                    ax[c].tick_params(axis='both', which='major', labelsize=self.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(self.path, 'images', 'cluster',
                                         'mean_activity_' + plot_type + 's_' + str(time_beg) + 's_' + str(
                                             time_end) + 's_raw'), dpi=self.my_dpi)
        return

    def plot_cv2_stacked_traces(self, isi_cv2, traces_type, roi_plot, trials, colors_session, plot_data, print_plots):
        """Plot stacked traces with CV2 information for a single ROI or a cluster
        Inputs:
        isi_cv2: (dataframe) with isi cv2 information
        traces_type: (str) raw or deconv
        roi_plot: (int) roi or cluster to plot
        trials: (list) list of trials in a session
        colors_session: (list) colors for each trial in the session
        plot_data: boolean
        print_plots: boolean"""
        int_find = ''.join(x for x in isi_cv2.columns[2] if x.isdigit())
        int_find_idx = isi_cv2.columns[2].find(int_find)
        if isi_cv2.columns[2][:int_find_idx] == 'ROI':
            df_type = 'ROI'
        else:
            df_type = 'cluster'
        idx_nr = df_type + str(roi_plot)
        if plot_data:
            fig, ax = plt.subplots(figsize=(15, 20), tight_layout=True)
            count_t = len(trials)
            for count_c, t in enumerate(trials):
                isi_cv2_trial = isi_cv2.loc[(isi_cv2['trial'] == t) & (isi_cv2['roi'] == idx_nr), 'cv2']
                time_trial = isi_cv2.loc[(isi_cv2['trial'] == t) & (isi_cv2['roi'] == idx_nr), 'time']
                ax.plot(time_trial, isi_cv2_trial + (count_t * 2), color=colors_session[count_c])
                ax.set_yticks(trials * 2)
                ax.set_yticklabels(map(str, trials[::-1]))
                ax.set_xlabel('Time (s)', fontsize=self.fsize - 2)
                ax.set_ylabel('Trials', fontsize=self.fsize - 2)
                ax.set_title('CV2 for ' + df_type + ' ' + str(roi_plot), fontsize=self.fsize - 2)
                ax.spines['left'].set_visible(False)
                plt.xticks(fontsize=self.fsize - 2)
                plt.yticks(fontsize=self.fsize - 2)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                count_t -= 1
            if print_plots:
                if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type)):
                    os.mkdir(os.path.join(self.path, 'images', 'events', traces_type))
                if df_type == 'ROI':
                    if not os.path.exists(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'events', traces_type, 'ROI' + str(roi_plot),
                                             'cv2_stacked_traces_' + traces_type), dpi=self.my_dpi)
                if df_type == 'cluster':
                    if not os.path.exists(
                            os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot))):
                        os.mkdir(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot)))
                    plt.savefig(os.path.join(self.path, 'images', 'cluster', traces_type, 'Cluster' + str(roi_plot),
                                             'Cluster' + str(roi_plot) + '_cv2_stacked_traces_' + traces_type),
                                dpi=self.my_dpi)
        return

    @staticmethod
    def get_contour_cluster(coord_ext, idx_roi_cluster_ordered, cluster):
        """Get the edges around a group of ROIs (a cluster for example).
        Outputs x andy coordinates of the edges.
        Inputs:
         coord_ext: (list) coordinates for each ROI
         idx_roi_cluster_ordered: for each ROI says the cluster index (starts at 1)
         cluster: cluster number to plot the edges around"""
        def alpha_shape(points, only_outer, alpha):
            """Compute the alpha shape (concave hull) of a set
            of points.
            param points: Iterable container of points.
            param alpha: alpha value to influence the
                gooeyness of the border. Smaller numbers
                don't fall inward as much as larger numbers.
                Too large, and you lose everything!
            only_outer: boolean value to specify if we keep only the outer border
               or also inner edges."""
            assert points.shape[0] > 3, "Need at least four points"
            def add_edge(edges, i, j):
                """Add an edge between the i-th and j-th points,
                if not in the list already"""
                if (i, j) in edges or (j, i) in edges:  # already added
                    assert (j, i) in edges, "Can't go twice over same directed edge right?"
                    if only_outer:
                        edges.remove((j, i))  # if both neighboring triangles are in shape, it's not a boundary edge
                    return
                edges.add((i, j))
            tri = Delaunay(points)
            edges = set()
            for ia, ib, ic in tri.vertices:
                pa = points[ia]
                pb = points[ib]
                pc = points[ic]
                a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)  # Lengths of sides of triangle
                b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
                c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
                s = (a + b + c) / 2.0  # Semiperimeter of triangle
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # Area of triangle by Heron's formula
                circum_r = np.divide(a * b * c, (4.0 * area))
                if circum_r < (1.0 / alpha):  # Here's the radius filter.
                    add_edge(edges, ia, ib)
                    add_edge(edges, ib, ic)
                    add_edge(edges, ic, ia)
            return edges
        idx_cluster = np.where(idx_roi_cluster_ordered == cluster)[0]
        rois_coordinates_cluster = np.array(list(chain.from_iterable(coord_ext[idx_cluster])))
        edges = alpha_shape(rois_coordinates_cluster, 1, alpha=0.4)
        plt.figure()
        plt.scatter(rois_coordinates_cluster[:, 0], rois_coordinates_cluster[:, 1], color='blue')
        edges_coordinates = []
        for i, j in edges:
            edges_coordinates.append([rois_coordinates_cluster[[i, j], 0], rois_coordinates_cluster[[i, j], 1]])
        edges_coordinates_array = np.array(edges_coordinates)
        plt.scatter(edges_coordinates_array[:, 0], edges_coordinates_array[:, 1], color='black')
        return edges_coordinates_array

    @staticmethod
    def sort_rois_clust(df_events, clusters_rois):
        if len(clusters_rois) > 1:
            clusters_rois_flat = np.transpose(sum(clusters_rois, []))
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
            cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
            df_events_sorted = df_events[clusters_rois_flat]
        else:
            df_events_sorted = df_events
            cluster_transition_idx = np.array([0])
        return df_events_sorted, cluster_transition_idx

    @staticmethod
    def sta(df_events, variable, bcam_time, window, trials):
        '''Compute spike-triggered average for each ROI
        Inputs:
            - df_events: dataframe of events
            - variable: 1D array of the variable
            - bcam_time: timestamps of behavior
            - window: peri-event epoch (samples)
            - trials: 1D array of trial numbers
        '''
        signal_chunks_allrois = []
        sta_allrois = []
        for n in range(2, df_events.shape[1]):
            sta = np.empty((0, len(window)))
            signal_chunks_tr = []
            for tr_idx, tr in enumerate(trials):
                signal_chunks = np.empty((0, len(window)))
                df_tr = df_events[df_events['trial']==tr]
                events_idx = np.array(df_tr.index[df_tr.iloc[:, n] == 1])
                events_ts = df_tr['time'].loc[events_idx].values
                matching_ts_idx = [np.abs(bcam_time[tr_idx] - ts).argmin() for ts in events_ts]
                # Extract traces around each event for one ROI
                for i in matching_ts_idx:
                    if i + window[0] >= 0 and i + window[-1] < len(variable[tr_idx]):
                        extracted_signal = variable[tr_idx][i + window[0]:i + window[-1] + 1]
                        # extracted_signal = (extracted_signal - np.nanmean(extracted_signal))/np.std(extracted_signal)
                        # List of raw traces for one ROI 'n' and trial 'tr'
                        signal_chunks = np.vstack((signal_chunks, extracted_signal))
                signal_chunks_tr.append(signal_chunks) # Array of traces for one ROI all trials
            # Compute STA by trial for one ROI
            sta = np.vstack([np.nanmean(signal_chunks_tr[tr_idx], axis=0) for tr_idx, _ in enumerate(trials)])
            # List of raw traces for each ROI whole session
            signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis = 0))
            # STA by trial for all ROIs
            sta_allrois.append(sta)
        return sta_allrois, signal_chunks_allrois

    @staticmethod
    def shuffle_spikes_ts(df_events, iter_n):
        ''' Shuffle timestamps of events for multiple iterations. This code shuffle the ISIs of each trial.
        Inputs:
            - df_events: dataframe of events for multiple ROIs. Column 'time' contains timestamps, column 'trial' indicates trial ID
            - iter_n: number of shuffling iterations
        '''
        trials = np.unique(df_events['trial'])
        shuffled_spikes_ts_allrois = []
        for n in range(2, df_events.shape[1]):
            shuffled_spikes_ts = []
            # Find all timestamps of events for all trials for ROI 'n'
            for tr in trials:
                df_events_tr = df_events[df_events.trial == tr] # Extract trial 'tr'
                events_idx = np.array(df_events_tr.index[df_events_tr.iloc[:, n] == 1]) # Find indexes of events for ROI 'n' and trial 'tr'
                spikes_ts_tr = np.array(df_events_tr.time[events_idx])  # Find timestamps of events for ROI 'n' and trial 'tr'
                isi = np.diff(spikes_ts_tr) # Compute ISI
                for _ in range(iter_n):
                    shuffled_spikes_ts_tr = []
                    np.random.shuffle(isi) # Shuffle ISI
                    shuffled_spikes_ts_tr = np.insert(np.cumsum(isi), 0, 0) # Find new timestamps
                shuffled_spikes_ts.append(shuffled_spikes_ts_tr)
            shuffled_spikes_ts_allrois.append(shuffled_spikes_ts)
        return shuffled_spikes_ts_allrois

    @staticmethod
    def sta_shuffled(spikes_ts, variable, bcam_time, window, trials):
        '''Compute spike-triggered average for single ROIs with shuffled event timings.
        Inputs:
            - spikes_ts = nested lists of spikes timestamps by trial for each neuron
            - variable: 1D array of the variable
            - bcam_time: timestamps of behavior
            - window: peri-event epoch (samples)
        '''
        signal_chunks_allrois = []
        sta_allrois = []
        for n in range(len(spikes_ts)):
            sta = np.empty((0, len(window)))
            signal_chunks_tr = []
            for tr_idx, tr in enumerate(trials):
                signal_chunks = np.empty((0, len(window)))
                events_ts = np.array(spikes_ts[n][tr_idx]) # Find timestamps of events for ROI 'n'
                matching_ts_idx = [np.abs(bcam_time[tr_idx] - ts).argmin() for ts in events_ts] # Find timestamps of behavior matching the ones of events
                # Extract traces around each event for one ROI
                for i in matching_ts_idx:
                    if i + window[0] >= 0 and i + window[-1] < len(variable[tr_idx]):
                        extracted_signal = variable[tr_idx][i + window[0]:i + window[-1] + 1]
                        # extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
                        signal_chunks = np.vstack((signal_chunks, extracted_signal))
                signal_chunks_tr.append(signal_chunks) # Array of traces for one ROI by trial
            # Compute STA by trial for one ROI
            for tr_idx, _ in enumerate(trials):
                sta_trial = np.nanmean(signal_chunks_tr[tr_idx], axis = 0)
                sta = np.vstack((sta, sta_trial))
            # STA all rois
            signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis = 0)) # List of raw traces for each ROI whole session
            sta_allrois.append(sta)
        return sta_allrois

    def paw_diff(self, tracks, p1, p2):
        ''' Compute displacement or phase difference between two paws.
        Inputs:
            - tracks: list of limbs coordinates for each trial
            - p1: reference paw (FR=0, HR=1, FL=2, HL=3)
            - P2: secondary paw
        '''
        paw_difference = []
        for tr in range(len(tracks)):
            ref = (tracks[tr][0, p1, :]*self.pixel_to_mm_behavior) - (np.nanmean(tracks[tr][0, :4, :], axis=0)*self.pixel_to_mm_behavior)
            sec = (tracks[tr][0, p2, :]*self.pixel_to_mm_behavior) - (np.nanmean(tracks[tr][0, :4, :], axis=0)*self.pixel_to_mm_behavior)
            paw_difference.append(ref - sec)
        return paw_difference

    def get_coordinates_cluster(self, centroid_ext, fov_coord, idx_roi_cluster_ordered):
        """Get the coordinates of the clusters based on the mean of the centroids.
        Put the coordinates in a global scale (based on histology)
        Inputs:
            fov_coord: coordinates of the center of the FOV (based on histology)
            idx_roi_cluster_ordered: list of cluster if for each ROI index
            centroid_ext: list of coordinates of ROIs centroids"""
        centroid_cluster_mean = np.zeros((len(np.unique(idx_roi_cluster_ordered)), 2))
        for count_i, i in enumerate(np.unique(idx_roi_cluster_ordered)):
            cluster_idx = np.where(idx_roi_cluster_ordered == i)[0]
            centroid_cluster = np.zeros((len(cluster_idx), 2))
            for count_c, c in enumerate(cluster_idx):
                centroid_cluster[count_c, :] = centroid_ext[c]
            centroid_mean = np.nanmean(centroid_cluster, axis=0)
            centroid_cluster_mean[count_i, 0] = -centroid_mean[0]  # because we are in the negative area of bregma
            centroid_cluster_mean[count_i, 1] = centroid_mean[1]
        fov_corner = np.array([fov_coord[0] + 0.5, fov_coord[1] - 0.5])
        centroid_cluster_dist_corner = (centroid_cluster_mean * 0.001) + fov_corner
        if not os.path.exists(self.path + 'processed files'):
            os.mkdir(self.path + 'processed files')
        np.save(os.path.join(self.path, 'processed files', 'cluster_coords.npy'), centroid_cluster_dist_corner)
        return centroid_cluster_dist_corner

    @staticmethod
    def get_peakamp_latency(data, xaxis):
        """Get the last peak time before 0 and amplitude of a sinusoidal curve such as difference between paws
        Inputs:
            data: vector with sinusoid (e.g. difference between paws)
            xaxis: vector with time of data points"""
        idx_time0 = np.where(xaxis == 0)[0][0]
        data_filt = sp.medfilt(data - np.nanmean(data), 11)
        peaks_idx = sp.find_peaks(data_filt, width=10)[0]
        peaks_idx_before0 = peaks_idx[np.where(peaks_idx < idx_time0)[0]]
        if len(peaks_idx_before0)>0:
            idx_closest_peak = np.argmax(peaks_idx_before0)
            amp = data[peaks_idx[idx_closest_peak]]
            latency = xaxis[peaks_idx[idx_closest_peak]]
        else:
            amp = np.nan
            latency = np.nan
        return amp, latency

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
