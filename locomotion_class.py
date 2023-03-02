#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:23:29 2020

@author: anagoncalves
"""

import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from itertools import chain
from math import pi
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import resample
import nptdms as tdms

#to call class
#os.chdir('/Users/anagoncalves/Documents/PhD/Code/Miniscope pipeline/')
#import locomotion_class
#loco = locomotion_class.loco_class(path)
#and then loco.(function name)

#to check class documentation (after importing class)
#help(loco.read_h5)

#to see all methods in class
#dir(loco)

class loco_class:
    
    def __init__(self,path):
        self.path = path
        self.delim = self.path[-1]
        path_split = self.path.split(self.delim)
        self.experiment = path_split[-3]
        self.pixel_to_mm = 1/1.955 #dana's setup
        self.sr = 330 #sampling rate of behavior camera for treadmill
        self.sr_F = 30
        self.my_dpi = 96 #resolution for plotting
        self.floor = 152*self.pixel_to_mm #Dana's setup
        # floor = 155*pixel_to_mm #Jovin's setup    

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
    def param_continuous_sym(param_trials, st_strides_trials, trials, p1, p2, sym, remove_nan):
        """Compute a parameter across all trials and correspondent time, if wanted compute symmetry using another paw
        Inputs:
        param_trials. list with the param values for each trial all strides
        st_strides_trials: list with strides for all trials
        trials: trial list
        p1: paw to compute the parameter (FR, HR, FL, HL)
        p2: paw to relate to p1, to compute the parameter as symmetry (FR, HR, FL, HL)
        sym: boolean
        remove_nan: boolean"""
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
        param_all = []
        param_all_time = []
        cumulative_idx = []
        for count_t, t in enumerate(trials):
            trial_index = np.where(trials == t)[0][0]
            sl_p1 = param_trials[trial_index][p1_idx]
            sl_p2 = param_trials[trial_index][p2_idx]
            strides_p1 = st_strides_trials[trial_index][p1_idx]
            strides_p2 = st_strides_trials[trial_index][p2_idx]
            # get events between the strides
            param = np.zeros(np.shape(strides_p1)[0])
            param[:] = np.nan
            param_time = np.zeros(np.shape(strides_p1)[0])
            param_time[:] = np.nan
            for s in range(np.shape(strides_p1)[0]):
                stride_contra = \
                np.where((strides_p2[:, 0, 0] > strides_p1[s, 0, 0]) & (strides_p2[:, 0, 0] < strides_p1[s, 1, 0]))[0]
                if sym:
                    if len(stride_contra) == 1:  # one event in the stride
                        param[s] = sl_p1[s] - sl_p2[stride_contra]
                    if len(stride_contra) > 1:  # more than two events in a stride
                        param[s] = sl_p1[s] - np.nanmean(sl_p2[stride_contra])
                else:
                    param[s] = sl_p1[s]
                param_time[s] = strides_p1[s, 0, 0] / 1000
                if count_t == 0 and s == 0:
                    cumulative_idx.append(1)
                else:
                    cumulative_idx.append(cumulative_idx[-1] + 1)
            param_all.extend(param)
            if count_t > 0:  # cumulative time
                param_all_time.extend(param_time + param_all_time[-1])
            else:
                param_all_time.extend(param_time)
        if remove_nan:
            param_all_notnan = np.array(param_all)[~np.isnan(param_all)]
            param_all_time_notnan = np.array(param_all_time)[~np.isnan(param_all)]
            cumulative_idx_array = np.array(cumulative_idx)[~np.isnan(param_all)]
        else:
            param_all_notnan = np.array(param_all)
            param_all_time_notnan = np.array(param_all_time)
            cumulative_idx_array = np.array(cumulative_idx)
        return cumulative_idx_array, param_all_time_notnan, param_all_notnan

    def get_session_id(self):
        session = int(self.path.split(self.delim)[-2].split()[-2][1:])
        return session
        
    def read_h5(self,filename,threshold,frames_loco):
        """Function to read output of DLC (as h5 file) and output the matrices final_tracks
        tracks_tail, joints_wrist, joints_elbow, ear and bodycenter for locomotion analysis
        Inputs:
            filename (str)
            threshold: to consider valid tracking
            frames_loco: list with the number of frames per trial removed in the Miniscope video (after triggering)"""
        track_df = pd.read_hdf(self.path+filename,'df_with_missing')
        likelihood_df = track_df[track_df.columns[2::3]]
        #tracks with likelihood below 0.9 are NaN
        xcol = np.arange(0,track_df.shape[1],3)
        trackcols = np.concatenate((xcol.reshape((-1,1)),xcol.reshape((-1,1))+1),axis=1) #array with indices of the tracks (x,y all views)
        for l in range(np.shape(trackcols)[0]):
            for c in range(np.shape(trackcols)[1]):
                idx_to_nan = np.array(likelihood_df.iloc[:,l]<threshold)
                track_df.iloc[idx_to_nan,trackcols[l,c]] = np.nan        
        #build final_tracks like variable: [x y z xside; paws nose; frames]
        #initialize arrays
        final_tracks = np.zeros((4,5,track_df.shape[0]))
        tracks_tail = np.zeros((4,15,track_df.shape[0]))
        joints_wrist = np.zeros((4,track_df.shape[0],2))
        joints_elbow = np.zeros((4,track_df.shape[0],2))
        ear = np.zeros((4,track_df.shape[0]))
        bodycenter = np.zeros((4,track_df.shape[0]))
        #fill with tracks
        final_tracks[0,:,:] = np.transpose(track_df.iloc[:,[30,36,33,39,3]]) #x
        final_tracks[1,:,:] = np.transpose(track_df.iloc[:,[31,37,34,40,4]]) #y
        final_tracks[3,:,:] = np.transpose(track_df.iloc[:,[19,25,22,28,1]]) #z
        final_tracks[2,:,:] = np.transpose(track_df.iloc[:,[18,24,21,27,0]]) #x side
        tracks_tail[0,:,:] = np.transpose(track_df.iloc[:,np.arange(111,track_df.shape[1],3)]) #x
        tracks_tail[1,:,:] = np.transpose(track_df.iloc[:,np.arange(112,track_df.shape[1],3)]) #y
        tracks_tail[2,:,:] = np.transpose(track_df.iloc[:,np.arange(67,110,3)]) #side z
        tracks_tail[3,:,:] = np.transpose(track_df.iloc[:,np.arange(66,110,3)]) #side x
        joints_wrist[:,:,0] = np.transpose(track_df.iloc[:,[42,48,45,51]]) #paws, frames x,z 
        joints_wrist[:,:,1] = np.transpose(track_df.iloc[:,[43,49,46,52]])
        joints_elbow[:,:,0] = np.transpose(track_df.iloc[:,[54,60,57,63]]) #x
        joints_elbow[:,:,1] = np.transpose(track_df.iloc[:,[55,61,58,64]])
        ear = np.array(np.transpose(track_df.iloc[:,[9,10,7,6]])) #x y z, sidex
        bodycenter = np.array(np.transpose(track_df.iloc[:,[15,16,13,12]])) #x y z, sidex
        if frames_loco>0:
            final_tracks = final_tracks[:,:,frames_loco:]
            tracks_tail = tracks_tail[:,:,frames_loco:]
            joints_wrist = joints_wrist[:,frames_loco:,:]
            joints_elbow = joints_elbow[:,frames_loco:,:]
            ear = ear[:,frames_loco:]
            bodycenter = bodycenter[:,frames_loco:]
        return final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter

    def compute_joint_angles(self,final_tracks,tracks_tail,joints_elbow,joints_wrist):
        """Compute body axis angles, tail axis angles and wrist angles"""
        body_axis_xy = np.radians(np.sin((final_tracks[0,4,:]-tracks_tail[0,0,:])/(final_tracks[1,4,:]-tracks_tail[1,0,:])))
        body_axis_xz = np.radians(np.sin((final_tracks[2,4,:]-tracks_tail[2,0,:])/(final_tracks[3,4,:]-tracks_tail[3,0,:])))
        tail_axis_xy = np.radians(np.sin((tracks_tail[0,0,:]-tracks_tail[0,:,:])/(tracks_tail[1,0,:]-tracks_tail[1,:,:])))
        tail_axis_xz = np.radians(np.sin((tracks_tail[2,0,:]-tracks_tail[2,:,:])/(tracks_tail[3,0,:]-tracks_tail[3,:,:])))
        for p in range(4):
            elxz = joints_elbow[p,:,:]-joints_wrist[p,:,:]
            toexz = np.concatenate((final_tracks[2,p,:].reshape(-1,1),final_tracks[3,p,:].reshape(-1,1)),axis=1)-joints_wrist[p,:,:]
        wrist_angles = np.arctan(elxz[:,0]/elxz[:,1])+np.arctan(toexz[:,0]/toexz[:,1])
        return body_axis_xy, body_axis_xz, tail_axis_xy, tail_axis_xz, wrist_angles
    
    def get_sw_st_matrices(self,final_tracks,exclusion):
        """Computes swing and stance points of a trial from x axis of the bottom view tracking.
        It excludes strides based on a distribution of some gait parameters
        Input: final_tracks (4x5xframes)
               exclusion - boolean to exclude strides 
        Output: st_strides_mat (stridesx2x5)
                sw_pts_mat (stridesx1x5)
        columns: st/sw in ms; x(st/sw); y(st/sw); z(st/sw); st idx/sw idx
        2 middle columns for beginning and end of stride"""
        #convert to mm and interpolate NaNs
        X = final_tracks[0,:,:]*self.pixel_to_mm
        Y = final_tracks[1,:,:]*self.pixel_to_mm
        Z = final_tracks[3,:,:]*self.pixel_to_mm
        X_interp = self.inpaint_nans(X)
        Y_interp = self.inpaint_nans(Y)
        Z_interp = self.inpaint_nans(Z)
        #peak detection
        swing_mat = []
        stance_mat = []
        for p in range(4):
            data_filt = savgol_filter(X[p,:], window_length = 11, polyorder = 1)
            peaks = find_peaks(data_filt)
            throughs = find_peaks(-data_filt)
            stance = peaks[0]
            swing = throughs[0]
            swing_mat.append(np.column_stack((swing/self.sr*1000,X_interp[p,swing],Y_interp[p,swing],Z_interp[p,swing],swing)))
            stance_mat.append(np.column_stack((stance/self.sr*1000,X_interp[p,stance],Y_interp[p,stance],Z_interp[p,stance],stance)))
        #stride sorting
        st_strides_mat = []
        sw_pts_mat = []
        for p in range(4):
            st_strides = np.zeros((len(stance_mat[p]),2,5))
            sw_pts = np.zeros((len(stance_mat[p]),1,5))
            for s in range(np.shape(stance_mat[p])[0]-1):
                #define stride from st to st onset
                st_strides[s,:,0] = [stance_mat[p][s,0],stance_mat[p][s+1,0]-1]
                st_strides[s,:,1] = [stance_mat[p][s,1],stance_mat[p][s+1,1]-1]
                st_strides[s,:,2] = [stance_mat[p][s,2],stance_mat[p][s+1,2]-1]
                st_strides[s,:,3] = [stance_mat[p][s,3],stance_mat[p][s+1,3]-1]
                st_strides[s,:,4] = [stance_mat[p][s,4],stance_mat[p][s+1,4]-1]
                #find swing point between those st onsets
                sw_pts[s,:,0] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),0][0]
                sw_pts[s,:,1] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),1][0]
                sw_pts[s,:,2] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),2][0]
                sw_pts[s,:,3] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),3][0]
                sw_pts[s,:,4] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),4][0]
            st_strides_mat.append(st_strides)
            sw_pts_mat.append(sw_pts)            
        if exclusion:
            #compute some gait parameters
            stride_duration_mat = []
            swing_duration_mat = []
            stance_duration_mat = []
            swing_length_mat = []
            swing_velocity_mat = []
            for p in range(4):
                stride_duration = st_strides_mat[p][:,1,0]-st_strides_mat[p][:,0,0]
                swing_duration = st_strides_mat[p][:,1,0]-sw_pts_mat[p][:,0,0]
                stance_duration = sw_pts_mat[p][:,0,0]-st_strides_mat[p][:,0,0]
                swing_length = X_interp[p,st_strides_mat[p][:,1,4].astype(int)]-X_interp[p,sw_pts_mat[p][:,0,4].astype(int)]
                swing_velocity = swing_length/swing_duration
                stride_duration_mat.append(stride_duration)
                swing_duration_mat.append(swing_duration)
                stance_duration_mat.append(stance_duration)
                swing_length_mat.append(swing_length)
                swing_velocity_mat.append(swing_velocity)
            #exclude strides
            exclusion_paws = []
            for p in range(4):               
                exclusion_mat = [np.where(stride_duration_mat[p]>600)[0],np.where(stride_duration_mat[p]<75)[0],np.where(swing_duration_mat[p]>275)[0],np.where(swing_duration_mat[p]<25)[0],np.where(swing_length_mat[p]>90)[0],np.where(swing_length_mat[p]<10)[0],np.where(stance_duration_mat[p]>550)[0],np.where(stance_duration_mat[p]<30)[0],np.where(swing_velocity_mat[p]<0)[0]]
                exclusion_paws.append(np.unique(list(chain.from_iterable(exclusion_mat))))
            #make excluded strides nan
            st_strides_mat_new = []
            sw_pts_mat_new = []
            for p in range(4):
                if len(exclusion_paws[p])>0:
                    st_strides_mat[p][exclusion_paws[p],:,:] = np.nan
                    st_strides_excl = st_strides_mat[p]
                    #remove nans
                    st_strides_excl = st_strides_excl[~np.isnan(st_strides_excl[:,0,0]),:,:]
                    st_strides_mat_new.append(st_strides_excl)
                    sw_pts_mat[p][exclusion_paws[p],:,:] = np.nan
                    sw_pts_excl = sw_pts_mat[p]
                    #remove nans
                    sw_pts_excl = sw_pts_excl[~np.isnan(sw_pts_excl[:,0,0]),:,:]
                    sw_pts_mat_new.append(sw_pts_excl)
                else:
                    st_strides_mat_new.append(st_strides_mat[p])
                    sw_pts_mat_new.append(sw_pts_mat[p])
        else:
            st_strides_mat_new = st_strides_mat
            sw_pts_mat_new = sw_pts_mat
        return st_strides_mat_new, sw_pts_mat_new

    @staticmethod
    def final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, phase_type):
        """Transform the final tracks trials in phase
        Inputs:
        final_tracks_trials: (list) final tracks for each trial
        trials: (list) trials in the session
        st_strides_trials: (list) with stride structure st to st
        sw_strides_trials: (list) with stride structure sw to sw
        phase_type: (str) st-st or st-sw-st
        """
        final_tracks_trials_phase = []
        for count_t, t in enumerate(trials):
            final_tracks_phase = np.zeros(np.shape(final_tracks_trials[count_t]))
            final_tracks_phase[:] = np.nan
            for a in range(4):
                for p in range(5):
                    if p == 4:  # if nose, do the phase in relation to the FR paw
                        p = 0
                    excursion_phase = np.zeros(len(final_tracks_trials[count_t][0, p, :]))
                    excursion_phase[:] = np.nan
                    for s in range(len(st_strides_trials[count_t][p][:, 0, -1])):
                        st_on = np.int64(st_strides_trials[count_t][p][s, 0, -1])
                        sw_on = np.int64(sw_strides_trials[count_t][p][s, 0, -1])
                        st_off = np.int64(st_strides_trials[count_t][p][s, 1, -1])
                        if phase_type == 'st-sw-st':
                            nr_st = len(final_tracks_trials[count_t][0, p, st_on:sw_on])
                            nr_sw = len(final_tracks_trials[count_t][0, p, sw_on:st_off])
                            excursion_phase[st_on - 1:sw_on] = np.linspace(0, 0.5, nr_st + 1)
                            excursion_phase[sw_on - 1:st_off] = np.linspace(0.5, 1, nr_sw + 1)
                            excursion_phase[st_off] = 0  # put it there -1
                        if phase_type == 'st-st':
                            nr_st = len(final_tracks_trials[count_t][0, p, st_on:st_off])
                            excursion_phase[st_on - 1:st_off] = np.linspace(0, 1, nr_st + 1)
                            excursion_phase[st_off] = 0  # put it there -1
                    final_tracks_phase[a, p, :] = excursion_phase
            final_tracks_trials_phase.append(final_tracks_phase)
        return final_tracks_trials_phase

    def get_sw_st_matrices_JR(self,final_tracks,dict_swst,exclusion):
        """Computes swing and stance points of a trial from x axis of the bottom view tracking.
        It excludes strides based on a distribution of some gait parameters
        Input: final_tracks (4x5xframes)
                dict_swst (dictionary with swing and stance info)              
                exclusion - boolean to exclude strides 
        Output: st_strides_mat (stridesx2x5)
                sw_pts_mat (stridesx1x5)
        columns: st/sw in ms; x(st/sw); y(st/sw); z(st/sw); st idx/sw idx
        2 middle columns for beginning and end of stride"""
        #convert to mm and interpolate NaNs
        X = final_tracks[0,:,:]*self.pixel_to_mm
        Y = final_tracks[1,:,:]*self.pixel_to_mm
        Z = final_tracks[3,:,:]*self.pixel_to_mm
        X_interp = self.inpaint_nans(X)
        Y_interp = self.inpaint_nans(Y)
        Z_interp = self.inpaint_nans(Z)
        #swst detection from JR
        paws = ['FR','HR','FL','HL']
        swing = []
        stance = []
        for p in paws:
            sw_pts = dict_swst[p]['Swing Onset F val.']
            swing.append(sw_pts)
            st_pts = dict_swst[p]['Stance Onset F val.']
            stance.append(st_pts)
        swing_mat = []
        stance_mat = []
        for p in range(4):
            swing_mat.append(np.column_stack((swing[p]/self.sr*1000,X_interp[p,swing[p]],Y_interp[p,swing[p]],Z_interp[p,swing[p]],swing[p])))
            stance_mat.append(np.column_stack((stance[p]/self.sr*1000,X_interp[p,stance[p]],Y_interp[p,stance[p]],Z_interp[p,stance[p]],stance[p])))
        #stride sorting
        st_strides_mat = []
        sw_pts_mat = []
        for p in range(4):
            st_strides = np.zeros((len(stance_mat[p])-1,2,5))
            sw_pts = np.zeros((len(stance_mat[p])-1,1,5))
            for s in range(np.shape(stance_mat[p])[0]-1):
                #define stride from st to st onset
                st_strides[s,:,0] = [stance_mat[p][s,0],stance_mat[p][s+1,0]-1]
                st_strides[s,:,1] = [stance_mat[p][s,1],stance_mat[p][s+1,1]-1]
                st_strides[s,:,2] = [stance_mat[p][s,2],stance_mat[p][s+1,2]-1]
                st_strides[s,:,3] = [stance_mat[p][s,3],stance_mat[p][s+1,3]-1]
                st_strides[s,:,4] = [stance_mat[p][s,4],stance_mat[p][s+1,4]-1]
                #find swing point between those st onsets
                sw_pts[s,:,0] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),0][0]
                sw_pts[s,:,1] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),1][0]
                sw_pts[s,:,2] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),2][0]
                sw_pts[s,:,3] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),3][0]
                sw_pts[s,:,4] = swing_mat[p][(swing_mat[p][:,4]>=stance_mat[p][s,4]) &  (swing_mat[p][:,4]<=stance_mat[p][s+1,4]),4][0]
            st_strides_mat.append(st_strides)
            sw_pts_mat.append(sw_pts)            
        if exclusion:
            #compute some gait parameters
            stride_duration_mat = []
            swing_duration_mat = []
            stance_duration_mat = []
            swing_length_mat = []
            swing_velocity_mat = []
            for p in range(4):
                stride_duration = st_strides_mat[p][:,1,0]-st_strides_mat[p][:,0,0]
                swing_duration = st_strides_mat[p][:,1,0]-sw_pts_mat[p][:,0,0]
                stance_duration = sw_pts_mat[p][:,0,0]-st_strides_mat[p][:,0,0]
                swing_length = X_interp[p,st_strides_mat[p][:,1,4].astype(int)]-X_interp[p,sw_pts_mat[p][:,0,4].astype(int)]
                swing_velocity = swing_length/swing_duration
                stride_duration_mat.append(stride_duration)
                swing_duration_mat.append(swing_duration)
                stance_duration_mat.append(stance_duration)
                swing_length_mat.append(swing_length)
                swing_velocity_mat.append(swing_velocity)
            #exclude strides
            exclusion_paws = []
            for p in range(4):               
                exclusion_mat = [np.where(stride_duration_mat[p]>600)[0],np.where(stride_duration_mat[p]<75)[0],np.where(swing_duration_mat[p]>275)[0],np.where(swing_duration_mat[p]<25)[0],np.where(swing_length_mat[p]>90)[0],np.where(swing_length_mat[p]<10)[0],np.where(stance_duration_mat[p]>550)[0],np.where(stance_duration_mat[p]<30)[0],np.where(swing_velocity_mat[p]<0)[0]]
                exclusion_paws.append(np.unique(list(chain.from_iterable(exclusion_mat))))
            #make excluded strides nan
            st_strides_mat_new = []
            sw_pts_mat_new = []
            for p in range(4):
                if len(exclusion_paws[p])>0:
                    st_strides_mat[p][exclusion_paws[p],:,:] = np.nan
                    st_strides_excl = st_strides_mat[p]
                    #remove nans
                    st_strides_excl = st_strides_excl[~np.isnan(st_strides_excl[:,0,0]),:,:]
                    st_strides_mat_new.append(st_strides_excl)
                    sw_pts_mat[p][exclusion_paws[p],:,:] = np.nan
                    sw_pts_excl = sw_pts_mat[p]
                    #remove nans
                    sw_pts_excl = sw_pts_excl[~np.isnan(sw_pts_excl[:,0,0]),:,:]
                    sw_pts_mat_new.append(sw_pts_excl)
                else:
                    st_strides_mat_new.append(st_strides_mat[p])
                    sw_pts_mat_new.append(sw_pts_mat[p])
        else:
            st_strides_mat_new = st_strides_mat
            sw_pts_mat_new = sw_pts_mat
        return st_strides_mat_new, sw_pts_mat_new
    
    def get_sw_st_pts(self,st_strides_mat,sw_pts_mat):
        """Gets swing and stance points from the respective matrices
        Input: st_strides_mat (stridesx2x5)
               sw_pts_mat (stridesx1x5)"""
        st_pts = []
        sw_pts = []
        for p in range(4):
            st_pts.append(st_strides_mat[p][:,0,4])
            sw_pts.append(sw_pts_mat[p][:,0,4])
        return st_pts, sw_pts
    
    def get_paws_rel(self,final_tracks,axis):
        """Gets paw positions relative to any of the body center axis or to the nose
        Input:  final_tracks (4x5xframes)
                axis - 'X','Y','Z','nose'"""
        #compute paws relative to body or nose
        paws_rel = []
        for p in range(5):
            if axis == 'X':
                X = final_tracks[0,:,:]*self.pixel_to_mm
                X_interp = self.inpaint_nans(X)
                paws_rel.append(X_interp[p,:]-np.nanmean(X_interp[:4,:],axis=0))
            if axis == 'Y':
                Y = final_tracks[1,:,:]*self.pixel_to_mm
                Y_interp = self.inpaint_nans(Y)
                paws_rel.append(Y_interp[p,:]-np.nanmean(Y_interp[:4,:],axis=0))
            if axis == 'Z':
                Z = final_tracks[3,:,:]*self.pixel_to_mm
                Z_interp = self.inpaint_nans(Z)
                paws_rel.append(Z_interp[p,:]-np.nanmean(Z_interp[:4,:],axis=0))
            if axis == 'nose':
                X = final_tracks[0,:,:]*self.pixel_to_mm
                X_interp = self.inpaint_nans(X)
                paws_rel.append(X_interp[p,:]-X_interp[4,:])
        return paws_rel

    def stride_time(self,st_strides_all, frame_time, paw):
        """Gets the miniscope timestamps within one stride for a designated paw
        Input:  st_strides_all (list of stridesx2x5 per paw, one for each trial)
                frame_time (list with miniscope timestamps for each trial)
                paw (string)"""
        strides_time = []
        for t in range(len(frame_time)):
            frames_strides = []
            if paw == 'FR':
                for s in range(np.shape(st_strides_all[t][0])[0]):
                    frames_strides.append(frame_time[t][np.where((frame_time[t]>=st_strides_all[t][0][s,0,0]/1000)&(frame_time[t]<=st_strides_all[t][0][s,1,0]/1000))])
            if paw == 'HR':
                for s in range(np.shape(st_strides_all[t][1])[0]):
                    frames_strides.append(frame_time[t][np.where((frame_time[t]>=st_strides_all[t][1][s,0,0]/1000)&(frame_time[t]<=st_strides_all[t][1][s,1,0]/1000))])
            if paw == 'FL':
                for s in range(np.shape(st_strides_all[t][2])[0]):
                    frames_strides.append(frame_time[t][np.where((frame_time[t]>=st_strides_all[t][2][s,0,0]/1000)&(frame_time[t]<=st_strides_all[t][2][s,1,0]/1000))])
            if paw == 'HL':
                for s in range(np.shape(st_strides_all[t][3])[0]):
                    frames_strides.append(frame_time[t][np.where((frame_time[t]>=st_strides_all[t][3][s,0,0]/1000)&(frame_time[t]<=st_strides_all[t][3][s,1,0]/1000))])       
            strides_time.append(frames_strides)
        return strides_time
    
    def compute_gait_param(self,bodycenter,final_tracks,paws_rel,st_strides_mat,sw_pts_mat,param):
        """Computes gait parameters for all four paws
        Input:  bodycenter (dataframe 4xframes)
                final_tracks (4x5xframes)
                paws_rel (1xframes per paw)
                st_strides_mat (stridesx2x5 per paw)
                sw_pts_mat (stridesx1x5 per paw)
                param - variable name (check outputs)
        Outputs:stride_duration, swing_duration, stance_duration, swing_length, swing_velocity
        swinglength_rel, stance_speed, body_center_x_stride, body_speed_x, duty_factor, cadence, coo, coo_stance
        coo_swing, body_speed_x_cv, step_length, double_support, phase_st[ref_paw][paw]"""
        X = final_tracks[0,:,:]*self.pixel_to_mm
        X_interp = self.inpaint_nans(X)
        bodycenter_x = np.nanmean(X_interp,axis=0)
        #compute gait parameters
        p_sl = np.array([2, 3, 0, 1]) #paw order for contralateral paw
        param_mat = []
        for p in range(4):
            if param == 'stride_duration':
                param_mat.append(st_strides_mat[p][:,1,0]-st_strides_mat[p][:,0,0])
            if param == 'swing_duration':
                param_mat.append(st_strides_mat[p][:,1,0]-sw_pts_mat[p][:,0,0])
            if param == 'stance_duration':
                param_mat.append(sw_pts_mat[p][:,0,0]-st_strides_mat[p][:,0,0])
            if param == 'swing_length':
                param_mat.append(X_interp[p,st_strides_mat[p][:,1,4].astype(int)]-X_interp[p,sw_pts_mat[p][:,0,4].astype(int)])
            if param == 'swing_velocity':
                param_mat.append((X_interp[p,st_strides_mat[p][:,1,4].astype(int)]-X_interp[p,sw_pts_mat[p][:,0,4].astype(int)])/(st_strides_mat[p][:,1,0]-sw_pts_mat[p][:,0,0]))
            if param == 'swinglength_rel':
                param_mat.append(paws_rel[p][st_strides_mat[p][:,1,4].astype(int)]-paws_rel[p][st_strides_mat[p][:,0,4].astype(int)]) 
            if param == 'stance_speed':
                param_mat.append((sw_pts_mat[p][:,0,1]-st_strides_mat[p][:,0,1])/(sw_pts_mat[p][:,0,0]-st_strides_mat[p][:,0,0]))   
            if param == 'stance_length':
                param_mat.append(sw_pts_mat[p][:,0,1]-st_strides_mat[p][:,0,1])      
            if param == 'body_center_x_stride':
                bodycenter_stride = np.zeros((np.shape(st_strides_mat[p])[0]))
                for s in range(np.shape(st_strides_mat[p])[0]):
                    bodycenter_stride[s] = np.nanmean(bodycenter_x[st_strides_mat[p][s,0,4].astype(int):st_strides_mat[p][s,1,4].astype(int)])
                param_mat.append(bodycenter_stride) 
            if param == 'body_speed_x':
                bodyspeed = np.zeros((np.shape(st_strides_mat[p])[0]))
                for s in range(np.shape(st_strides_mat[p])[0]):
                    space_beg = bodycenter_x[st_strides_mat[p][s,0,4].astype(int)]*self.pixel_to_mm
                    space_end = bodycenter_x[st_strides_mat[p][s,1,4].astype(int)]*self.pixel_to_mm
                    bodyspeed[s] = (space_end-space_beg)/(st_strides_mat[p][s,1,0]-st_strides_mat[p][s,0,0])
                param_mat.append(bodyspeed)
            if param == 'duty_factor':
                param_mat.append((sw_pts_mat[p][:,0,0]-st_strides_mat[p][:,0,0])/(st_strides_mat[p][:,1,0]-st_strides_mat[p][:,0,0])*100)  
            if param == 'cadence':
                param_mat.append(1/(st_strides_mat[p][:,1,0]-st_strides_mat[p][:,0,0]))    
            if param == 'coo':
                param_mat.append(np.nanmean(np.column_stack((paws_rel[p][st_strides_mat[p][:,0,4].astype(int)],paws_rel[p][sw_pts_mat[p][:,0,4].astype(int)])),axis=1))   
            if param == 'coo_stance':
                param_mat.append(paws_rel[p][st_strides_mat[p][:,0,4].astype(int)])    
            if param == 'coo_swing':
                param_mat.append(paws_rel[p][sw_pts_mat[p][:,0,4].astype(int)])      
            if param == 'body_speed_x_cv':
                bodyspeed = np.zeros((np.shape(st_strides_mat[p])[0]))
                for s in range(np.shape(st_strides_mat[p])[0]):
                    space_beg = bodycenter_x[st_strides_mat[p][s,0,4].astype(int)]*self.pixel_to_mm
                    space_end = bodycenter_x[st_strides_mat[p][s,1,4].astype(int)]*self.pixel_to_mm
                    bodyspeed[s] = (space_end-space_beg)/(st_strides_mat[p][s,1,0]-st_strides_mat[p][s,0,0])
                param_mat.append(np.nanstd(bodyspeed)/np.nanmean(bodyspeed))
            if param == 'step_length':
                param_mat.append(X_interp[p,st_strides_mat[p][:,0,4].astype(int)]-X_interp[p_sl[p],st_strides_mat[p][:,0,4].astype(int)])   
            if param == 'double_support':
                ds = np.zeros((np.shape(st_strides_mat[p])[0]))
                ds[:] = np.nan
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc25 = np.round(0.25*(st_strides_mat[p][s,1,4]-st_strides_mat[p][s,0,4]))
                    pt_next = sw_pts_mat[p_sl[p]][np.logical_and(sw_pts_mat[p_sl[p]][:,0,4]>=(st_strides_mat[p][s,0,4]-perc25),(sw_pts_mat[p_sl[p]][:,0,4]<=sw_pts_mat[p][s,0,4])),0,4]
                    if len(pt_next) == 0:
                        ds[s] = 0
                    else:
                        ds[s] = (pt_next[0]-st_strides_mat[p][s,0,4])/(st_strides_mat[p][s,1,4]-st_strides_mat[p][s,0,4])*100
                param_mat.append(ds)
            if param == 'phase_st':
                #to do average do circular mean
                phase_st_paw = []
                for paw in range(4):  
                    phase_st_radians = np.zeros((np.shape(st_strides_mat[p])[0]))
                    phase_st_radians[:] = np.nan
                    for s in range(np.shape(st_strides_mat[p])[0]):      
                        val = st_strides_mat[paw][(st_strides_mat[paw][:,0,4]>=st_strides_mat[p][s,0,4])&(st_strides_mat[paw][:,0,4]<=st_strides_mat[p][s,1,4]),0,4]
                        if len(val)>0:
                            phase_st = (val[0]-st_strides_mat[p][s,0,4])/(st_strides_mat[p][s,1,4]-st_strides_mat[p][s,0,4])
                            phase_st_radians[s] = phase_st*2*pi
                    phase_st_paw.append(phase_st_radians)
                param_mat.append(phase_st_paw)
        return param_mat

    def animals_within_session(self):
        """See which animals and sessions are in the folder with tracks"""
        delim = self.path[-1]
        h5files = glob.glob(self.path+'*.h5')
        animal_session = []
        for f in h5files:
            path_split = f.split(delim)
            filename_split = path_split[-1].split('_')
            animal_session.append([filename_split[0],filename_split[7]])
        #check which sessions exist
        unique_list = []       
        for x in animal_session: 
            if x not in unique_list: 
                unique_list.append(x) 
        return unique_list
    
    def get_track_files(self,animal,session):
        """Gets list of .h5 files with tracks for that session of that animal
        Input: animal - animal name (str)
               session - session number (int)"""
        delim = self.path[-1]
        h5files = glob.glob(self.path+'*.h5')
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0]
            session_nr = int(filename_split[7])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[8][:-3]))     
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        return files_ordered

    def plot_gait_adaptation(self, param_sym, param, animal, session, print_plots):
        """Plots nicely the symmetry parameter that was computed for the whole session
        Input: param_sym - array with gait symmetry value across trials
               param - gait parameter (string with _)
               animal - animal name (str)
               session - session number (int)
               print_plots - boolean for printing and saving in path/images/"""
        fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
        rectangle = plt.Rectangle((4.5,min(param_sym)), 10, max(param_sym)-min(param_sym), fc='grey',alpha=0.3) 
        plt.gca().add_patch(rectangle)
        plt.hlines(0,1,len(param_sym),colors='grey',linestyles='--')
        plt.plot(np.linspace(1,len(param_sym),len(param_sym)),param_sym, color = 'black')
        ax.set_xlabel('Trial', fontsize = 20)
        ax.set_ylabel(param.replace('_',' '), fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title(animal+' session '+str(session)+' '+param.replace('_',' ')+' symmetry',color='black',fontsize = 16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(self.path+'images'):
                os.mkdir(self.path+'images')
            plt.savefig(os.path.join(self.path, 'images', param+'_'+animal+'_'+str(session)), dpi=self.my_dpi)
    
    def get_stride_trajectories(self,excursion,align_array,paw,axis,interpolation,stride_points):
        """Gets the trajectories of a joint aligned to the stance or swing onset of that paw
        Input: excursion (array) - final_tracks,tracks_tail,joints_wrist,joints_elbow,ear or bodycenter
               align_array (array) - array with stance (sx2x5) or swing points (sx1x5)
               paw (str) - 'FR','FL','HR','HL','nose' (if nose the stride to align is FR paw)
               axis (str) - 'X','Y' or 'Z' (X is from bottom view)
               interpolation (boolean) - make strides all the same length
               stride_points (int) - number of points in interpolation
        Output: Stride array (interpolation=1) or stride list (interpolation=0)"""
        #check paw id
        if paw == 'FR':
            p = 0 #paw of tracking
            p_stride = 0 #paw of stride alignment
        if paw == 'HR':
            p = 1
            p_stride = 1
        if paw == 'FL':
            p = 2
            p_stride = 2
        if paw == 'HL':
            p = 3
            p_stride = 3
        if paw == 'nose':
            p = 4
            p_stride = 0
        #compute excursion for the desired paw and body axis
        if np.shape(excursion)[0] == 4 and np.shape(excursion)[1] == 5:
            if axis == 'X':
                axis_id = 0
            if axis == 'Y':
                axis_id = 1
            if axis == 'Z':
                axis_id = 3
            excursion = self.inpaint_nans(excursion[axis_id,p,:]*self.pixel_to_mm)
        if np.shape(excursion)[0] == 4 and np.shape(excursion)[1] == 15:
            if axis == 'X':
                axis_id = 0
            if axis == 'Y':
                axis_id = 1
            if axis == 'Z':
                axis_id = 3
            excursion = self.inpaint_nans(excursion[axis_id,:,:]*self.pixel_to_mm)
        if np.shape(excursion)[0] == 4 and np.shape(excursion)[2] == 2:
            if axis == 'X':
                axis_id = 0
            if axis == 'Z':
                axis_id = 1
            excursion = self.inpaint_nans(excursion[p,:,axis_id]*self.pixel_to_mm)
        if np.shape(excursion)[0] == 4 and np.shape(excursion)[1] > 50:
            if axis == 'X':
                axis_id = 0
            if axis == 'Z':
                axis_id = 1
            excursion = self.inpaint_nans(excursion[axis_id,:]*self.pixel_to_mm)
        #redo alignment matrix
        if np.shape(align_array[p_stride])[1] == 2:   
            align_mat = np.column_stack((align_array[p_stride][:,0,4].astype(int),align_array[p_stride][:,1,4].astype(int)))
        if np.shape(align_array[p_stride])[1] == 1:
            align_mat = np.column_stack((align_array[p_stride][:-1,0,4].astype(int),align_array[p_stride][1:,0,4].astype(int)-1))
        if interpolation == 1:
            if len(np.shape(excursion)) == 2: #tracks_tail
                strides = np.zeros((np.shape(excursion)[0],np.shape(align_mat)[0],stride_points))
                strides[:] = np.nan
            else:
                strides = np.zeros((np.shape(align_mat)[0],stride_points))
                strides[:] = np.nan
        else:
            strides = []
        strides_length = np.zeros(np.shape(align_mat)[0])
        for s in range(np.shape(align_mat)[0]):
            if len(np.shape(excursion)) == 2: #tracks_tail
                single_stride = excursion[:,align_mat[s,0]:align_mat[s,1]]
                strides_length[s] = np.shape(single_stride)[1]
                if interpolation == 1:
                    #interpolate to 100 points
                    if strides_length[s]<stride_points:
                        x = np.linspace(0,np.shape(single_stride)[1],np.shape(single_stride)[1])
                        #loop over tail segments
                        stride_interp = np.zeros((np.shape(excursion)[0],stride_points))
                        for t in range(np.shape(excursion)[0]):
                            f_interp = interp1d(x,single_stride[t,:],kind='nearest',fill_value='extrapolate')
                            stride_interp[t,:] = f_interp(range(stride_points))
                            strides[:,s,:] = single_stride
                    #downsample to 100 points with FFT
                    if strides_length[s]>stride_points:
                        stride_interp = np.zeros((np.shape(excursion)[0],stride_points))
                        stride_interp[:] = np.nan
                        for t in range(np.shape(excursion)[0]):
                            stride_interp[t,:] = resample(single_stride[t,:],stride_points,window = 20)
                            #blips from downsampling
                            if abs(np.diff(stride_interp[t,:])[0])>abs(np.diff(stride_interp[t,:])[1])+10:
                                stride_interp[t,0] = stride_interp[t,1]
                                strides[:,s,:] = single_stride
                            else:
                                strides[:,s,:] = single_stride
                    if strides_length[s] == stride_points:
                        strides[:,s,:] = single_stride
                else:
                    strides.append(single_stride)
            else:
                single_stride = excursion[align_mat[s,0]:align_mat[s,1]]
                strides_length[s] = len(single_stride)
                if interpolation == 1:
                    #interpolate to 75 points
                    if strides_length[s]<stride_points:
                        x = np.linspace(0,len(single_stride),len(single_stride))
                        f_interp = interp1d(x,single_stride,kind='nearest',fill_value='extrapolate')
                        stride_interp = f_interp(range(stride_points))
                        strides[s,:] = stride_interp
                    #downsample to 75 points with FFT
                    if strides_length[s]>stride_points:
                        stride_interp = resample(single_stride,stride_points,window = 20)
                        #blips from downsampling
                        if abs(np.diff(stride_interp)[0])>abs(np.diff(stride_interp)[1])+10:
                            stride_interp[0] = stride_interp[1]
                            strides[s,:] = stride_interp
                        else:
                            strides[s,:] = stride_interp
                    if strides_length[s] == stride_points:
                        strides[s,:] = single_stride
                else:
                    strides.append(single_stride)
        return strides
    
    def bin_strides(self,param,strides,paw,bin_range):
        """Bins the strides and strides trajectories by a certain parameter
        Input: param (list of arrays) - array with the parameter values for each stride
               strides (array) - array of stride trajectories
               paw (str) - 'FR','FL','HR','HL'
               bin_range (vector) - 100:400:50 (e.g)
        Output: stride_idx_bins (idx of the strides per bin), param_mat_bins 
        (parameter values of the strides per bin), stride_trajectory_bins 
        (stride trajectories per bin)"""
        if paw == 'FR':
            p = 0 #paw of tracking
        if paw == 'HR':
            p = 1
        if paw == 'FL':
            p = 2
        if paw == 'HL':
            p = 3
        param_mat_sort = np.sort(param[p]) #sort parameter values
        param_mat_argsort = np.argsort(param[p]) #get idx of sorted parameter values
        [count,bin_edges] = np.histogram(param[p],bin_range) #get bin edges 
        bin_edge_beg = bin_edges[0:-1] #begining of each edge
        bin_edge_end = bin_edges[1:] #end of each edge
        param_mat_bins = []
        stride_idx_bins = []
        stride_trajectory_bins = []
        for b in range(len(bin_edge_beg)):
            #idx of beginning of bin edge in the idx of the sorted parameters
            bin_beg_idx = np.argmin(np.abs(param_mat_sort-bin_edge_beg[b]))
            #idx of end of bin edge in the idx of the sorted parameters
            bin_end_idx = np.argmin(np.abs(param_mat_sort-bin_edge_end[b])) 
            stride_idx_bins.append(param_mat_argsort[bin_beg_idx:bin_end_idx])
            param_mat_bins.append(param_mat_sort[bin_beg_idx:bin_end_idx])
            strides_bin = []
            for s in range(len(stride_idx_bins[b])):
                strides_bin.append(strides[stride_idx_bins[b][s]])
            stride_trajectory_bins.append(strides_bin)
        return stride_idx_bins, param_mat_bins, stride_trajectory_bins
    
    def group_trials_speed(self,filelist):
        """Bins the trials by speed of the belts
        Input: 
            filelist - list of trial filenames
        Output:
            speed_unique - list with the different speeds set
            speed_trials_idx - list with the trials for each speed bin"""
        speed_trials = np.zeros((len(filelist),2))
        count = 0
        #get speed of each trial
        for f in filelist:
            filelist_split = f.split('_')
            speed_trials[count,0] = float(filelist_split[5].replace(',','.'))
            speed_trials[count,1] = float(filelist_split[6].replace(',','.'))
            count += 1
        speed_unique = np.unique(speed_trials,axis=0)
        speed_trials_idx = []
        for s in range(len(speed_unique)):
            trials_speed = []
            for t in range(len(speed_trials)):
                #if the speed of the trial matches speed category append trial id
                if speed_trials[t,0] == speed_unique[s][0] and speed_trials[t,1] == speed_unique[s][1]:
                    trials_speed.append(t+1)
            speed_trials_idx.append(trials_speed)
        return speed_unique,speed_trials_idx

    def final_tracks_forwardlocomotion(self, final_tracks, st_strides_mat):
        """Retains in final tracks only the periods of forward locomotion that were good strides
        Inputs:
        final_tracks (array)
        st_strides_mat (list)"""
        final_tracks_FR_interp = self.inpaint_nans(final_tracks[:, :, :])
        final_tracks_FR_interp_forwadloco = np.zeros(np.shape(final_tracks))
        final_tracks_FR_interp_forwadloco[:] = np.nan
        idx_keep = []
        for s in range(np.shape(st_strides_mat[0])[0]):
            idx_keep.extend(np.int64(np.arange(st_strides_mat[0][s, 0, -1], st_strides_mat[0][s, 1, -1])))
        final_tracks_FR_interp_forwadloco[:, :, idx_keep] = final_tracks_FR_interp[:, :, idx_keep]
        return final_tracks_FR_interp_forwadloco

    def z_score(self,A):
        """Normalizes a vector by z-scoring it"""        
        A_norm = (A-np.nanmean(A))/np.nanstd(A)
        return A_norm
    
    def compute_bodyspeed(self,bodycenter):
        """Computes body speed with a Savitzky-Golay filter"""
        bodyspeed = savgol_filter(self.inpaint_nans(bodycenter),51,3,deriv=1)
        return bodyspeed
    
    def compute_bodycenter(self,final_tracks,axis_name):
        """ Computes bodycenter as the mean of the four paws for the desired axis"""
        if axis_name == 'X':
            axis_id = 0
        if axis_name == 'Y':
            axis_id = 1
        if axis_name == 'Z':
            axis_id = 3
        return np.nanmean(final_tracks[axis_id,:4,:],axis=0)*self.pixel_to_mm

    def compute_bodyacc(self,bodycenter):
        """Computes body acceleration with a Savitzky-Golay filter"""
        bodyacc = savgol_filter(self.inpaint_nans(bodycenter),51,3,deriv=2)
        return bodyacc

    def get_trials_split(self,filelist): 
        """Gets the trials that are split-belt"""
        trials_split = []
        for f in filelist:
            if f.split('_')[5] !=  f.split('_')[6]:
                trials_split.append(int(f.split('_')[8][:-3]))
        return np.array(trials_split)
    
    def trials_ordered(self,filelist): 
        """Gets the trials in order"""
        trials_ordered = []
        for f in filelist:
            trials_ordered.append(int(f.split('_')[8][:-3]))
        return np.array(trials_ordered)
   
    def plot_gait_catch(self,param_sym,param,animal,session,trials_split,print_plots):
        """Plots nicely the symmetry parameter that was computed for the whole session
        Input: param_sym - array with gait symmetry value across trials
               param - gait parameter (string with _)
               animal - animal name (str)
               session - session number (int)
               trials_split - array with the trials split-belt
               print_plots - boolean for printing and saving in path/images/"""
        fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
        for t in trials_split:
            rectangle = plt.Rectangle((t-0.5,min(param_sym)), 1, max(param_sym)-min(param_sym), fc='grey',alpha=0.3) 
            plt.gca().add_patch(rectangle)
        plt.hlines(0,1,len(param_sym),colors='grey',linestyles='--')
        plt.plot(np.linspace(1,len(param_sym),len(param_sym)),param_sym, color = 'black')
        ax.set_xlabel('Trial', fontsize = 15)
        ax.set_ylabel(param.replace('_',' '), fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title(animal+' session '+str(session)+' '+param.replace('_',' ')+' symmetry',color='black',fontsize = 16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(self.path+'images'):
                os.mkdir(self.path+'images')
            plt.savefig(self.path+'images/'+ param+'_'+animal+'_'+str(session), dpi=self.my_dpi)
        return
   
    def compute_angle_trajectories(self,st_strides_mat,sw_pts_mat,body_axis_xy,body_axis_xz,tail_axis_xy,tail_axis_xz,wrist_angles,param):
        """Computes angle trajectories during swing phase
        Input:  st_strides_mat (stridesx2x5 per paw)
                sw_pts_mat (stridesx1x5 per paw)
                body_axis_xy (1xframes)
                body_axis_xz (1xframes)
                tail_axis_xy (15xframes)
                tail_axis_xz (15xframes)
                wrist_angles (1xframes)
                st_strides_mat (stridesx2x5 per paw)
                sw_pts_mat (stridesx1x5 per paw)
                param - variable name (check outputs)
        Outputs: paw_angle, body_axisXYswing, body_axisXZswing, tail_axis_XYswing, tail_axis_XZswing"""
        stride_points = np.linspace(0,100,100)
        body_axis_xy_filt = savgol_filter(self.inpaint_nans(body_axis_xy*self.pixel_to_mm),window_length = 5, polyorder = 1)
        body_axis_xz_filt = savgol_filter(self.inpaint_nans(body_axis_xz*self.pixel_to_mm),window_length = 5, polyorder = 1)
        tail_axis_xy_filt = savgol_filter(self.inpaint_nans(tail_axis_xy*self.pixel_to_mm),window_length = 5, polyorder = 1)
        tail_axis_xz_filt = savgol_filter(self.inpaint_nans(tail_axis_xz*self.pixel_to_mm),window_length = 5, polyorder = 1)
        wrist_angles_filt = savgol_filter(self.inpaint_nans(wrist_angles*self.pixel_to_mm),window_length = 5, polyorder = 1)
        param_mat = []
        for p in range(4):
            if param == 'paw_angle':
                paw_angle = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<len(wrist_angles):
                        paw_angle.append(wrist_angles_filt[int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                paw_angle_norm = np.zeros((len(paw_angle),len(stride_points)))
                for s in range(len(paw_angle)):
                    xaxis = np.linspace(0,len(stride_points)-1,len(paw_angle[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = paw_angle[s]
                    if len(np.where(np.isnan(vec))[0]) == len(vec):
                        paw_angle_norm[s,:] = np.zeros(len(stride_points))
                        paw_angle_norm[s,:] = np.nan
                    else:
                        paw_angle_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(paw_angle_norm)
            if param == 'body_axisXYswing':
                body_axisXYswing = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<len(wrist_angles):
                        body_axisXYswing.append(body_axis_xy_filt[int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                body_axisXYswing_norm = np.zeros((len(body_axisXYswing),len(stride_points)))
                for s in range(len(body_axisXYswing)):
                    xaxis = np.linspace(0,len(stride_points)-1,len(body_axisXYswing[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = body_axisXYswing[s]
                    if len(np.where(np.isnan(vec))[0]) == len(vec):
                        body_axisXYswing_norm[s,:] = np.zeros(len(stride_points))
                        body_axisXYswing_norm[s,:] = np.nan
                    else:
                        body_axisXYswing_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(body_axisXYswing_norm)
            if param == 'body_axisXZswing':
                body_axisXZswing = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<len(wrist_angles):
                        body_axisXZswing.append(body_axis_xz_filt[int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                body_axisXZswing_norm = np.zeros((len(body_axisXZswing),len(stride_points)))
                for s in range(len(body_axisXZswing)):
                    xaxis = np.linspace(0,len(stride_points)-1,len(body_axisXZswing[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = body_axisXZswing[s]
                    if len(np.where(np.isnan(vec))[0]) == len(vec):
                        body_axisXZswing_norm[s,:] = np.zeros(len(stride_points))
                        body_axisXZswing_norm[s,:] = np.nan
                    else:
                        body_axisXZswing_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(body_axisXZswing_norm)
            if param == 'tail_axis_XYswing':
                tail_axis_XYswing = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    tail_axis_XYswing.append(tail_axis_xy_filt[:,int(sw_pts_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])])
                tail_axis_XYswing_norm = np.zeros((len(tail_axis_XYswing),15,len(stride_points)))
                for s in range(len(tail_axis_XYswing)):
                    for t in range(15):
                        xaxis = np.linspace(0,len(stride_points)-1,len(tail_axis_XYswing[s][t,:]))
                        xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                        vec = np.zeros(len(stride_points))
                        vec[:] = np.nan
                        vec[xaxis_idx] = tail_axis_XYswing[s][t,:]
                        if len(np.where(np.isnan(vec))[0]) == len(vec):
                            tail_axis_XYswing_norm[s,t,:] = np.zeros(len(stride_points))
                            tail_axis_XYswing_norm[s,t,:] = np.nan
                        else:
                            tail_axis_XYswing_norm[s,t,:] = self.inpaint_nans(vec)
                param_mat.append(tail_axis_XYswing_norm)
            if param == 'tail_axis_XZswing':
                tail_axis_XZswing = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    tail_axis_XZswing.append(tail_axis_xz_filt[:,int(sw_pts_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])])
                tail_axis_XZswing_norm = np.zeros((len(tail_axis_XZswing),15,len(stride_points)))
                for s in range(len(tail_axis_XZswing)):
                    for t in range(15):
                        xaxis = np.linspace(0,len(stride_points)-1,len(tail_axis_XZswing[s][t,:]))
                        xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                        vec = np.zeros(len(stride_points))
                        vec[:] = np.nan
                        vec[xaxis_idx] = tail_axis_XZswing[s][t,:]
                        if len(np.where(np.isnan(vec))[0]) == len(vec):
                            tail_axis_XZswing_norm[s,t,:] = np.zeros(len(stride_points))
                            tail_axis_XZswing_norm[s,t,:] = np.nan
                        else:
                            tail_axis_XZswing_norm[s,t,:] = self.inpaint_nans(vec)
                param_mat.append(tail_axis_XZswing_norm)
        return param_mat
    
    def compute_trajectories(self,paws_rel,bodycenter,final_tracks,joints_elbow,joints_wrist,tracks_tail,st_strides_mat,sw_pts_mat,param):
        """Computes trajectories for all four paws normalized to 100 points
        Input:  paws_rel (4xframes) paws relative to center of the body for a certain dimension
                bodycenter (4xframes)
                final_tracks (4x5xframes)
                joints_elbow (4xframesx2)
                joints_wrist (4xframesx2)
                st_strides_mat (stridesx2x5 per paw)
                sw_pts_mat (stridesx1x5 per paw)
                param - variable name (check outputs)
        Outputs: swing_inst_vel, swing_z, swing_z_elbow, swing_z_wrist, swing_z_pos, swing_z_elbow_pos, swing_z_wrist_pos, tail_y_relbase, 
        tail_z_relbase, swing_y_rel, swing_x_rel, stride_nose_yrel,  stride_nose_zrel"""
        stride_points = np.linspace(0,100,100)
        timestep = 1/self.sr
        X = savgol_filter(self.inpaint_nans(final_tracks[0,:,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        joints_elbow_filt_Z = savgol_filter(self.inpaint_nans(joints_elbow[:,:,1]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        joints_wrist_filt_Z = savgol_filter(self.inpaint_nans(joints_wrist[:,:,1]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        joints_elbow_filt_X = savgol_filter(self.inpaint_nans(joints_elbow[:,:,0]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        joints_wrist_filt_X = savgol_filter(self.inpaint_nans(joints_wrist[:,:,0]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        Z = savgol_filter(self.inpaint_nans(final_tracks[3,:,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        X_side = savgol_filter(self.inpaint_nans(final_tracks[2,:,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        Y = savgol_filter(self.inpaint_nans(final_tracks[1,:,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        tail_y = np.zeros(np.shape(tracks_tail[1,:,:]))
        for st in range(np.shape(tail_y)[0]):
            if len(np.where(np.isnan(tracks_tail[1,st,:]))[0])<(0.7*np.shape(tracks_tail)[2]):
                tail_y[st,:] = savgol_filter(self.inpaint_nans(tracks_tail[1,st,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)-np.nanmean(Y[:4,:],axis=0)
        tail_z = np.zeros(np.shape(tracks_tail[3,:,:]))
        for st in range(np.shape(tail_z)[0]):
            if len(np.where(np.isnan(tracks_tail[3,st,:]))[0])<(0.7*np.shape(tracks_tail)[2]):
                tail_z[st,:] = savgol_filter(self.inpaint_nans(tracks_tail[3,st,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        nose_rely = savgol_filter(self.inpaint_nans(final_tracks[1,4,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)-np.nanmean(Y[:4,:],axis=0)
        nose_z = savgol_filter(self.inpaint_nans(final_tracks[3,4,:]*self.pixel_to_mm),window_length = 5, polyorder = 1)
        param_mat = []
        for p in range(4):
            if param == 'swing_inst_vel':
                #plot swing instantaneous forward velocity trajectory
                swing_inst_vel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(X)[1]:
                        traj = np.diff(X[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])/timestep/1000
                        swing_inst_vel.append(traj)
                sw_inst_vel_norm = np.zeros((len(swing_inst_vel),len(stride_points)))
                for s in range(len(swing_inst_vel)):
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_inst_vel[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_inst_vel[s]
                    sw_inst_vel_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(sw_inst_vel_norm)
            if param == 'tail_y_relbase':
                #tail y
                tail_y_rel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    taily_norm = tail_y[:,int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]-tail_y[0,int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]
                    tail_y_rel.append(taily_norm-np.transpose(np.tile(taily_norm[:,0],(np.shape(taily_norm)[1],1))))
                tail_y_rel_norm = np.zeros((len(tail_y_rel),15,len(stride_points)))
                for s in range(len(tail_y_rel)):
                    for st in range(15):
                        xaxis = np.linspace(0,len(stride_points)-1,len(tail_y_rel[s][st,:]))
                        xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                        vec = np.zeros(len(stride_points))
                        vec[:] = np.nan
                        vec[xaxis_idx] = tail_y_rel[s][st,:]
                        tail_y_rel_norm[s,st,:] = self.inpaint_nans(vec)
                param_mat.append(tail_y_rel_norm)
            if param == 'tail_z_relbase':
                #tail z
                tail_z_rel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    tailz_norm = tail_z[:,int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]-tail_z[0,int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]
                    tail_z_rel.append(tailz_norm-np.transpose(np.tile(tailz_norm[:,0],(np.shape(tailz_norm)[1],1))))
                tail_z_rel_norm = np.zeros((len(tail_z_rel),15,len(stride_points)))
                for s in range(len(tail_z_rel)):
                    for st in range(15):
                        xaxis = np.linspace(0,len(stride_points)-1,len(tail_z_rel[s][st,:]))
                        xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                        vec = np.zeros(len(stride_points))
                        vec[:] = np.nan
                        vec[xaxis_idx] = tail_z_rel[s][st,:]
                        tail_z_rel_norm[s,st,:] = self.inpaint_nans(vec)
                param_mat.append(tail_z_rel_norm)
            if param == 'swing_z_elbow':
                swing_z_elbow = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(joints_elbow_filt_Z)[1]:
                        swing_z_elbow.append(self.floor-joints_elbow_filt_Z[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                swing_z_elbow_norm = np.zeros((len(swing_z_elbow),len(stride_points)))
                for s in range(len(swing_z_elbow)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_z_elbow[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_z_elbow[s]
                    swing_z_elbow_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_z_elbow_norm)
            if param == 'swing_z_wrist':
                swing_z_wrist = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(joints_wrist_filt_Z)[1]:
                        swing_z_wrist.append(self.floor-joints_wrist_filt_Z[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                swing_z_wrist_norm = np.zeros((len(swing_z_wrist),len(stride_points)))
                for s in range(len(swing_z_wrist)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_z_wrist[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_z_wrist[s]
                    swing_z_wrist_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_z_wrist_norm)
            if param == 'swing_z':
                sw_z = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(Z)[1]:
                        Z_excursion = np.median(Z[p,int(st_strides_mat[p][s,0,4]):int(sw_pts_mat[p][s,0,4])])
                        Z_excursion_perc = Z[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)] 
                        sw_z.append(Z_excursion-Z_excursion_perc)
                sw_z_norm = np.zeros((len(sw_z),len(stride_points)))
                for s in range(len(sw_z)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(sw_z[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = sw_z[s]
                    sw_z_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(sw_z_norm)
            if param == 'swing_z_elbow_pos':
                swing_z_elbow_pos = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(joints_elbow_filt_X)[1]:
                        swing_z_elbow_pos.append(joints_elbow_filt_X[p,int(sw_pts_mat[p][s,0,4]-perc_50)]-joints_elbow_filt_X[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                swing_z_elbow_pos_norm = np.zeros((len(swing_z_elbow_pos),len(stride_points)))
                for s in range(len(swing_z_elbow_pos)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_z_elbow_pos[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_z_elbow_pos[s]
                    swing_z_elbow_pos_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_z_elbow_pos_norm)
            if param == 'swing_z_wrist_pos':
                swing_z_wrist_pos = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(joints_wrist_filt_X)[1]:
                        swing_z_wrist_pos.append(joints_wrist_filt_X[p,int(sw_pts_mat[p][s,0,4]-perc_50)]-joints_wrist_filt_X[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)])
                swing_z_wrist_pos_norm = np.zeros((len(swing_z_wrist_pos),len(stride_points)))
                for s in range(len(swing_z_wrist_pos)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_z_wrist_pos[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_z_wrist_pos[s]
                    swing_z_wrist_pos_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_z_wrist_pos_norm)
            if param == 'swing_z_pos':
                sw_z_pos = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    perc_50 = 0.5*len(np.arange(sw_pts_mat[p][s,0,4],st_strides_mat[p][s,1,4]))
                    if sw_pts_mat[p][s,0,4]-perc_50>0 and st_strides_mat[p][s,1,4]+perc_50<np.shape(Z)[1]:
                        X_excursion = np.median(X_side[p,int(st_strides_mat[p][s,0,4]):int(sw_pts_mat[p][s,0,4])])
                        X_excursion_perc = X_side[p,int(sw_pts_mat[p][s,0,4]-perc_50):int(st_strides_mat[p][s,1,4]+perc_50)] 
                        sw_z_pos.append(X_excursion-X_excursion_perc)
                sw_z_pos_norm = np.zeros((len(sw_z_pos),len(stride_points)))
                for s in range(len(sw_z_pos)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(sw_z_pos[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = sw_z_pos[s]
                    sw_z_pos_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(sw_z_pos_norm)
            if param == 'swing_y_rel':
                swing_y_rel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    swing_y_rel.append(paws_rel[p][int(sw_pts_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])])
                swing_y_rel_norm = np.zeros((len(swing_y_rel),len(stride_points)))
                for s in range(len(swing_y_rel)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_y_rel[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_y_rel[s]
                    swing_y_rel_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_y_rel_norm)
            if param == 'swing_x_rel':
                swing_x_rel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    swing_x_rel.append(paws_rel[p][int(sw_pts_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])])
                swing_x_rel_norm = np.zeros((len(swing_x_rel),len(stride_points)))
                for s in range(len(swing_x_rel)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(swing_x_rel[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = swing_x_rel[s]
                    swing_x_rel_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(swing_x_rel_norm)
            if param == 'stride_nose_yrel': 
                stride_nose_yrel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    stride_nose_yrel.append(nose_rely[int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]-nose_rely[int(st_strides_mat[p][s,0,4])])
                stride_nose_yrel_norm = np.zeros((len(stride_nose_yrel),len(stride_points)))
                for s in range(len(stride_nose_yrel)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(stride_nose_yrel[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = stride_nose_yrel[s]
                    stride_nose_yrel_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(stride_nose_yrel_norm)
            if param == 'stride_nose_zrel':
                stride_nose_zrel = []
                for s in range(np.shape(st_strides_mat[p])[0]):
                    nose_norm = self.floor-nose_z[int(st_strides_mat[p][s,0,4]):int(st_strides_mat[p][s,1,4])]
                    stride_nose_zrel.append(nose_norm-nose_norm[0])
                stride_nose_zrel_norm = np.zeros((len(stride_nose_zrel),len(stride_points)))
                for s in range(len(stride_nose_zrel)): 
                    xaxis = np.linspace(0,len(stride_points)-1,len(stride_nose_zrel[s]))
                    xaxis_idx = np.round(xaxis, decimals=0).astype(int)
                    vec = np.zeros(len(stride_points))
                    vec[:] = np.nan
                    vec[xaxis_idx] = stride_nose_zrel[s]
                    stride_nose_zrel_norm[s,:] = self.inpaint_nans(vec)
                param_mat.append(stride_nose_zrel_norm)
        return param_mat
    
    def get_supports(self,final_tracks,st_strides_mat,sw_pts_mat):
        """Computes the supports for all strides (relative to reference paw)
        Inputs:
            final_tracks (4x5xframes)
            st_strides_mat (stridesx2x5 per paw)
            sw_pts_mat (stridesx1x5 per paw)
        Output: supports (list of 4 reference paws) order: 4 paw support, 3 supports, 
        diagonal FL, diagonal FR, homolateral supports, 1 paw supports, 0 paw support"""
        all_pts = [] #get all range of points
        for s in range(np.shape(st_strides_mat[1])[0]):    
            all_pts.extend(np.arange(st_strides_mat[1][s,0,4],sw_pts_mat[1][s,0,4]))
        supports = []   
        for ref in range(4):
            state = np.zeros((4,np.shape(final_tracks)[2]))
            for p in range(4):
                for s in range(np.shape(st_strides_mat[p])[0]):
                    state[p,int(st_strides_mat[p][s,0,4]):int(sw_pts_mat[p][s,0,4])] = 1
            supports_ref = np.zeros((np.shape(st_strides_mat[ref])[0],7))
            supports_ref[:] = np.nan
            for s in range(np.shape(st_strides_mat[ref])[0]):
                include = np.zeros(4)
                include[:] = np.nan
                for p in range(4):
                    if len(all_pts<st_strides_mat[ref][s,0,4])==0 or len(all_pts>st_strides_mat[ref][s,1,4])==0:
                        include[p] = 1
                    else:
                        include[p] = 0
                stride = np.transpose(state[:,int(st_strides_mat[ref][s,0,4]):int(st_strides_mat[ref][s,1,4])])
                supports_cat = np.zeros((np.shape(stride)[0],7))
                supports_cat[:] = np.nan
                if np.sum(include)>1:
                    supports_ref[s,:] = np.nan
                else:
                    for q in range(np.shape(stride)[0]):
                        if (stride[q,:]==np.array([1, 1, 1, 1])).all(): #4 paw support
                            supports_cat[q,0] = 1
                        else:
                            supports_cat[q,0] = 0
                        if (stride[q,:]==np.array([1, 1, 0, 1])).all() or (stride[q,:]==np.array([1, 1, 1, 0])).all() or (stride[q,:]==np.array([0, 1, 1, 1])).all() or (stride[q,:]==np.array([1, 0, 1, 1])).all(): #3 paw support
                            supports_cat[q,1] = 1
                        else:
                            supports_cat[q,1] = 0
                        if (stride[q,:]==np.array([0, 1, 1, 0])).all(): #diagonal FL
                            supports_cat[q,2] = 1
                        else:
                            supports_cat[q,2] = 0
                        if (stride[q,:]==np.array([1, 0, 0, 1])).all(): #diagonal FR
                            supports_cat[q,3] = 1
                        else:
                            supports_cat[q,3] = 0
                        if (stride[q,:]==np.array([1, 0, 1, 0])).all(): #homolateral supports
                            supports_cat[q,4] = 1
                        else:
                            supports_cat[q,4] = 0
                        if (stride[q,:]==np.array([1, 0, 0, 0])).all() or (stride[q,:]==np.array([0, 1, 0, 0])).all() or (stride[q,:]==np.array([0, 0, 1, 0])).all() or (stride[q,:]==np.array([0, 0, 0, 1])).all(): #1 paw support
                            supports_cat[q,5] = 1
                        else:
                            supports_cat[q,5] = 0
                        if (stride[q,:]==np.array([0, 0, 0, 0])).all(): #0 paw support
                            supports_cat[q,6] = 1
                        else:
                            supports_cat[q,6] = 0
                supports_ref[s,:] = np.sum(supports_cat,axis=0)/np.shape(stride)[0]*100
            supports.append(supports_ref)
        return supports
        
    def get_stride_speed(self,speed_L,final_tracks,st_strides_mat):
        """Compute stride speed using instantaneous speed across the four paws (adding the speed of the belts)
        Inputs:
            speed_L: float with belt speed (either right or left)
            final_tracks: 4x5xframes
            st_strides_mat: (stridesx2x5 per paw)"""
        timestep = 1/self.sr
        speed_center_rel = np.nancumsum(np.nanmean(np.diff(final_tracks[0,:,:],axis=1)/timestep/1000,axis=0)+speed_L)*timestep*1000
        stride_speed_center_rel = []
        for p in range(4):
            speed_stride = np.zeros((np.shape(st_strides_mat[p])[0]))
            for s in range(np.shape(st_strides_mat[p])[0]):
                speed_stride[s] = (speed_center_rel[int(st_strides_mat[p][s,1,4])]-speed_center_rel[int(st_strides_mat[p][s,0,4])])/(st_strides_mat[p][s,1,0]-st_strides_mat[p][s,0,0])
            stride_speed_center_rel.append(speed_stride)     
        return stride_speed_center_rel
    
    def save_raw_tracking(self,filename,final_tracks,path_loco):
        """Plot a subset of the paws x excursion for each tracking file"""
        plt.subplots(figsize=(15,15), tight_layout = True)
        plt.plot(np.transpose(final_tracks[0,:,1000:3000]))
        plt.savefig(path_loco+filename[:-3]+'.png', dpi=self.my_dpi)
        return
    
    def save_raw_tracking_tail(self,filename,tracks_tail,path_loco):
        """Plot a subset of the tail z excursion for each tracking file"""
        plt.subplots(figsize=(15,15), tight_layout = True)
        plt.plot(np.transpose(tracks_tail[3,:,1000:3000]))
        plt.savefig(path_loco+filename[:-3]+'_tail.png', dpi=self.my_dpi)
        return
    
    def get_tdms_frame_start(self,animal,session,black_frames):
        """From the tdms files get the length of the triggers, strobes and
        the first behavioral frame when miniscope started - for alignment
        Inputs:
            animal: (str) animal name
            session: (int) session number
        Outputs:
            triggers_ordered: (array) number of bcam triggers in each trial 
            strobes_ordered: (array) number of bcam strobes in each trial (effective frames)
            frame_rec_start_full: (array) number of bcam frames to be discarded in beginning of trial
            mscope_align_time_ordered: (array) time start of bcam after mscope cam
            frame_time_bcam_ordered: (list) timestamps of bcam frames for each trial
            """
        delim = self.path[-1]
        tdmslist = glob.glob(self.path+'*.tdms') 
        h5list = glob.glob(self.path+'*.h5') 
        tdms_sr = 10000
        h5filename = []
        for f in h5list:
            if delim == '/':
                h5filename.append(f.split('/')[-1])
            if delim == '\\':
                h5filename.append(f.split('\\')[-1])
        h5_trialorder = []
        h5list_animal = []
        for f in h5filename:
            if f.split('_')[0] == animal and int(f.split('_')[7]) == session:
                h5_trialorder.append(int(f.split('_')[8][:-3]))
                h5list_animal.append(self.path+f)               
        frame_nr = []
        trial = []
        triggers = []
        strobes = []
        frame_rec_start = []
        mscope_align_time = [] #if miniscope start is before behavior cam (should never happen)
        frame_time_bcam = []
        for f in tdmslist:
            if delim == '/':
                filename = f.split('/')[-1]
            else:
                filename = f.split('\\')[-1]
            if filename.split('_')[0]==animal and int(filename.split('_')[7])==session:
                tdms_file = tdms.TdmsFile.read(f)
                trial.append(int(filename.split('_')[-1][:-5]))   
                h5_trial = np.where(np.array(h5_trialorder) == int(filename.split('_')[-1][:-5]))[0][0]
                track_df = pd.read_hdf(h5list_animal[h5_trial],'df_with_missing')
                frame_nr.append(track_df.shape[0])
                tdms_groups = tdms_file.groups()
                tdms_channels = tdms_groups[0].channels()
                ch1 = tdms_channels[0].data #triggers
                ch2 = tdms_channels[1].data #miniscope pulse
                ch3 = tdms_channels[2].data #strobes
                triggers.append(len(np.where(np.diff(ch1)>1)[0]+1))
                idx_peaks = np.where(np.diff(ch3)>1)[0]+1
                cam_start = idx_peaks[0]
                strobes.append(len(idx_peaks))
                idx_miniscope_start = np.where(ch2)[0][0] #tdms time point from bcam at which mscope frame started
                frame_behavior_miniscope = len(np.where(idx_peaks<idx_miniscope_start)[0])+1 #frame from bcam at which mscope frame started (tdms starts before trigger)
                if frame_behavior_miniscope == 1: #if miniscope start is before behavior cam (should never happen)
                    mscope_idx_0 = np.where(ch2==1)[0][0]
                    mscope_start = (cam_start-mscope_idx_0)/tdms_sr 
                    mscope_align_time.append(mscope_start)
                    frame_rec_start.append(0)
                    frame_time_bcam.append(((idx_peaks-cam_start)/tdms_sr)+mscope_start) #need to add to bcam time the difference of time since mscope started
                else:
                    frame_rec_start.append(frame_behavior_miniscope)
                    mscope_start = 0
                    mscope_align_time.append(mscope_start)
                    cam_frames_time = (idx_peaks-idx_peaks[0])/tdms_sr
                    frame_time_bcam.append(cam_frames_time) #bcam time starts from 0
        trial_order = np.argsort(trial) #reorder trials  
        triggers_ordered = np.zeros(len(triggers))
        strobes_ordered = np.zeros(len(triggers))
        frame_rec_start_ordered = np.zeros(len(triggers)) 
        frame_nr_ordered = np.zeros(len(triggers))
        mscope_align_time_ordered = np.zeros(len(triggers))
        frame_time_bcam_ordered = []
        for count_t, t in enumerate(trial_order):
            triggers_ordered[t] = triggers[count_t]
            strobes_ordered[t] = strobes[count_t]
            frame_rec_start_ordered[t] = frame_rec_start[count_t]
            frame_nr_ordered[t] = frame_nr[count_t]
            mscope_align_time_ordered[t] = mscope_align_time[count_t]
            frame_time_bcam_ordered.append(frame_time_bcam[count_t])
        if len(black_frames) == 0:
            frame_rec_start_full = frame_rec_start_ordered
        else:
            frame_rec_start_full = np.zeros(len(triggers))
            for i in range(len(trial_order)):
                frame_rec_start_full[i] = frame_rec_start_ordered[i]+(black_frames[i]/30)*self.sr
        #change frame_time_bcam to account for black frames that were excluded
        bcam_time = []
        for t in range(len(frame_time_bcam_ordered)):
            if frame_rec_start_full[t]/self.sr > frame_time_bcam_ordered[t][0]: #if black frames is larger than the gap between mscope start and bcam start
                bcam_time_clean = frame_time_bcam_ordered[t][int(frame_rec_start_full[t]):]
                bcam_time.append(bcam_time_clean-bcam_time_clean[0]) #because extra frames will be deleted from tracking
            else:
                bcam_time.append(frame_time_bcam_ordered[t])
                frame_rec_start_full[t] = 0
        return triggers_ordered, strobes_ordered,frame_rec_start_full,mscope_align_time_ordered,bcam_time
    
    def backwards_locomotion(self,bodycenter,plot_data):
        """Function to get indices of trial where animal was being dragged backwards by treadmill
        Input:
            bodycenter"""
        backwards_loco = []
        for b in range(len(bodycenter)-1):
            if (bodycenter[b+1]-bodycenter[b])<0:
                backwards_loco.append(b)
        if plot_data:
            plt.scatter(backwards_loco,bodycenter[backwards_loco],s=1,color='red')
        return backwards_loco

    def bcam_strobe_number(self):
        """Get number of frames from txt file"""
        txtlist = glob.glob(self.path+'*.txt')
        strobe_nr_txt = np.zeros(len(txtlist))
        txt_count = 0
        for t in txtlist:
            with open(t) as f:
                fc_count_trial = f.readlines() 
            strobe_nr_txt[txt_count] = len(fc_count_trial)
            txt_count += 1
        return strobe_nr_txt
        
    def trial_start_blips(self):
        """Trial start trigger (square pulse) shows from time to time blips. These blips don't affect miniscope recording duration.
        This function says the number of blips per trial"""
        tdmslist = glob.glob(self.path+'*.tdms') 
        blip_nr = np.zeros((len(tdmslist)))
        count_tdms = 0
        for k in tdmslist:
            tdms_file = tdms.TdmsFile.read(k)
            tdms_groups = tdms_file.groups()
            tdms_channels = tdms_groups[0].channels()
            ch2 = tdms_channels[1].data 
            trial_start_rise = np.where(ch2>0)[0][0]
            trial_start_end = np.where(ch2==1)[0][-1]
            if trial_start_end>0:
                trial_start_blips = np.where(ch2[trial_start_rise:trial_start_end]<1)[0]
            else:
                trial_start_blips = np.where(ch2[trial_start_rise:]<1)[0]
            blip_nr[count_tdms] = len(trial_start_blips)
            count_tdms += 1
        return blip_nr


        







    

    
    
