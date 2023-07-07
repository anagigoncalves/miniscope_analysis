# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:46:24 2023

@author: User
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.signal import savgol_filter
import SlopeThreshold as ST


class df_behav_analysis:  
    
    
    def __init__(self, path):
        self.save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'
        self.pixel_to_mm = 1/1.955 #dana's setup
        self.sr_cam = 330 #sampling rate of behavior camera for treadmill
        self.sr = 30
        self.my_dpi = 128 #resolution for plotting
        self.trial_length = 60
        self.tied_speed = 0.225
        self.split_speed = [0.150, 0.300]
        
        
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
    
    
    def sort_rois_clust(self, df_events, clusters_rois):
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
    
    
    def trial_transition_idx(self, df):
        ''' detect indexes of trial transitions
        Input:
            - df: dataframe of neural activity where a column named 'trial' indicate the trial id of each frame
        '''
        tr_trans_idx = np.where(np.diff(np.asarray(df['trial'].values)) != 0)[0] + 1
        tr_trans_idx = np.concatenate(([0], tr_trans_idx, [len(df)]))
        return tr_trans_idx


    def split_expblocks(self, trials_ses):
        '''Sub-divide experimental blocks'''
        block_halflen = (trials_ses[1][1] - trials_ses[1][0]+1)//2
        split_blocks = np.array(([trials_ses[0][0]-1, trials_ses[0][1]], 
                                 [trials_ses[1][0]-1, trials_ses[1][0]-1 + block_halflen], 
                                 [trials_ses[1][0]-1 + block_halflen, trials_ses[1][1]], 
                                 [trials_ses[2][0]-1, trials_ses[2][0]-1 + block_halflen], 
                                 [trials_ses[2][0]-1 + block_halflen, trials_ses[2][1]]))
        return split_blocks


    def find_behav_ts(self, df, bcam_time):
        ''' Find index of timestamps of behavioral recording matching the ones of neural
        activity
        Inputs:
            - df: dataframe of calcium activity with a column containing timestamps
            - bcam_time: timestamps of behavioral recording'''
        # Get a list of dF/F timestamps for each trial
        ts_trial = df.groupby('trial')['time'].apply(list)
        df_ts = [np.array(tr) for tr in ts_trial] 
        # Find corresponding timestamps in behavioral recording
        behav_ts_idx = []
        for tr in range(len(df_ts)):
            ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in df_ts[tr]])
            behav_ts_idx.append(ts_idx)
        return behav_ts_idx


    def body_kinematic_aligned(self, final_tracks_trials, trials, behav_ts_idx, win_len, polyorder):
        ''' Compute body location (mean of the x position of the four paws), body speed (1st derivative of position)
        body acceleration (2nd derivative of position). All the kinematic variables are downsampled to the same
        frame rate of neural activity.
        Inputs:
            - final_tracks_trials: 3D array with paw coordinates
            - trials: array of trial numbers
            - behav_ts_idx: array of indexes of behavioral recording frames aligned to neural activity
            - win_len = length of savgol_filter kernel
            - polyorder = polynomial order for savgol_filter
        '''
        bodycenter = []
        bodyspeed = []
        bodyacc = []
        for tr in range(trials[-1]):
                bodycenter_trial = np.nanmean(final_tracks_trials[tr][0,:4, :],axis=0)*self.pixel_to_mm # Get bodycenter position (x-axis) for the desired trial and interval
                bodycenter.append(bodycenter_trial)        
                bodyspeed_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=1) # Get body speed
                bodyspeed.append(bodyspeed_trial)
                bodyacc_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=2) # Get body acceleration
                bodyacc.append(bodyacc_trial)
        bodycenter_aligned = [[bodycenter[i][j] for j in behav_ts_idx[i]] for i in range(len(bodycenter))]
        bodycenter_aligned = np.concatenate(bodycenter_aligned).flatten()
        bodyspeed_aligned = [[bodyspeed[i][j] for j in behav_ts_idx[i]] for i in range(len(bodyspeed))]
        bodyspeed_aligned = np.concatenate(bodyspeed_aligned).flatten()
        bodyacc_aligned = [[bodyacc[i][j] for j in behav_ts_idx[i]] for i in range(len(bodyacc))]
        bodyacc_aligned = np.concatenate(bodyacc_aligned).flatten()
        # Visualize the effect of different savgol filter's window lengths
        win_list = [win_len-30, win_len, win_len+30]
        bodycenter_check = np.nanmean(final_tracks_trials[trials[2]][0,:4, :15*self.sr_cam],axis=0)*self.pixel_to_mm # Get bodycenter position (x-axis) for the first 15 seconds of the last baseline trial
        for i in range(1,3):
            plt.figure()
            for size in win_list:
                kin_var = savgol_filter(self.inpaint_nans(bodycenter_check),size,polyorder,deriv=i)
                plt.plot(kin_var)
                plt.xlabel('Samples', fontsize=18)
                if i == 1:
                    plt.ylabel('Speed (m/s)', fontsize=18)
                else:
                    plt.ylabel('Acceleration (m/s^2)', fontsize=18)
            plt.legend(win_list, title='''Savgol win size (samples)''')
        # Visualize kinematic variables for sanity check
        plt.figure(figsize=(10, 8))
        # Plot 1: Body Position
        plt.subplot(2, 1, 1)
        plt.plot(bodycenter_check, color='gray')
        plt.ylabel('Body position (mm)', fontsize=18)
        # Plot 2: Speed and Acceleration
        plt.subplot(2, 1, 2)
        bodyspeed_check = savgol_filter(self.inpaint_nans(bodycenter_check), win_len, polyorder, deriv=1)
        bodyacc_check = savgol_filter(self.inpaint_nans(bodycenter_check), win_len, polyorder, deriv=2)
        plt.plot(bodyspeed_check, color='navy')
        plt.xlabel('Samples', fontsize=18)
        plt.ylabel('Speed (m/s)', color='navy', fontsize=18)
        plt.twinx()
        plt.plot(bodyacc_check, color='hotpink')
        plt.ylabel('Acceleration (m/s^2)', color='hotpink', fontsize=18)
        return bodycenter_aligned, bodyspeed_aligned, bodyacc_aligned


    def body_kinematic(self, final_tracks_trials, trials, win_len, polyorder):  
        ''' Compute body location (mean of the x position of the four paws), body speed (1st derivative of position)
        body acceleration (2nd derivative of position). 
        Inputs:
            - final_tracks_trials: 3D array with paw coordinates
            - trials: array of trial numbers
            - win_len = length of savgol_filter kernel
            - polyorder = polynomial order for savgol_filter
        '''
        bodycenter = []
        bodyspeed = []
        bodyacc = []
        for tr in range(len(final_tracks_trials)): #len(trials)
            bodycenter_trial = np.nanmean(final_tracks_trials[tr][0,:4, :],axis=0)*self.pixel_to_mm # Get bodycenter position (x-axis) for the desired trial and interval
            bodycenter.append(bodycenter_trial)        
            bodyspeed_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=1) # Get body speed
            bodyspeed.append(bodyspeed_trial)
            bodyacc_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=2) # Get body acceleration
            bodyacc.append(bodyacc_trial)
        return bodycenter, bodyspeed, bodyacc
    
    
    def paw_kinematic(self, tracks, p, ax, win_len, polyorder):
        '''Compute paw location, speed and acceleration for the preferred paw and axis.
        Inputs:
            - tracks: list of paws coordinates by trial
            - p: paw (FR = 0, HR = 1, FL = 2, HL = 3)
            - ax: axis (x = 0 or 2, y = 1, z = 3)
            - win_len = length of savgol_filter kernel
            - polyorder = polynomial order for savgol_filter       
        '''
        paw_pos = []
        paw_speed = []
        paw_acc = []
        for tr in range(len(tracks)):
            paw_pos_tr = (tracks[tr][ax, p, :] * self.pixel_to_mm).astype(int)
            paw_pos.append(paw_pos_tr)
            paw_speed.append(savgol_filter(self.inpaint_nans(paw_pos_tr),win_len,polyorder,deriv=1))
            paw_acc.append(savgol_filter(self.inpaint_nans(paw_pos_tr),win_len,polyorder,deriv=2))
        fig, axs = plt.subplots(3,1)
        axs[0].plot(paw_pos[1][self.sr_cam*5:self.sr_cam*15], c = 'grey')
        axs[0].set_ylabel('Paw position (mm)')
        axs[1].plot(paw_speed[1][self.sr_cam*5:self.sr_cam*15], c = 'navy')
        axs[1].set_ylabel('Speed (m/s)')
        axs[2].plot(paw_acc[1][self.sr_cam*5:self.sr_cam*15], c = 'hotpink')
        axs[2].set_ylabel('Acceleration (m/s^2)')
        axs[2].set_xlabel('Samples')
        return paw_pos, paw_speed, paw_acc
    
    
    def paw_diff(self, tracks, p1, p2):
        ''' Compute displacement or phase difference between two paws.
        Inputs:
            - tracks: list of limbs coordinates for each trial
            - p1: reference paw (FR=0, HR=1, FL=2, HL=3)
            - P2: secondary paw
        '''
        paw_difference = []
        for tr in range(len(tracks)):
            ref = tracks[tr][0,p1,:] - np.nanmean(tracks[tr][0,p1,:])
            sec = tracks[tr][0,p2,:] - np.nanmean(tracks[tr][0,p2,:])
            paw_difference.append(ref - sec) 
        return paw_difference
        
    
    def df_behav_align(self, df, clusters_rois, frame_time, final_tracks_trials, sl_time_all_array, sl_sym_all_array, trials, plot_type, window, save_plot):
        '''Align dF/F (population heatmap or clusters traces) to behavior and plot the result for desired trials and windows. 
        Behaviors computed by the function are: body position (x-axis), speed, acceleration, step-length symmetry.
        Inputs:
            - df = DataFrame of fluorescence or events for each ROI or cluster
            - clusters_rois = list of ROIs belonging to each cluster
            - frame_time = list of miniscope timestamps for each trial
            - final_tracks_trials = list of final tracks for each trial, each item of the list is (4x5xframes)
            - sl_time_all_array = array of step-length symmetry timestamps
            - sl_sym_all_array = array of step-length symmetry values
            - trials = list of trials
            - plot_type = 'popul_heatmap', 'cluster_traces' or 'popul_raster'
            - window = list with beginning and end of your desired time window
            - save_plot = boolean (1 = save figures)
        '''
        # Initialize variables
        beg = window[0]
        end = window[1]
        
        # Sort ROIs by cluster
        if plot_type == 'popul_heatmap' or plot_type == 'popul_raster':
            clusters_rois_flat = np.transpose(sum(clusters_rois, []))
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
            clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
            cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
            df = df[clusters_rois_flat]
        
        # Loop through trials
        for trial in range(trials[0], len(trials)+1):
            height_ratios = [2, 1, 1, 1, 1]
            gs_kw = dict(height_ratios=height_ratios)
            fig, axs = plt.subplots(5, 1, figsize=(12, 8), gridspec_kw=gs_kw)
        # Neural activity
            if plot_type == 'popul_heatmap': # Population dF/F heatmap
                df_trial = df.loc[df['trial'] == trial].iloc[beg*self.sr:end*self.sr, 2:] # Get df/f for the desired trial and interval
                sns.heatmap(df_trial.T, cbar=False, cmap='viridis', ax=axs[0])
                axs[0].set(xticklabels=[])
                axs[0].set(yticklabels=[])
                axs[0].set_ylabel('ROIs')
                axs[0].spines['right'].set_visible(False)
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['bottom'].set_visible(False)
                axs[0].tick_params(left=False, bottom=False)
                for c in cluster_transition_idx: # Lines to mark clusters in the heatmap
                    axs[0].hlines(c + 1, *axs[0].get_xlim(), color='white', linestyle='dashed')
            elif plot_type == 'clust_traces': # Clusters dF/F traces
                    idx_trial = np.where(trials==trial)[0][0]
                    df_trial = df.loc[df['trial'] == trial].iloc[beg*self.sr:end*self.sr, 2:]  # Get df/f for the desired trial and interval
                    count_r = 0
                    for r in df.columns[2:]: # To plot stacked traces
                        axs[0].plot(frame_time[idx_trial][beg*self.sr:end*self.sr], df_trial[r] + (count_r / 2))
                        count_r += 1
                        axs[0].set_xlim([beg, end])
                        axs[0].set_ylabel('Clusters')
                        axs[0].spines['right'].set_visible(False)
                        axs[0].spines['top'].set_visible(False)
                        axs[0].spines['bottom'].set_visible(False)
                        axs[0].tick_params(left=False, bottom=False)
                        axs[0].set(xticklabels=[])
            elif plot_type == 'popul_raster': # Population raster plot
                    spikes = df.loc[df['trial'] == trial].iloc[beg*self.sr:end*self.sr, 2:].values
                    ts = df.loc[df['trial'] == trial].iloc[beg*self.sr:end*self.sr, 1].values
                    for i in range(spikes.shape[1]):
                        # Get the indices of all spike events and respective timestamps for each column
                        spikes_idx = np.where(spikes[:, i] == 1)[0]
                        spikes_ts = ts[spikes_idx]
                        axs[0].vlines(spikes_ts, (i+1) - 0.8, (i+1) + 0.8, color = 'grey')
                    axs[0].set_xlim([0, ts[-1]])
                    axs[0].set_ylabel('ROIs')
                    axs[0].spines['right'].set_visible(False)
                    axs[0].spines['top'].set_visible(False)
                    axs[0].spines['bottom'].set_visible(False)
                    axs[0].tick_params(left=False, bottom=False)
                    axs[0].set(xticklabels=[]) 
                    axs[0].set(yticklabels=[])
                    plt.show()
                            
        # Behavior
            bodycenter_trial = np.nanmean(final_tracks_trials[trial-1][0,:4, beg*self.sr_cam:end*self.sr_cam],axis=0)*self.pixel_to_mm # Get bodycenter position (x-axis) for the desired trial and interval
            t = np.linspace(beg, end, len(bodycenter_trial)) # Create x-axis time values
            sns.lineplot(x=t, y=bodycenter_trial, ax=axs[1])
            axs[1].set_xlim([beg, end])
            axs[1].set_ylim([0, 300])
            axs[1].set(xticklabels=[])
            axs[1].set_ylabel('Body Center (mm)')
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['bottom'].set_visible(False)
            axs[1].tick_params(left=False, bottom=False)
            bodyspeed_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),51,3,deriv=1) # Get body speed
            sns.lineplot(x=t, y=bodyspeed_trial, ax=axs[2])
            axs[2].set_xlim([beg, end])
            axs[2].set_ylim([-1, 2])            
            axs[2].set(xticklabels=[])
            axs[2].set_ylabel('Body Speed (m/s)')
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['top'].set_visible(False)
            axs[2].spines['bottom'].set_visible(False)
            axs[2].tick_params(left=False, bottom=False)
            bodyacc_trial = savgol_filter(self.inpaint_nans(bodycenter_trial),51,3,deriv=2) # Get body acceleration
            sns.lineplot(x=t, y=bodyacc_trial, ax=axs[3])
            axs[3].set_xlim([beg, end])
            axs[3].set_ylim([-0.1, 0.1])                       
            axs[3].set_ylabel('Body Acceleration (m/s^2)')
            axs[3].spines['right'].set_visible(False)
            axs[3].spines['top'].set_visible(False)
            axs[3].spines['bottom'].set_visible(False)
            axs[3].tick_params(left=False, bottom=False)
            axs[3].set(xticklabels=[])    
            
        # Errors
            if trial == 1:
                beg_of_trial = beg # First timestamp of the desired interval
            else:
                beg_of_trial = ((trial - 1) * self.trial_length) + beg
            end_of_trial = beg_of_trial + (end-beg) # Last timestamp of the desired interval
            end_idx = np.abs(sl_time_all_array - end_of_trial).argmin() # Find index of the first timestamp of the desired trial and interval
            beg_idx = np.abs(sl_time_all_array - beg_of_trial).argmin() # Find index of the last timestamp of the desired trial and interval
            if trial == 1:
                sl_trial_time = sl_time_all_array[beg_idx:end_idx] # Find timestamps of step-length values for the desired interval (note: timestamps are not equally spaced because sl sym is computed for each stride)
            else:
                sl_trial_time = sl_time_all_array[beg_idx:end_idx] - self.trial_length*(trial-1)
            sl_trial_values = sl_sym_all_array[beg_idx:end_idx] # Find step-length values for the desired interval
            axs[4].plot(sl_trial_time, sl_trial_values)
            axs[4].set_xlabel('Time (s)') # Change with strides
            axs[4].set_ylabel('SL symmetry (mm)')
            axs[4].spines['right'].set_visible(False)
            axs[4].spines['top'].set_visible(False)
            axs[4].set_xlim([beg, end])
            axs[4].set_ylim([-40, 40])     
            
        # Show figure
            fig.suptitle('Trial ' + str(trial))
            plt.tight_layout()
            plt.show()  
            
        # Save
            if save_plot:
                if plot_type == 'popul_heatmap':
                    if not os.path.exists(os.path.join(self.save_path, 'dF_heatmap_behav_aligned_beg')):
                        os.mkdir(os.path.join(self.save_path, 'dF_heatmap_behav_aligned_beg'))
                    plt.savefig(os.path.join(self.save_path, 'dF_heatmap_behav_aligned_beg\\','dF_heatmap_behav_trial' + str(trial) + 'beg.png'), dpi=self.my_dpi)
                elif plot_type == 'clust_traces':
                    if not os.path.exists(os.path.join(self.save_path, 'dF_traces_behav_aligned_beg')):
                        os.mkdir(os.path.join(self.save_path, 'dF_traces_behav_aligned_beg'))
                    plt.savefig(os.path.join(self.save_path, 'dF_traces_behav_aligned_beg\\','dF_traces_behav_trial' + str(trial) + 'beg.png'), dpi=self.my_dpi)


    def sta(self, df_events, variable, bcam_time, window, trials):
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
                        extracted_signal = (extracted_signal - np.nanmean(extracted_signal))/np.std(extracted_signal)
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


    def plot_sta_rois(self, sta_allrois, signal_chunks_allrois, window, trials, trials_ses, colors_session, rois_sorted, var_name, mouse_id, session, save_plot):
        ''' Plot STA for each ROI.
        Inputs:
        - sta_allrois: list of STA for all the trials for each ROI
        - signal_chunks_allrois: raw traces of signal around events for each ROI
        - window: epoch of interest around CS
        - trials: array with all the trial numbers
        - trials_ses: 2D array or list with the first and last trial of each experimental block
        - colors_session = colors of experimental blocks
        - rois_sorted: 1D array with all the ROIs sorted by cluster
        - var_name: name of the variable
        - mouse_id: mouse name (str)
        - session: ipsi (S1) or contra (S2) (str)
        - save_plot (boolean)
        '''
        # Define font size for plot labels
        font_size = 15
                
        # Define tick labels for plots
        y_tick_labels = [block[1] for block in trials_ses]
        y_tick_locations = [block[1] - 1 for block in trials_ses]
        
        # Find min and max to set limits of the axis
        max_val = np.max(np.concatenate(signal_chunks_allrois, axis=0))
        min_val = np.min(np.concatenate(signal_chunks_allrois, axis=0))
        max_val_sta = np.max(np.concatenate(sta_allrois, axis=0))
        min_val_sta = np.min(np.concatenate(sta_allrois, axis=0))
        
        # STA for each ROI
        for n in range(len(sta_allrois)):
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            # Subplot 1: heatmap of all the raw traces for each ROI    
            sns.heatmap(signal_chunks_allrois[n], cbar = False, cmap='viridis', ax=axs[0], vmin = min_val, vmax = max_val) # heatmap of traces for one neuron whole session
            axs[0].axvline(x=window[-1], color='white', linestyle='--')
            axs[0].set_ylabel('Events', fontsize = font_size)
            axs[0].set(xticklabels=[])
            axs[0].set(yticklabels=[])
            axs[0].tick_params(left=False, bottom=False)
            # Subplot 2: heatmap of STA for each trial for each ROI   
            sns.heatmap(sta_allrois[n], cbar = False, cmap='viridis', ax=axs[1], vmin = min_val_sta, vmax = max_val_sta) # heatmap of STA for one neuron by trial
            axs[1].set_ylabel('Trials', fontsize = font_size)
            axs[1].set_yticks(y_tick_locations)
            axs[1].set_yticklabels(y_tick_labels)
            axs[1].axvline(x=window[-1], color='white', linestyle='--')
            axs[1].set(xticklabels=[])
            axs[1].tick_params(bottom=False)
            # Subplot 3: STA traces for each trial for each ROI        
            for tr_idx, _ in enumerate(trials):
                axs[2].plot(window * 1/self.sr_cam, sta_allrois[n][tr_idx], c = 'lightgray')
            axs[2].axvline(x=0, color='black', linestyle='--')
            for b in trials_ses:
                start_idx = b[0]
                end_idx = b[1]
                axs[2].plot(window * 1/self.sr_cam, np.mean(sta_allrois[n][start_idx:end_idx], axis=0), c=colors_session[start_idx], linewidth=2.3)
            axs[2].set_ylabel(var_name + ' (z-score)', fontsize = font_size)
            axs[2].set_xlabel('Time around event (s)', fontsize = font_size)
            axs[2].spines['right'].set_visible(False)
            axs[2].spines['top'].set_visible(False)
            axs[2].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[2].set_ylim([min_val_sta, max_val_sta])
            fig.suptitle('STA ' + var_name + ' ' + str(rois_sorted[n]), fontsize = font_size)
            # Save plots
            if save_plot:
                if not os.path.exists(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session)):
                    os.mkdir(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session))
                plt.savefig(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id  + '_' + session + '\\', 'STA_' + var_name + '_' + str(rois_sorted[n]) + '.png'), dpi=self.my_dpi) 
                plt.close()


    def plot_sta_popul(self, sta_allrois, window, cluster_transition_idx, colors_cluster, colors_session, trials, split_blocks, var_name, interval, mouse_id, session, save_plot):
        ''' Plot STA for the whole population.
        Inputs:
        - sta_allrois: list of STA for all the trials for each ROI
        - window: epoch of interest around CS
        - cluster_transition_idx: array of indexes of cluster ends
        - colors_cluster: array with clusters colorcode
        - colors_session: dictionary with colorcode for each trial
        - trials: array with all the trial numbers
        - split_blocks = 2D array of sub-divided experimental blocks
        - var_name: name of the variable
        - interval: start and end of window around event for peak detection
        - mouse_id: mouse name (str)
        - session: ipsi (S1) or contra (S2) (str)
        - save_plot (boolean)
        '''
        # Define font size for plot labels
        font_size = 15
        
        # Define colorcode for experimental blocks
        color_blocks = []
        for b in split_blocks:
            color_blocks.append(colors_session[b[0]+1])
                     
        # Re-sort data to have a list of the STA of all the ROIs for each trial
        sta_tr_allrois = [[sta_roi[tr_idx] for sta_roi in sta_allrois] for tr_idx, _ in enumerate(trials)] # List of the STA of all the ROIs for each trial

        # Compute STA of all the ROIs for each block
        sta_blocks_allrois = [np.mean(np.array(sta_tr_allrois[start:end]), axis=0) for start, end in split_blocks] # List of the STA of all the ROIs for block

        # Define tick labels for plots
        x_tick_values = [round(window[0]/self.sr_cam,1), round((1/2)*window[0]/self.sr_cam,1), 0, round((1/2)*window[-1]/self.sr_cam,1), round(window[-1]/self.sr_cam,1)]
        x_ticks = np.linspace(0, len(sta_tr_allrois[0][0]), len(x_tick_values)).astype(int)

        # Plot 1: Heatmap STA of the population by trial 
        max_val = np.max(np.concatenate(sta_tr_allrois, axis=0))
        min_val = np.min(np.concatenate(sta_tr_allrois, axis=0))
        for tr_idx, tr in enumerate(trials): #WARNING: for tr in range(len(trial_changes)-1):
            plt.figure()
            hm = sns.heatmap(sta_tr_allrois[tr_idx], cmap='viridis', vmin = min_val, vmax = max_val)  # heatmap STA whole population by trial 
            plt.ylabel('ROIs', fontsize = font_size)
            plt.xlabel('Time (s)', fontsize=font_size)
            cbar = hm.collections[0].colorbar
            cbar.set_label(var_name + ' (z-score)', fontsize=font_size)
            plt.axvline(x=window[-1], color='white', linestyle='--')
            plt.yticks([])
            plt.tick_params(left=False)
            plt.xticks(x_ticks, [f"{tick}" for tick in x_tick_values])
            plt.title('STA ' + var_name + ' trial ' + str(tr), fontsize=font_size)
            if save_plot:
                if not os.path.exists(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id  + '_' + session)):
                    os.mkdir(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id  + '_' + session))
                plt.savefig(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id   + '_' + session + '\\', 'STA_' + var_name + '_trial' + str(tr) + '.png'), dpi=self.my_dpi)
                plt.close()
    
        # Plot 2: Heatmap STA of the population by block 
        max_val = np.max(np.concatenate(sta_blocks_allrois, axis=0))
        min_val = np.min(np.concatenate(sta_blocks_allrois, axis=0))
        fig, axs = plt.subplots(len(sta_blocks_allrois),1, figsize = (12, 12))
        for b in range(len(sta_blocks_allrois)):
            hm = sns.heatmap(sta_blocks_allrois[b], cmap='viridis', ax = axs[b], vmin = min_val, vmax = max_val)  # heatmap STA whole population by block 
            axs[b].axvline(x=window[-1], color='white', linestyle='--')
            axs[b].set_ylabel('ROIs', fontsize = font_size)
            axs[b].set(xticklabels=[])
            axs[b].set(yticklabels=[])
            if b < len(sta_blocks_allrois)-1:
                axs[b].tick_params(left=False, bottom=False)
            if b == int((len(sta_blocks_allrois)-1)/2):
                cbar = hm.collections[0].colorbar
                cbar.set_label(var_name + ' (z-score)', fontsize=font_size)
            plt.xlabel('Time (s)', fontsize=font_size)
            plt.axvline(x=window[-1], color='white', linestyle='--')
            plt.yticks([])
            plt.tick_params(left=False)
            plt.xticks(x_ticks, [f"{tick}" for tick in x_tick_values], fontsize = 12)
            fig.suptitle('STA ' + var_name, fontsize = font_size)
            for c in cluster_transition_idx: # Mark clusters in the heatmap
                axs[b].hlines(c + 1, *axs[b].get_xlim(), color='white', linestyle='dashed', linewidth = 0.5)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session)):
                os.mkdir(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session))
            plt.savefig(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session + '\\', 'STA_' + var_name + '_blocks' + '.png'), dpi=self.my_dpi)
            plt.close()
        
        # Plot 3: STA traces of one cluster across blocks     
        clust_sta = np.zeros((len(sta_blocks_allrois), len(cluster_transition_idx), len(window))) # STA of clusters by block
        for block_idx, block in enumerate(sta_blocks_allrois):
            start = 0
            for clust_idx, clust_end in enumerate(cluster_transition_idx):
                if len(cluster_transition_idx)-1 == clust_idx:
                    end = len(sta_blocks_allrois[0])
                else:
                    end = clust_end
                clust_sta[block_idx, clust_idx] = np.mean(block[start:end], axis = 0)
                start = clust_end
        fig, axs = plt.subplots(nrows=1, ncols=len(cluster_transition_idx), figsize = (15, 5))
        if len(cluster_transition_idx) == 1:
            axs = [axs]  # Convert single axis to a list for indexing consistency
        for clust_idx, _ in enumerate(cluster_transition_idx):
            for block_idx, _ in enumerate(sta_blocks_allrois):
                current_ax = axs[clust_idx]  # Get the current axis
                current_ax.plot(window * 1/self.sr_cam, clust_sta[block_idx, clust_idx], c=color_blocks[block_idx], linewidth=2)
            current_ax.axvline(x=0, color='black', linestyle='--')
            current_ax.spines['right'].set_visible(False)
            current_ax.spines['top'].set_visible(False)
            current_ax.set_ylim([min_val, max_val])
            current_ax.set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            current_ax.set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            current_ax.set_xlabel('Time around event (s)', fontsize=font_size)
            if clust_idx == 0:
                current_ax.set_ylabel(var_name + ' (z-score)', fontsize=font_size)
            if clust_idx > 0:
                current_ax.set(yticklabels=[])
                current_ax.tick_params(left=False)
                current_ax.spines['left'].set_visible(False)
            current_ax.set_title('Cluster ' + str(clust_idx+1), fontsize=font_size, c=colors_cluster[clust_idx])       
        if save_plot:
            save_dir = os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)            
            filename = 'STA_' + var_name + '_clusters_blocks.png'
            save_path = os.path.join(save_dir, filename)            
            plt.savefig(save_path, dpi=self.my_dpi)
            plt.close()


    def tuning_learn(self, interval, sta, cluster_transition_idx, trials, colors_cluster, var_name, mouse_id, session, save_plot):
        font_size = 15
        start = self.sr_cam+interval[0]
        end = self.sr_cam+interval[1]
        sta_max = np.zeros((len(cluster_transition_idx), len(trials)))
        latency_max = np.zeros((len(cluster_transition_idx), len(trials)))
        sta_min = np.zeros((len(cluster_transition_idx), len(trials)))
        latency_min = np.zeros((len(cluster_transition_idx), len(trials)))
        for tr in range(len(trials)):
            clust_start = 0
            for clust_idx, clust_end in enumerate(cluster_transition_idx):
                if clust_idx == len(cluster_transition_idx) - 1:
                    clust_end = len(sta)
                sta_clust = sta[clust_start:clust_end, tr]
                sta_max[clust_idx, tr] = np.mean(np.max(sta_clust[:, start:end], axis=1))
                latency_max[clust_idx, tr] = np.mean(((np.argmax(sta_clust[:, start:end], axis=1)/self.sr_cam)+round(interval[0]/self.sr_cam, 2)))
                sta_min[clust_idx, tr] = np.mean(np.min(sta_clust[:, start:end], axis=1))
                latency_min[clust_idx, tr] = np.mean(((np.argmin(sta_clust[:, start:end], axis=1)/self.sr_cam)+round(interval[0]/self.sr_cam, 2)))                
                clust_start = clust_end
    
        fig, axs = plt.subplots(2,2, figsize = (12, 8))
        for c in range(len(cluster_transition_idx)):
            axs[0,0].plot(trials, sta_max[c,:], c=colors_cluster[c], linewidth = 2.5)
            axs[0,1].plot(trials, latency_max[c,:], c=colors_cluster[c], linewidth = 2.5)
            axs[1,0].plot(trials, sta_min[c,:], c=colors_cluster[c], linewidth = 2.5)
            axs[1,1].plot(trials, latency_min[c,:], c=colors_cluster[c], linewidth = 2.5)
        axs[0,0].set_xlim(1, trials[-1])
        axs[0,1].set_xlim(1, trials[-1])
        axs[1,0].set_xlim(1, trials[-1])
        axs[1,1].set_xlim(1, trials[-1])
        axs[1,0].set_xlabel('Trial', fontsize = font_size)
        axs[1,1].set_xlabel('Trial', fontsize = font_size)
        axs[0,0].set_ylabel('Max z-score ', fontsize = font_size)
        axs[1,0].set_ylabel('Min z-score ', fontsize = font_size)
        axs[0,1].set_ylabel('Latency (s)', fontsize = font_size)
        axs[1,1].set_ylabel('Latency (s)', fontsize = font_size)
        axs[0,0].set(xticklabels=[])
        axs[0,1].set(xticklabels=[])
        axs[0,0].tick_params(bottom=False)
        axs[0,1].tick_params(bottom=False)
        axs[0,0].spines['right'].set_visible(False)
        axs[0,0].spines['top'].set_visible(False)
        axs[0,0].spines['bottom'].set_visible(False)
        axs[0,1].spines['right'].set_visible(False)
        axs[0,1].spines['top'].set_visible(False)
        axs[0,1].spines['bottom'].set_visible(False)
        axs[1,1].spines['right'].set_visible(False)
        axs[1,1].spines['top'].set_visible(False)
        axs[1,0].spines['right'].set_visible(False)
        axs[1,0].spines['top'].set_visible(False)
        fig.subplots_adjust(wspace=0.3)
        plt.suptitle(var_name, fontsize = font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session)):
                os.mkdir(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session))
            plt.savefig(os.path.join(self.save_path, 'STA_' + var_name + '_' + mouse_id + '_' + session + '\\', 'STA_' + var_name + 'peaks_latency' + '.png'), dpi=self.my_dpi)
            plt.close()


    def shuffle_spikes_ts(self, df_events, iter_n):
        ''' Shuffle timestamps of events for multiple iterations.
        Inputs:
            - df_events: dataframe of events for multiple ROIs. Column 'time' contains timestamps, column 'trial' indicates trial ID
            - iter_n: number of shuffling iterations
        '''
        trial_len = round(max(df_events['time']))
        trials = np.unique(df_events['trial'])
        cumul_tr_len = np.arange(0, len(trials) * trial_len + trial_len, trial_len, dtype=int)
        shuffled_spikes_ts_allrois = []
        for n in range(2, df_events.shape[1]):
            all_spikes_ts = np.array([])
            # Find all timestamps of events for all trials for ROI 'n'
            for tr in trials:
                df_events_tr = df_events[df_events.trial == tr] # Extract trial 'tr'
                events_idx = np.array(df_events_tr.index[df_events_tr.iloc[:, n] == 1]) # Find indexes of events for ROI 'n' and trial 'tr'
                spikes_ts = np.array(df_events_tr.time[events_idx]) + trial_len*(tr-1) # Find timestamps of events for ROI 'n' and trial 'tr'
                all_spikes_ts = np.concatenate((all_spikes_ts, spikes_ts)) # Concatenate timestamps of events of each trial for ROI 'n'
            isi = np.diff(all_spikes_ts) # Compute ISI 
            for _ in range(iter_n):
                shuffled_spikes_ts_tr = []
                np.random.shuffle(isi) # Shuffle ISI
                shuffled_spikes_ts = np.insert(np.cumsum(isi), 0, 0) # Find new timestamps (whole session)
                for tr in trials:
                    shuffled_spikes_ts_tr.append(shuffled_spikes_ts[(cumul_tr_len[tr-1] < shuffled_spikes_ts) & (shuffled_spikes_ts <= cumul_tr_len[tr])] - (trial_len*(tr-1))) # List of shuffled timestamps for each trial for ROI 'n'
            shuffled_spikes_ts_allrois.append(shuffled_spikes_ts_tr) 
        return shuffled_spikes_ts_allrois


    def sta_shuffled(self, spikes_ts, variable, bcam_time, window, trials):
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
                        extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
                        signal_chunks = np.vstack((signal_chunks, extracted_signal))
                signal_chunks_tr.append(signal_chunks) # Array of traces for one ROI by trial
            # Compute STA by trial for one ROI     
            for tr_idx, _ in enumerate(trials): 
                sta_trial = np.mean(signal_chunks_tr[tr_idx], axis = 0)
                sta = np.vstack((sta, sta_trial))
            # STA all rois
            signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis = 0)) # List of raw traces for each ROI whole session
            sta_allrois.append(sta)
        return sta_allrois
    
    
    def plot_sta_shuffled(self, sta_zs, sta_roi, sta_chance, window, var_name, trials_ses, rois_sorted, mouse_id, session, colors_session, save_plot):
        ''' Plot heatmaps and traces of STA for observed data, shuffled data
        and observed data standardized on shuffled data.
        Inputs:
            - sta_zs: list of observed STA standardized on STA computed with shuffled data for each ROI
            - sta_roi: list of observed STA for each ROI
            - stsd_chance: list the standard deviations of the STA computed with shuffled data for each ROI
            - window: peri-event epoch (samples)
            - var_name: name of the variable (str)
            - trials_ses: 2D arrays with beginning and end trial for each experimental block
            - rois_sorted: array with all the ROIs sorted by cluster
            - mouse_id: mouse name (str)
            - session: ipsi (S1) or contra (S2) (str)
            - colors_session: dict with trials colorcode
            - save_plot (boolean)
        '''
        font_size = 15
        tick_labels = [block[1] for block in trials_ses]
        tick_locations = [block[1] - 1 for block in trials_ses]
        # Plot data
        for n in range(len(sta_roi)):
            fig, axs = plt.subplots(2, 3, figsize=(20, 8))
            vmin = min(sta_roi[n].min(), sta_chance[n].min())
            vmax = max(sta_roi[n].max(), sta_chance[n].max())
            # Subplots 1: heatmaps
            sns.heatmap(sta_roi[n], cmap='viridis', cbar = False, ax=axs[0, 0], vmin=vmin, vmax=vmax) # Observed STA
            sns.heatmap(sta_chance[n], cmap='viridis', cbar = False, ax=axs[0, 1], vmin=vmin, vmax=vmax) # Chance STA
            sns.heatmap(sta_zs[n], cmap='viridis', cbar = False, ax=axs[0, 2]) # Z-scored STA
            axs[0, 0].set_ylabel('Trials', fontsize = font_size)
            axs[0, 0].set_yticks(tick_locations)
            axs[0, 0].set_yticklabels(tick_labels)
            axs[0, 0].axvline(x=window[-1], color='white', linestyle='--')
            axs[0, 0].set(xticklabels=[])
            axs[0, 0].tick_params(bottom=False)
            axs[0, 1].axvline(x=window[-1], color='white', linestyle='--')
            axs[0, 1].set(xticklabels=[])
            axs[0, 1].set(yticklabels=[])
            axs[0, 1].tick_params(bottom=False, left=False)
            axs[0, 2].axvline(x=window[-1], color='white', linestyle='--')
            axs[0, 2].set(xticklabels=[])
            axs[0, 2].set(yticklabels=[])
            axs[0, 2].tick_params(bottom=False, left=False)
            # Subplots 2: traces
            for tr in range(sta_zs.shape[1]):
                axs[1, 0].plot(window * 1/self.sr_cam, sta_roi[n][tr], c = colors_session[tr+1]) # Observed STA
                axs[1, 1].plot(window * 1/self.sr_cam, sta_chance[n][tr], c = colors_session[tr+1]) # Chance STA
                axs[1, 2].plot(window * 1/self.sr_cam, sta_zs[n][tr], c = colors_session[tr+1]) # Z-scored STA
            ymin = min(axs[1, 0].get_ylim()[0], axs[1, 1].get_ylim()[0])
            ymax = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
            axs[1, 0].set_ylim(ymin, ymax)
            axs[1, 1].set_ylim(ymin, ymax)
            axs[1, 0].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 1].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 2].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 0].set_ylabel(var_name + ' (z-scored)', fontsize = font_size)
            axs[1, 0].set_xlabel('Time around event (s)', fontsize = font_size)
            axs[1, 0].spines['right'].set_visible(False)
            axs[1, 0].spines['top'].set_visible(False)
            axs[1, 0].axvline(x=0, color='k', linestyle='--')
            axs[1, 1].set_xlabel('Time around event (s)', fontsize = font_size)
            axs[1, 1].spines['right'].set_visible(False)
            axs[1, 1].spines['top'].set_visible(False)
            axs[1, 1].axvline(x=0, color='k', linestyle='--')
            axs[1, 2].set_xlabel('Time around event (s)', fontsize = font_size)
            axs[1, 2].spines['right'].set_visible(False)
            axs[1, 2].spines['top'].set_visible(False)
            axs[1, 2].axhline(y=-2, color='k', linestyle='--')
            axs[1, 2].axhline(y=2, color='k', linestyle='--')
            axs[1, 2].axvline(x=0, color='k', linestyle='--')
            # Save plots
            if save_plot:
                if not os.path.exists(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session)):
                    os.mkdir(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session))
                plt.savefig(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session + '\\', 'STA_' + var_name + '_' + str(rois_sorted[n]) + '.png'), dpi=self.my_dpi) 
                plt.close()


    def maxzs_distr(self, sta_zs, interval, signif_thresh, var_name, mouse_id, session, save_plot):
        ''' Find the absolute max z-score (so either the positive maximum or the negative maximum) of STA and the latency 
        of the peak from the event.
        Inputs:
            - sta_zs: array of z-scored STA for each ROI
            - interval = interval before and after the event to compute max abs z-score
            - signif_thresh: threshold of significance of the peak
            - var_name: name of the behavioral variable (str)
            - mouse_id: mouse name (str)
            - session: ipsi=1, contra = 2 (int)
            - save_plot (boolean)
        '''
        # Detect min and max z-score for every peak
        start = self.sr_cam+interval[0]
        end = self.sr_cam+interval[1]
        max_zs = np.zeros((sta_zs.shape[0], sta_zs.shape[1]))
        max_zs[:] = np.nan
        min_zs = np.zeros((sta_zs.shape[0], sta_zs.shape[1]))
        min_zs[:] = np.nan
        latency = np.zeros((sta_zs.shape[0], sta_zs.shape[1]))
        latency[:] = np.nan
        for n in range(sta_zs.shape[0]):
            for tr in range(sta_zs.shape[1]):
                max_zs[n, tr] = max(sta_zs[n, tr, start:end])
                min_zs[n, tr] = min(sta_zs[n, tr, start:end]) 
                latency[n, tr] = (np.argmax(sta_zs[n, tr, start:end])/self.sr_cam)+round(interval[0]/self.sr_cam, 2)

        # Plot distribution of max z-score for the whole session
        fig, axs = plt.subplots(2,1, figsize=(8,12))
        median_max_zs = np.nanmean(max_zs, axis=1)
        median_min_zs = np.nanmean(min_zs, axis=1)
        min_val = min(median_max_zs)
        max_val = max(median_max_zs)
        bin_size = 0.5
        bin_edges = np.arange(int(min_val), int(max_val) + bin_size, bin_size)
        hist, _ = np.histogram(median_max_zs, bins=bin_edges)
        axs[0].bar(bin_edges[:-1], hist, bin_size, color='white')
        for i in range(len(bin_edges)-1):
            if bin_edges[i] >= signif_thresh:
                axs[0].bar(bin_edges[i], hist[i], bin_size, color='yellow', edgecolor='black', linewidth=1.2)
            else:
                axs[0].bar(bin_edges[i], hist[i], bin_size, color='black', edgecolor='black', linewidth=1.2)
        axs[0].set_xlabel('Max z-score', fontsize = 15)
        axs[0].set_ylabel('Count', fontsize = 15)
        min_val = min(median_min_zs)
        max_val = max(median_min_zs)
        bin_edges = np.arange(int(min_val), int(max_val) + bin_size, bin_size)
        hist, _ = np.histogram(median_min_zs, bins=bin_edges)
        axs[1].bar(bin_edges[:-1], hist, bin_size, color='white')
        for i in range(len(bin_edges)-1):
            if bin_edges[i+1] <= -signif_thresh:
                axs[1].bar(bin_edges[i], hist[i], bin_size, color='blue', edgecolor='black', linewidth=1.2)
            else:
                axs[1].bar(bin_edges[i], hist[i], bin_size, color='black', edgecolor='black', linewidth=1.2)
        axs[1].set_xlabel('Min z-score', fontsize = 15)
        axs[1].set_ylabel('Count', fontsize = 15)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session)):
                os.mkdir(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session))
            plt.savefig(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session + '\\', 'Zs_distr.png'), dpi=self.my_dpi) 
        
        # Boxplot latency of max z-score for the whole session
        latency_negzs = latency[np.where(min_zs < signif_thresh)]
        latency_poszs = latency[np.where(max_zs > signif_thresh)]
        fig, ax = plt.subplots()
        boxplot = ax.boxplot([latency_negzs, latency_poszs], showfliers=False, patch_artist=True)
        ax.set_xticklabels([]) 
        ax.tick_params(bottom=False)
        colors = ['blue', 'yellow']
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        for median in boxplot['medians']:
            median.set(color='black', linewidth=1.7)
        for whisker in boxplot['whiskers']:
            whisker.set(color='black', linewidth=1.7)
        for cap in boxplot['caps']:
            cap.set(color='black', linewidth=1.7)
        ax.set_ylabel('Latency (s)', fontsize = 15)
        plt.show()
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session)):
                os.mkdir(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session))
            plt.savefig(os.path.join(self.save_path, 'STA_zs_' + var_name + '_' + mouse_id + '_' + session + '\\', 'Latency.png'), dpi=self.my_dpi) 
        
        return max_zs, min_zs, latency   

    
    def peak_detection(self, bodycenter, ampl, TimePntThres, trials):
        '''Use the derivative method to detect body position peaks, onsets and offsets for each trial and plot them
        Inputs:
            - bodycenter: 
            - ampl: minimal amplitude change considered as a valid event
            - TimePntThres: number of points that a change is expected to happen
            - trials = array with trial numbers
        '''
        onsets_tr = []
        peaks_tr = []
        offsets_tr =[]
        for tr_idx, tr in enumerate(trials):
            data = -bodycenter[tr_idx]
            # Derive noise amplitude
            TrueStd, deriv_mean, deriv_std = ST.DerivGauss_NoiseEstim(data, thres=2)
            AmpPntThres = TrueStd*ampl
            # Use that noise amplitude to define regions of slope change / stability
            IncremSet, DecremSet, F_Values = ST.SlopeThreshold(data, AmpPntThres, TimePntThres, CollapSeq=True, acausal=True, graph=None)
            # Detect onsets and peaks with the derivative method
            peaks = []
            onsets_pos = []
            if len(IncremSet) > 0:
                if type(IncremSet[0]) is tuple:
                    for i in range(len(IncremSet)):
                        onsets_pos.append(IncremSet[i][0])
                        if IncremSet[i][0] + TimePntThres >= len(data):
                            values_idx = np.arange(IncremSet[i][0], len(data))
                        else:
                            values_idx = np.arange(IncremSet[i][0], IncremSet[i][0] + TimePntThres)
                        peak_idx = np.argmax(data[values_idx])
                        peaks.append(IncremSet[i][0] + peak_idx)
            peaks_tr.append(peaks)
            onsets_tr.append(onsets_pos)
            # Detect offsets with the derivative method
            onsets_neg = []
            if len(DecremSet) > 0:
                if type(DecremSet[0]) is tuple:
                    for i in range(len(DecremSet)):        
                        onsets_neg.append(DecremSet[i][1]) # DecremSet[i][1] because we want the offset of the peak
            offsets_tr.append(onsets_neg)
            # Plot peaks by trial
            plt.figure()
            plt.plot(data)
            plt.scatter(onsets_pos, data[onsets_pos], s=80, marker = '.', c='g')
            plt.ylabel('- Body position', fontsize = 18)
            plt.xlabel('Samples', fontsize = 18)
            plt.scatter(peaks, data[peaks], s=80, marker = '.', c='k')
            plt.scatter(onsets_neg, data[onsets_neg], s=80, marker = '.', c='r')
            plt.title('Trial ' + str(tr))
        return peaks_tr, onsets_tr, offsets_tr


    def phasemap(self, final_tracks_trials_phase, df_events_trace_clusters, bcam_time, p1, colors_clusters, st_align, save_plot):
        '''Plot the phasemap for couple of paws and overimpose spikes. CODE NOT OPTIMIZED YET!!!
        Inputs:
            - final_tracks_trials_phase
            - df_events_trace_clusters
            - bcam_time
            - p1
            - colors_clusters = list of cluster colors
            - save_plot
            * Add sw-st-sw option
        '''
        p = np.array([0, 1, 2, 3])
        p = np.delete(p, np.where(p == p1))
        paws = ['FR','HR','FL','HL']
        for n in range(0,len(p)):
            if st_align == False:
                for tr in range(0,23):
                    fig, ax = plt.subplots()
                    plt.scatter(final_tracks_trials_phase[tr][0, p[n], :], final_tracks_trials_phase[tr][0, p1, :], s=3, c="gray", marker=".")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(1.2)
                    ax.spines['left'].set_linewidth(1.2)
                    ax.set_xlabel(paws[p[n]] + ' St-St Phase', fontsize=18)
                    ax.set_ylabel(paws[p1] + ' St-St Phase', fontsize=18)
                    fig.suptitle('Trial ' + str(tr+1), fontsize=18)
                    plt.show()
                    df_tr = df_events_trace_clusters.loc[df_events_trace_clusters['trial'] == tr+1].iloc[:, 2:]
                    event_idx = []
                    event_clust = []
                    for i in range(len(df_tr.index)):
                          for j in range(len(df_tr.columns)):
                              if df_tr.iloc[i, j] == 1:
                                event_idx.append(i)
                                event_clust.append(j)
                    event_ts = [df_events_trace_clusters['time'][i] for i in event_idx]
                    ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in event_ts])
                    colors = [colors_clusters[cluster] for cluster in event_clust]
                    plt.scatter(final_tracks_trials_phase[tr][0, p[n], ts_idx], final_tracks_trials_phase[tr][0,  p1, ts_idx], s=5, c=colors, marker="o")
                    if save_plot:
                        if not os.path.exists(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1])):
                            os.mkdir(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1]))
                        plt.savefig(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1] + '\\','trial' + str(tr+1) + '.png'), dpi=self.my_dpi)
            else:    
                st_pts_trials = []
                for tr in range (0,23):
                    st_pts = np.where(final_tracks_trials_phase[tr][0, p[n], :] == 0)[0]
                    st_pts_trials.append(st_pts)
                    # plt.figure()
                    # plt.plot(final_tracks_trials_phase[tr][0, 0, :])
                    # plt.scatter(st_pts, final_tracks_trials_phase[tr][0, 0, st_pts], color='red', marker='o')
                    
                st_centered_phase = []
                for tr in range(0,23):
                    center_on_st = [val - 1 if val > 0.5 else val for val in final_tracks_trials_phase[tr][0, p[n], :]]
                    st_centered_phase.append(center_on_st)
                st_centered_phase = [np.array(l) for l in st_centered_phase]
                
                for tr in range(0,23):
                    fig, ax = plt.subplots()
                    plt.scatter(st_centered_phase[tr], final_tracks_trials_phase[tr][0, p1, :], s=3, c="gray", marker=".")
                    ax.set_xlim(-0.5, 0.5)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([-0.5, 0, 0.5])
                    ax.set_yticks([0, 1])
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(1.2)
                    ax.spines['left'].set_linewidth(1.2)
                    ax.axvline(x=0, color='red', linestyle = '--')
                    ax.text(0, 1.02, 'St', ha='center', fontsize=12, color='red')
                    ax.set_xlabel(paws[p[n]] + ' Stance Phase', fontsize=18)
                    ax.set_ylabel(paws[p1] + ' St-St Phase', fontsize=18)
                    fig.suptitle('Trial ' + str(tr+1), fontsize=18)
                    plt.show()
                    df_tr = df_events_trace_clusters.loc[df_events_trace_clusters['trial'] == tr+1].iloc[:, 2:]
                    event_idx = []
                    event_clust = []
                    for i in range(len(df_tr.index)):
                          for j in range(len(df_tr.columns)):
                              if df_tr.iloc[i, j] == 1:
                                event_idx.append(i)
                                event_clust.append(j)
                    event_ts = [df_events_trace_clusters['time'][i] for i in event_idx]
                    ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in event_ts])
                    colors = [colors_clusters[cluster] for cluster in event_clust]
                    plt.scatter(st_centered_phase[tr][ts_idx], final_tracks_trials_phase[tr][0,  p1, ts_idx], s=5, c=colors, marker="o")
                    if save_plot:
                        if not os.path.exists(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1])):
                            os.mkdir(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1]))
                        plt.savefig(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1] + '\\','trial' + str(tr+1) + '.png'), dpi=self.my_dpi)                        