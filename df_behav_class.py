# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:46:24 2023

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from scipy.signal import savgol_filter, find_peaks
from scipy import stats


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
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
        df_events_sorted = df_events[clusters_rois_flat]
        return df_events_sorted, cluster_transition_idx
    
    
    def fr_distr_trial(self, df, trials, clusters_rois, colors_clusters, colors_session, save_plot):
        ''' Compute and plot firing rate distribution per trial and mean firing rate per trial (both average and by cluster)
        Inputs:
            - df = DataFrame of events for each ROI
            - trials = list of trials
            - clusters_rois = list of ROIs belonging to each cluster
            - colors_clusters = list of cluster colors
            - colors_session = list of colors for the session
            - save_plot = boolean (1 = save figures)
        '''
        trial_fr = []
        for trial in range(trials[0], len(trials)+1):
            trial_data = df[df['trial'] == trial].iloc[:, 2:] # Extract frames for the desired trial
            trial_fr.append(trial_data.mean(axis=0) * self.sr)  # Compute mean firing rate (multiply by 30 to get Hz)
        # std_fr = np.array([np.std(series) for series in trial_fr])  # Compute firing rate std
        # Plot mean firing rate across trials
        mean_fr = [np.mean(x) for x in trial_fr]
        fig2, ax2 = plt.subplots(figsize=(6, 8))
        ax2.plot(range(trials[0], len(trials)+1), mean_fr, c='k', alpha = 0.7, linewidth = 1.5)
        # ax2.errorbar(range(trials[0], len(trials)+1), mean_fr, yerr=std_fr, fmt='none', ecolor='k', alpha = 0.7, capsize=3)
        for count_t, t in enumerate(trials):
            idx_trial = np.where(trials==t)[0][0]
            ax2.scatter(t, mean_fr[idx_trial], s=100, color=colors_session[t], alpha = 1)
        ax2.axvspan(3.5, 13.5, facecolor='gray', alpha=0.4)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        ax2.set_xlabel('Trials', fontsize=18)
        ax2.set_ylabel('Mean Firing Rate (Hz)', fontsize=18)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        plt.show()
        # Save
        if save_plot:
            plt.savefig(os.path.join(self.save_path, 'fr_distr\\','fr_mean.png'), dpi=self.my_dpi)
            
        # Plot median firing rate across trials by cluster
        # Sort ROIs by cluster
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
        df = df[clusters_rois_flat]
        # Compute firing rate by cluster
        trial_fr=[]
        for trial in range(1,len(trials)+2): 
            arr = df[df['trial'] == trial].iloc[:, 2:] # Extract frames for the desired trial
            trial_fr.append(arr.mean(axis=0) * self.sr)  # Compute mean firing rate (multiply by 30 to get Hz)
        trial_fr = pd.concat(trial_fr, axis=1)
        means = {}
        for col in trial_fr.columns:
            col_data = trial_fr[col]
            sections = np.split(col_data, cluster_transition_idx[:-1])
            section_means = [np.mean(section) for section in sections]
            means[col] = section_means
        fr_clust = pd.DataFrame(means)
        t = np.arange(1, len(trials)+2)
        # Plot
        fig = plt.figure(figsize=(6, 8))
        for i in range(0, len(fr_clust)):
            plt.plot(t, fr_clust.iloc[i, :], marker='.', c=colors_clusters[i], markersize=10, linewidth=1.5)
        plt.xlabel('Trials', fontsize=18)
        plt.ylabel('Mean firing rate (Hz)', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        ax.axvspan(3.5, 13.5, facecolor='gray', alpha=0.4)
        plt.show()
        # Save
        if save_plot:
            plt.savefig(os.path.join(self.save_path, 'fr_distr\\','fr_mean_clust.png'), dpi=self.my_dpi)


    def find_behav_ts(self, df, bcam_time):
        ''' Find timestamps of behavioral recording that match the ones of neural
        activity
        Inputs:
            - df: dataframe of calcium activity with a column containing timestamps
            - bcam_time: timestamps of behavioral recording'''
            
        # Get a list of dF/F timestamps for each trial
        ts_trial = df.groupby('trial')['time'].apply(list) # group timestamps by trial and convert to list
        df_ts = [np.array(tr) for tr in ts_trial] # convert each group to a numpy array and put them in a list
        
        # Find corresponding timestamps in behavioral recording
        behav_ts_idx = []
        for tr in range(len(df_ts)):
            ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in df_ts[tr]])
            behav_ts_idx.append(ts_idx)
            
        return behav_ts_idx
        
    
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


    def phasemap(self, final_tracks_trials_phase, df_events_trace_clusters, bcam_time, p1, colors_clusters, st_align, save_plot):
        '''Plot the phasemap for couple of paws and overimpose spikes
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
                        

    # def sta(self, df_events, variable, var_name, window, save_plot):
    #     '''Compute and plot spike-triggered average for single ROIs and
    #     for the whole population. 
    #     The code assumes that neural activity and behavior are sampled at the same sampling rate.
    #     Inputs:
    #         - df_events: dataframe of events
    #         - variable: 1D array of the variable 
    #         - var_name: name of the variable (str)
    #         - window: peri-event epoch (samples)
    #         - save_plot (boolean)
    #     '''
    
    #     # Find index of trial transitions
    #     trial_changes = np.where(np.diff(np.asarray(df_events.iloc[:, 0].values)) != 0)[0] + 1
    #     trial_changes = np.concatenate(([0], trial_changes, [len(df_events)]))
        
    #     ##### EXTRACT TRACES AROUND EVENTS AND COMPUTE STA #####
    #     signal_chunks_allrois = []
    #     sta_allrois = []
    #     for n in range(2, df_events.shape[1]):
    #         signal_chunks = np.empty((0, len(window)))
    #         sta = np.empty((0, len(window)))
    #         # Get the indices of all events for one ROI
    #         events_idx = np.array(df_events.index[df_events.iloc[:, n] == 1])  
    #         events_tr = df_events['trial'][events_idx]
    #         # Extract traces around each event for one ROI
    #         for j in events_idx:
    #             if j + window[0] >= 0 and j + window[-1] < variable.shape[0]:
    #                 extracted_signal = variable[j + window[0]:j + window[-1] + 1]
    #                 extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
    #                 signal_chunks = np.vstack((signal_chunks, extracted_signal))  # Array of traces for one ROI for the whole session
    #             else:
    #                 del events_tr[j]
                    
    #         # Compute STA by trial for one ROI     
    #         for tr in range(1, len(trial_changes)): 
    #             sta_trial = np.mean(signal_chunks[np.array(events_tr) == tr], axis = 0)
    #             sta = np.vstack((sta, sta_trial))
    #         signal_chunks_allrois.append(signal_chunks) # List of raw traces for each ROI for the whole session
    #         sta_allrois.append(sta)
            
    #     # Sort data to have a list of the STA of all the ROIs for each trial
    #     sta_tr_allrois = []
    #     for tr in range(len(trial_changes)-1):
    #         sta_tr = [sta_roi[tr] for sta_roi in sta_allrois]
    #         sta_tr_allrois.append(sta_tr) # List of the STA of all the ROIs for each trial
            
    #     ##### PLOT DATA #####
    #     # Find min and max to set limits of the axis
    #     max_val = np.max(np.concatenate(signal_chunks_allrois, axis=0))
    #     min_val = np.min(np.concatenate(signal_chunks_allrois, axis=0))
    #     max_val_sta = np.max(np.concatenate(sta_allrois, axis=0))
    #     min_val_sta = np.min(np.concatenate(sta_allrois, axis=0))
    #     for n in range(df_events.shape[1]-2):
    #         fig, axs = plt.subplots(3, 1)
        
    #         # Plot heatmaps of all the raw traces for each single ROIs    
    #         hm = sns.heatmap(signal_chunks_allrois[n], cmap='viridis', ax=axs[0], vmin = min_val, vmax = max_val) # heatmap of traces for one neuron whole session
    #         axs[0].axvline(x=window[-1], color='white', linestyle='--')
    #         axs[0].set_ylabel('Events', fontsize = 15)
    #         axs[0].set(xticklabels=[])
    #         axs[0].set(yticklabels=[])
    #         axs[0].tick_params(left=False, bottom=False)
    #         cbar = hm.collections[0].colorbar
    #         cbar.set_label(var_name, fontsize = 15)
            
    #         # Plot heatmaps of STA for each trial for each single ROI   
    #         hm = sns.heatmap(sta_allrois[n], cmap='viridis', ax=axs[1], vmin = min_val_sta, vmax = max_val_sta) # heatmap of STA for one neuron by trial
    #         axs[1].set_ylabel('Trials', fontsize = 15)
    #         cbar = hm.collections[0].colorbar
    #         cbar.set_label(var_name, fontsize = 15)
    #         axs[1].axvline(x=window[-1], color='white', linestyle='--')
    #         axs[1].set(xticklabels=[])
    #         axs[1].set(yticklabels=[])
    #         axs[1].tick_params(left=False, bottom=False)
            
    #         # Plot STA traces for each trial for each single ROI        
    #         for tr in range(len(trial_changes)-1):
    #             axs[2].plot(window * 1/self.sr, sta_allrois[n][tr])
    #             axs[2].axvline(x=0, color='black', linestyle='--')
    #         axs[2].set_ylabel(var_name, fontsize = 15)
    #         axs[2].set_xlabel('Time around event (s)', fontsize = 15)
    #         axs[2].spines['right'].set_visible(False)
    #         axs[2].spines['top'].set_visible(False)
    #         axs[2].set_xlim(window[0] * 1/self.sr, window[-1] * 1/self.sr)
    #         axs[2].set_ylim([min_val_sta, max_val_sta])
            
    #         fig.suptitle('STA ' + df_events.columns[n], fontsize=15)
        
    #     # Plot STA of all the ROIs by trial          
    #     tick_values = [window[0]/self.sr, (1/2)*window[0]/self.sr, 0, (1/2)*window[-1]/self.sr, window[-1]/self.sr]
    #     max_val = np.max(np.concatenate(sta_tr_allrois, axis=0))
    #     min_val = np.min(np.concatenate(sta_tr_allrois, axis=0))
    #     for tr in range(len(trial_changes)-1):
    #         plt.figure()
    #         hm = sns.heatmap(sta_tr_allrois[tr], cmap='viridis', vmin = min_val, vmax = max_val)  # heatmap STA whole population by trial
    #         plt.ylabel('ROIs', fontsize=15)
    #         plt.xlabel('Time (s)', fontsize=15)
    #         cbar = hm.collections[0].colorbar
    #         cbar.set_label(var_name, fontsize=15)
    #         plt.axvline(x=window[-1], color='white', linestyle='--')
    #         plt.yticks([])
    #         plt.tick_params(left=False)
    #         x_ticks = np.linspace(0, len(sta_tr_allrois[tr][0]), len(tick_values)).astype(int)
    #         plt.xticks(x_ticks, [f"{tick}" for tick in tick_values])
    #         plt.title('STA trial ' + str(tr+1), fontsize=15)

    #     return sta_tr_allrois, sta_allrois, signal_chunks_allrois


    def sta(self, df_events, variable, var_name, bcam_time, window, zs_signal, plot_data, save_plot):
        '''Compute and plot spike-triggered average for single ROIs and
        for the whole population (sorted by cluster).
        The code works also neural traces and behavioral traces sampled at different sampling rates.
        Inputs:
            - df_events: dataframe of events
            - variable: 1D array of the variable 
            - var_name: name of the variable (str)
            - bcam_time: timestamps of behavior
            - window: peri-event epoch (samples)
            - zs_signal (boolean)
            - plot_data (boolean)
            - save_plot (boolean)
        '''
            
        # Find index of trial transitions
        trial_changes = np.where(np.diff(np.asarray(df_events.iloc[:, 0].values)) != 0)[0] + 1
        trial_changes = np.concatenate(([0], trial_changes, [len(df_events)]))
        
        blocks = [(1, 3), (3, 8), (8, 13), (13, 18), (18, 23)]
                
        ##### EXTRACT TRACES AROUND EVENTS AND COMPUTE STA #####
        signal_chunks_allrois = []
        sta_allrois = []
        for n in range(2, df_events.shape[1]):
            sta = np.empty((0, len(window)))
            signal_chunks_tr = []
            for tr in range(23):
                signal_chunks = np.empty((0, len(window)))
                df_tr = df_events[df_events['trial']==tr+1]
                events_idx = np.array(df_tr.index[df_tr.iloc[:, n] == 1])  
                events_ts = np.array(df_tr['time'][events_idx])
                matching_ts_idx = [np.abs(bcam_time[tr] - ts).argmin() for ts in events_ts]
                # Extract traces around each event for one ROI
                for j in matching_ts_idx:
                    if j + window[0] >= 0 and j + window[-1] < len(variable[tr]):
                        extracted_signal = variable[tr][j + window[0]:j + window[-1] + 1]
                        if zs_signal == True:
                            extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
                        signal_chunks = np.vstack((signal_chunks, extracted_signal))
                signal_chunks_tr.append(signal_chunks) # Array of traces for one ROI by trial
                    
            # Compute STA by trial for one ROI     
            for tr in range(23): 
                sta_trial = np.mean(signal_chunks_tr[tr], axis = 0)
                sta = np.vstack((sta, sta_trial))
            
            signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis = 0)) # List of raw traces for each ROI whole session
            sta_allrois.append(sta)
            
        # Sort data to have a list of the STA of all the ROIs for each trial
        sta_tr_allrois = []
        for tr in range(len(trial_changes)-1):
            sta_tr = [sta_roi[tr] for sta_roi in sta_allrois]
            sta_tr_allrois.append(sta_tr) # List of the STA of all the ROIs for each trial

        # Compute STA of all the ROIs for each block
        sta_blocks_allrois = []
        for start, end in blocks:
            sta_block =  np.mean(np.array(sta_tr_allrois[start:end]), axis=0)
            sta_blocks_allrois.append(sta_block) # List of the STA of all the ROIs for block
            
        ##### PLOT DATA #####
        if plot_data == True:
            # Find min and max to set limits of the axis
            max_val = np.max(np.concatenate(signal_chunks_allrois, axis=0))
            min_val = np.min(np.concatenate(signal_chunks_allrois, axis=0))
            max_val_sta = np.max(np.concatenate(sta_allrois, axis=0))
            min_val_sta = np.min(np.concatenate(sta_allrois, axis=0))
            for n in range(df_events.shape[1]-2): 
                fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            
                # Plot heatmaps of all the raw traces for each single ROIs    
                hm = sns.heatmap(signal_chunks_allrois[n], cbar = False, cmap='viridis', ax=axs[0], vmin = min_val, vmax = max_val) # heatmap of traces for one neuron whole session
                axs[0].axvline(x=window[-1], color='white', linestyle='--')
                axs[0].set_ylabel('Events', fontsize = 15)
                axs[0].set(xticklabels=[])
                axs[0].set(yticklabels=[])
                axs[0].tick_params(left=False, bottom=False)
                # cbar = hm.collections[0].colorbar
                # cbar.set_label(var_name, fontsize = 15)
                
                # Plot heatmaps of STA for each trial for each single ROI   
                hm = sns.heatmap(sta_allrois[n], cbar = False, cmap='viridis', ax=axs[1], vmin = min_val_sta, vmax = max_val_sta) # heatmap of STA for one neuron by trial
                axs[1].set_ylabel('Trials', fontsize = 15)
                tick_locations = [2, 12, 22]  # Indices of the desired tick locations
                tick_labels = [3, 13, 23]  # Corresponding tick labels
                axs[1].set_yticks(tick_locations)
                axs[1].set_yticklabels(tick_labels)
                # cbar = hm.collections[0].colorbar
                # cbar.set_label(var_name, fontsize = 15)
                axs[1].axvline(x=window[-1], color='white', linestyle='--')
                axs[1].set(xticklabels=[])
                axs[1].tick_params(bottom=False)
                
                # Plot STA traces for each trial for each single ROI        
                for tr in range(23):
                    axs[2].plot(window * 1/self.sr_cam, sta_allrois[n][tr], c = 'lightgray')
                axs[2].axvline(x=0, color='black', linestyle='--')
                axs[2].plot(window * 1/self.sr_cam, np.mean(sta_allrois[n][0:3], axis=0), c='k', linewidth=2.3)
                axs[2].plot(window * 1/self.sr_cam, np.mean(sta_allrois[n][3:13], axis=0), c='crimson', linewidth=2.3)
                axs[2].plot(window * 1/self.sr_cam, np.mean(sta_allrois[n][13:23], axis=0), c='navy', linewidth=2.3)
                axs[2].set_ylabel(var_name, fontsize = 15)
                axs[2].set_xlabel('Time around event (s)', fontsize = 15)
                axs[2].spines['right'].set_visible(False)
                axs[2].spines['top'].set_visible(False)
                axs[2].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
                axs[2].set_ylim([min_val_sta, max_val_sta])
                
                fig.suptitle('STA ' + var_name + ' ' + df_events.columns[n+2], fontsize=15)
                
                if save_plot:
                    if not os.path.exists(os.path.join(self.save_path, 'STA')):
                        os.mkdir(os.path.join(self.save_path, 'STA'))
                    plt.savefig(os.path.join(self.save_path, 'STA\\', 'STA_' + var_name + '_' + df_events.columns[n+2] + '.png'), dpi=self.my_dpi) 
        
            # Plot STA of all the ROIs by trial 
            tick_values = [round(window[0]/self.sr_cam,1), round((1/2)*window[0]/self.sr_cam,1), 0, round((1/2)*window[-1]/self.sr_cam,1), round(window[-1]/self.sr_cam,1)]
            max_val = np.max(np.concatenate(sta_tr_allrois, axis=0))
            min_val = np.min(np.concatenate(sta_tr_allrois, axis=0))
            for tr in range(len(trial_changes)-1):
                plt.figure()
                hm = sns.heatmap(sta_tr_allrois[tr], cmap='viridis', vmin = min_val, vmax = max_val)  # heatmap STA whole population by trial 
                plt.ylabel('ROIs (sorted by ML dist)', fontsize=15)
                plt.xlabel('Time (s)', fontsize=15)
                cbar = hm.collections[0].colorbar
                cbar.set_label(var_name, fontsize=15)
                plt.axvline(x=window[-1], color='white', linestyle='--')
                plt.yticks([])
                plt.tick_params(left=False)
                x_ticks = np.linspace(0, len(sta_tr_allrois[tr][0]), len(tick_values)).astype(int)
                plt.xticks(x_ticks, [f"{tick}" for tick in tick_values])
                plt.title('STA ' + var_name + ' trial ' + str(tr+1), fontsize=15)
                if save_plot:
                    if not os.path.exists(os.path.join(self.save_path, 'STA')):
                        os.mkdir(os.path.join(self.save_path, 'STA'))
                    plt.savefig(os.path.join(self.save_path, 'STA\\', 'STA_' + var_name + '_trial' + str(tr+1) + '.png'), dpi=self.my_dpi)
                
            # Plot STA of all the ROIs by block 
            max_val = np.max(np.concatenate(sta_blocks_allrois, axis=0))
            min_val = np.min(np.concatenate(sta_blocks_allrois, axis=0))
            fig, axs = plt.subplots(5,1, figsize = (12, 12))
            for block in range(len(blocks)):
                hm = sns.heatmap(sta_blocks_allrois[block], cmap='viridis', ax = axs[block], vmin = min_val, vmax = max_val)  # heatmap STA whole population by block 
                axs[block].axvline(x=window[-1], color='white', linestyle='--')
                axs[block].set_ylabel('ROIs', fontsize = 15)
                axs[block].set(xticklabels=[])
                axs[block].set(yticklabels=[])
                if block < 4:
                    axs[block].tick_params(left=False, bottom=False)
                if block == 2:
                    cbar = hm.collections[0].colorbar
                    cbar.set_label(var_name, fontsize=15)
                plt.xlabel('Time (s)', fontsize=15)
                plt.axvline(x=window[-1], color='white', linestyle='--')
                plt.yticks([])
                plt.tick_params(left=False)
                x_ticks = np.linspace(0, len(sta_tr_allrois[tr][0]), len(tick_values)).astype(int)
                plt.xticks(x_ticks, [f"{tick}" for tick in tick_values], fontsize = 12)
                fig.suptitle('STA ' + var_name, fontsize = 15)
            if save_plot:
                if not os.path.exists(os.path.join(self.save_path, 'STA')):
                    os.mkdir(os.path.join(self.save_path, 'STA'))
                plt.savefig(os.path.join(self.save_path, 'STA\\', 'STA_' + var_name + '_blocks' + '.png'), dpi=self.my_dpi)
        
        return sta_tr_allrois, sta_allrois, signal_chunks_allrois


    def sta_shuffled(self, spikes_ts, variable, var_name, bcam_time, window):
        '''Compute and plot spike-triggered average for single ROIs and
        for the whole population (sorted by cluster).
        The code works also neural traces and behavioral traces sampled at different sampling rates.
        Inputs:
            - spikes_ts = nested lists of spikes timestamps by trial for each neuron
            - variable: 1D array of the variable 
            - var_name: name of the variable (str)
            - bcam_time: timestamps of behavior
            - window: peri-event epoch (samples)
        '''
    
        ##### EXTRACT TRACES AROUND EVENTS AND COMPUTE STA #####
        signal_chunks_allrois = []
        sta_allrois = []
        for n in range(len(spikes_ts)):
            sta = np.empty((0, len(window)))
            signal_chunks_tr = []
            for tr in range(23):
                signal_chunks = np.empty((0, len(window)))
                events_ts = np.array(spikes_ts[n][tr])
                matching_ts_idx = [np.abs(bcam_time[tr] - ts).argmin() for ts in events_ts]
                # Extract traces around each event for one ROI
                for j in matching_ts_idx:
                    if j + window[0] >= 0 and j + window[-1] < len(variable[tr]):
                        extracted_signal = variable[tr][j + window[0]:j + window[-1] + 1]
                        extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
                        signal_chunks = np.vstack((signal_chunks, extracted_signal))
                signal_chunks_tr.append(signal_chunks) # Array of traces for one ROI by trial
            
            # Compute STA by trial for one ROI     
            for tr in range(23): 
                sta_trial = np.mean(signal_chunks_tr[tr], axis = 0)
                sta = np.vstack((sta, sta_trial))
            
            signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis = 0)) # List of raw traces for each ROI whole session
            sta_allrois.append(sta)
        return sta_allrois
    
    
    def shuffle_spikes_ts(self, df_events_extract, iter_n):
        shuffled_spikes_ts_alliter = []
        trial_len = round(max(df_events_extract.iloc[:,1]))
        trials_n = np.unique(df_events_extract.iloc[:,0])[-1]
        cum_tr_len = np.arange(0, (trials_n * trial_len) + trial_len, trial_len, dtype=int)
        for i in range(iter_n):
            shuffled_spikes_ts_all = []
            for n in range(2, df_events_extract.shape[1]):
                all_spikes_ts = np.array([])
                shuffled_spikes_ts_tr = []
                for tr in range(1, trials_n + 1):
                    df_events_tr = df_events_extract[df_events_extract.trial == tr]
                    events_idx = np.array(df_events_tr.index[df_events_tr.iloc[:, n] == 1])
                    spikes_ts = np.array(df_events_tr.time[events_idx]) + trial_len*(tr-1)
                    all_spikes_ts = np.concatenate((all_spikes_ts, spikes_ts)) 
                isi = np.diff(all_spikes_ts)
                np.random.shuffle(isi)
                shuffled_spikes_ts = np.insert(np.cumsum(isi), 0, 0)
                for j in range(1, trials_n + 1):
                    shuffled_spikes_ts_tr.append(shuffled_spikes_ts[(cum_tr_len[j-1] < shuffled_spikes_ts) & (shuffled_spikes_ts <= cum_tr_len[j])] - (60*(j-1)))
                shuffled_spikes_ts_all.append(shuffled_spikes_ts_tr)
            shuffled_spikes_ts_alliter.append(shuffled_spikes_ts_all)
        return shuffled_spikes_ts_alliter
    
    
    def plot_sta_shuffled(self, sta_roi, sta_chance, stsd_chance, df_events, window, var_name):
        sta_zs = np.zeros((df_events_extract.shape[1]-2, trials_n, len(window)))
        trials_n = np.unique(df_events_extract.iloc[:,0])[-1]
        for n in range(df_events_extract.shape[1]-2):
            for tr in range(trials_n):
                sta_mean_sub = (sta_roi[n][tr] - sta_chance[n,tr])
                sta_zs[n, tr] = [x / y for x, y in zip(sta_mean_sub, stsd_chance[n,tr])]
        
        tick_locations = [2, 12] 
        tick_labels = [3, 13]
        for n in range(df_events_extract.shape[1]-2):
            fig, axs = plt.subplots(2, 3, figsize=(20, 8))
            vmin = min(sta_roi[n].min(), sta_chance[n].min())
            vmax = max(sta_roi[n].max(), sta_chance[n].max())
            # Heatmaps
            sns.heatmap(sta_roi[n], cmap='viridis', cbar = False, ax=axs[0, 0], vmin=vmin, vmax=vmax)
            sns.heatmap(sta_chance[n], cmap='viridis', cbar = False, ax=axs[0, 1], vmin=vmin, vmax=vmax)
            sns.heatmap(sta_zs[n], cmap='viridis', cbar = False, ax=axs[0, 2])
            axs[0, 0].set_ylabel('Trials', fontsize = 15)
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
            # Traces
            for tr in range(trials_n):
                axs[1, 0].plot(window * 1/self.sr_cam, sta_roi[n][tr])
                axs[1, 1].plot(window * 1/self.sr_cam, sta_chance[n][tr])
                axs[1, 2].plot(window * 1/self.sr_cam, sta_zs[n][tr])
            ymin = min(axs[1, 0].get_ylim()[0], axs[1, 1].get_ylim()[0])
            ymax = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
            axs[1, 0].set_ylim(ymin, ymax)
            axs[1, 1].set_ylim(ymin, ymax)
            axs[1, 0].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 1].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 2].set_xlim(window[0] * 1/self.sr_cam, window[-1] * 1/self.sr_cam)
            axs[1, 0].set_ylabel(var_name, fontsize = 15)
            axs[1, 0].set_xlabel('Time around event (s)', fontsize = 15)
            axs[1, 0].spines['right'].set_visible(False)
            axs[1, 0].spines['top'].set_visible(False)
            axs[1, 0].axvline(x=0, color='k', linestyle='--')
            axs[1, 1].set_xlabel('Time around event (s)', fontsize = 15)
            axs[1, 1].spines['right'].set_visible(False)
            axs[1, 1].spines['top'].set_visible(False)
            axs[1, 1].axvline(x=0, color='k', linestyle='--')
            axs[1, 2].set_xlabel('Time around event (s)', fontsize = 15)
            axs[1, 2].spines['right'].set_visible(False)
            axs[1, 2].spines['top'].set_visible(False)
            axs[1, 2].axhline(y=-2, color='k', linestyle='--')
            axs[1, 2].axhline(y=2, color='k', linestyle='--')
            axs[1, 2].axvline(x=0, color='k', linestyle='--')
            
            if not os.path.exists(os.path.join(self.save_path, 'STA_zs')):
                os.mkdir(os.path.join(self.save_path, 'STA_zs'))
            plt.savefig(os.path.join(self.save_path, 'STA_zs\\', 'STA_' + var_name + '_' + df_events.columns[n+2] + '.png'), dpi=128) 
        
        return sta_zs
    
    
    def kinematic(self, final_tracks_trials, trials, behav_ts_idx, win_len, polyorder):
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