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
        self.sr_cam = 326 #sampling rate of behavior camera for treadmill
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
                        

    def sta(self, df_events, behavior, behav_name, window, save_plot):
        signal_chunks = []
        sta = []
        # sta_std = []
        # Loop over each column 
        for i in range(df_events.shape[1]):
            # Get the indices of all spike events for this column
            spike_idx = np.where(df_events.iloc[:, i] == 1)[0]
            # Loop over each spike index and extract the corresponding signal chunk
            for j in spike_idx:
                # Check that the window around this index does not extend past the beginning or end of the signal
                if j + window[0] >= 0 and j + window[-1] < behavior.shape[0]:
                    # Extract the signal chunk and append it to the list
                    signal_chunks.append(behavior[j + window[0]:j + window[-1] + 1])
            
            sta.append(np.mean(signal_chunks, axis=0))
            # sta_std.append(np.std(signal_chunks, axis=0))    
        max_val = max(max(arr) for arr in sta)
        min_val = min(min(arr) for arr in sta)
        for i in range(len(sta)):
            plt.figure()
            plt.plot(window*(1/self.sr), sta[i])
            # upper_bound = sta[i] + sta_std[i]
            # lower_bound = sta[i] - sta_std[i]
            # plt.fill_between(window*(1/30), upper_bound, lower_bound, alpha=0.2)
            plt.ylim([min_val, max_val])
            plt.axvline(x=0, color='k', linewidth = 1.5)
            plt.xlabel('Time around spike event (s)')
            plt.ylabel(behav_name)
            plt.title(df_events.columns[i])
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if save_plot:
                if not os.path.exists(os.path.join(self.save_path, 'STA')):
                    os.mkdir(os.path.join(self.save_path, 'STA'))
                plt.savefig(os.path.join(self.save_path, 'STA_' + df_events.columns[i] + '.png'), dpi=self.my_dpi)                        
        plt.show()