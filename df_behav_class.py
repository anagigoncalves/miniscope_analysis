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


class df_behav_analysis:
    
    
    def __init__(self,path):
        self.save_path = 'C:\\Users\\User\\Desktop\\Climbing fibers and instructive error signals\\Figures\\'
        self.pixel_to_mm = 1/1.955 #dana's setup
        self.sr_cam = 326 #sampling rate of behavior camera for treadmill
        self.sr = 30
        self.my_dpi = 128 #resolution for plotting
        self.trial_length = 0
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
    
    
    def df_behav_align(self, df_zs, df_clust, clusters_rois, frame_time, final_tracks_trials, sl_time_all_array, sl_sym_all_array, trials, plot_type, window, save_plot):
        '''Align dF/F (population heatmap or clusters traces) to behavior and plot the result for desired trials and windows. 
        Behaviors computed by the function are: body position (x-axis), speed, acceleration, step-length symmetry.
        Inputs:
            - df_zs = DataFrame of z-scored fluorescence for each ROI
            - df_clust = DataFrame of z-scored fluorescence for clusters
            - clusters_rois = list of ROIs belonging to each cluster
            - frame_time = list of miniscope timestamps for each trial
            - final_tracks_trials = list of final tracks for each trial, each item of the list is (4x5xframes)
            - sl_time_all_array = array of step-length symmetry timestamps
            - sl_sym_all_array = array of step-length symmetry values
            - trials = list of trials
            - plot_type = 'popul_heatmap' or 'cluster_traces'
            - window = list with beginning and end of your desired time window
            - save_plot = boolean (1 = save figures)
        '''
        # Initialize variables
        beg = window[0]
        end = window[1]
        data_list = []
        
        # Sort ROIs by cluster
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
        df_zs = df_zs[clusters_rois_flat]
        
        # Loop through trials
        for trial in range(trials[0], len(trials)+1):
            fig, axs = plt.subplots(5, 1, figsize=(12, 8))
        # Neural activity
            if plot_type == 'popul_heatmap': # Population dF/F heatmap
                df_trial = df_zs.loc[df_zs['trial'] == trial].iloc[beg*self.sr:end*self.sr, 2:] # Get df/f for the desired trial and interval
                data_list.append(df_trial)
                data = np.concatenate(data_list, axis=0)
                sns.heatmap(data.T, cbar=False, cmap='viridis', ax=axs[0])
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
                    df_trial = df_clust.loc[df_clust['trial'] == trial].iloc[beg*self.sr:end*self.sr, 2:]  # Get df/f for the desired trial and interval
                    count_r = 0
                    for r in df_clust.columns[2:]: # To plot stacked traces
                        axs[0].plot(frame_time[idx_trial][beg*self.sr:end*self.sr], df_trial[r] + (count_r / 2))
                        count_r += 1
                        axs[0].set_xlim([beg, end])
                        axs[0].set_ylabel('Clusters')
                        axs[0].spines['right'].set_visible(False)
                        axs[0].spines['top'].set_visible(False)
                        axs[0].spines['bottom'].set_visible(False)
                        axs[0].tick_params(left=False, bottom=False)
                        axs[0].set(xticklabels=[])
                        
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
            # if trial < 4 or trial > 13:
            #     bodyspeed_trial = bodyspeed_trial + self.tied_speed
            # else:
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