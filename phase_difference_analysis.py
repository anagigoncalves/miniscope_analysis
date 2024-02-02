# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:55:35 2024

@author: User
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import warnings
from scipy.ndimage import gaussian_filter1d
from scipy.stats import circmean
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

# Import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
import behav_locked_neural_activity_class
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
blna = behav_locked_neural_activity_class.behav_locked_neural_activity('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
    
path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'

plot_data = True
save_plot = True
bins_time = np.arange(-0.250, 0.252, 0.0125) # 12.5 ms
paws = ['FR', 'HR', 'FL', 'HL']
color_paws = ['red', 'magenta', 'blue', 'cyan']
p1 = 0; p2 = 3

temporal_dimension = 'time'
align = 'async_pk'

# Utils 
def remove_outliers(arr):
    # Identify NaN values
    nan_values = np.isnan(arr)
    # Calculate the 1st and 99th percentiles excluding NaN values
    lower_bound = np.percentile(arr[~nan_values], 0)
    upper_bound = np.percentile(arr[~nan_values], 95)
    print(lower_bound, upper_bound)
    # Identify outliers excluding NaN values
    outliers = ((arr < lower_bound) | (arr > upper_bound))
    # Replace outliers with NaN
    arr_with_nan = np.copy(arr)
    arr_with_nan[outliers] = np.nan
    return arr_with_nan

def flatten_list(my_list):
    return [item for array in my_list for item in array]

def change_angle_range(array):
    return array - 2 * np.pi * np.round(array / (2 * np.pi))

# def circular_dist(angle1, angle2):
#     difference = angle1 - angle2
#     return (difference + math.pi) % (2 * math.pi) - math.pi # Normalize the result to [-π, π)
def circular_dist(angle1, angle2):
    return angle1 - angle2

def get_events(df_events):
    trials = np.unique(df_events['trial'])
    events_ts_allrois = []
    for n in range(2, df_events.shape[1]):
        events_ts = []
        for tr_idx, tr in enumerate(trials):
            df_events_trial = df_events[df_events['trial'] == tr]
            df_events_trial.reset_index(drop = True, inplace=True)
            events_idx = np.where(df_events_trial.iloc[:, n] == 1)[0]
            events_ts.append(df_events_trial['time'].loc[events_idx].values)
        events_ts_allrois.append(events_ts)
    return events_ts_allrois

def map_timestamps(timestamps1, timestamps2):    
    return np.array([np.argmin(np.abs(timestamps2 - t)) for t in timestamps1])

def find_peaks_with_nans(data, lower_threshold=None, upper_threshold=None, distance=None):
    # Find peaks in the data ignoring NaN values
    peaks, _ = find_peaks(data, height=lower_threshold, distance=distance)
    # Apply upper threshold to filter out peaks above a certain value
    if upper_threshold is not None:
        peaks = [peak for peak in peaks if data[peak] <= upper_threshold]
    # Identify onset and offset indices for each peak
    onset_indices = []
    offset_indices = []
    for peak in peaks:
        # Find the onset index
        onset_index = np.argmax(data[:peak] >= lower_threshold)
        onset_indices.append(onset_index)
        # Find the offset index
        offset_index = np.argmax(data[peak:] < lower_threshold) + peak
        offset_indices.append(offset_index)
    return onset_indices, peaks, offset_indices

def find_closest_timestamps(t1, t2):
    closest_indices = np.searchsorted(t2, t1)
    closest_indices = np.clip(closest_indices, 1, len(t2)-1)
    before = t2[closest_indices - 1]
    after = t2[closest_indices]
    closest_timestamps = np.where(np.abs(t1 - before) < np.abs(t1 - after), before, after)
    return closest_timestamps, closest_indices - 1, closest_indices

def sort_array(array1, array2):
    # Pair elements from both arrays and sort based on values in array1
    sorted_indices = sorted(range(len(array1)), key=lambda k: array1[k], reverse=True)
    # Rearrange elements in array2 based on sorted indices
    sorted_array2 = [array2[i] for i in sorted_indices]
    # Return the sorted array1 and corresponding sorted array2
    return sorted(array1, reverse=True), sorted_array2

firing_rate_blocks_allanimals = []
for s in range(len(session_data)):
    # Load animal info for a session
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = ses_info[0]
    
    # Import classes
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Load session data
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [_, _, _, _, df_events_extract_rawtrace, _, _, _, trials,
      clusters_rois, colors_cluster, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
    [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, _, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)
    # Sort ROIs in the dataframe of neural activity by cluster
    df_spikes, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois)
    # Flatten the list of ROIs 'clusters_rois'
    rois_sorted = []
    for i in range(len(clusters_rois)):
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
    # Further sub-divide experimental blocks
    if session_id == 'tied':
        split_blocks = trials_ses
    else:
        split_blocks = blna.split_expblocks(trials_ses)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    # Compute gait parameters
    st_strides_trials = []
    sw_strides_trials = []
    final_tracks_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, _, _, _, _, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-sw-st')

################################################ STEP 1: BEHAVIOR ##################################################
    # Get phase difference (in degrees)
    trial_id = np.array(flatten_list([np.repeat(tr+1, trial.shape[-1]) for tr, trial in enumerate(final_tracks_trials_phase)]))
    phase_difference = np.array(flatten_list(loco.phase_diff(final_tracks_trials_phase, paws[p1], paws[p2], 'X')[0]))
    phase_difference_mean = np.array([circmean(phase_difference[np.where(trial_id == tr)[0]], nan_policy='omit') for tr in trials])
    interlimb_asym = [np.array(circular_dist(phase_difference[np.where(trial_id == tr)[0]], phase_difference_mean[tr_idx])) for tr_idx, tr in enumerate(trials)]
    interlimb_asym = [np.degrees(interlimb_asym[tr_idx]) for tr_idx, _ in enumerate(trials)]
    
    # Set
    temporal_dataset = bcam_time
    bins = bins_time
        
    # Find peaks and onsets/offsets indices and timestamps
    on_idx = []; pk_idx = []; off_idx = []; on_ts = []; pk_ts = []; off_ts = []
    for tr_idx, data_tr in enumerate(interlimb_asym):
        # _, pk, _ = find_peaks_with_nans(data_tr, threshold=50, distance=50)
        _, pk, _ = find_peaks_with_nans(data_tr, lower_threshold=40, upper_threshold=100, distance=50)
        pk_idx.append(pk)
        pk_ts.append(temporal_dataset[tr_idx][pk])
        on, _, _ = find_closest_timestamps(temporal_dataset[tr_idx][pk]-0.300, bcam_time[tr_idx])
        on_ts.append(on)
        off, _, _ = find_closest_timestamps(temporal_dataset[tr_idx][pk]+0.300, bcam_time[tr_idx])
        off_ts.append(off)
        if plot_data:
            plt.figure(figsize = (20, 10))
            plt.plot(data_tr)
            plt.scatter(pk, data_tr[pk], s=150, marker = '.', c='k')
            plt.title(f'trial {tr_idx+1}')
            # Save
            if save_plot:
                if not os.path.exists(os.path.join(save_path, f'{align} {animal} {session_id}')):
                    os.mkdir(os.path.join(save_path, f'{align} {animal} {session_id}'))
                plt.savefig(os.path.join(save_path, f'{align} {animal} {session_id}\\', f'{align} trial{tr_idx+1} {animal} {session_id}.png'))
            plt.close()

    # In the behavioral dataset, find index of spikes occurring near each behavioral event                      
    spikesIdx_behavData = {}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    spikesIdx_behavData = []
    for n in range(2, df_spikes.shape[1]):
        spikesIdx_behavData_roi = blna.get_spikes_behav(df_spikes.iloc[:, [0,1,n]], on_ts, off_ts, pk_ts, bins, temporal_dataset, temporal_dimension)
        spikesIdx_behavData.append(spikesIdx_behavData_roi)
    
    # Find time or phase at which each spike occurred with respect to the peak
    spikes_timing = []
    for n in range(len(spikesIdx_behavData)):
        spikes_timing_roi = blna.get_spikes_timing(spikesIdx_behavData[n], temporal_dataset, pk_ts, temporal_dimension)
        spikes_timing.append(spikes_timing_roi)
    
    # Bin spikes in time or phase 
    spikes_count = []
    for n in range(len(spikes_timing)):
        spikes_count_roi = blna.bin_spikes(spikes_timing[n], bins)
        spikes_count.append(spikes_count_roi)
    
    # Compute firing rate
    firing_rate = []
    for n in range(len(spikes_count)):
        firing_rate.append([np.sum(np.vstack(spikes_count[n][tr_idx]), axis = 0)/len(spikes_count[n][tr_idx]) for tr_idx, _ in enumerate(trials)])
    
    # Pre-process firing rate
    firing_rate = [gaussian_filter1d(firing_rate_roi, sigma=1.5, axis=1) for firing_rate_roi in firing_rate]

    # Plot firing rate aligned to asymmetry peaks for each ROIs by block
    font_size = 15  
    x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
    x_ticks = [0, (len(bins)-1)/2, len(bins)-1] 
    for n in range(len(firing_rate)):                
        fig, axs = plt.subplots(3,1, figsize=(6, 15))
        spikes_timing_all = [ts for tr in spikes_timing[n] for ts in tr]
        peak_idx = [np.full_like(ts, idx) for idx, ts in enumerate(spikes_timing_all)]
        axs[0].scatter(np.concatenate(spikes_timing_all), np.concatenate(peak_idx), c = 'dimgrey', marker = '.', s = 5)
        axs[0].set_xlim(bins[0], bins[-1])
        sns.heatmap(np.flipud(firing_rate[n]), cmap = 'viridis', cbar = False, ax=axs[1])
        [axs[2].plot(np.nanmean(firing_rate[n][start:end], axis = 0), c = colors_session[start+1]) for start, end in split_blocks]
        axs[2].set_xlim(0, len(bins)-2)
        axs[0].axvline(x=0, linestyle = '-', color = 'crimson')
        axs[1].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'white')
        axs[2].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'k')
        axs[0].set_ylabel('Asymmetry peaks', fontsize = font_size)
        axs[1].set_ylabel('Trials', fontsize = font_size)
        axs[2].set_ylabel('p(CSpk)', fontsize = font_size)
        axs[0].set(xticklabels=[])
        axs[0].tick_params(bottom=False)
        axs[1].set(xticklabels=[])
        axs[1].tick_params(bottom=False)
        axs[1].set(yticklabels=[])
        axs[1].tick_params(left=False)
        axs[2].set_xticks(x_ticks)
        axs[2].set_xticklabels(x_tick_labels, fontsize = font_size)
        for i in range(3):
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
        plt.tight_layout
        plt.suptitle(f'Neural activity {rois_sorted[n]} locked to asymmetry peaks', fontsize = font_size)
        plt.show()
        # Save
        if save_plot:
            if not os.path.exists(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
                os.mkdir(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
            plt.savefig(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} {rois_sorted[n]} {animal} {session_id}.png'))
        plt.close()
    
    # Compute firing rate population for each block & plot it
    firing_rate_blocks = []
    for start, end in split_blocks:
        firing_rate_block = []
        for n in range(len(firing_rate)):
            firing_rate_block.append(np.nanmean(firing_rate[n][start:end], axis = 0))
        firing_rate_blocks.append(firing_rate_block)
        
    # Plot firing rate aligned to asymmetry peaks for each ROIs by block    
    fig, axs = plt.subplots(len(firing_rate_blocks), 1, figsize=(4, 20))
    for b in range(len(firing_rate_blocks)):                
        sns.heatmap(np.flipud(firing_rate_blocks[b]), cmap = 'viridis', cbar = True, ax=axs[b], vmin = 0.01, vmax = 0.06, cbar_kws={'label': 'p(CSpks)'})
        axs[b].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'white')
        axs[b].set(xticklabels=[])
        axs[b].tick_params(bottom=False)
        axs[b].set(yticklabels=[])
        axs[b].tick_params(left=False)
        axs[b].spines['top'].set_visible(False)
        axs[b].spines['right'].set_visible(False)
        plt.tight_layout
        plt.show()
    axs[2].set_ylabel('Asymmetry peaks', fontsize = font_size)
    axs[4].set_xticks(x_ticks)
    axs[4].set_xticklabels(x_tick_labels, fontsize = font_size)
    # Save
    if save_plot:
        if not os.path.exists(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
            os.mkdir(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
        plt.savefig(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} population {animal} {session_id}.png'))
    plt.close()
        
    firing_rate_blocks_allanimals.append(firing_rate_blocks) 
    
    # Find peaks values and sort rasters accordingly (instead of time). Then plot activity clustered by percentile range
    peaks_values_tr = [interlimb_asym[tr_idx][pk_idx[tr_idx]] for tr_idx, _ in enumerate(trials)]
    peaks_values = flatten_list(peaks_values_tr)
    sorted_rasters = []
    for n in range(len(spikes_timing)):
        spikes_timing_all = [ts for tr in spikes_timing[n] for ts in tr]
        sorted_rasters.append(sort_array(peaks_values, spikes_timing_all)[1])
    for n in range(len(firing_rate)):
        plt.figure(figsize=(4, 4))
        peak_idx = [np.full_like(ts, idx) for idx, ts in enumerate(sorted_rasters[n])]
        plt.scatter(np.concatenate(sorted_rasters[n]), np.concatenate(peak_idx), c='dimgrey', marker='.', s=5)
        plt.xlim(bins[0], bins[-1])
        plt.axvline(x=0, linestyle='-', color='crimson')
        plt.ylabel('Asymmetry peaks', fontsize=font_size)
        plt.xticks([])  
        plt.tick_params(bottom=False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.suptitle(f'Neural activity {rois_sorted[n]} locked to sorted asymmetry peaks', fontsize=font_size)
        plt.show()
        # Save
        if save_plot:
            if not os.path.exists(os.path.join(save_path, f'sorted {align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
                os.mkdir(os.path.join(save_path, f'sorted {align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
            plt.savefig(os.path.join(save_path, f'sorted {align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'sorted {align}-locked neural activity {temporal_dimension} {rois_sorted[n]} {animal} {session_id}.png'))
        plt.close()    

# Plot firing rate aligned to body acceleration peaks for each ROIs by block    
animal_transition_idx = np.cumsum(np.array([len(firing_rate_animal[0]) for firing_rate_animal in firing_rate_blocks_allanimals]))
firing_rate_blocks_all = [np.concatenate([firing_rate_animal[b] for firing_rate_animal in firing_rate_blocks_allanimals]) for b in range(len(split_blocks))]
fig, axs = plt.subplots(1, len(firing_rate_blocks_all), figsize=(20, 4))
for b in range(len(firing_rate_blocks_all)):                
    sns.heatmap(np.flipud(firing_rate_blocks_all[b]), cmap = 'viridis', cbar = True, ax=axs[b], vmin=0.005, vmax=0.05, cbar_kws={'label': 'p(CSpks)'})
    axs[b].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'white')
    axs[b].set(xticklabels=[])
    axs[b].tick_params(bottom=False)
    axs[b].set(yticklabels=[])
    axs[b].tick_params(left=False)
    axs[b].set_xticks(x_ticks)
    axs[b].set_xticklabels(x_tick_labels, fontsize = font_size)
    axs[b].spines['top'].set_visible(False)
    axs[b].spines['right'].set_visible(False)
    plt.tight_layout
    plt.show()
axs[0].set_ylabel('ROIs', fontsize = font_size)
# Save
if save_plot:
    if not os.path.exists(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {session_id}')):
        os.mkdir(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {session_id}'))
    plt.savefig(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} population all animals {session_id}.png'))
plt.close()
    