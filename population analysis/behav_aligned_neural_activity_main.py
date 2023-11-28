# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:26:31 2023

@author: User
"""
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import warnings
import pickle
from scipy.ndimage import gaussian_filter1d
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
save_plot = False
save_data = False
bins_phase = np.arange(0, 1.01, 0.05) # 5 deg    
bins_time = np.arange(-0.250, 0.252, 0.0125) # 12.5 ms

temporal_dimension = 'time'
align = 'acc_pk'

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
    bodycenter, bodyspeed, bodyacc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 81, polyorder = 3)

    # Set
    if temporal_dimension == 'time':
        temporal_dataset = bcam_time
        bins = bins_time
    else:
        temporal_dataset = final_tracks_trials_phase
        bins = bins_phase
        
    # Find peaks and onsets/offsets indices and timestamps
    on_idx = []; pk_idx = []; off_idx = []; on_ts = []; pk_ts = []; off_ts = []
    for tr_idx, data_tr in enumerate(bodyacc):
        on, pk, off = blna.peak_detection(data_tr, 20, 100)
        on_idx.append(on)
        pk_idx.append(pk)
        off_idx.append(off)
        on_ts.append(temporal_dataset[tr_idx][on])
        pk_ts.append(temporal_dataset[tr_idx][pk])
        off_ts.append(temporal_dataset[tr_idx][off])
    events_idx = {'onsets idx': on_idx, 'peaks idx': pk_idx, 'offset idx': off_idx}
    
    # Plot detected peaks for an example trace
    if plot_data:
        plt.figure()
        plt.plot(data_tr, linewidth = 3)
        plt.scatter(on, data_tr[on], s=150, marker = '.', c='g')
        plt.scatter(pk, data_tr[pk], s=150, marker = '.', c='k')
        plt.scatter(off, data_tr[off], s=150, marker = '.', c='r')

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
    
    # Save data
    file_path = os.path.join(path_session_data, 'Acceleration-locked neural activity', f'{animal}_{session_id}_{align}-locked_spikes_{temporal_dimension}.npy')
    if save_data:
        dataset = {'Acceleration events idx': events_idx, 'spikes timing': spikes_timing}
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)
    
    # Bin spikes in time or phase 
    spikes_count = []
    for n in range(len(spikes_timing)):
        spikes_count_roi = blna.bin_spikes(spikes_timing[n], bins)
        spikes_count.append(spikes_count_roi)
    
    # Compute firing rate
    firing_rate = []
    for n in range(len(spikes_count)):
        firing_rate_roi, _ = blna.neural_activity_bin(spikes_count[n], temporal_dataset, bins)
        firing_rate.append(firing_rate_roi) 
    
    # Pre-process firing rate
    firing_rate_smoothed = [gaussian_filter1d(firing_rate_roi, sigma=1.5, axis=1) for firing_rate_roi in firing_rate]

    # # Plot firing rate aligned to body acceleration peaks for each ROIs
    # font_size = 15  
    # x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
    # x_ticks = [0, (len(bins)-1)/2, len(bins)-1] 
    # for n in range(len(firing_rate_smoothed)):                
    #     fig, axs = plt.subplots(3,1, figsize=(6, 15))
    #     spikes_timing_all = [ts for tr in spikes_timing[n] for ts in tr]
    #     peak_idx = [np.full_like(ts, idx) for idx, ts in enumerate(spikes_timing_all)]
    #     axs[0].scatter(np.concatenate(spikes_timing_all), np.concatenate(peak_idx), c = 'dimgrey', marker = '.', s = 5)
    #     axs[0].set_xlim(bins[0], bins[-1])
    #     sns.heatmap(np.flipud(firing_rate_smoothed[n]), cmap = 'viridis', cbar = False, ax=axs[1])
    #     [axs[2].plot(np.nanmean(firing_rate_smoothed[n][start:end], axis = 0), c = colors_session[start+1]) for start, end in split_blocks]
    #     axs[2].set_xlim(0, len(bins)-2)
    #     axs[0].axvline(x=0, linestyle = '-', color = 'crimson')
    #     axs[1].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'white')
    #     axs[2].axvline(x=(len(bins)-1)/2, linestyle = '--', color = 'k')
    #     axs[0].set_ylabel('Acceleration peaks', fontsize = font_size)
    #     axs[1].set_ylabel('Trials', fontsize = font_size)
    #     axs[2].set_ylabel('Firing rate (Hz)', fontsize = font_size)
    #     axs[0].set(xticklabels=[])
    #     axs[0].tick_params(bottom=False)
    #     axs[1].set(xticklabels=[])
    #     axs[1].tick_params(bottom=False)
    #     axs[1].set(yticklabels=[])
    #     axs[1].tick_params(left=False)
    #     axs[2].set_xticks(x_ticks)
    #     axs[2].set_xticklabels(x_tick_labels, fontsize = font_size)
    #     for i in range(3):
    #         axs[i].spines['top'].set_visible(False)
    #         axs[i].spines['right'].set_visible(False)
    #     plt.tight_layout
    #     plt.suptitle(f'Neural activity {rois_sorted[n]} locked to body acceleration peak', fontsize = font_size)
    #     plt.show()
    #     # Save
    #     if save_plot:
    #         if not os.path.exists(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
    #             os.mkdir(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
    #         plt.savefig(os.path.join(save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} {rois_sorted[n]} {animal} {session_id}.png'))
    #     plt.close()
    
    # Compute firing rate population for each block & plot it
    firing_rate_blocks = []
    for start, end in split_blocks:
        firing_rate_block = []
        for n in range(len(firing_rate)):
            firing_rate_block.append(np.nanmean(firing_rate_smoothed[n][start:end], axis = 0))
        firing_rate_blocks.append(firing_rate_block)
        
    # Plot firing rate aligned to body acceleration peaks for the whole population
    if plot_data:
        blna.plot_behav_locked_activity_popul(firing_rate_blocks, bins, trials_ses, colors_session, colors_cluster, animal, session_id, align, save_plot, temporal_dimension, cluster_transition_idx)
    
    firing_rate_blocks_allanimals.append(firing_rate_blocks) 

# Plot firing rate aligned to body acceleration peaks for all the animals  
animal_transition_idx = np.cumsum(np.array([len(firing_rate_animal[0]) for firing_rate_animal in firing_rate_blocks_allanimals]))
firing_rate_blocks_all = [np.concatenate([firing_rate_animal[b] for firing_rate_animal in firing_rate_blocks_allanimals]) for b in range(len(split_blocks))]
if plot_data:
    blna.plot_behav_locked_activity_popul(firing_rate_blocks_all, bins, trials_ses, colors_session, colors_cluster, animal, session_id, align, save_plot, temporal_dimension, animal_transition_idx[:-1])
