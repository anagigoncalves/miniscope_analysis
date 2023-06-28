# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:14:36 2023

@author: Ana
"""
# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    traces_type = 'raw'
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)

     # Order ROIs mediolateral
    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    centroids_mediolateral = []
    for c in range(len(centroid_ext)):
        centroids_mediolateral.append(centroid_ext[c][0])
    distance_neurons_ordered = np.argsort(centroids_mediolateral)
    rois_ordered_distance_str = []
    for i in distance_neurons_ordered:
        rois_ordered_distance_str.append(roi_list[i])

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    param_trials = []
    st_strides_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        param_mat = loco.compute_gait_param(bodycenter,final_tracks,paws_rel,st_strides_mat,sw_pts_mat,'step_length')
        st_strides_trials.append(st_strides_mat)
        param_trials.append(param_mat)
    [param_all_idx, param_all_time, param_all] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, 'FR', 'FL', 1, 1)


    cmap = plt.get_cmap('magma')
    color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
    def get_colors_plot(animal_name, color_animals):
        if animal_name == 'MC8855':
            color_plot = color_animals[0]
        if animal_name == 'MC9194':
            color_plot = color_animals[1]
        if animal_name == 'MC10221':
            color_plot = color_animals[2]
        if animal_name == 'MC9513':
            color_plot = color_animals[3]
        if animal_name == 'MC9226':
            color_plot = color_animals[4]
        return color_plot
    
    window = 0.2
    event_prob = []
    event_time = []
    for count_roi, roi in enumerate(df_events_extract_rawtrace.columns[2:]): 
        event_trial_prob = []
        event_trial_time = []
        for count_trial, trial in enumerate(trials):
            trial_idx = np.where(trials == trial)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (df_events_extract_rawtrace['trial'] == trial), 'time'])
            bins = np.arange(0, bcam_trial[-1], window)
            bcam_trial_idx_bins = np.digitize(bcam_trial, bins) #returns the indices of the bins to which each bcam timestamp belongs
            events_trial_idx_bins = np.digitize(events_trial, bins)  # returns the indices of the bins to which each calcium event time belongs
            event_trial_prob.extend(np.array([np.divide(len(events_trial[events_trial_idx_bins[:len(events_trial)] == i]), len(events_trial)) for i in range(len(bins))]))
            event_trial_time.extend(bins+(count_trial*60))
        event_prob.append(event_trial_prob)
        event_time.append(event_trial_time)
    
    
    plt.figure()
    plt.scatter(event_time, event_prob, color='black')
    plt.plot(param_all_time, (param_all-np.min(param_all))/(np.max(param_all)-np.min(param_all)), color='blue')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
