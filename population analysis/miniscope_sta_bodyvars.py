# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\STA bodyvars\\split ipsi fast S1\\'

save_plot = True
plot_data = True
window = np.arange(-330, 330 + 1)  # Samples
interval = [-165, 0] # Samples (-0.5s to 0.25s)
zs_data = True # True if you want to standardize observed data on shuffled data
iter_n = 100 # Number of iterations of CS timestamps random shuffling

for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = session_type + '_' + ses_info[2]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    import df_behav_class
    nxb = df_behav_class.df_behav_analysis('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    param_name = 'step_length'
    p = 'FR'
    p2 = 'FL'
    st_strides_trials = []
    sw_strides_trials = []
    final_tracks_trials = []
    param_trials = []
    param_trials_fr_mean = np.zeros(len(trials))
    stride_duration_trials = []
    final_tracks_forwadloco_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        final_tracks_forwadloco = loco.final_tracks_forwardlocomotion(final_tracks, st_strides_mat)
        final_tracks_forwadloco_trials.append(final_tracks_forwadloco)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        param_trials.append(
            loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
        param_trials_fr_mean[count_trial] = np.nanmean(param_trials[-1][0]) - np.nanmean(param_trials[-1][2])
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials,
                                                        sw_strides_trials, 'st-st')
    [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials,
                                                                                  trials, p, p2, sym=1,
                                                                                  remove_nan=1)  # SL symmetry for each stride

    # Get kinematic variables (body position, speed, acceleration)
    bodycenter, bodyspeed, bodyacc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 81, polyorder = 3)

    # Get phase and displacement difference
    displ_diff_contra = nxb.paw_diff(final_tracks_trials, 0, 3)
    displ_diff_ipsi = nxb.paw_diff(final_tracks_trials, 0, 1)

    # Dictionary of all the indipendent variables on which computing the STA
    # ind_vars = {'FR-HL displacement difference': displ_diff_contra, 'FR-HR displacement difference': displ_diff_ipsi,
    #             'Body position': bodycenter, 'Body speed': bodyspeed, 'Body acceleration': bodyacc}
    ind_vars = {'Body position': bodycenter, 'Body speed': bodyspeed, 'Body acceleration': bodyacc}
    keys=list(ind_vars.keys())
    
    # Loop through indipendent variables to compute and plot STAs of each one
    for var in range(len(ind_vars)):
        var_name = keys[var]
        variable = ind_vars[var_name]
        
        # Compute spike-triggered average (STA) of kinematic variables
        df_events, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace,
                                                                clusters_rois)  # Sort ROIs by cluster
        sta_allrois, signal_chunks_allrois = nxb.sta(df_events, variable, bcam_time, window, trials)
        # Sort ROIs
        rois_sorted = []
        for i in range(len(clusters_rois)):  # flatten 'clusters_rois'
            rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
        # Sub-divide experimental blocks
        if session_type == 'split':
            split_blocks = nxb.split_expblocks(trials_ses)
        else:
            split_blocks = trials_ses
        # Plot STA
        if plot_data:
            nxb.plot_sta_rois(sta_allrois, signal_chunks_allrois, window, trials, trials_ses, colors_session, rois_sorted, var_name, animal, session_id, save_plot)
            nxb.plot_sta_popul(sta_allrois, window, cluster_transition_idx, colors_cluster, colors_session, trials, split_blocks, var_name, interval, animal, session_id, save_plot)
        
        # Standardize observed STA on STA computed with shuffled data
        if zs_data: 
            # Shuffle CS timestamps
            shuffled_spikes_ts = nxb.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)
            # Compute STA for shuffled data
            sta_shuffled_ts = np.array(nxb.sta_shuffled(shuffled_spikes_ts, variable, bcam_time, window, trials))
            # Standardize STA 
            mean_chance = np.nanmean(sta_shuffled_ts, axis=2)
            sd_chance = np.nanstd(sta_shuffled_ts, axis=2)
            sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
            for n in range(len(sta_allrois)):
                for tr in range(len(trials)):
                    sta_zs[n, tr] = (sta_allrois[n][tr] - mean_chance[n][tr]) / sd_chance[n][tr]
            if plot_data:
                # Plot changes in max and min z-score and their latency in a window before events
                nxb.tuning_learn(interval, sta_zs, cluster_transition_idx, trials, colors_cluster, var_name, animal, session_id, save_plot)
                # Plot observed STA, STA you would expect by chance and standardized STA
                nxb.plot_sta_shuffled(sta_zs, sta_allrois, sta_shuffled_ts, window, var_name, trials_ses, rois_sorted, animal, session_id, colors_session, save_plot)