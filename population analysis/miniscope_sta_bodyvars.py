# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class


path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S1.xlsx')

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
    import df_behav_class
    nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    traces_type = 'raw'
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
    win_len = 81  # In samples
    polyorder = 3
    bodycenter, bodyspeed, bodyacc = nxb.kinematic(final_tracks_trials, trials, win_len, polyorder)

    # Compute spike-triggered average (STA) of kinematic variables
    window = np.arange(-330, 330 + 1)  # In samples
    variable = bodyacc
    df_events, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace,
                                                            clusters_rois)  # Sort ROIs by cluster
    sta_allrois, signal_chunks_allrois = nxb.sta(df_events, variable, bcam_time, window, trials)
    # Plot STA
    save_plot = True
    plot_data = True
    var_name = 'Acceleration'
    rois_sorted = []
    for i in range(len(clusters_rois)):  # flatten 'clusters_rois'
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
    nxb.plot_sta(sta_allrois, signal_chunks_allrois, window, trials, trials_ses, colors_session, rois_sorted,
                 var_name, animal, save_plot)
    
    # Shuffle CS timestamps
    iter_n = 10
    shuffled_spikes_ts = nxb.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)
    # Compute STA for each iteration of CSs timestamps shuffling
    sta_shuffled_ts_alliter = []
    for i in range(iter_n):
        sta_shuffled_ts = nxb.sta_shuffled(shuffled_spikes_ts[i], variable, bcam_time, window, trials)
        sta_shuffled_ts_alliter.append(sta_shuffled_ts)   
    # Average the STA traces obtained from the different iterations of random shuffling of CS timestamps
    sta_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
    stsd_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
    for n in range(df_events.iloc[:, 2:].shape[1]):
        roi_sta_tr = [sublist[n] for sublist in sta_shuffled_ts_alliter]
        for tr in range(trials[-1]):
            sta_chance[n, tr] = np.mean([array[tr] for array in roi_sta_tr], axis=0)
            stsd_chance[n, tr] = np.std([array[tr] for array in roi_sta_tr], axis=0)
            
    # Standardize observed STA on STA computed with shuffled data
    sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
    for n in range(len(sta_allrois)):
        sta_zs[n] = (sta_allrois[n] - sta_chance[n]) / stsd_chance[n]
    # Plot observed STA, STA you would expect by chance and standardized STA
    nxb.plot_sta_shuffled(sta_zs, sta_allrois, sta_chance, window, var_name, trials_ses, rois_sorted, animal, save_plot)

    # Plot distribution of STA peaks amplitude (absolute max z-scores) and their latency with respect to the event
    signif_thresh = 2
    interval = [-82, 82] # In samples
    max_abs_zs, latency = nxb.maxzs_distr(sta_zs, interval, signif_thresh)