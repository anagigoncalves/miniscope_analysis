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
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S2.xlsx')
save_path = 'J:\\Miniscope processed files\\STA paw spatial diff\\split contra fast S1\\'

window = np.arange(-330, 330 + 1)  # Samples
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
    final_tracks_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        final_tracks_trials.append(final_tracks)

    # Get phase and displacement difference
    displ_diff_diag = nxb.paw_diff(final_tracks_trials, 0, 3)
    displ_diff_homo = nxb.paw_diff(final_tracks_trials, 0, 1)
    displ_diff_front = nxb.paw_diff(final_tracks_trials, 0, 2)

    # Dictionary of all the independent variables on which computing the STA
    ind_vars = {'FR-HL displacement difference': displ_diff_diag, 'FR-HR displacement difference': displ_diff_homo,
                'FR-FL displacement difference': displ_diff_front}
    keys=list(ind_vars.keys())

    # Loop through independent variables to compute and plot STAs of each one
    for var in range(len(ind_vars)):
        var_name = keys[var]
        variable = ind_vars[var_name]

        # Compute spike-triggered average (STA) of kinematic variables
        df_events_ordered, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace,
                                                                clusters_rois)  # Sort ROIs by cluster
        sta_allrois, signal_chunks_allrois = nxb.sta(df_events_ordered, variable, bcam_time, window, trials)
        # Sort ROIs
        rois_sorted = []
        for i in range(len(clusters_rois)):
            rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))

        # Standardize observed STA on STA computed with shuffled data
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
        if not os.path.exists(os.path.join(save_path, animal + ' ' + ses_info[0])):
            os.mkdir(os.path.join(save_path, animal + ' ' + ses_info[0]))
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                             'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'), sta_zs)
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
            'sta_bodyvars_' + var_name.replace(' ', '_') + '_cluster_transition_idx.npy'), cluster_transition_idx)