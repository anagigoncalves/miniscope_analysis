# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# import classes
path_code = 'C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\'
os.chdir(path_code)
import miniscope_session_class
import locomotion_class
import df_behav_class
nxb = df_behav_class.df_behav_analysis(path_code)

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\STA bodyvars\\'

save_plot = True
plot_data = True
window = np.arange(-330, 330 + 1)  # Samples
interval = [-165, 0]  # Samples (-0.5s to 0.25s)
zs_data = True  # True if you want to standardize observed data on shuffled data
iter_n = 100  # Number of iterations of CS timestamps random shuffling

for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = session_type + '_' + ses_info[2]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
     frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                             frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
        trials, session_type, animal, session)
    if session_type == 'tied':
        for i in range(np.shape(trials_ses)[0]):
            if cond_plot[i] == 'baseline':
                trials_baseline = np.arange(trials_ses[i][0], trials_ses[i][1]+1)
            if cond_plot[i] == 'fast':
                trials_fast = np.arange(trials_ses[i][0], trials_ses[i][1]+1)
            if cond_plot[i] == 'slow':
                trials_slow = np.arange(trials_ses[i][0], trials_ses[i][1]+1)

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
    bodycenter, bodyspeed, bodyacc = nxb.body_kinematic(final_tracks_trials, trials, win_len=81, polyorder=3)

    # Get phase and displacement difference
    displ_diff_contra = nxb.paw_diff(final_tracks_trials, 0, 3)
    displ_diff_ipsi = nxb.paw_diff(final_tracks_trials, 0, 1)

    # Dictionary of all the indipendent variables on which computing the STA
    # ind_vars = {'FR-HL displacement difference': displ_diff_contra, 'FR-HR displacement difference': displ_diff_ipsi,
    #             'Body position': bodycenter, 'Body speed': bodyspeed, 'Body acceleration': bodyacc}
    ind_vars = {'Body position': bodycenter, 'Body speed': bodyspeed, 'Body acceleration': bodyacc}
    keys = list(ind_vars.keys())

    # Loop through independent variables to compute and plot STAs of each one
    for var in range(len(ind_vars)):
        var_name = keys[var]
        variable = ind_vars[var_name]

        # Compute spike-triggered average (STA) of kinematic variables
        df_events, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace,
                                                                clusters_rois)  # Sort ROIs by cluster
        sta_allrois, signal_chunks_allrois = nxb.sta(df_events, variable, bcam_time, window, trials)

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

        xaxis = window/loco.sr
        xaxis_start = np.where(xaxis >= -0.5)[0][0]
        xaxis_end = np.where(xaxis >= 0)[0][0]
        if session_type == 'split':
            sta_zs_rois_bs = np.zeros((len(trials_baseline), xaxis_end-xaxis_start, len(clusters_rois)))
            for c in range(len(clusters_rois)):
                for count_t, trial in enumerate(trials_baseline):
                    trial_idx = np.where(trial == trials)[0][0]
                    sta_zs_rois_bs[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)
            sta_zs_rois_split = np.zeros((len(trials_split), xaxis_end-xaxis_start, len(clusters_rois)))
            for c in range(len(clusters_rois)):
                for count_t, trial in enumerate(trials_split):
                    trial_idx = np.where(trial == trials)[0][0]
                    sta_zs_rois_split[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)
            sta_zs_rois_washout = np.zeros((len(trials_washout), xaxis_end-xaxis_start, len(clusters_rois)))
            for c in range(len(clusters_rois)):
                for count_t, trial in enumerate(trials_washout):
                    trial_idx = np.where(trial == trials)[0][0]
                    sta_zs_rois_washout[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)

            if not os.path.exists(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'))):
                os.mkdir(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_')))
            np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_bs.npy'), sta_zs_rois_bs)
            np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_split.npy'), sta_zs_rois_split)
            np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_washout.npy'), sta_zs_rois_washout)

            fig, ax = plt.subplots(1, 3, figsize=(10, 5), tight_layout=True, sharey=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                ax[0].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0), color='black')
                ax[0].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)-np.nanstd(sta_zs_rois_bs[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)+np.nanstd(sta_zs_rois_bs[:, :, c], axis=0), color='black', alpha=0.3)
                ax[1].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_split[:, :, c], axis=0), color='red')
                ax[1].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_split[:, :, c], axis=0)-np.nanstd(sta_zs_rois_split[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_split[:, :, c], axis=0)+np.nanstd(sta_zs_rois_split[:, :, c], axis=0), color='red', alpha=0.3)
                ax[2].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_washout[:, :, c], axis=0), color='blue')
                ax[2].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_washout[:, :, c], axis=0)-np.nanstd(sta_zs_rois_washout[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_washout[:, :, c], axis=0)+np.nanstd(sta_zs_rois_washout[:, :, c], axis=0), color='blue', alpha=0.3)
            ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[0].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[1].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            ax[2].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[2].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['top'].set_visible(False)
            ax[2].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            plt.savefig(os.path.join(save_path, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal + '_' + ses_info[0].replace(' ', '_')), dpi=mscope.my_dpi)

        if session_type == 'tied':
            sta_zs_rois_bs = np.zeros((len(trials_baseline), xaxis_end-xaxis_start, len(clusters_rois)))
            for c in range(len(clusters_rois)):
                for count_t, trial in enumerate(trials_baseline):
                    trial_idx = np.where(trial == trials)[0][0]
                    sta_zs_rois_bs[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)
            if animal != 'MC8855':
                sta_zs_rois_slow = np.zeros((len(trials_slow), xaxis_end-xaxis_start, len(clusters_rois)))
                for c in range(len(clusters_rois)):
                    for count_t, trial in enumerate(trials_slow):
                        trial_idx = np.where(trial == trials)[0][0]
                        sta_zs_rois_slow[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)
            sta_zs_rois_fast = np.zeros((len(trials_fast), xaxis_end-xaxis_start, len(clusters_rois)))
            for c in range(len(clusters_rois)):
                for count_t, trial in enumerate(trials_fast):
                    trial_idx = np.where(trial == trials)[0][0]
                    sta_zs_rois_fast[count_t, :, c] = np.nanmean(sta_zs[np.where(idx_roi_cluster_ordered == c+1)[0], trial_idx, xaxis_start:xaxis_end], axis=0)

            if not os.path.exists(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'))):
                os.mkdir(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_')))
            np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_bs.npy'), sta_zs_rois_bs)
            if animal != 'MC8855':
                np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_slow.npy'), sta_zs_rois_slow)
            np.save(os.path.join(save_path, animal + '_' + ses_info[0].replace(' ', '_'), 'sta_bodyvars_' + var_name.replace(' ', '_') + '_fast.npy'), sta_zs_rois_fast)

            fig, ax = plt.subplots(1, 3, figsize=(10, 5), tight_layout=True, sharey=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                ax[0].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0), color='black')
                ax[0].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)-np.nanstd(sta_zs_rois_bs[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)+np.nanstd(sta_zs_rois_bs[:, :, c], axis=0), color='black', alpha=0.3)
                if animal != 'MC8855':
                    ax[1].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_slow[:, :, c], axis=0),
                               color='purple')
                    ax[1].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_slow[:, :, c], axis=0)-np.nanstd(sta_zs_rois_slow[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_slow[:, :, c], axis=0)+np.nanstd(sta_zs_rois_slow[:, :, c], axis=0), color='purple', alpha=0.3)
                ax[2].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_fast[:, :, c], axis=0), color='orange')
                ax[2].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_fast[:, :, c], axis=0)-np.nanstd(sta_zs_rois_fast[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_fast[:, :, c], axis=0)+np.nanstd(sta_zs_rois_fast[:, :, c], axis=0), color='orange', alpha=0.3)
            ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[0].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['top'].set_visible(False)
            ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[1].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['top'].set_visible(False)
            ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            ax[2].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
            ax[2].set_ylabel(var_name, fontsize=mscope.fsize - 4)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['top'].set_visible(False)
            ax[2].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
            plt.savefig(os.path.join(save_path, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal + '_' + ses_info[0].replace(' ', '_')), dpi=mscope.my_dpi)
        plt.close('all')