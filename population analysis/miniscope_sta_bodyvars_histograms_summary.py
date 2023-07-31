# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns
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
save_path = 'J:\\Miniscope processed files\\STA bodyvars\\split ipsi fast S1\\'

save_plot = True
plot_data = True
window = np.arange(-330, 330 + 1)  # Samples
interval = [-165, 0]  # Samples (-0.5s to 0.25s)
zs_data = True  # True if you want to standardize observed data on shuffled data
iter_n = 100  # Number of iterations of CS timestamps random shuffling
xaxis = window / 330
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    protocol = ses_info[0].replace(' ', '_')
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

        # Define font size for plot labels
        font_size = 30
        split_blocks = trials_ses

            # Re-sort data to have a list of the STA of all the ROIs for each trial
        sta_tr_allrois = [[sta_roi[tr_idx] for sta_roi in sta_allrois] for tr_idx, _ in
                          enumerate(trials)]  # List of the STA of all the ROIs for each trial

        # Compute STA of all the ROIs for each block
        sta_blocks_allrois = [np.mean(np.array(sta_tr_allrois[start:end]), axis=0) for start, end in
                              split_blocks]  # List of the STA of all the ROIs for block

        # Define tick labels for plots
        x_tick_values = [round(window[0] / nxb.sr_cam, 1), round((1 / 2) * window[0] / nxb.sr_cam, 1), 0,
                         round((1 / 2) * window[-1] / nxb.sr_cam, 1), round(window[-1] / nxb.sr_cam, 1)]
        x_ticks = np.linspace(0, len(sta_tr_allrois[0][0]), len(x_tick_values)).astype(int)

        max_val = np.nanmax(np.concatenate(sta_blocks_allrois, axis=0))
        min_val = np.nanmin(np.concatenate(sta_blocks_allrois, axis=0))
        fig, axs = plt.subplots(len(sta_blocks_allrois),1, figsize = (15, 15), tight_layout=True)
        for b in range(len(sta_blocks_allrois)):
            hm = sns.heatmap(sta_blocks_allrois[b], cmap='viridis', ax = axs[b], vmin = min_val, vmax = max_val)  # heatmap STA whole population by block
            axs[b].axvline(x=window[-1], color='white', linestyle='--')
            axs[b].set_ylabel('ROIs', fontsize = font_size)
            axs[b].set(xticklabels=[])
            axs[b].set(yticklabels=[])
            if b < len(sta_blocks_allrois)-1:
                axs[b].tick_params(left=False, bottom=False)
            if b == int((len(sta_blocks_allrois)-1)/2):
                cbar = hm.collections[0].colorbar
                cbar.set_label(var_name + ' (z-score)', fontsize=font_size)
                cbar.ax.tick_params(labelsize=26)
            plt.xlabel('Time (s)', fontsize=font_size)
            plt.axvline(x=window[-1], color='white', linestyle='--')
            plt.yticks([])
            plt.tick_params(left=False)
            plt.xticks(x_ticks, [f"{tick}" for tick in x_tick_values], fontsize = 26)
            for c in cluster_transition_idx: # Mark clusters in the heatmap
                axs[b].hlines(c + 1, *axs[b].get_xlim(), color='white', linestyle='dashed', linewidth = 4)
        plt.savefig(save_path + 'sta_bodyvars_' + animal + '_' + var_name.replace(' ', '_') + '_' + protocol, dpi=256)
        plt.close('all')