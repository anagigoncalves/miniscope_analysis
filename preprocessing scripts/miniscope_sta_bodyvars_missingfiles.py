# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S2.xlsx')
save_path = 'J:\\test\\'

save_plot = 0
plot_data = 1
window = np.arange(-330, 330 + 1)  # Samples
interval = [-165, 0]  # Samples (-0.5s to 0.25s)
zs_data = True  # True if you want to standardize observed data on shuffled data
iter_n = 100  # Number of iterations of CS timestamps random shuffling

# for s in range(1, len(session_data)):
s=3
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
import df_behav_class

nxb = df_behav_class.df_behav_analysis('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis')

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
    # Sort ROIs
    rois_sorted = []
    for i in range(len(clusters_rois)):  # flatten 'clusters_rois'
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
    # Sub-divide experimental blocks
    if session_type == 'split':
        split_blocks = nxb.split_expblocks(trials_ses)
    else:
        split_blocks = trials_ses

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

    # Define font size for plot labels
    font_size = 15

    # Define colorcode for experimental blocks
    color_blocks = []
    for b in split_blocks:
        color_blocks.append(colors_session[b[0] + 1])

    # Re-sort data to have a list of the STA of all the ROIs for each trial
    sta_tr_allrois = [[sta_roi[tr_idx] for sta_roi in sta_zs] for tr_idx, _ in
                      enumerate(trials)]  # List of the STA of all the ROIs for each trial

    # Compute STA of all the ROIs for each block
    sta_blocks_allrois = [np.nanmean(np.array(sta_tr_allrois[start:end]), axis=0) for start, end in
                          split_blocks]  # List of the STA of all the ROIs for block

    # Define tick labels for plots
    x_tick_values = [round(window[0] / nxb.sr_cam, 1), round((1 / 2) * window[0] / nxb.sr_cam, 1), 0,
                     round((1 / 2) * window[-1] / nxb.sr_cam, 1), round(window[-1] / nxb.sr_cam, 1)]
    x_ticks = np.linspace(0, len(sta_tr_allrois[0][0]), len(x_tick_values)).astype(int)

    # Plot 1: Heatmap STA of the population by trial
    max_val = np.nanmax(np.concatenate(sta_tr_allrois, axis=0))
    min_val = np.nanmin(np.concatenate(sta_tr_allrois, axis=0))
    for tr_idx, tr in enumerate(trials):  # WARNING: for tr in range(len(trial_changes)-1):
        plt.figure()
        hm = sns.heatmap(sta_tr_allrois[tr_idx], cmap='viridis', vmin = min_val, vmax = max_val)  # heatmap STA whole population by trial
        plt.ylabel('ROIs', fontsize=font_size)
        plt.xlabel('Time (s)', fontsize=font_size)
        cbar = hm.collections[0].colorbar
        cbar.set_label(var_name + ' (z-score)', fontsize=font_size)
        plt.axvline(x=window[-1], color='white', linestyle='--')
        plt.yticks([])
        plt.tick_params(left=False)
        plt.xticks(x_ticks, [f"{tick}" for tick in x_tick_values])
        plt.title('STA ' + var_name + ' trial ' + str(tr), fontsize=font_size)
        plt.savefig(save_path+'STA_'+var_name+'_trial'+str(tr))
        plt.close('all')

    # Plot 2: STA traces of one cluster across trials
    max_val = np.nanmax(np.concatenate(sta_tr_allrois, axis=0))
    min_val = np.nanmin(np.concatenate(sta_tr_allrois, axis=0))
    clust_sta_tr = np.zeros(
        (len(sta_tr_allrois), len(cluster_transition_idx), len(window)))  # STA of clusters by block
    for tr_idx, trial in enumerate(sta_tr_allrois):
        start = 0
        for clust_idx, clust_end in enumerate(cluster_transition_idx):
            if len(cluster_transition_idx) - 1 == clust_idx:
                end = len(sta_tr_allrois[0])
            else:
                end = clust_end
            clust_sta_tr[tr_idx, clust_idx] = np.mean(trial[start:end], axis=0)
            start = clust_end
    fig, axs = plt.subplots(nrows=1, ncols=len(cluster_transition_idx), figsize=(15, 5))
    if len(cluster_transition_idx) == 1:
        axs = [axs]  # Convert single axis to a list for indexing consistency
    for clust_idx, _ in enumerate(cluster_transition_idx):
        for tr in trials_ses.flatten():
            trial_idx = np.where(tr == trials)[0][0]
            current_ax = axs[clust_idx]  # Get the current axis
            current_ax.plot(window * 1 / nxb.sr_cam, clust_sta_tr[trial_idx, clust_idx], c=colors_session[tr],
                            linewidth=2)
        current_ax.axvline(x=0, color='black', linestyle='--')
        current_ax.spines['right'].set_visible(False)
        current_ax.spines['top'].set_visible(False)
        current_ax.set_ylim([min_val, max_val])
        current_ax.set_xlim(window[0] * 1 / nxb.sr_cam, window[-1] * 1 / nxb.sr_cam)
        current_ax.set_xlim(window[0] * 1 / nxb.sr_cam, window[-1] * 1 / nxb.sr_cam)
        current_ax.set_xlabel('Time around event (s)', fontsize=font_size)
        if clust_idx == 0:
            current_ax.set_ylabel(var_name + ' (z-score)', fontsize=font_size)
        if clust_idx > 0:
            current_ax.set(yticklabels=[])
            current_ax.tick_params(left=False)
            current_ax.spines['left'].set_visible(False)
        current_ax.set_title('Cluster ' + str(clust_idx + 1), fontsize=font_size, c=colors_cluster[clust_idx])
    plt.savefig(save_path+'STA_zs_'+var_name+'_clusters_trials')

    # Plot 3: Heatmap STA of the population by block
    max_val = np.nanmax(np.concatenate(sta_blocks_allrois, axis=0))
    min_val = np.nanmin(np.concatenate(sta_blocks_allrois, axis=0))
    fig, axs = plt.subplots(len(sta_blocks_allrois), 1, figsize=(12, 12))
    for b in range(len(sta_blocks_allrois)):
        hm = sns.heatmap(sta_blocks_allrois[b], cmap='viridis', ax=axs[b], vmin = min_val, vmax = max_val)  # heatmap STA whole population by block
        axs[b].axvline(x=window[-1], color='white', linestyle='--')
        axs[b].set_ylabel('ROIs', fontsize=font_size)
        axs[b].set(xticklabels=[])
        axs[b].set(yticklabels=[])
        if b < len(sta_blocks_allrois) - 1:
            axs[b].tick_params(left=False, bottom=False)
        if b == int((len(sta_blocks_allrois) - 1) / 2):
            cbar = hm.collections[0].colorbar
            cbar.set_label(var_name + ' (z-score)', fontsize=font_size)
        plt.xlabel('Time (s)', fontsize=font_size)
        plt.axvline(x=window[-1], color='white', linestyle='--')
        plt.yticks([])
        plt.tick_params(left=False)
        plt.xticks(x_ticks, [f"{tick}" for tick in x_tick_values], fontsize=12)
        fig.suptitle('STA ' + var_name, fontsize=font_size)
        for c in cluster_transition_idx:  # Mark clusters in the heatmap
            axs[b].hlines(c + 1, *axs[b].get_xlim(), color='white', linestyle='dashed', linewidth=0.5)
    plt.savefig(save_path+'STA_zs_'+var_name+'_blocks')

    # Plot 4: STA traces of one cluster across blocks
    clust_sta = np.zeros(
        (len(sta_blocks_allrois), len(cluster_transition_idx), len(window)))  # STA of clusters by block
    for block_idx, block in enumerate(sta_blocks_allrois):
        start = 0
        for clust_idx, clust_end in enumerate(cluster_transition_idx):
            if len(cluster_transition_idx) - 1 == clust_idx:
                end = len(sta_blocks_allrois[0])
            else:
                end = clust_end
            clust_sta[block_idx, clust_idx] = np.mean(block[start:end], axis=0)
            start = clust_end
    fig, axs = plt.subplots(nrows=1, ncols=len(cluster_transition_idx), figsize=(15, 5))
    if len(cluster_transition_idx) == 1:
        axs = [axs]  # Convert single axis to a list for indexing consistency
    for clust_idx, _ in enumerate(cluster_transition_idx):
        for block_idx, _ in enumerate(sta_blocks_allrois):
            current_ax = axs[clust_idx]  # Get the current axis
            current_ax.plot(window * 1 / nxb.sr_cam, clust_sta[block_idx, clust_idx], c=color_blocks[block_idx],
                            linewidth=2)
        current_ax.axvline(x=0, color='black', linestyle='--')
        current_ax.spines['right'].set_visible(False)
        current_ax.spines['top'].set_visible(False)
        current_ax.set_ylim([min_val, max_val])
        current_ax.set_xlim(window[0] * 1 / nxb.sr_cam, window[-1] * 1 / nxb.sr_cam)
        current_ax.set_xlim(window[0] * 1 / nxb.sr_cam, window[-1] * 1 / nxb.sr_cam)
        current_ax.set_xlabel('Time around event (s)', fontsize=font_size)
        if clust_idx == 0:
            current_ax.set_ylabel(var_name + ' (z-score)', fontsize=font_size)
        if clust_idx > 0:
            current_ax.set(yticklabels=[])
            current_ax.tick_params(left=False)
            current_ax.spines['left'].set_visible(False)
        current_ax.set_title('Cluster ' + str(clust_idx + 1), fontsize=font_size, c=colors_cluster[clust_idx])
    plt.savefig(save_path+'STA_zs_'+var_name+'_clusters_blocks')
    plt.close('all')
