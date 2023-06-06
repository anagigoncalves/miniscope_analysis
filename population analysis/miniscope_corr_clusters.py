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
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files\\'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data_all.xlsx')
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
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
    if session_type == 'split':
        colors_phases = ['black', 'crimson', 'teal']
    if session_type == 'tied':
        colors_phases = ['black', 'orange', 'purple']
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

    trials_baseline_idx = []
    for t in trials_baseline:
        idx_trial = np.where(trials_baseline==t)[0][0]
        trials_baseline_idx.append(idx_trial)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    corr_data_clusters_bs = np.zeros((len(clusters_rois), len(trials)))
    for c in range(len(clusters_rois)):
        corr_data_trials = np.zeros(len(trials))
        for count_t, t in enumerate(trials):
            # mean_corr = np.nanmean(np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == t, clusters_rois[c]].corr())[1:, 0])
            mean_corr = np.nanmean(np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == t, clusters_rois[c]].corr()).flatten())
            corr_data_trials[count_t] = mean_corr
        corr_data_clusters_bs[c, :] = corr_data_trials-np.nanmean(corr_data_trials[trials_baseline_idx])
        ax[0].plot(trials, corr_data_trials, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[1].plot(trials, corr_data_trials-np.nanmean(corr_data_trials[trials_baseline_idx]), marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
    ax[0].set_title('Mean cluster correlation', fontsize=mscope.fsize - 8)
    ax[1].set_title('Baseline subtracted', fontsize=mscope.fsize - 8)
    ax[0].axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax[0].axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax[1].axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax[1].axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    if print_plots:
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'corr_summary_raw'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(figsize=(4, 5), tight_layout=True)
    ax.plot(trials, np.nanmean(corr_data_clusters_bs, axis=0), marker='o', color='black', markersize=5, linewidth=2)
    ax.fill_between(trials, np.nanmean(corr_data_clusters_bs, axis=0)-np.nanstd(corr_data_clusters_bs, axis=0),
                    np.nanmean(corr_data_clusters_bs, axis=0) + np.nanstd(corr_data_clusters_bs, axis=0), color='black', alpha=0.3)
    ax.set_title('Mean cluster correlation bs', fontsize=mscope.fsize - 8)
    ax.axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax.axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if print_plots:
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'corr_summary_cluster_mean_raw'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    corr_data_clusters_events_bs = np.zeros((len(clusters_rois), len(trials)))
    for c in range(len(clusters_rois)):
        corr_data_trials_events = np.zeros(len(trials))
        for count_t, t in enumerate(trials):
            # mean_corr_events = np.nanmean(np.array(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, clusters_rois[c]].corr())[1:, 0])
            mean_corr_events = np.nanmean(np.array(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, clusters_rois[c]].corr()).flatten())
            corr_data_trials_events[count_t] = mean_corr_events
        corr_data_clusters_events_bs[c, :] = corr_data_trials_events-np.nanmean(corr_data_trials_events[trials_baseline_idx])
        ax[0].plot(trials, corr_data_trials_events, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[1].plot(trials, corr_data_trials_events-np.nanmean(corr_data_trials_events[trials_baseline_idx]), marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
    ax[0].set_title('Mean cluster events correlation', fontsize=mscope.fsize - 8)
    ax[1].set_title('Baseline subtracted', fontsize=mscope.fsize - 8)
    ax[0].axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax[0].axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax[1].axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax[1].axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    if print_plots:
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'corr_summary_events'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(figsize=(4, 5), tight_layout=True)
    ax.plot(trials, np.nanmean(corr_data_clusters_events_bs, axis=0), marker='o', color='black', markersize=5, linewidth=2)
    ax.fill_between(trials, np.nanmean(corr_data_clusters_events_bs, axis=0)-np.nanstd(corr_data_clusters_events_bs, axis=0),
                    np.nanmean(corr_data_clusters_events_bs, axis=0) + np.nanstd(corr_data_clusters_events_bs, axis=0), color='black', alpha=0.3)
    ax.set_title('Mean cluster events correlation bs', fontsize=mscope.fsize - 8)
    ax.axvline(trials_baseline[-1]+0.5, linestyle='dashed', color='black')
    ax.axvline(trials_split[-1] + 0.5, linestyle='dashed', color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if print_plots:
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'corr_summary_events_cluster_mean_raw'), dpi=mscope.my_dpi)
    plt.close('all')