# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import auc

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

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data.xlsx')
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

    # FOV coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    colors_session = mscope.colors_session(animal, session_type, trials, 1)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
    if session_type == 'split':
        colors_phases = ['black', 'crimson', 'teal']
    if session_type == 'tied':
        colors_phases = ['black', 'orange', 'purple']
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

    # # Clustering maps spatial and temporal
    # mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, plot_data, 0)

    if session_type == 'split':
        trials_plot = np.array([trials_baseline[-1], trials_split[0], trials_split[-1], trials_washout[0]])
    if session_type == 'tied':
        trials_plot = np.array(trials_ses[:, 1])

    # Order ROIs by cluster
    if len(clusters_rois) == 1:
        clusters_rois_flat = clusters_rois[0]
    else:
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
    clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
    clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
    cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
    df_events_extract_zscore_clustered = df_events_extract_rawtrace[clusters_rois_flat]
    df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')
    df_extract_rawtrace_detrended_zscore_clustered = df_extract_rawtrace_detrended_zscore[clusters_rois_flat]

    # raw signal clustered
    time_beg_vec = np.arange(0, 60, 5)
    time_end_vec = np.arange(5, 60+5, 5)
    # time_beg_vec = np.arange(0, 60, 1)
    # time_end_vec = np.arange(1, 60+1, 1)
    # time_beg_vec = np.arange(0, 60, 20)
    # time_end_vec = np.arange(20, 60+20, 20)
    mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore_clustered, [time_beg_vec[0]], [time_end_vec[0]], clusters_rois, cluster_transition_idx, 'raw', 'cluster', plot_data, print_plots)
    mscope.response_time_population_avg(df_events_extract_zscore_clustered, [time_beg_vec[0]], [time_end_vec[0]], clusters_rois, cluster_transition_idx, 'events', 'cluster', plot_data, print_plots)
    if plot_data:
        if len(clusters_rois) == 1:
            fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True, sharey=True)
            c = 0
            mean_data_trials = np.zeros((len(trials), len(time_beg_vec)))
            std_data_trials = np.zeros((len(trials), len(time_beg_vec)))
            for w in range(len(time_beg_vec)):
                for count_t, t in enumerate(trials):
                    data_trials = df_events_extract_zscore_clustered.loc[df_events_extract_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[w] * mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
                    mean_data_trials[count_t, w] = data_trials.mean()
                    std_data_trials[count_t, w] = data_trials.std()
            ax.add_patch(plt.Rectangle((trials_baseline[-1] + 0.5, np.min(mean_data_trials[:, 0] - std_data_trials[:, 0])), len(trials_split), np.max(mean_data_trials[:, 0] + std_data_trials[:, 0]) - np.min(mean_data_trials[:, 0] - std_data_trials[:, 0]), fc='grey', alpha=0.3))
            for w in range(len(time_beg_vec)):
                ax.plot(trials, mean_data_trials[:, w], color='gray', linewidth=0.5)
            ax.plot(trials, mean_data_trials[:, 0], marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
            ax.fill_between(trials, mean_data_trials[:, 0] - std_data_trials[:, 0], mean_data_trials[:, 0] + std_data_trials[:, 0], color=colors_cluster[c], alpha=0.3)
            ax.hlines(np.nanmean(mean_data_trials[trials_baseline-1, 0]), 1, len(mean_data_trials[:, 0]), colors='black', linestyles='--', linewidth=2)
            ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
            ax.set_ylabel('Mean activity (dFF)', fontsize=mscope.fsize - 8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(mscope.path, 'images', 'cluster',
                                         'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_events'),
                            dpi=mscope.my_dpi)
        else:
            fig, ax = plt.subplots(1, len(clusters_rois), figsize=(15, 5), tight_layout=True, sharey=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                mean_data_trials = np.zeros((len(trials), len(time_beg_vec)))
                std_data_trials = np.zeros((len(trials), len(time_beg_vec)))
                for w in range(len(time_beg_vec)):
                    for count_t, t in enumerate(trials):
                        data_trials = df_events_extract_zscore_clustered.loc[df_events_extract_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[w] * mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
                        mean_data_trials[count_t, w] = data_trials.mean()
                        std_data_trials[count_t, w] = data_trials.std()
                ax[c].add_patch(plt.Rectangle((trials_baseline[-1] + 0.5, np.min(mean_data_trials[:, 0] - std_data_trials[:, 0])), len(trials_split), np.max(mean_data_trials[:, 0] + std_data_trials[:, 0]) - np.min(mean_data_trials[:, 0] - std_data_trials[:, 0]), fc='grey', alpha=0.3))
                for w in range(len(time_beg_vec)):
                    ax[c].plot(trials, mean_data_trials[:, w], color='gray', linewidth=0.5)
                ax[c].plot(trials, mean_data_trials[:, 0], marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
                ax[c].fill_between(trials, mean_data_trials[:, 0]-std_data_trials[:, 0], mean_data_trials[:, 0]+std_data_trials[:, 0], color=colors_cluster[c], alpha=0.3)
                ax[c].hlines(np.nanmean(mean_data_trials[trials_baseline-1, 0]), 1, len(mean_data_trials), colors='black', linestyles='--', linewidth=2)
                ax[c].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
                ax[c].set_ylabel('Mean activity (dFF)', fontsize=mscope.fsize - 8)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_events'), dpi=mscope.my_dpi)

        if len(clusters_rois) == 1:
            fig, ax = plt.subplots(figsize=(15, 5), tight_layout=True, sharey=True)
            c = 0
            mean_data_trials = np.zeros((len(trials), len(time_beg_vec)))
            std_data_trials = np.zeros((len(trials), len(time_beg_vec)))
            for w in range(len(time_beg_vec)):
                for count_t, t in enumerate(trials):
                    data_trials = df_extract_rawtrace_detrended_zscore_clustered.loc[df_extract_rawtrace_detrended_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[w] * mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
                    mean_data_trials[count_t, w] = data_trials.mean()
                    std_data_trials[count_t, w] = data_trials.std()
            ax.add_patch(plt.Rectangle((trials_baseline[-1] + 0.5, np.min(mean_data_trials[:, 0] - std_data_trials[:, 0])), len(trials_split), np.max(mean_data_trials[:, 0] + std_data_trials[:, 0]) - np.min(mean_data_trials[:, 0] - std_data_trials[:, 0]), fc='grey', alpha=0.3))
            for w in range(len(time_beg_vec)):
                ax.plot(trials, mean_data_trials[:, w], color='gray', linewidth=0.5)
            ax.plot(trials, mean_data_trials[:, 0], marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
            ax.fill_between(trials, mean_data_trials[:, 0] - std_data_trials[:, 0], mean_data_trials[:, 0] + std_data_trials[:, 0], color=colors_cluster[c], alpha=0.3)
            ax.hlines(np.nanmean(mean_data_trials[trials_baseline-1, 0]), 1, len(mean_data_trials[:, 0]), colors='black', linestyles='--', linewidth=2)
            ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
            ax.set_ylabel('Mean activity (dFF)', fontsize=mscope.fsize - 8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(mscope.path, 'images', 'cluster',
                                         'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_raw'),
                            dpi=mscope.my_dpi)
        else:
            fig, ax = plt.subplots(1, len(clusters_rois), figsize=(15, 5), tight_layout=True, sharey=True)
            ax = ax.ravel()
            for c in range(len(clusters_rois)):
                mean_data_trials = np.zeros((len(trials), len(time_beg_vec)))
                std_data_trials = np.zeros((len(trials), len(time_beg_vec)))
                for w in range(len(time_beg_vec)):
                    for count_t, t in enumerate(trials):
                        data_trials = df_extract_rawtrace_detrended_zscore_clustered.loc[df_extract_rawtrace_detrended_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[w] * mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
                        mean_data_trials[count_t, w] = data_trials.mean()
                        std_data_trials[count_t, w] = data_trials.std()
                ax[c].add_patch(plt.Rectangle((trials_baseline[-1] + 0.5, np.min(mean_data_trials[:, 0] - std_data_trials[:, 0])), len(trials_split), np.max(mean_data_trials[:, 0] + std_data_trials[:, 0]) - np.min(mean_data_trials[:, 0] - std_data_trials[:, 0]), fc='grey', alpha=0.3))
                for w in range(len(time_beg_vec)):
                    ax[c].plot(trials, mean_data_trials[:, w], color='gray', linewidth=0.5)
                ax[c].plot(trials, mean_data_trials[:, 0], marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
                ax[c].fill_between(trials, mean_data_trials[:, 0]-std_data_trials[:, 0], mean_data_trials[:, 0]+std_data_trials[:, 0], color=colors_cluster[c], alpha=0.3)
                ax[c].hlines(np.nanmean(mean_data_trials[trials_baseline-1, 0]), 1, len(mean_data_trials), colors='black', linestyles='--', linewidth=2)
                ax[c].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
                ax[c].set_ylabel('Mean activity (dFF)', fontsize=mscope.fsize - 8)
                ax[c].spines['right'].set_visible(False)
                ax[c].spines['top'].set_visible(False)
                ax[c].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            if print_plots:
                plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_raw'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(2, 2, figsize=(7, 5), tight_layout=True)
    ax = ax.ravel()
    for c in range(len(clusters_rois)):
        mean_data_trials = np.zeros(len(trials))
        for count_t, t in enumerate(trials):
            data_trials = df_extract_rawtrace_detrended_zscore_clustered.loc[df_extract_rawtrace_detrended_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[0] * mscope.sr:time_end_vec[0] * mscope.sr].mean(axis=0)
            mean_data_trials[count_t] = data_trials.mean()
        mean_data_1sttrial = []
        for t in np.array([trials_baseline[-1], trials_split[0], trials_washout[0]]):
            idx_trial = np.where(trials==t)[0]
            mean_data_1sttrial.append(mean_data_trials[idx_trial])
        ax[0].plot(np.arange(0, 3), mean_data_1sttrial, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[0].set_xticks(np.arange(0, 3))
        ax[0].set_xticklabels(['last baseline', 'first split', 'first washout'])
        ax[0].set_title('Last baseline, 1st split, 1st washout', fontsize=mscope.fsize - 8)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        trials_phases = np.array([[trials_baseline], [trials_split], [trials_washout]])
        mean_data_phases = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0]
            mean_data_phases.append(np.nanmean(mean_data_trials[idx_trials]))
        ax[1].plot(range(3), mean_data_phases, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[1].set_xticks(range(3))
        ax[1].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[1].set_title('Trials mean of session phase', fontsize=mscope.fsize - 8)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        auc_data_phases = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0]
            auc_data_phases.append(auc(idx_trials, mean_data_trials[idx_trials])/len(idx_trials))
        ax[2].plot(range(3), auc_data_phases, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[2].set_xticks(range(3))
        ax[2].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[2].set_title('AUC of session phase', fontsize=mscope.fsize - 8)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        auc_data_phases_3trials = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0][:3]
            auc_data_phases_3trials.append(auc(idx_trials, mean_data_trials[idx_trials])/len(idx_trials))
        ax[3].plot(range(3), auc_data_phases_3trials, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[3].set_xticks(range(3))
        ax[3].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[3].set_title('AUC of session phase (1st 3 trials)', fontsize=mscope.fsize - 8)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        if print_plots:
            plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'summary_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + '_raw'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(2, 2, figsize=(7, 5), tight_layout=True)
    ax = ax.ravel()
    for c in range(len(clusters_rois)):
        mean_data_trials = np.zeros(len(trials))
        for count_t, t in enumerate(trials):
            data_trials = df_events_extract_zscore_clustered.loc[df_events_extract_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[0] * mscope.sr:time_end_vec[0] * mscope.sr].mean(axis=0)
            mean_data_trials[count_t] = data_trials.mean()
        mean_data_1sttrial = []
        for t in np.array([trials_baseline[-1], trials_split[0], trials_washout[0]]):
            idx_trial = np.where(trials==t)[0]
            mean_data_1sttrial.append(mean_data_trials[idx_trial])
        ax[0].plot(np.arange(0, 3), mean_data_1sttrial, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[0].set_xticks(np.arange(0, 3))
        ax[0].set_xticklabels(['last baseline', 'first split', 'first washout'])
        ax[0].set_title('Last baseline, 1st split, 1st washout', fontsize=mscope.fsize - 8)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        trials_phases = np.array([[trials_baseline], [trials_split], [trials_washout]])
        mean_data_phases = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0]
            mean_data_phases.append(np.nanmean(mean_data_trials[idx_trials]))
        ax[1].plot(range(3), mean_data_phases, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[1].set_xticks(range(3))
        ax[1].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[1].set_title('Trials mean of session phase', fontsize=mscope.fsize - 8)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        auc_data_phases = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0]
            auc_data_phases.append(auc(idx_trials, mean_data_trials[idx_trials])/len(idx_trials))
        ax[2].plot(range(3), auc_data_phases, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[2].set_xticks(range(3))
        ax[2].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[2].set_title('AUC of session phase', fontsize=mscope.fsize - 8)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        auc_data_phases_3trials = []
        for t in range(3):
            idx_trials = np.in1d(trials, trials_phases[t][0]).nonzero()[0][:3]
            auc_data_phases_3trials.append(auc(idx_trials, mean_data_trials[idx_trials])/len(idx_trials))
        ax[3].plot(range(3), auc_data_phases_3trials, marker='o', color=colors_cluster[c], markersize=5, linewidth=2)
        ax[3].set_xticks(range(3))
        ax[3].set_xticklabels(['Baseline', 'Split', 'Washout'])
        ax[3].set_title('AUC of session phase (1st 3 trials)', fontsize=mscope.fsize - 8)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        if print_plots:
            plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'summary_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + '_events'), dpi=mscope.my_dpi)
    plt.close('all')