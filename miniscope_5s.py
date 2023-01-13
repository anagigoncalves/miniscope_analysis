# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# path inputs
path = 'D:\\Miniscope processed files\\TM RAW FILES\\split contra fast\\MC10221\\2021_09_24\\'
path_loco = 'D:\\Miniscope processed files\\TM TRACKING FILES\\split contra fast S3 240921\\'
line_ratio = 3
session_type = path.split('\\')[-4].split(' ')[0]
version_mscope = 'v4'
plot_data = 1
print_plots = 1
save_data = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
traces_type = 'raw'
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, idx_roi_cluster_ordered] = mscope.load_processed_files()

# #FOV coordinates
# centroid_ext = mscope.get_roi_centroids(coord_ext)
# fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
# centroid_cluster_mean = np.zeros((len(np.unique(idx_roi_cluster_ordered)), 2))
# for count_i, i in enumerate(np.unique(idx_roi_cluster_ordered)):
#     cluster_idx = np.where(idx_roi_cluster_ordered==i)[0]
#     centroid_cluster = np.zeros((len(cluster_idx), 2))
#     for count_c, c in enumerate(cluster_idx):
#         centroid_cluster[count_c, :] = centroid_ext[c]
#     centroid_mean = np.nanmean(centroid_cluster, axis=0)
#     centroid_cluster_mean[count_i, 0] = -centroid_mean[0] #because we are in the negative area of bregma
#     centroid_cluster_mean[count_i, 1] = centroid_mean[1]
# fov_corner = np.array([fov_coord[0]+0.5, fov_coord[1]-0.5])
# centroid_cluster_dist_corner = (centroid_cluster_mean*0.001)+fov_corner
# #TODO sanity checks with centroids of clusters in ref_image

frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
colors_session = mscope.colors_session(session_type, trials, 1)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
if session_type == 'split':
    colors_phases = ['black', 'crimson', 'teal']
if session_type == 'tied':
    colors_phases = ['black', 'orange', 'purple']
time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
[df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

# Clustering maps spatial and temporal
import matplotlib.image as img
ref_image = img.imread(os.path.join(mscope.path, 'images', 'cluster', 'roi_clustering_fov.png'))
mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, plot_data, 0)

if session_type == 'split':
    trials_plot = np.array([trials_baseline[-1], trials_split[0], trials_split[-1], trials_washout[0]])
if session_type == 'tied':
    trials_plot = np.array(trials_ses[:, 1])

# Order ROIs by cluster
clusters_rois_flat = np.transpose(sum(clusters_rois, []))
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
df_events_extract_zscore_clustered = df_events_extract_rawtrace[clusters_rois_flat]

# raw signal clustered
mscope.response_time_population_avg(df_events_extract_zscore_clustered, [0], [5], clusters_rois, cluster_transition_idx, 'events', 'cluster', plot_data, print_plots)
time_beg_vec = np.arange(0, 60, 5)
time_end_vec = np.arange(5, 60+5, 5)
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
            ax[c].spines['right'].set_visible(False)
            ax[c].spines['top'].set_visible(False)
            ax[c].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        if print_plots:
            plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_events'), dpi=mscope.my_dpi)

for cluster_plot in np.arange(1, len(clusters_rois)+1):
    mscope.plot_stacked_traces_singleROI(df_trace_clusters_ave, traces_type, cluster_plot, trials, colors_session, line_ratio, plot_data, print_plots)