# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns

# path inputs
path = 'I:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'I:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
plot_data = 1
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

# create plots folders
if delim == '/':
    path_images = path + '/images/'
    path_cluster = path + '/images/cluster/'
    path_events = path + '/images/events/'
else:
    path_images = path + '\\images\\'
    path_cluster = path + '\\images\\cluster\\'
    path_events = path + '\\images\\events\\'
if not os.path.exists(path_images):
    os.mkdir(path_images)
if not os.path.exists(path_cluster):
    os.mkdir(path_cluster)
if not os.path.exists(path_events):
    os.mkdir(path_events)

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()
trials = mscope.get_trial_id()
frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
strobe_nr_txt = loco.bcam_strobe_number()
trial_start_blip_nr = loco.trial_start_blips()
frame_time = mscope.get_miniscope_frame_time(trials, frames_dFF, version_mscope)  # get frame time for each trial
ref_image = mscope.get_ref_image()
session_type = path.split(delim)[-4].split(' ')[0]  # tied or split
if session_type == 'tied' and animal == 'MC8855':
    trials_ses = np.array([3, 6])
    trials_ses_name = ['baseline speed', 'fast speed']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[0] + 1, trials_ses[1] + 1)]
    cond_plot = ['baseline', 'fast']
if session_type == 'tied' and animal != 'MC8855':
    trials_ses = np.array([6, 12, 18])
    trials_ses_name = ['baseline speed', 'fast speed']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[0] + 1, trials_ses[1] + 1),
                np.arange(trials_ses[1] + 1, trials_ses[2] + 1)]
    cond_plot = ['baseline', 'slow', 'fast']
if session_type == 'split' and animal == 'MC8855':
    trials_ses = np.array([3, 4, 13, 14])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[1], trials_ses[2] + 1),
                np.arange(trials_ses[3], len(trials) + 1)]
    cond_plot = ['baseline', 'split', 'washout']
if session_type == 'split' and animal != 'MC8855':
    trials_ses = np.array([6, 7, 16, 17])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[1], trials_ses[2] + 1),
                np.arange(trials_ses[3], len(trials) + 1)]
    cond_plot = ['baseline', 'split', 'washout']
if len(trials) == 23:
    trials_baseline = np.array([1, 2, 3])
    trials_split = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    trials_washout = np.array([14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
elif len(trials) == 26:
    trials_baseline = np.array([1, 2, 3, 4, 5, 6])
    trials_split = np.array([7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    trials_washout = np.array([17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
elif len(trials) < 23:
    trials_baseline = trials
greys = mp.cm.get_cmap('Greys', 14)
reds = mp.cm.get_cmap('Reds', 23)
blues = mp.cm.get_cmap('Blues', 23)
if len(trials) == 23:
    colors_session = [greys(12), greys(7), greys(4), reds(23), reds(21), reds(19), reds(17), reds(15),
                      reds(13),
                      reds(11), reds(9), reds(7), reds(5), blues(23), blues(21), blues(19), blues(17),
                      blues(15), blues(13),
                      blues(11), blues(9), blues(7), blues(5)]
if len(trials) == 26:
    colors_session = [greys(14), greys(12), greys(10), greys(8), greys(6), greys(4), reds(23), reds(21),
                      reds(19), reds(17), reds(15), reds(13),
                      reds(11), reds(9), reds(7), reds(5), blues(23), blues(21), blues(19), blues(17),
                      blues(15), blues(13),
                      blues(11), blues(9), blues(7), blues(5)]

# summary gait parameters
filelist = loco.get_track_files(animal, session)
param_sym_name = 'coo_stance'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
stride_duration_trials = []
param_sym = np.zeros(len(trials))
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_sym_name))
    param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_sym_name)
    param_sym[count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
    stride_duration_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'stride_duration'))

# baseline subtracion of parameter
param_sym_bs = np.zeros(np.shape(param_sym))
bs_mean = np.nanmean(param_sym[:trials_baseline[-1]-1])
param_sym_bs = param_sym - bs_mean

# Population plots  split ipsi fast
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, trials,
 coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()
df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')
df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')

# Clustering
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
trial_plot = 3
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                  distance_neurons, trial_plot, th_cluster,
                                                                  colormap_cluster, plot_data)
[clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                 idx_roi_cluster, centroid_ext)
furthest_neuron = np.argmax(np.array(centroid_ext)[:, 0])
neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
roi_list = df_extract_rawtrace.columns[2:]
roi_list_ordered = roi_list[neuron_order].insert(0, 'time').insert(0, 'trial')
df_extract_rawtrace_detrended_zscore_distance = df_extract_rawtrace_detrended_zscore[roi_list_ordered]
cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1

# Order ROIs by cluster
clusters_rois_flat = np.transpose(sum(clusters_rois, []))
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
df_extract_rawtrace_detrended_zscore_clustered = df_extract_rawtrace_detrended_zscore[clusters_rois_flat]
df_events_extract_rawtrace_clustered = df_events_extract_rawtrace[clusters_rois_flat]

time_beg = np.arange(0, 60, 5)
time_end = np.arange(5, 65, 5)
# # raw signal unclustered
# traces_type = 'raw'
# plot_type = 'raw'
# trial_avg_raw = mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore, time_beg, time_end, clusters_rois, cluster_transition_idx, plot_type, trials_baseline, trials_split, colors_session, save_data)
# # raw signal clustered
# traces_type = 'raw'
# plot_type = 'cluster'
# trial_avg_cluster = mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore_clustered, time_beg, time_end, clusters_rois, cluster_transition_idx, plot_type, trials_baseline, trials_split, colors_session, save_data)
# # raw signal distance order
# traces_type = 'raw'
# plot_type = 'distance'
# trial_avg_distance = mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore_distance, time_beg, time_end, clusters_rois, cluster_transition_idx, plot_type, trials_baseline, trials_split, colors_session, save_data)

# if save_data:
#     # Median activity for all ROIs and all trials
#     fig, ax = plt.subplots(3, 2, figsize=(30,20), tight_layout=True)
#     ax = ax.ravel()
#     ax[0].add_patch(
#         plt.Rectangle((np.where(df_extract_rawtrace_detrended_zscore_clustered['trial']==trials_baseline[-1])[0][-1], -2),
#                       np.where(df_extract_rawtrace_detrended_zscore_clustered['trial']==trials_split[-1])[0][-1] - np.where(df_extract_rawtrace_detrended_zscore_clustered['trial']==trials_baseline[-1])[0][-1],
#                       10,
#                       fc='lightgray', alpha=0.7))
#     ax[0].plot(df_extract_rawtrace_detrended_zscore_clustered.iloc[:,2:].median(axis=1), color='black')
#     for t in trials:
#         ax[0].axvline(x = np.where(df_extract_rawtrace_detrended_zscore_clustered['trial']==t)[0][-1], color='red', linestyle = 'dashed')
#         ax[0].set_title('All ROIs', fontsize=mscope.fsize - 2)
#         ax[0].set_ylabel('Median DeltaF/F', fontsize=mscope.fsize - 4)
#         ax[0].set_xlabel('Frames', fontsize=mscope.fsize - 4)
#         ax[0].spines['right'].set_visible(False)
#         ax[0].spines['top'].set_visible(False)
#         ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 4)
#     for c in range(len(clusters_rois)):
#         df_rois_plot = df_extract_rawtrace_detrended_zscore_clustered[clusters_rois[c]]
#         ax[c+1].add_patch(
#             plt.Rectangle(
#                 (np.where(df_extract_rawtrace_detrended_zscore_clustered['trial'] == trials_baseline[-1])[0][-1], -2),
#                 np.where(df_extract_rawtrace_detrended_zscore_clustered['trial'] == trials_split[-1])[0][-1] -
#                 np.where(df_extract_rawtrace_detrended_zscore_clustered['trial'] == trials_baseline[-1])[0][-1],
#                 10,
#                 fc='lightgray', alpha=0.7))
#         ax[c+1].plot(df_rois_plot.iloc[:, 2:].median(axis=1), color='black')
#         for t in trials:
#             ax[c+1].axvline(x=np.where(df_extract_rawtrace_detrended_zscore_clustered['trial'] == t)[0][-1], color='red',
#                           linestyle='dashed')
#         ax[c+1].set_title('Cluster '+str(c+1), fontsize=mscope.fsize - 2)
#         ax[c+1].set_ylabel('Median DeltaF/F', fontsize=mscope.fsize - 4)
#         ax[c+1].set_xlabel('Frames', fontsize=mscope.fsize - 4)
#         ax[c+1].spines['right'].set_visible(False)
#         ax[c+1].spines['top'].set_visible(False)
#         ax[c+1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 4)
#     plt.savefig(os.path.join(mscope.path, 'images', 'raw','median_activity_all_trials_clusters'), dpi=mscope.my_dpi)

# # event signal unclustered
# traces_type = 'events'
# plot_type = 'raw'
# trial_avg_events_raw = mscope.response_time_population_events(df_events_extract_rawtrace, time_beg, time_end, cluster_rois_ordered, cluster_transition_idx, plot_type, trials_baseline, trials_split, colors_session, 1)
# # event signal clustered
# traces_type = 'events'
# plot_type = 'cluster'
# trial_avg_events_cluster = mscope.response_time_population_events(df_events_extract_rawtrace_clustered, time_beg, time_end, cluster_rois_ordered, cluster_transition_idx, plot_type, trials_baseline, trials_split, colors_session, 1)

# Step length symmetry analysis
traces_type = 'raw'
step_size = 2
p1 = 'FR'
p2 = 'FL'
roi_list = df_extract_rawtrace_detrended.columns[2:]
roi_plot = []
for r in range(len(roi_list)):
    roi_plot.append(np.int64(roi_list[r][3:]))
sl_front_events_baseline = np.zeros((len(roi_plot),51))
sl_front_events_split = np.zeros((len(roi_plot),51))
sl_front_events_washout = np.zeros((len(roi_plot),51))
for count_r, roi in enumerate(roi_plot):
    for count_t, trials_compute in enumerate(idx_plot):
        [bins, sl_p1_events_trials, sl_p1_events_trials_shuffled, t_stat, p_value] = mscope.param_events_plot(param_trials, st_strides_trials,
                                                               df_events_extract_rawtrace, param_sym_name, roi, p1,
                                                               p2, step_size, trials_compute, trials,
                                                               traces_type, cond_plot[count_t], stride_duration_trials,
                                                               0)
        if count_t == 0:
            sl_front_events_baseline[count_r,:] = sl_p1_events_trials
        if count_t == 1:
            sl_front_events_split[count_r,:] = sl_p1_events_trials
        if count_t == 2:
            sl_front_events_washout[count_r,:] = sl_p1_events_trials

clusters_rois_flat_idx = [np.where(i == df_extract_rawtrace_detrended.columns[2:])[0][0] for i in clusters_rois_flat[2:]]
sl_front_events_baseline_clustered = sl_front_events_baseline[clusters_rois_flat_idx, :]
sl_front_events_split_clustered = sl_front_events_split[clusters_rois_flat_idx, :]
sl_front_events_washout_clustered = sl_front_events_washout[clusters_rois_flat_idx, :]
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_baseline[:, ~np.isnan(sl_front_events_baseline[0,:])], cmap='viridis')
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) unclustered for baseline',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(df_events_extract_rawtrace.columns[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_baseline[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_baseline_unclustered'),
                dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_baseline_clustered[:, ~np.isnan(sl_front_events_baseline_clustered[0,:])], cmap='viridis')
for c in cluster_transition_idx:
    ax.hlines(c+1, *ax.get_xlim(), color='white', linestyle='dashed')  # +1 puts in beginning of cluster
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) clustered for baseline',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(clusters_rois_flat[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_baseline[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_baseline_clustered'),
                dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_split[:, ~np.isnan(sl_front_events_split[0,:])], cmap='viridis')
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) unclustered for split',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(df_events_extract_rawtrace.columns[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_split[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_split_unclustered'),
                dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_split_clustered[:, ~np.isnan(sl_front_events_split_clustered[0,:])], cmap='viridis')
for c in cluster_transition_idx:
    ax.hlines(c+1, *ax.get_xlim(), color='white', linestyle='dashed')  # +1 puts in beginning of cluster
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) clustered for split',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(clusters_rois_flat[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_split[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_split_clustered'),
                dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_washout[:, ~np.isnan(sl_front_events_washout[0,:])], cmap='viridis')
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) unclustered for washout',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(df_events_extract_rawtrace.columns[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_washout[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_washout_unclustered'),
                dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(15, 5), tight_layout = True)
sns.heatmap(sl_front_events_washout_clustered[:, ~np.isnan(sl_front_events_washout_clustered[0,:])], cmap='viridis')
for c in cluster_transition_idx:
    ax.hlines(c+1, *ax.get_xlim(), color='white', linestyle='dashed')  # +1 puts in beginning of cluster
ax.set_title(param_sym_name.replace('_', ' ') + ' symmetry event proportion (stride duration normalized) clustered for washout',
             fontsize=mscope.fsize - 4)
ax.set_yticks(np.arange(0, len(df_events_extract_rawtrace.columns[2:]), 6))
ax.set_yticklabels(clusters_rois_flat[2::6], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticklabels(map(str, bins[~np.isnan(sl_front_events_washout[0,:])]), fontsize=mscope.fsize - 10)
ax.set_xlabel(param_sym_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 8)
if save_data:
    if not os.path.exists(os.path.join(mscope.path, 'images','events','raw')):
        os.mkdir(os.path.join(mscope.path, 'images','events','raw'))
    plt.savefig(os.path.join(mscope.path, 'images','events','raw',
                             'heatmap_' + param_sym_name + '_washout_clustered'),
                dpi=mscope.my_dpi)