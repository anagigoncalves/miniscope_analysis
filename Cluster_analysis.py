# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# path inputs
path = 'I:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'I:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
# path = 'I:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\'
# path_loco = 'I:\\TM TRACKING FILES\\tied baseline\\tied baseline S1 040421\\'
session_type = path.split('\\')[2].split(' ')[0]
version_mscope = 'v4'
load_data = 1
plot_data = 1
print_plots = 0
save_data = 0
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
path_images = os.path.join(path, 'images')
path_cluster = os.path.join(path, 'images', 'cluster')
path_events = os.path.join(path, 'images', 'events')
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
ops_s2p = mscope.get_s2p_parameters()
session_type = path.split(mscope.delim)[-4].split(' ')[0]  # tied or split
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
colors_session = mscope.colors_session(session_type, trials, 1)
colors_session_boxplot = mscope.colors_session(session_type, trials, 0)

[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()
time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
trial_plot = 3
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data, print_plots)
[clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended, idx_roi_cluster, centroid_ext)
[df_events_extract_rawtrace_clustered, df_extract_rawtrace_detrended_clustered] = mscope.compute_clustered_traces_events_correlations(df_events_extract_rawtrace, df_extract_rawtrace_detrended, clusters_rois, trials_baseline, plot_data, print_plots)
if load_data == 0:
    [df_trace_clusters_ave, df_trace_clusters_std] = mscope.clusters_dataframe(df_extract_rawtrace_detrended, clusters_rois, save_data)
    df_events_trace_clusters = mscope.get_events(df_trace_clusters_ave, 0, 'df_events_trace_clusters')
if load_data == 1:
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
stride_duration_trials = []
final_tracks_forwadloco_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_forwadloco = loco.final_tracks_forwardlocomotion(final_tracks, st_strides_mat)
    final_tracks_forwadloco_trials.append(final_tracks_forwadloco)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-sw-st')

traces_type = 'raw'
mscope.plot_stacked_traces(frame_time, df_trace_clusters_ave, traces_type, trials_ses, plot_data, print_plots)  # input can be one trial or trials_ses

# Calcium events stats
# ISI
isi_events_cluster = mscope.compute_isi(df_events_trace_clusters, traces_type, 'isi_events_clusters')
# CV of ISI
[isi_cv, isi_cv2] = mscope.compute_isi_cv(isi_events_cluster, trials)

# Ratio between ISI values
range_isiratio = [[0, 0.5], [0.8, 1.5]]
isi_ratio = mscope.compute_isi_ratio(isi_events_cluster, range_isiratio, trials)
# ISI analysis for shuffled spikes
df_events_trace_clusters_shuffle = mscope.shuffle_events(df_events_trace_clusters, 1000)
df_events_extract_rawtrace_shuffle = mscope.shuffle_events(df_events_extract_rawtrace, 1000)
isi_events_cluster_shuffle = mscope.compute_isi(df_events_trace_clusters_shuffle, traces_type, 'isi_events_clusters_shuffle')
isi_ratio_shuffle = mscope.compute_isi_ratio(isi_events_cluster_shuffle, range_isiratio, trials)

# Synchronous events
th_sync = 0.7
df_events_sync = mscope.number_sync_events_cluster(df_extract_rawtrace_detrended, df_events_extract_rawtrace, clusters_rois, trials, colors_session, traces_type, th_sync, plot_data, print_plots, save_data)
df_events_sync_shuffle = mscope.shuffle_events(df_events_sync, 1000)

# loop over clusters
align = 'stride'
traj_type = ['time', 'phase']
event_type = ['cluster', 'sync', 'shuffle']
event_type_df = [df_events_trace_clusters, df_events_sync, df_events_trace_clusters_shuffle]
final_tracks_traj = [final_tracks_forwadloco_trials, final_tracks_trials_phase]
for cluster_plot in np.arange(1, len(clusters_rois)+1):
    # mscope.plot_stacked_traces_singleROI(frame_time, df_trace_clusters_ave, traces_type, cluster_plot, trials, colors_session, plot_data, print_plots)
    # mscope.plot_single_cluster_map(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, traces_type, 1, 1)
    # cs_waveforms_mean_all, cs_waveforms_sem_all = mscope.compute_event_waveform(df_trace_clusters_ave, traces_type, df_events_trace_clusters, cluster_plot, animal, session_type, trials_ses, trials, plot_data, print_plots)
    # event_count_loco = mscope.get_event_count_locomotion(df_events_trace_clusters, traces_type, colors_session, trials, bcam_time, st_strides_trials, cluster_plot, plot_data, print_plots)
    # event_count_all = mscope.get_event_count_wholetrial(df_events_trace_clusters, traces_type, colors_session, trials, cluster_plot, plot_data, print_plots)
    # pixel_step = 20
    # for paw in paws:
    #     event_corridor = mscope.event_corridor_distribution(df_events_trace_clusters, final_tracks_trials, bcam_time, cluster_plot, paw, trials_baseline, pixel_step, traces_type, plot_data, print_plots)
    #     df_cs_stride = mscope.events_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, paw, cluster_plot, align, save_data)
    #     event_probability = mscope.event_probability_plot(df_cs_stride, df_events_trace_clusters, traces_type, colors_session, paw, cluster_plot, plot_data, print_plots)
    # mscope.plot_cv_session(cluster_plot, isi_cv, traces_type, colors_session, trials, 'cv', plot_data, print_plots)
    # mscope.plot_cv_session(cluster_plot, isi_cv2, traces_type, colors_session_boxplot, trials, 'cv2', plot_data, print_plots)
    # mscope.plot_isi_ratio_session(cluster_plot, isi_ratio, isi_ratio_shuffle, traces_type, colors_session, range_isiratio, trials, plot_data, print_plots)
    # mscope.plot_isi_boxplots(cluster_plot, isi_events_cluster, traces_type, session_type, animal, trials, plot_data, print_plots)
    # for trial_plot in trials:
    #     [hist_xaxis, hist_norm] = mscope.plot_isi_single_trial(trial_plot, cluster_plot, isi_events_cluster, traces_type, plot_data, print_plots)
    for count_traj, traj in enumerate(traj_type):
        data_cv_events = []
        traj_diff_front_ave_events = []
        traj_diff_front_sem_events = []
        traj_diff_hind_ave_events = []
        traj_diff_hind_sem_events = []
        for count_event, event in enumerate(event_type):
            traj_cluster_trials = []
            traj_cv_cluster_trials = []
            for t in trials:
                # trajectories around event onset
                [traj_time, traj_cluster] = mscope.events_align_trajectory(event_type_df[count_event], traces_type, traj, bcam_time, final_tracks_traj[count_traj], t, trials, cluster_plot, event, plot_data, print_plots)
                # coefficient of variation of trajectories around event onset
                traj_cv_cluster = mscope.events_cv_trajectory(event_type_df[count_event], traj, bcam_time, final_tracks_traj[count_traj], t, trials, cluster_plot)
                traj_cluster_trials.append(traj_cluster)
                traj_cv_cluster_trials.append(traj_cv_cluster)
                plt.close('all')
            # plot trajectories around event onset
            mscope.events_align_trajectory_plot_all(traj_cluster_trials, traj, cluster_plot, traces_type, event, trials, colors_session, plot_data, print_plots)
            # plot coefficient of variation of trajectories around event onset
            data_cv = mscope.events_cv_trajectory_plot(traj_cv_cluster_trials, traj, cluster_plot, traces_type, event, trials, colors_session, plot_data, print_plots)
            # plot difference of trajectories around evet onset
            [traj_diff_cluster_front_ave, traj_diff_cluster_front_sem, traj_diff_cluster_hind_ave, traj_diff_cluster_hind_sem] = mscope.diff_paws_around_event(event_type_df[count_event], traj, cluster_plot, traces_type, event, bcam_time, final_tracks_traj[count_traj], trials, colors_session, plot_data, print_plots)
            plt.close('all')
            data_cv_events.append(data_cv)
            traj_diff_front_ave_events.append(traj_diff_cluster_front_ave)
            traj_diff_front_sem_events.append(traj_diff_cluster_front_sem)
            traj_diff_hind_ave_events.append(traj_diff_cluster_hind_ave)
            traj_diff_hind_sem_events.append(traj_diff_cluster_hind_sem)
        # # comparison cluster and sync
        # fig, ax = plt.subplots(2, 2, tight_layout=True, sharex=True)
        # ax = ax.ravel()
        # for p in range(4):
        #     data_line = np.zeros(len(trials))
        #     for t in trials:
        #         ax[p].scatter(t, data_cv_events[1][p][t - 1], color=colors_session[t - 1])
        #         ax[p].scatter(t, data_cv_events[0][p][t - 1], color='gray')
        #     ax[p].plot(trials, data_cv_events[1][p], color='black')
        #     ax[p].plot(trials, data_cv_events[0][p], color='gray')
        #     ax[p].set_xlabel('Trials', fontsize=mscope.fsize - 8)
        #     ax[p].set_ylabel('CV', fontsize=mscope.fsize - 8)
        #     ax[p].spines['right'].set_visible(False)
        #     ax[p].spines['top'].set_visible(False)
        #     ax[p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # plt.suptitle('Cluster ' + str(cluster_plot) + ' for time of event peak sync colors all gray')
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_cv_event_onset_sync_cluster'), dpi=mscope.my_dpi)
        # fig, ax = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
        # ax = ax.ravel()
        # for t in trials:
        #     ax[0].scatter(t, traj_diff_front_ave_events[1][t-1], s=30, color=colors_session[t-1])
        #     ax[0].scatter(t, traj_diff_front_ave_events[0][t-1], s=30, color='gray')
        #     ax[1].scatter(t, traj_diff_hind_ave_events[1][t-1], s=30, color=colors_session[t - 1])
        #     ax[1].scatter(t, traj_diff_hind_ave_events[0][t-1], s=30, color='gray')
        # ax[0].plot(trials, traj_diff_front_ave_events[1], color='black')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[1]-traj_diff_front_sem_events[1], traj_diff_front_ave_events[1]+traj_diff_front_sem_events[1], color='black', alpha=0.3)
        # ax[0].plot(trials, traj_diff_front_ave_events[0], color='gray')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[0]-traj_diff_front_sem_events[0], traj_diff_front_ave_events[0]+traj_diff_front_sem_events[0], color='gray', alpha=0.3)
        # ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[0].set_ylabel('Front paw difference', fontsize=mscope.fsize - 8)
        # ax[0].spines['right'].set_visible(False)
        # ax[0].spines['top'].set_visible(False)
        # ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[0].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # ax[1].plot(trials, traj_diff_hind_ave_events[1], color='black')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[1]-traj_diff_hind_sem_events[1], traj_diff_hind_ave_events[1]+traj_diff_hind_sem_events[1], color='black', alpha=0.3)
        # ax[1].plot(trials, traj_diff_hind_ave_events[0], color='gray')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[0]-traj_diff_hind_sem_events[0], traj_diff_hind_ave_events[0]+traj_diff_hind_sem_events[0], color='gray', alpha=0.3)
        # ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[1].set_ylabel('Hind paw difference', fontsize=mscope.fsize - 8)
        # ax[1].spines['right'].set_visible(False)
        # ax[1].spines['top'].set_visible(False)
        # ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[1].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_diff_ave_sync_cluster'), dpi=mscope.my_dpi)
        # # comparison sync and shuffle
        # fig, ax = plt.subplots(2, 2, tight_layout=True, sharex=True)
        # ax = ax.ravel()
        # for p in range(4):
        #     data_line = np.zeros(len(trials))
        #     for t in trials:
        #         ax[p].scatter(t, data_cv_events[1][p][t - 1], color=colors_session[t - 1])
        #         ax[p].scatter(t, data_cv_events[2][p][t - 1], color='gray')
        #     ax[p].plot(trials, data_cv_events[1][p], color='black')
        #     ax[p].plot(trials, data_cv_events[2][p], color='gray')
        #     ax[p].set_xlabel('Trials', fontsize=mscope.fsize - 8)
        #     ax[p].set_ylabel('CV', fontsize=mscope.fsize - 8)
        #     ax[p].spines['right'].set_visible(False)
        #     ax[p].spines['top'].set_visible(False)
        #     ax[p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # plt.suptitle('Cluster ' + str(cluster_plot) + ' for time of event peak sync colors shuffle gray')
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_cv_event_onset_sync_shuffle'), dpi=mscope.my_dpi)
        # fig, ax = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
        # ax = ax.ravel()
        # for t in trials:
        #     ax[0].scatter(t, traj_diff_front_ave_events[1][t-1], s=30, color=colors_session[t-1])
        #     ax[0].scatter(t, traj_diff_front_ave_events[2][t-1], s=30, color='gray')
        #     ax[1].scatter(t, traj_diff_hind_ave_events[1][t-1], s=30, color=colors_session[t - 1])
        #     ax[1].scatter(t, traj_diff_hind_ave_events[2][t-1], s=30, color='gray')
        # ax[0].plot(trials, traj_diff_front_ave_events[1], color='black')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[1]-traj_diff_front_sem_events[1], traj_diff_front_ave_events[1]+traj_diff_front_sem_events[1], color='black', alpha=0.3)
        # ax[0].plot(trials, traj_diff_front_ave_events[2], color='gray')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[2]-traj_diff_front_sem_events[2], traj_diff_front_ave_events[0]+traj_diff_front_sem_events[2], color='gray', alpha=0.3)
        # ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[0].set_ylabel('Front paw difference', fontsize=mscope.fsize - 8)
        # ax[0].spines['right'].set_visible(False)
        # ax[0].spines['top'].set_visible(False)
        # ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[0].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # ax[1].plot(trials, traj_diff_hind_ave_events[1], color='black')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[1]-traj_diff_hind_sem_events[1], traj_diff_hind_ave_events[1]+traj_diff_hind_sem_events[1], color='black', alpha=0.3)
        # ax[1].plot(trials, traj_diff_hind_ave_events[2], color='gray')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[2]-traj_diff_hind_sem_events[2], traj_diff_hind_ave_events[0]+traj_diff_hind_sem_events[2], color='gray', alpha=0.3)
        # ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[1].set_ylabel('Hind paw difference', fontsize=mscope.fsize - 8)
        # ax[1].spines['right'].set_visible(False)
        # ax[1].spines['top'].set_visible(False)
        # ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[1].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_diff_ave_sync_shuffle'), dpi=mscope.my_dpi)
        # # comparison cluster and shuffle (cluster shuffle)
        # fig, ax = plt.subplots(2, 2, tight_layout=True, sharex=True)
        # ax = ax.ravel()
        # for p in range(4):
        #     data_line = np.zeros(len(trials))
        #     for t in trials:
        #         ax[p].scatter(t, data_cv_events[0][p][t - 1], color=colors_session[t - 1])
        #         ax[p].scatter(t, data_cv_events[2][p][t - 1], color='gray')
        #     ax[p].plot(trials, data_cv_events[0][p], color='black')
        #     ax[p].plot(trials, data_cv_events[2][p], color='gray')
        #     ax[p].set_xlabel('Trials', fontsize=mscope.fsize - 8)
        #     ax[p].set_ylabel('CV', fontsize=mscope.fsize - 8)
        #     ax[p].spines['right'].set_visible(False)
        #     ax[p].spines['top'].set_visible(False)
        #     ax[p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # plt.suptitle('Cluster ' + str(cluster_plot) + ' for time of event peak sync colors shuffle gray')
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_cv_event_onset_cluster_shufflecluster'), dpi=mscope.my_dpi)
        # fig, ax = plt.subplots(2, 1, figsize=(7, 10), tight_layout=True)
        # ax = ax.ravel()
        # for t in trials:
        #     ax[0].scatter(t, traj_diff_front_ave_events[0][t-1], s=30, color=colors_session[t-1])
        #     ax[0].scatter(t, traj_diff_front_ave_events[2][t-1], s=30, color='gray')
        #     ax[1].scatter(t, traj_diff_hind_ave_events[0][t-1], s=30, color=colors_session[t - 1])
        #     ax[1].scatter(t, traj_diff_hind_ave_events[2][t-1], s=30, color='gray')
        # ax[0].plot(trials, traj_diff_front_ave_events[0], color='black')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[0]-traj_diff_front_sem_events[0], traj_diff_front_ave_events[1]+traj_diff_front_sem_events[0], color='black', alpha=0.3)
        # ax[0].plot(trials, traj_diff_front_ave_events[2], color='gray')
        # ax[0].fill_between(trials, traj_diff_front_ave_events[2]-traj_diff_front_sem_events[2], traj_diff_front_ave_events[0]+traj_diff_front_sem_events[2], color='gray', alpha=0.3)
        # ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[0].set_ylabel('Front paw difference', fontsize=mscope.fsize - 8)
        # ax[0].spines['right'].set_visible(False)
        # ax[0].spines['top'].set_visible(False)
        # ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[0].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # ax[1].plot(trials, traj_diff_hind_ave_events[0], color='black')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[1]-traj_diff_hind_sem_events[0], traj_diff_hind_ave_events[1]+traj_diff_hind_sem_events[0], color='black', alpha=0.3)
        # ax[1].plot(trials, traj_diff_hind_ave_events[2], color='gray')
        # ax[1].fill_between(trials, traj_diff_hind_ave_events[2]-traj_diff_hind_sem_events[2], traj_diff_hind_ave_events[0]+traj_diff_hind_sem_events[2], color='gray', alpha=0.3)
        # ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        # ax[1].set_ylabel('Hind paw difference', fontsize=mscope.fsize - 8)
        # ax[1].spines['right'].set_visible(False)
        # ax[1].spines['top'].set_visible(False)
        # ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        # ax[1].set_title('Cluster ' + str(cluster_plot) + ' homolateral paw subtraction events in ' + traj)
        # plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot),
        #                          'event_paw_' + traj + '_diff_ave_cluster_shufflecluster'), dpi=mscope.my_dpi)

# # Cumulative activity for each cluster
# mscope.cumulative_activity_cluster(df_events_trace_clusters, time_cumulative, clusters_rois, trials, colors_cluster, plot_data, print_plots)

# # CV2 stacked traces
# for cluster_plot in range(len(clusters_rois)):
#     mscope.plot_cv2_stacked_traces(isi_cv2, traces_type, cluster_plot+1, trials, colors_session, plot_data, print_plots)
# plt.close('all')

# # Order ROIs by cluster
# clusters_rois_flat = np.transpose(sum(clusters_rois, []))
# clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
# clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
# cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
# df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')
# df_extract_rawtrace_detrended_zscore_clustered = df_extract_rawtrace_detrended_zscore[clusters_rois_flat]

# # raw signal clustered
# time_beg_vec = np.arange(0, 60, 5)
# time_end_vec = np.arange(5, 65, 5)
# mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore_clustered, time_beg_vec, time_end_vec, clusters_rois, cluster_transition_idx, 'cluster', plot_data, print_plots)
# plt.close('all')
# Continuous event parameter computations
# [param_all_idx, param_all_time, param_all] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, 'FR', 'FL', 1)
# window = 1
# for count_t, t in enumerate(time_beg_vec):
#     if plot_data:
#         # step length symmetry
#         mscope.plot_param_continuous(param_all_time, param_all, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # mean activity all ROIs
#         mscope.mean_activity_time_period(df_extract_rawtrace_detrended_zscore_clustered, 'roi', clusters_rois, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # cv2 all ROIs
#         mscope.cv2_time_period(df_events_extract_rawtrace, 'roi', clusters_rois, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # mean activity clusters
#         mscope.mean_activity_time_period(df_trace_clusters_ave, 'cluster', clusters_rois, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # cv2 clusters
#         mscope.cv2_time_period(df_events_trace_clusters, 'cluster', clusters_rois, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # nr of rois sync
#         mscope.rois_sync_time_period(df_events_extract_rawtrace, clusters_rois, trials, time_beg_vec[count_t], time_end_vec[count_t], colors_session, plot_data, print_plots)
#         # event probability
#         mscope.event_probability_time_period(df_events_trace_clusters, window, clusters_rois, trials, 'cluster', colors_session, time_beg_vec[count_t], time_end_vec[count_t], plot_data, print_plots)
#         plt.close('all')
#
# # # TODO PCA - On raw and deconv data do PCA and check in a baseline, split, and washout trial the variance of their components
# # # TODO Simulate each ROI with Poisson distribution, can take nr of all the ROIs as parameter, bin is 1/sr
# # # TODO recheck event corridor computation and do per session phase
# # # TODO distinguish moments of synchrony and moments of high activity
