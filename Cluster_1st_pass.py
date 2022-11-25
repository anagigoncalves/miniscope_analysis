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
plot_data = 1
load_data = 1
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
if session_type == 'split':
    colors_phases = ['black', 'red', 'blue']
if session_type == 'tied':
    colors_phases = ['black', 'orange', 'purple']

if load_data == 0:
    # Load ROIs and traces - EXTRACT
    thrs_spatial_weights = 0
    [coord_ext, df_extract_allframes] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trials)
    # # Load ROIs and traces - IMAGEJ
    # norm = 0
    # [coord_fiji, df_fiji_allframes] = mscope.get_imagej_output(frame_time, trials, norm)

    # Good periods after motion correction
    [x_offset, y_offset, corrXY] = mscope.get_reg_data() # registration bad moments - correct
    th = 0.006 # change with the notes from EXCEL
    # [idx_to_nan,df_fiji] = mscope.corr_FOV_movement(th, df_fiji_allframes, corrXY)
    [idx_to_nan,df_extract] = mscope.corr_FOV_movement(th, df_extract_allframes, corrXY)
    [width_roi_ext, height_roi_ext, aspect_ratio_ext] = mscope.get_roi_stats(coord_ext)
    [coord_ext, df_extract] = mscope.rois_larger_motion(df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi_ext, height_roi_ext, 1)
    trial_corr = 2
    mscope.correlation_signal_motion(df_extract, x_offset, y_offset, trial_corr, idx_to_nan, 1)

    # ROI curation
    trial_curation = 2
    # ROI spatial stats
    [width_roi_rois_nomotion, height_roi_rois_nomotion, aspect_ratio_rois_nomotion] = mscope.get_roi_stats(coord_ext)
    [coord_ext_curated, df_extract_curated] = mscope.roi_curation(ref_image, df_extract, coord_ext, aspect_ratio_rois_nomotion, trial_curation)

    # # Get background subtracted trace from ImageJ segmentation
    # coeff_sub = 1
    # ## ITS GIVING SOME ROIs AS ALL NULL VALUES
    # df_trace_bgsub = mscope.compute_bg_roi_fiji(coord_fiji, trials, frame_time, df_fiji_allframes, coeff_sub)

    # Get raw trace from EXTRACT ROIs
    roi_list = list(df_extract_curated.columns[2:])
    df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext_curated, df_extract_curated, roi_list, trials, frame_time)

    # Find calcium events - label as synchronous or asynchronous
    df_events_extract = mscope.get_events(df_extract_curated, 0, 'df_events_extract')
    df_events_extract_rawtrace = mscope.get_events(df_extract_rawtrace, 1, 'df_events_extract_rawtrace')
    roi_plot = np.int64(np.random.choice(roi_list)[3:])
    trial_plot = np.random.choice(trials)
    mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace, df_events_extract_rawtrace, 0)

    # Detrend calcium trace
    df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')

    # Save data
    mscope.save_processed_files(df_extract_curated, trials, df_events_extract,  df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext_curated, th, idx_to_nan)

    # Data as clusters
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    th_cluster = 0.6
    colormap_cluster = 'hsv'
    trial_plot = 3
    [colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                      distance_neurons, trial_plot, th_cluster,
                                                                      colormap_cluster, plot_data, print_plots)
    [clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                     idx_roi_cluster, centroid_ext)
    [df_events_extract_rawtrace_clustered,
     df_extract_rawtrace_detrended_clustered] = mscope.compute_clustered_traces_events_correlations(
        df_events_extract_rawtrace, df_extract_rawtrace_detrended, clusters_rois, trials_baseline, plot_data,
        print_plots)
    [df_trace_clusters_ave, df_trace_clusters_std] = mscope.clusters_dataframe(df_extract_rawtrace_detrended, clusters_rois, save_data)
    df_events_trace_clusters = mscope.get_events(df_trace_clusters_ave, 0, 'df_events_trace_clusters')

if load_data:
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    th_cluster = 0.6
    colormap_cluster = 'hsv'
    trial_plot = 3
    [colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                      distance_neurons, trial_plot, th_cluster,
                                                                      colormap_cluster, plot_data, print_plots)
    [clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                     idx_roi_cluster, centroid_ext)


# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
param_trials_fr_mean = np.zeros(len(trials))
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
    param_trials_fr_mean[count_trial] = np.nanmean(param_trials[-1][0])-np.nanmean(param_trials[-1][2])
final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')

# SL sym curve across trials
mscope.plot_sl_sym_session(param_trials_fr_mean, trials_ses, trials, session_type, colors_session, plot_data, print_plots)

traces_type = 'raw'
if session_type == 'split':
    trials_plot = np.array([trials_baseline[-1], trials_split[0], trials_split[-1], trials_washout[0]])
if session_type == 'tied':
    trials_plot = np.array([trials_baseline[-1], trials_ses[:, 0]])
mscope.plot_stacked_traces(frame_time, df_trace_clusters_ave, traces_type, trials_plot, plot_data, print_plots)  # input can be one trial or trials_ses

# Order ROIs by cluster
clusters_rois_flat = np.transpose(sum(clusters_rois, []))
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')
df_extract_rawtrace_detrended_zscore_clustered = df_extract_rawtrace_detrended_zscore[clusters_rois_flat]

# raw signal clustered
time_beg_vec = [0]
time_end_vec = [5]
mscope.response_time_population_avg(df_extract_rawtrace_detrended_zscore_clustered, time_beg_vec, time_end_vec, clusters_rois, cluster_transition_idx, 'cluster', plot_data, print_plots)
if plot_data:
    fig, ax = plt.subplots(1, len(clusters_rois), figsize=(15, 5), tight_layout=True, sharey=True)
    ax = ax.ravel()
    for c in range(len(clusters_rois)):
        mean_data_trials = np.zeros(len(trials))
        std_data_trials = np.zeros(len(trials))
        for t in trials:
            data_trials = df_extract_rawtrace_detrended_zscore_clustered.loc[df_extract_rawtrace_detrended_zscore_clustered['trial'] == t, clusters_rois[c]].iloc[time_beg_vec[0] * mscope.sr:time_end_vec[0] * mscope.sr].mean(axis=0)
            mean_data_trials[t-1] = data_trials.mean()
            std_data_trials[t-1] = data_trials.std()
        ax[c].add_patch(plt.Rectangle((trials_baseline[-1] + 0.5, np.min(mean_data_trials-std_data_trials)), len(trials_split), np.max(mean_data_trials+std_data_trials)-np.min(mean_data_trials-std_data_trials), fc='grey', alpha=0.3))
        ax[c].hlines(0, 1, len(mean_data_trials), colors='grey', linestyles='--')
        ax[c].plot(trials, mean_data_trials, marker='o', color=colors_cluster[c], markersize=5)
        ax[c].fill_between(trials, mean_data_trials-std_data_trials, mean_data_trials+std_data_trials, color=colors_cluster[c], alpha=0.3)
        ax[c].set_xlabel('Time (s)', fontsize=mscope.fsize - 8)
        ax[c].set_ylabel('Mean activity (dFF)', fontsize=mscope.fsize - 8)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
    if print_plots:
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'avg_activity_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_raw'), dpi=mscope.my_dpi)

traj = 'time'
time_window = 0.2
sym = 1
remove_nan = 0
for cluster_plot in np.arange(1, len(clusters_rois)+1):
    mscope.plot_stacked_traces_singleROI(frame_time, df_trace_clusters_ave, traces_type, cluster_plot, trials, colors_session, plot_data, print_plots)
    mscope.plot_single_cluster_map(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, traces_type, plot_data, print_plots)
    align_str = ['st', 'sw']
    for align in align_str:
        # raster
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
        ax = ax.ravel()
        for count_p, p in enumerate(paws):
            [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
            ax[count_p].scatter(events_stride_trial, cumulative_idx, s=1, color='black')
            ax[count_p].axvline(x=0, color='black')
            ax[count_p].axhline(y=np.where(trial_id == trials_ses[0, 1])[0][-1], color='black', linestyle='dashed')
            ax[count_p].axhline(y=np.where(trial_id == trials_ses[1, 1])[0][-1], color='black', linestyle='dashed')
            ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            ax[count_p].set_ylabel('Aligned to ' + str(align), fontsize=mscope.fsize - 8)
            ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
            ax[count_p].spines['right'].set_visible(False)
            ax[count_p].spines['top'].set_visible(False)
            ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot))):
            os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot)))
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_raster_trial_order_' + align + '_' + traces_type), dpi=mscope.my_dpi)
        # histogram for the different session phases
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
        ax = ax.ravel()
        for count_p, p in enumerate(paws):
            [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters,
                                                                                       st_strides_trials, sw_strides_trials,
                                                                                       align, trials, p, cluster_plot,
                                                                                       time_window, traj)
            for t in range(np.shape(trials_ses)[0]):
                [hist_result, xaxis] = np.histogram(events_stride_trial[np.where(trial_id == trials_ses[t, 0])[0][0]:
                                                                        np.where(trial_id == trials_ses[t, 1])[0][-1]],
                                                    range=(-time_window * 1000, time_window * 1000), bins=20)
                ax[count_p].plot(xaxis[:-1], hist_result / np.nanmax(hist_result), color=colors_phases[t], linewidth=2)
            ax[count_p].axvline(x=0, color='black')
            ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            ax[count_p].set_ylabel('Aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
            ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
            ax[count_p].spines['right'].set_visible(False)
            ax[count_p].spines['top'].set_visible(False)
            ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot))):
            os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot)))
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_hist_trial_order_' + align + '_' + traces_type), dpi=mscope.my_dpi)
        # raster sorted by sl sym directional
        fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
        ax = ax.ravel()
        for count_p, p in enumerate(paws):
            [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
            if p == 'FR':
                p2 = 'FL'
            if p == 'HR':
                p2 = 'HL'
            if p == 'FL':
                p2 = 'FR'
            if p == 'HL':
                p2 = 'HR'
            [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, p, p2, sym, remove_nan)  # SL symmetry for each stride
            sl_idx_all_sorted = np.argsort(sl_sym_all_array)
            ax[count_p].scatter(events_stride_trial[sl_idx_all_sorted], cumulative_idx, s=1, color='black')
            ax[count_p].axvline(x=0, color='black')
            ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            ax[count_p].set_ylabel('Aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
            ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
            ax[count_p].spines['right'].set_visible(False)
            ax[count_p].spines['top'].set_visible(False)
            ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot))):
            os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot)))
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_raster_sl_sym_' + align + '_' + traces_type), dpi=mscope.my_dpi)
        # raster sorted by sl sym directional -
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        for count_p, p in enumerate(paws):
            [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
            if p == 'FR':
                p2 = 'FL'
            if p == 'HR':
                p2 = 'HL'
            if p == 'FL':
                p2 = 'FR'
            if p == 'HL':
                p2 = 'HR'
            [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, p, p2, sym, remove_nan)  # SL symmetry for each stride
            sl_idx_all_sorted = np.argsort(sl_sym_all_array)
            events_near_stsw_idx = np.where((events_stride_trial[sl_idx_all_sorted] > -50) & (events_stride_trial[sl_idx_all_sorted] < 50))[0]
            sl_sym_stsw = sl_sym_all_array[sl_idx_all_sorted][events_near_stsw_idx]
            sl_sym_stsw_notnan = sl_sym_stsw[~np.isnan(sl_sym_stsw)]
            [hist_result, xaxis] = np.histogram(sl_sym_stsw_notnan, bins=20)
            ax.plot(xaxis[:-1], hist_result / np.nanmax(hist_result), color=paw_colors[count_p], linewidth=2)
        ax.axvline(x=0, color='black')
        ax.set_xlabel('Step length symmetry', fontsize=mscope.fsize - 8)
        ax.set_ylabel('Number of events aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
        if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot))):
            os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot)))
        plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_summary_sl_sym_' + align + '_' + traces_type), dpi=mscope.my_dpi)
        plt.close('all')
