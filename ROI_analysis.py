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
plot_figures = 1
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

if load_data == 1:
    # Load data
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()
    df_extract_rawtrace_detrended_norm = mscope.norm_traces(df_extract_rawtrace_detrended, 'min_max', 'session')
    df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')

centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext, distance_neurons, trials_baseline, th_cluster, colormap_cluster, plot_data, print_plots)
[clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended, idx_roi_cluster, centroid_ext)
[df_events_extract_rawtrace_clustered, df_extract_rawtrace_detrended_clustered] = mscope.compute_clustered_traces_events_correlations(df_events_extract_rawtrace, df_extract_rawtrace_detrended, clusters_rois, trials_baseline)

# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'coo_stance'
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
    stride_duration_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'stride_duration'))

if plot_figures:
    traces_type = 'raw'
    # Standard plots - example traces, ROI masks, heatmap for baseline speed trials
    mscope.plot_stacked_traces(frame_time, df_extract_rawtrace_detrended_norm, traces_type, trials_ses, plot_data) # input can be one trial or trials_ses
    mscope.plot_rois_ref_image(ref_image, coord_ext, plot_data)
    mscope.plot_heatmap_baseline(df_extract_rawtrace_detrended_norm, traces_type, plot_data)

    # # Microzones plots, order correlation matrix by distance between neurons - do this for raw signals
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    th_cluster = 0.6
    colormap_cluster = 'hsv'
    trial_plot = trials_baseline[-1]
    [colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
    [clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended, idx_roi_cluster, centroid_ext)
    mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, plot_data, print_plots)
    mscope.plot_roi_clustering_temporal(df_extract_rawtrace_detrended, frame_time, centroid_ext, distance_neurons, trial_plot, colors_cluster, idx_roi_cluster_ordered, plot_data, print_plots)

    # Plot trace with events - examples and session
    roi_plot = 12
    trial_plot = 2
    mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace_detrended, traces_type, df_events_extract_rawtrace, plot_data)

    # Calcium events stats
    # ISI
    isi_events = mscope.compute_isi(df_events_extract_rawtrace, traces_type, 'isi_events_extract')
    # CV of ISI
    [isi_cv, isi_cv2] = mscope.compute_isi_cv(isi_events, trials)
    # Ratio between ISI values
    range_isiratio = [[0,0.5],[0.8,1.5]]
    isi_ratio = mscope.compute_isi_ratio(isi_events, range_isiratio, trials)

    roi_list = df_extract_rawtrace_detrended.columns[2:]
    sl_front_events_baseline = np.zeros((len(roi_list), 51))
    sl_front_events_split = np.zeros((len(roi_list), 51))
    sl_front_events_washout = np.zeros((len(roi_list), 51))
    sl_front_events_baseline_shuffle = np.zeros((len(roi_list), 51))
    sl_front_events_split_shuffle = np.zeros((len(roi_list), 51))
    sl_front_events_washout_shuffle = np.zeros((len(roi_list), 51))
    t_stat_baseline = np.zeros(len(roi_list))
    p_value_baseline = np.zeros(len(roi_list))
    t_stat_split = np.zeros(len(roi_list))
    p_value_split = np.zeros(len(roi_list))
    t_stat_washout = np.zeros(len(roi_list))
    p_value_washout = np.zeros(len(roi_list))
    roi_plot = []
    for r in range(len(roi_list)):
        roi_plot.append(np.int64(roi_list[r][3:]))
    for count_r, roi in enumerate(roi_plot):
        mscope.plot_cv_session(roi, isi_cv, traces_type, session_type, trials, 'cv', plot_data)
        mscope.plot_cv_session(roi, isi_cv2, traces_type, session_type, trials, 'cv2', plot_data)
        mscope.plot_isi_ratio_session(roi, isi_ratio, traces_type, session_type, range_isiratio, trials, plot_data)
        mscope.plot_isi_boxplots(roi, isi_events, traces_type, session_type, trials, plot_data)
        plt.close('all')
        mscope.plot_stacked_traces_singleROI(frame_time, df_extract_rawtrace_detrended, traces_type, roi, session_type, trials, colors_session, plot_data)
        mscope.plot_single_roi_ref_image(ref_image, coord_ext, roi, traces_type, roi_list, colors_cluster, idx_roi_cluster_ordered, 1)
        for trial_plot in trials:
            mscope.plot_isi_single_trial(trial_plot, roi, isi_events, traces_type, 1)
            plt.close('all')
        # Event waveform
        mscope.compute_event_waveform(df_extract_rawtrace_detrended_zscore, traces_type, df_events_extract_rawtrace, roi, animal, session_type, trials_ses, trials, plot_data)
        # Event count
        mscope.get_event_count_wholetrial(df_events_extract_rawtrace, traces_type, session_type, trials, roi, plot_data)
        event_count_loco = mscope.get_event_count_locomotion(df_events_extract_rawtrace, traces_type, session_type, trials, bcam_time, st_strides_trials, roi, plot_data)
        # Proportion of events in strides
        paw = 'FR'
        align = 'stride'
        df_events_stride_all = mscope.events_stride(df_events_extract_rawtrace, st_strides_trials, sw_strides_trials, paw, roi, align)
        event_probability = mscope.event_probability_plot(df_events_stride_all, df_events_extract_rawtrace, traces_type, session_type, paw, roi, plot_data)
        plt.close('all')
        for t in trials:
            mscope.events_align_trajectory(df_events_extract_rawtrace, traces_type, bcam_time, final_tracks_forwadloco_trials, t, trials, roi, plot_data)
            plt.close('all')
        # Bin step length asymmetry and check CS firing
        step_size = 2
        p1 = 'FR'
        p2 = 'FL'
        nr_strides = 10
        for count_t, trials_compute in enumerate(idx_plot):
            [bins, sl_p1_events_trials, sl_p1_events_trials_shuffled, t_stat, p_value] = mscope.param_events_plot(param_trials, st_strides_trials,
                                                                   df_events_extract_rawtrace, param_name, roi, p1,
                                                                   p2, step_size, trials_compute, trials,
                                                                   traces_type, cond_plot[count_t], stride_duration_trials, plot_data)
            plt.close('all')
        step_size = 2
        p1 = 'FR'
        p2 = 'FL'
        nr_strides = 10
        [bins, sl_p1_events_trials, sl_p1_events_trials_shuffled, t_stat, p_value] = mscope.param_events_plot(param_trials, st_strides_trials,
                                                                   df_events_extract_rawtrace, param_name, roi, p1,
                                                                   p2, step_size, trials, trials,
                                                                   traces_type, 'all', stride_duration_trials, plot_data)
        if count_t == 0:
            sl_front_events_baseline[count_r,:] = sl_p1_events_trials
            sl_front_events_baseline_shuffle[count_r, :] = sl_p1_events_trials_shuffled
            t_stat_baseline[count_r] = t_stat
            p_value_baseline[count_r] = p_value
        if count_t == 1:
            sl_front_events_split[count_r,:] = sl_p1_events_trials
            sl_front_events_split_shuffle[count_r, :] = sl_p1_events_trials_shuffled
            t_stat_split[count_r] = t_stat
            p_value_split[count_r] = p_value
        if count_t == 2:
            sl_front_events_washout[count_r,:] = sl_p1_events_trials
            sl_front_events_washout_shuffle[count_r, :] = sl_p1_events_trials_shuffled
            t_stat_washout[count_r] = t_stat
            p_value_washout[count_r] = p_value
        plt.close('all')



