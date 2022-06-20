# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import pandas as pd
import time
import seaborn as sns

# path inputs
path = 'I:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'I:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
# path = 'I:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\'
# path_loco = 'I:\\TM TRACKING FILES\\tied baseline\\tied baseline S1 040421\\'
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
load_data = 1
improve_eventdet = 0
plot_figures = 0
plot_data = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR','HR','FL','HL']
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
if session_type == 'tied' and animal != 'MC8855':
    trials_ses = np.array([6, 12, 18])
    trials_ses_name = ['baseline speed', 'fast speed']
if session_type == 'split':
    trials_ses = np.array([3, 4, 13, 14])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
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

# TODO
#  do analysis for raw and deconvoluted traces, do sw and st for raw signal, check correlation between all ROIs for each frame and see how close highly correlated frames are from stance/sw

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
    [coord_ext, df_extract] = mscope.rois_larger_motion(df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi_ext, height_roi_ext, 1) #needs to remove ROIs from df trace
    trial_corr = 2
    mscope.correlation_signal_motion(df_extract, x_offset, y_offset, trial_corr, idx_to_nan, 1)

    # ROI curation
    trial_curation = 2
    # ROI spatial stats
    [width_roi_rois_nomotion, height_roi_rois_nomotion, aspect_ratio_rois_nomotion] = mscope.get_roi_stats(coord_ext)
    [coord_ext_curated, df_extract_curated] = mscope.roi_curation(ref_image, df_extract, coord_ext, aspect_ratio_rois_nomotion, trial_curation)

    # # keep_rois = [ 0, 61, 58, 37, 48, 29, 39, 71, 24, 27, 53,  9, 38, 59, 40, 23, 46,
    # #        63,  4, 28, 14, 25,  6, 22,  3, 65, 49, 13, 18, 56, 51,  1, 17, 30,
    # #        52, 57, 50, 21, 44,  7, 16,  8, 15, 54, 60, 73, 72, 45, 70,  5, 55,
    # #        36, 31, 69, 11, 68,  2, 64, 41, 74, 42]

    # # Get background subtracted trace from ImageJ segmentation
    # coeff_sub = 1
    # ## ITS GIVING SOME ROIs AS ALL NULL VALUES
    # df_trace_bgsub = mscope.compute_bg_roi_fiji(coord_fiji, trials, frame_time, df_fiji_allframes, coeff_sub)

    # Find calcium events - label as synchronous or asynchronous
    timeT = 7
    amp_vec = []
    [df_events_extract, amp_arr] = mscope.get_events(df_extract_curated, timeT, amp_vec, 'df_events_extract')
    roi_plot = 3
    trial_plot = 3
    mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_curated, df_events_extract, 0)
    # df_dFF_fiji = mscope.compute_dFF(df_fiji_allframes)
    # df_dFF_fiji_norm = mscope.norm_traces(df_dFF_fiji, 'zscore', 'trial')
    # df_fiji_bgsub_norm = mscope.norm_traces(df_trace_bgsub, 'zscore', 'trial')
    # df_events_all = mscope.get_events(df_dFF_fiji_norm, timeT, 'df_events_all')
    # df_events_unsync = mscope.get_events(df_fiji_bgsub_norm, timeT, 'df_events_unsync')
    # # Save data
    roi_list = list(df_extract_curated.columns[2:])
    df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext, roi_list, trials, frame_time)
    mscope.save_processed_files(df_extract_curated, trials, df_events_extract,  df_extract_rawtrace, coord_ext_curated, th, amp_arr, idx_to_nan)
    # # mscope.save_processed_files_ext_fiji(df_fiji, df_trace_bgsub, df_extract, df_events_all, df_events_unsync, trials, coord_fiji, coord_ext, th, idx_to_nan)

    # #delete after
    # df_extract = pd.read_csv(mscope.path + '\\processed files\\' + 'df_extract.csv')
    # df_events_extract = pd.read_csv(mscope.path + '\\processed files\\' + 'df_events_extract.csv')
    # coord_ext = np.load(mscope.path + '\\processed files\\' + 'coord_ext.npy', allow_pickle=True)
    # trials = np.load(mscope.path + '\\processed files\\' + 'trials.npy')
    # reg_th = np.load(mscope.path + '\\processed files\\' + 'reg_th.npy')
    # amp_arr = np.load(mscope.path + '\\processed files\\' + 'amplitude_events.npy')
    # reg_bad_frames = np.load(mscope.path + '\\processed files\\' + 'frames_to_exclude.npy')
    # roi_list = list(df_extract.columns[2:])
    # df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext, roi_list, trials, frame_time)
    # df_extract_rawtrace.to_csv(mscope.path + '\\processed files\\' + 'df_extract_raw.csv', sep=',', index=False)


if load_data == 1:
    # Load data
    [df_extract, df_events_extract, df_extract_rawtrace, trials, coord_ext, reg_th, amp_arr, reg_bad_frames] = mscope.load_processed_files()
    # df_dFF_fiji = mscope.compute_dFF(df_fiji)
    # df_dFF_fiji_norm = mscope.norm_traces(df_dFF_fiji, 'zscore', 'trial')
    # df_fiji_bgsub_norm = mscope.norm_traces(df_fiji_bgsub, 'zscore', 'trial')
    if improve_eventdet == 1:
        timeT = 7
        roi_list = df_extract.columns[2:]
        amp_vec = np.zeros(len(roi_list))
        count_r = 0
        for r in roi_list:
            fig, ax = plt.subplots(figsize=(30,7),tight_layout=True)
            ax.plot(df_extract[r][:10000], color='black', linewidth=0.5)
            ax.set_title(r, fontsize=mscope.fsize)
            ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
            ax.set_ylabel('Amplitude of F values', fontsize=mscope.fsize - 4)
            ax.tick_params(axis='x', labelsize=mscope.fsize - 4)
            ax.tick_params(axis='y', labelsize=mscope.fsize - 4)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            coord = plt.ginput(n=1, timeout=0)
            amp_vec[count_r] = coord[0][1]
            count_r += 1
            plt.close('all')
        [df_events_extract, amp_arr] = mscope.get_events(df_extract, timeT, amp_vec, 'df_events_extract')
        roi_plot = 3
        trial_plot = 3
        mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract, df_events_extract, 0)
        mscope.save_processed_files(df_extract, trials, df_events_extract, df_extract_rawtrace, coord_ext, reg_th, amp_arr, reg_bad_frames)

# Load behavioral data
filelist = loco.get_track_files(animal, session)
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
count_trial = 0
for f in filelist:
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    count_trial += 1

if plot_figures:
    # # Standard plots - example traces, ROI masks, heatmap for baseline speed trials
    df_extract_norm = mscope.norm_traces(df_extract, 'min_max', 'session')
    mscope.plot_stacked_traces(frame_time, df_extract_norm, trials_ses, plot_data) # input can be one trial or trials_ses
    mscope.plot_rois_ref_image(ref_image, coord_ext, plot_data)
    mscope.plot_heatmap_baseline(df_extract_norm, plot_data)
    # df_dFF_fiji = mscope.compute_dFF(df_fiji)
    # df_dFF_fiji_bgsub = mscope.compute_dFF(df_fiji_bgsub)
    # df_dFF_fiji_norm = mscope.norm_traces(df_dFF_fiji, 'min_max', 'session')
    # df_dFF_fiji_bgsub_norm = mscope.norm_traces(df_dFF_fiji_bgsub, 'min_max', 'session')

    # # Microzones plots, order correlation matrix by distance between neurons - do this for raw signals
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    th_cluster = 0.6
    colormap_cluster = 'hsv'
    trial_plot = 2
    [colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract, centroid_ext, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
    mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster, coord_ext, plot_data)
    mscope.plot_roi_clustering_temporal(df_extract, frame_time, centroid_ext, distance_neurons, trial_plot, colors_cluster, idx_roi_cluster, plot_data)
    # centroid_fiji = mscope.get_roi_centroids(coord_fiji)
    # distance_neurons = mscope.distance_neurons(centroid_fiji, 0)
    # th_cluster = 0.6
    # colormap_cluster = 'hsv'
    # [colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_dFF_fiji, centroid_fiji, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
    # mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster, coord_fiji, plot_data)
    # mscope.plot_roi_clustering_temporal(df_dFF_fiji, frame_time, centroid_fiji, distance_neurons, trial_plot, colors_cluster, idx_roi_cluster, plot_data)

    # # Plot EXTRACT ROIs with manual ROIs
    # [coord_fiji, df_fiji] = mscope.get_imagej_output(frame_time,trials,0)
    # plt.figure(figsize=(10, 10), tight_layout=True)
    # for r in range(len(coord_ext)):
    #     plt.scatter(coord_ext[r][:, 0], coord_ext[r][:, 1], color='blue', s=1, alpha=0.6)
    # for r in range(len(coord_fiji)):
    #     plt.scatter(coord_fiji[r][:, 0], coord_fiji[r][:, 1], color='yellow', s=1, alpha=0.6)
    # plt.imshow(ref_image, cmap='gray',
    #            extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um, 0])
    # plt.title('ROIs grouped by activity', fontsize=mscope.fsize)
    # plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
    # plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
    # plt.xticks(fontsize=mscope.fsize - 4)
    # plt.yticks(fontsize=mscope.fsize - 4)

    # # Plot trace with events - examples and session
    roi_plot = 8
    trial_plot = 2
    mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract, df_events_extract, plot_data)
    # mscope.plot_events_roi_examples_bgsub(trial_plot, roi_plot, frame_time, df_dFF_fiji_norm, df_dFF_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data)
    # mscope.plot_events_roi_trial_bgsub(trial_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data)

    # Calcium events stats
    # ISI
    isi_events = mscope.compute_isi(df_events_extract, 'isi_events_extract')
    # isi_all_events = mscope.compute_isi(df_events_all, 'isi_all_events')
    # isi_unsync_events = mscope.compute_isi(df_events_unsync, 'isi_unsync_events')
    # CV of ISI
    [isi_cv, isi_cv2] = mscope.compute_isi_cv(isi_events, trials)
    # [isi_cv_unsync, isi_cv2_unsync] = mscope.compute_isi_cv(isi_unsync_events)
    # Ratio between ISI values
    range_isiratio = [[0,0.5],[0.8,1.5]]
    isi_ratio = mscope.compute_isi_ratio(isi_events, range_isiratio, trials)
    # isi_ratio_unsync = mscope.compute_isi_ratio(isi_unsync_events, range_isiratio)

    roi_list = df_extract.columns[2:]
    roi_plot = []
    for r in range(len(roi_list)):
        roi_plot.append(np.int64(roi_list[r][3:]))
    event_proportion_all = np.zeros((len(roi_plot),len(trials)))
    count_r = 0
    for roi in roi_plot:
        mscope.plot_stacked_traces_singleROI(frame_time, df_extract_norm, roi, session_type, trials, plot_data)
        for trial_plot in trials:
            mscope.plot_isi_single_trial(trial_plot, roi, isi_events, plot_data)
        mscope.plot_isi_session(roi, isi_events, animal, session_type, trials, trials_ses, plot_data)
        mscope.plot_cv_session(roi, isi_cv, trials, 'cv', plot_data)
        mscope.plot_cv_session(roi, isi_cv2, trials, 'cv2', plot_data)
        mscope.plot_isi_ratio_session(roi, isi_ratio, range_isiratio, trials, plot_data)
        plt.close('all')
        # Event waveform
        mscope.compute_event_waveform(df_extract_norm, df_events_extract, roi, animal, session_type, trials_ses, trials, plot_data)
        # Event count
        mscope.get_event_count_wholetrial(df_events_extract, trials, roi, plot_data)
        mscope.get_event_count_locomotion(df_events_extract, trials, bcam_time, st_strides_trials, roi, plot_data)
        # Proportion of events in strides
        paw = 'FR'
        align = 'stride'
        df_events_stride_all = mscope.events_stride(df_events_extract, st_strides_trials, sw_strides_trials, paw, roi, align)
        event_proportion = mscope.event_proportion_plot(df_events_stride_all, paw, roi, plot_data)
        event_proportion_all[count_r,:] = event_proportion
        np.save(mscope.path + '\\processed files\\' + 'event_proportion_allrois.npy', event_proportion_all)
        plt.close('all')
        # Align events with stance/swing periods
        time_window = 0.2
        bin_size = 20
        align = 'stance'
        for paw in paws:
            event_stance_paws = mscope.events_align_st_sw(df_events_extract, st_strides_trials, sw_strides_trials, time_window, bin_size, paw, roi, align, session_type, trials_ses, plot_data)
        align = 'swing'
        for paw in paws:
            event_swing_paws = mscope.events_align_st_sw(df_events_extract, st_strides_trials, sw_strides_trials, time_window, bin_size, paw, roi, align, session_type, trials_ses, plot_data)
        plt.close('all')
        for t in trials:
            mscope.events_align_trajectory(df_events_extract, bcam_time, final_tracks_trials, t, roi, plot_data)
        count_r += 1
        plt.close('all')

df_extract_rawtrace_norm = mscope.norm_traces(df_extract_rawtrace, 'min_max', 'trial')
time_window = 1
paw = 'FR'
align = 'stance'
roi_list = mscope.get_roi_list(df_extract_rawtrace_norm)
for roi in roi_list:
    # mscope.plot_stacked_traces_singleROI(frame_time, df_extract_rawtrace_norm, np.int64(roi[3:]), session_type, trials, plot_data)
    raw_signal_st = mscope.raw_signal_align_st_sw(df_extract_rawtrace, st_strides_trials, sw_strides_trials, time_window, paw, np.int64(roi[3:]), align, session_type, trials_ses, plot_data)
    plt.close('all')














