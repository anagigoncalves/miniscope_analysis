# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import pandas as pd

# path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'E:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
session_type = 'tied'
delim = path[-1]
version_mscope = 'v4'
load_data = 0
plot_data = 0
paw_colors = ['red', 'magenta', 'blue', 'cyan']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
# import locomotion_class
# loco = locomotion_class.loco_class(path_loco)

# create plots folders
if delim == '/':
    path_images = path + '/images/'
    path_cluster = path + '/images/cluster/'
    path_stats = path + '/images/stats/'
    path_events = path + '/images/events/'
else:
    path_images = path + '\\images\\'
    path_cluster = path + '\\images\\cluster\\'
    path_stats = path + '\\images\\stats\\'
    path_events = path + '\\images\\events\\'
if not os.path.exists(path_images):
    os.mkdir(path_images)
if not os.path.exists(path_cluster):
    os.mkdir(path_cluster)
if not os.path.exists(path_stats):
    os.mkdir(path_stats)
if not os.path.exists(path_events):
    os.mkdir(path_events)

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
# session = loco.get_session_id()
trials = mscope.get_trial_id()
frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
# [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
# strobe_nr_txt = loco.bcam_strobe_number()
# trial_start_blip_nr = loco.trial_start_blips()
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

if load_data == 0:
    # Load ROIs and traces - EXTRACT - NEEDS TO BE FOR THE WHOLE SESSION
    trial = 2
    thrs_spatial_weights = 0
    [coord_ext, df_extract_allframes] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trial)

    # # Load ROIs and traces - IMAGEJ
    # norm = 0
    # [coord_fiji, df_fiji_allframes] = mscope.get_imagej_output(frame_time, trials, norm)

    # Good periods after motion correction
    [x_offset, y_offset, corrXY] = mscope.get_reg_data() # registration bad moments - correct
    th = 0.006 # change with the notes from EXCEL
    # [idx_to_nan,df_fiji] = mscope.corr_FOV_movement(th, df_fiji_allframes, corrXY)
    [idx_to_nan,df_extract] = mscope.corr_FOV_movement(th, df_extract_allframes, corrXY)
    [coord_ext, df_extract] = mscope.rois_larger_motion(df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi_ext, height_roi_ext, 1)
    mscope.correlation_signal_motion(df_extract, x_offset, y_offset, trial, idx_to_nan, 1)

    # ROI curation
    trial_curation = 2
    # ROI spatial stats
    [width_roi_ext, height_roi_ext, aspect_ratio_ext] = mscope.get_roi_stats(coord_ext)
    # [keep_rois, df_dFF_clean] = mscope.roi_curation(ref_image, df_fiji, coord_fiji, trial_curation)
    # # keep_rois = [ 0, 61, 58, 37, 48, 29, 39, 71, 24, 27, 53,  9, 38, 59, 40, 23, 46,
    # #        63,  4, 28, 14, 25,  6, 22,  3, 65, 49, 13, 18, 56, 51,  1, 17, 30,
    # #        52, 57, 50, 21, 44,  7, 16,  8, 15, 54, 60, 73, 72, 45, 70,  5, 55,
    # #        36, 31, 69, 11, 68,  2, 64, 41, 74, 42]

    # Get background subtracted trace from ImageJ segmentation
    coeff_sub = 1
    df_trace_bgsub = mscope.compute_bg_roi_fiji(coord_fiji, trials, frame_time, df_fiji, coeff_sub)

    # Find calcium events - label as synchronous or asynchronous
    df_dFF_fiji = mscope.compute_dFF(df_fiji)
    df_dFF_fiji_bgsub = mscope.compute_dFF(df_fiji_bgsub)
    df_dFF_fiji_norm = mscope.norm_traces(df_dFF_fiji, 'zscore')
    df_dFF_fiji_bgsub_norm = mscope.norm_traces(df_dFF_fiji_bgsub, 'zscore')
    timeT = 7
    df_events_all = mscope.get_events(df_dFF_fiji_norm, timeT, 'df_events_all')
    df_events_unsync = mscope.get_events(df_dFF_fiji_bgsub_norm, timeT, 'df_events_unsync')

if load_data:
    # Load data
    [df_fiji, df_fiji_bgsub, df_extract, trials, coord_fiji, coord_ext, reg_th,
     reg_bad_frames] = mscope.load_processed_files()
    df_events_all = pd.read_csv(mscope.path+'\\processed files\\'+'df_events_all.csv')
    df_events_unsync = pd.read_csv(mscope.path+'\\processed files\\'+'df_events_unsync.csv')

# Load behavioral data
# filelist = loco.get_track_files(animal, session)
# st_strides_trials = []
# sw_strides_trials = []
# count_trial = 0
# for f in filelist:
#     [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
#     [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
#     st_strides_trials.append(st_strides_mat)
#     sw_strides_trials.append(sw_pts_mat)
#     count_trial += 1

# Standard plots - example traces and ROI masks
trial_plot = 2
df_dFF_fiji = mscope.compute_dFF(df_fiji)
df_dFF_fiji_bgsub = mscope.compute_dFF(df_fiji_bgsub)
df_dFF_fiji_norm = mscope.norm_traces(df_dFF_fiji, 'min_max')
df_dFF_fiji_bgsub_norm = mscope.norm_traces(df_dFF_fiji_bgsub, 'min_max')
mscope.plot_stacked_traces(frame_time, df_dFF_fiji_norm, trial_plot, plot_data) # input can be one trial or trials_ses
mscope.plot_rois_ref_image(ref_image, coord_fiji, plot_data)

# Microzones plots, order correlation matrix by distance between neurons - do this for raw signals
centroid_fiji = mscope.get_roi_centroids(coord_fiji)
distance_neurons = mscope.distance_neurons(centroid_fiji, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_dFF_fiji, centroid_fiji, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster, coord_fiji, plot_data)
mscope.plot_roi_clustering_temporal(df_dFF_fiji, frame_time, centroid_fiji, distance_neurons, trial_plot, colors_cluster, idx_roi_cluster, plot_data)

# Plot trace with events - examples and session
roi_plot = 25
mscope.plot_events_roi_examples(trial_plot, roi_plot, frame_time, df_dFF_fiji_norm, df_dFF_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data)
# mscope.plot_events_roi_trial(trial_plot, frame_time, df_fiji_norm, df_fiji_bgsub_norm, df_events_all, df_events_unsync, plot_data)

# Calcium events stats
# ISI
isi_all_events = mscope.compute_isi(df_events_all, 'isi_all_events')
isi_unsync_events = mscope.compute_isi(df_events_unsync, 'isi_unsync_events')
mscope.plot_isi_single_trial(trial_plot, roi_plot, isi_all_events, plot_data)
mscope.plot_isi_session(roi_plot, isi_all_events, animal, session_type, trials, trials_ses, plot_data)
# CV of ISI
[isi_cv_all, isi_cv2_all] = mscope.compute_isi_cv(isi_all_events)
[isi_cv_unsync, isi_cv2_unsync] = mscope.compute_isi_cv(isi_unsync_events)
mscope.plot_cv_session(roi_plot, isi_cv_all, trials, plot_data)
# Ratio between ISI values
range_isiratio = [[0,0.5],[0.8,1.5]]
isi_ratio_all = mscope.compute_isi_ratio(isi_all_events, range_isiratio)
isi_ratio_unsync = mscope.compute_isi_ratio(isi_unsync_events, range_isiratio)
mscope.plot_isi_ratio_session(roi_plot, isi_ratio_all, range_isiratio, trials, plot_data)

# Event waveform
mscope.compute_event_waveform(df_fiji_norm, df_events_all, roi_plot, animal, session_type, trials_ses, trials, plot_data)

# Event count
mscope.get_event_count_wholetrial(df_events_all, trials, roi_plot, plot_data)
mscope.get_event_count_locomotion(df_events_all, trials, bcam_time, st_strides_trials, roi_plot, plot_data)

# Proportion of events in strides
paw = 'FR'
align = 'stride'
df_events_stride_all = mscope.events_stride(df_events_all, st_strides_trials, sw_strides_trials, paw, roi_plot, align)
mscope.event_proportion_plot(df_events_stride_all, paw, roi_plot, plot_data)

# Align events with stance/swing periods
align = 'stance'
time_window = 0.2
bin_size = 20
paw = 'FR'
event_stance_FR = mscope.events_align_st_sw(df_events_all, st_strides_trials, sw_strides_trials, time_window, bin_size, paw, roi_plot, align, session_type, trials_ses, plot_data)

# Save data
# mscope.save_processed_files(df_fiji, df_trace_bgsub, df_extract, trials, coord_fiji, coord_ext, th, idx_to_nan)
# df_events_all.to_csv(mscope.path + '\\processed files\\' + 'df_events_all.csv', sep=',', index=False)
# df_events_unsync.to_csv(mscope.path + '\\processed files\\' + 'df_events_unsync.csv', sep=',', index=False)