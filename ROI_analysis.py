# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp

# path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'E:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
load_data = 1
plot_data = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
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
session = loco.get_session_id()
trials = mscope.get_trial_id()
frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
strobe_nr_txt = loco.bcam_strobe_number()
trial_start_blip_nr = loco.trial_start_blips()
frame_time = mscope.get_miniscope_frame_time(trials, frames_dFF, version_mscope)  # get frame time for each trial
ref_image = mscope.get_ref_image()
session_type = path.split(delim)[-4].split(' ')[0]  # tied or split
if session_type == 'tied':
    trials_ses = np.array([3, 4])
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
elif len(trials) < 23:
    trials_baseline = trials

if load_data == 0:
    # Load ROIs and traces - EXTRACT - NEEDS TO BE FOR THE WHOLE SESSION
    trial = 2
    thrs_spatial_weights = 0.3
    [coord_ext, df_extract] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trial)

    # Load ROIs and traces - IMAGEJ
    norm = 0
    [coord_fiji, df_fiji] = mscope.get_imagej_output(frame_time, trials, norm)

    # Good periods after motion correction
    [x_offset, y_offset, corrXY] = mscope.get_reg_data() # registration bad moments - correct
    th = 0.006 # change with the notes from EXCEL
    [idx_to_nan,df_dFF] = mscope.corr_FOV_movement(th, df_fiji, corrXY)

    # ROI curation
    # trial_curation = 3
    # [keep_rois, df_dFF_clean] = mscope.roi_curation(ref_image, df_fiji, coord_fiji, trial_curation)
    # # keep_rois = [ 0, 61, 58, 37, 48, 29, 39, 71, 24, 27, 53,  9, 38, 59, 40, 23, 46,
    # #        63,  4, 28, 14, 25,  6, 22,  3, 65, 49, 13, 18, 56, 51,  1, 17, 30,
    # #        52, 57, 50, 21, 44,  7, 16,  8, 15, 54, 60, 73, 72, 45, 70,  5, 55,
    # #        36, 31, 69, 11, 68,  2, 64, 41, 74, 42]

    # Get background subtracted trace from ImageJ segmentation
    coeff_sub = 1
    df_trace_bgsub = mscope.compute_bg_roi_fiji(coord_fiji, trials, frame_time, df_fiji, coeff_sub)

if load_data:
    # Load data
    [df_fiji, df_fiji_bgsub, df_extract, trials, coord_fiji, coord_ext, reg_th,
     reg_bad_frames] = mscope.load_processed_files()
    df_fiji_norm = mscope.minmax_norm_traces(df_fiji)
    df_fiji_bgsub_norm = mscope.minmax_norm_traces(df_fiji_bgsub)

# Standard plots - example traces and ROI masks
trial_plot = 2
mscope.plot_stacked_traces(frame_time, df_fiji_norm, trial_plot, plot_data) # input can be one trial or trials_ses
mscope.plot_rois_ref_image(ref_image, coord_fiji, plot_data)

# Microzones plots, order correlation matrix by distance between neurons - do this for raw signals
centroid_fiji = mscope.get_roi_centroids(coord_fiji)
distance_neurons = mscope.distance_neurons(centroid_fiji, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_fiji_norm, centroid_fiji, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster, coord_fiji, plot_data)
mscope.plot_roi_clustering_temporal(df_fiji_norm, frame_time, centroid_fiji, distance_neurons, trial_plot, colors_cluster, idx_roi_cluster, plot_data)

# Find calcium events - label as synchronous or asynchronous
timeT = 7
thrs_amp = 10
df_events_all = mscope.get_events(coord_fiji, df_fiji, timeT, thrs_amp)
thrs_amp_bgsub = 4
df_events_unsync = mscope.get_events(coord_fiji, df_fiji_bgsub, timeT, thrs_amp_bgsub)

df_fiji_trial_norm = df_fiji_norm.loc[df_fiji_norm['trial'] == trial_plot]  # get dFF for the desired trial
df_fiji_bgsub_trial_norm = df_fiji_bgsub_norm.loc[df_fiji_bgsub_norm['trial'] == trial_plot]  # get dFF for the desired trial
fig, ax = plt.subplots(1, 2, figsize=(20, 20), tight_layout=True)
ax = ax.ravel()
for r in range(df_fiji_trial_norm.shape[1] - 2):
    ax[0].plot(frame_time[trial_plot - 1], df_fiji_trial_norm['ROI' + str(r + 1)] + (r/2), color='black')
    events_plot = np.where(df_events_all.loc[df_fiji_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
    for e in events_plot:
        ax[0].scatter(frame_time[trial_plot - 1][e], df_fiji_trial_norm.iloc[e, r+2] + (r/2), s=20, color='gray')
    events_unsync_plot = np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
    for e in events_unsync_plot:
        ax[0].scatter(frame_time[trial_plot - 1][e], df_fiji_bgsub_trial_norm.iloc[e, r+2] + (r/2), s=20, color='orange')
ax[0].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax[0].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax[0].get_yticklabels(), visible=False)
ax[0].tick_params(axis='y', which='y', length=0)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
for r in range(df_fiji_bgsub_trial_norm.shape[1] - 2):
    ax[1].plot(frame_time[trial_plot - 1], df_fiji_bgsub_trial_norm['ROI' + str(r + 1)] + (r/2), color='black')
    events_unsync_plot = np.where(df_events_unsync.loc[df_fiji_bgsub_norm['trial'] == trial_plot, 'ROI' + str(r + 1)])[0]
    for e in events_unsync_plot:
        ax[1].scatter(frame_time[trial_plot - 1][e], df_fiji_bgsub_trial_norm.iloc[e, r+2] + (r/2), s=20, color='gray')
ax[1].set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax[1].set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax[1].get_yticklabels(), visible=False)
ax[1].tick_params(axis='y', which='y', length=0)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)

# Calcium events stats

# Align events with stance/swing periods

# # Save data
# mscope.save_processed_files(df_fiji, df_trace_bgsub, df_extract, trials, coord_fiji, coord_ext, th, idx_to_nan)
