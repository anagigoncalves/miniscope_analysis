# %% Inputs
import os
import numpy as np

# path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'E:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
plot_data = 0
print_plots = 0
paw_colors = ['red', 'magenta', 'blue', 'cyan']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\Miniscope pipeline\\')
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

# Load ROIs and traces - EXTRACT - NEEDS TO BE FOR THE WHOLE SESSION
trial = 2
thrs_spatial_weights = 0.3
[coord_cell, df_extract] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trial)
centroid_cell = mscope.get_roi_centroids(coord_cell)

# Load ROIs and traces - IMAGEJ
[coord_fiji, df_fiji] = mscope.get_imagej_output(frame_time, trials)
centroid_fiji = mscope.get_roi_centroids(coord_fiji)

# Good periods after motion correction
[x_offset, y_offset, corrXY] = mscope.get_reg_data() # registration bad moments - correct
th = 0.006 # change with the notes from EXCEL
[idx_to_nan,df_dFF] = mscope.corr_FOV_movement(th, df_fiji, corrXY)

# ROI curation
df_dFF = df_fiji
coord_cell = coord_fiji
centroid_cell = centroid_fiji
# trial_curation = 3
# [keep_rois, df_dFF_clean] = mscope.roi_curation(ref_image, df_fiji, coord_fiji, trial_curation)
# # keep_rois = [ 0, 61, 58, 37, 48, 29, 39, 71, 24, 27, 53,  9, 38, 59, 40, 23, 46,
# #        63,  4, 28, 14, 25,  6, 22,  3, 65, 49, 13, 18, 56, 51,  1, 17, 30,
# #        52, 57, 50, 21, 44,  7, 16,  8, 15, 54, 60, 73, 72, 45, 70,  5, 55,
# #        36, 31, 69, 11, 68,  2, 64, 41, 74, 42]
mscope.plot_stacked_traces(frame_time, df_dFF, 2, 1) #input can be one trial or trials_ses
mscope.plot_rois_ref_image(ref_image, coord_cell, 1)

# Microzones plots, order correlation matrix by distance between neurons - do this for raw signals
distance_neurons = mscope.distance_neurons(centroid_cell, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_dFF, centroid_cell, distance_neurons, 2, th_cluster, colormap_cluster, 1)
mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster, coord_cell, 1)
mscope.plot_roi_clustering_temporal(df_dFF, frame_time, centroid_cell, distance_neurons, 2, colors_cluster, idx_roi_cluster, 1)

# Find calcium events - label as synchronous or asynchronous
timeT = 10
thrs_amp = 50
df_events_all = mscope.get_events(coord_cell, df_dFF, timeT, thrs_amp)
coeff_sub = 1
df_events_bgsub = mscope.compute_bg_roi_fiji(coord_cell, df_dFF, coeff_sub)


# Calcium events stats

# Align events with stance/swing periods

# Save data