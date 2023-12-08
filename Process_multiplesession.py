#THIS SCRIPT PROCESS ONE SESSION OF TWO SESSIONS THAT WERE REGISTERED TOGETHER IN SUITE2P
# %% Inputs
import os
import numpy as np

# path inputs
path = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast 405 processed with 480\\MC13420\\2022_05_31\\'
path_loco = 'D:\\Miniscopes\\TM TRACKING FILES\\split contra fast S1 310522\\'
session_process = 2 #480 is the second session
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

if session_process == 1:
    start = 0
if session_process == 2:
    start = 1

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()
trials = mscope.get_trial_id()
trials = trials[start::2]
strobe_nr_txt = loco.bcam_strobe_number()
trial_start_blip_nr = loco.trial_start_blips()
ops_s2p = mscope.get_s2p_parameters()
print(ops_s2p)
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
if session_type == 'split':
    colors_phases = ['black', 'crimson', 'teal']
if session_type == 'tied':
    colors_phases = ['black', 'orange', 'purple']
traces_type = 'raw'

ref_image = mscope.get_ref_image()
frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
frames_dFF = frames_dFF[start::2]
frame_time = mscope.get_miniscope_frame_time(np.arange(start+1, len(trials)*2+1, 2), frames_dFF, version_mscope)  # get frame time for each trial
trial_length_cumsum = mscope.cumulative_trial_length(frame_time)
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)

# Load ROIs and traces - EXTRACT
thrs_spatial_weights = 0
[coord_ext, df_extract_allframes] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trials)

# Good periods after motion correction
th = 0 # change with the notes from EXCEL
[x_offset, y_offset, corrXY] = mscope.get_reg_data()  # registration bad moments
[idx_to_nan, df_extract] = mscope.corr_FOV_movement(th, df_extract_allframes, corrXY)
[width_roi_ext, height_roi_ext, aspect_ratio_ext] = mscope.get_roi_stats(coord_ext)
#DONT DO THESE STEPS FOR 480 AND 405 TOGETHER BECAUSE THERES A LOT OF JITTER BETWEEN IMAGES
# [coord_ext, df_extract] = mscope.rois_larger_motion(df_extract, coord_ext, idx_to_nan, x_offset, y_offset, width_roi_ext, height_roi_ext, 1)
# corr_rois_motion = mscope.correlation_signal_motion(df_extract, x_offset, y_offset, trials_baseline[-1], idx_to_nan, traces_type, plot_data, print_plots)

# # ROI spatial stats
# [width_roi_rois_nomotion, height_roi_rois_nomotion, aspect_ratio_rois_nomotion] = mscope.get_roi_stats(coord_ext)
# ROI curation
[coord_ext_curated, df_extract_curated] = mscope.roi_curation(ref_image, df_extract, coord_ext, aspect_ratio_ext, trials_baseline[-1])
# [coord_ext_curated, df_extract_curated] = mscope.roi_curation(ref_image, df_extract, coord_ext, aspect_ratio_rois_nomotion, trials_baseline[-1])

# Get raw trace from EXTRACT ROIs
roi_list = list(df_extract_curated.columns[2:])
trials = np.arange(1, 27)
df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext_curated, df_extract_curated, roi_list, trials, frame_time)
# Find calcium events - label as synchronous or asynchronous
df_events_extract = mscope.get_events(df_extract_curated, 0, 'df_events_extract') # 0 for no detrending
df_events_extract_rawtrace = mscope.get_events(df_extract_rawtrace, 1, 'df_events_extract_rawtrace')  # 1 for detrending"
roi_plot = np.int64(np.random.choice(roi_list)[3:])
trial_plot = np.random.choice(trials)
mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace, traces_type, df_events_extract_rawtrace, trials, plot_data, print_plots)

# Detrend calcium trace
df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')

# Save data
mscope.save_processed_files(df_extract_curated, trials, df_events_extract,  df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext_curated, th, idx_to_nan)


