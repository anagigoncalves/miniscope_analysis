#THIS SCRIPT TAKES ROIS FROM CORRESPONDING 480 SESSION AND PLOTS ACTIVITY ON THE 405 SESSION
# %% Inputs
import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

# path inputs
path = 'E:\\Miniscopes\\TM RAW FILES\\split contra fast 405 processed with 480\\MC13419\\2022_05_31\\'
path_loco = 'E:\\Miniscopes\\TM TRACKING FILES\\split contra fast 405nm S2 310522\\'
path_480 = 'E:\\Miniscopes\\TM RAW FILES\\split contra fast 480 processed with 405\\MC13419\\2022_05_31\\'
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
path_events = os.path.join(path, 'images', 'events')
if not os.path.exists(path_images):
    os.mkdir(path_images)
if not os.path.exists(path_events):
    os.mkdir(path_events)

start = 0
# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()
trials = mscope.get_trial_id()
trials = np.arange(1,27)
strobe_nr_txt = loco.bcam_strobe_number()
trial_start_blip_nr = loco.trial_start_blips()
ops_s2p = mscope.get_s2p_parameters()
print(ops_s2p)
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
traces_type = 'raw'

ref_image = mscope.get_ref_image()
frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
frames_dFF = frames_dFF[start::2]
frame_time = mscope.get_miniscope_frame_time(np.arange(start+1, len(trials)*2+1, 2), frames_dFF, version_mscope)  # get frame time for each trial
# trial_length_cumsum = mscope.cumulative_trial_length(frame_time)
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
# [trials, trial_start, strobe_nr, bcam_time, colors_session, frame_time, frames_dFF, frames_loco,
#  del_trials_index] = mscope.correct_for_deleted_trials(trials, trial_start, strobe_nr, bcam_time, colors_session,
#                                                        frame_time, frames_dFF, frames_loco)
# Load ROIs from corresponding 480 session
coord_ext_480 = np.load(os.path.join(path_480, 'processed files', 'coord_ext.npy'), allow_pickle=True)
df_extract_rawtrace_detrended_480 = pd.read_csv(
    os.path.join(path_480, 'processed files', 'df_extract_rawtrace_detrended.csv'))

#Compute traces from ROIs
roi_list = list(df_extract_rawtrace_detrended_480.columns[2:])
ext_trace_trials = []
for t in trials:
    idx_trial = np.where(trials == t)[0][0]
    tiff_stack = tiff.imread(
        os.path.join(mscope.path, 'Registered video') + mscope.delim + 'T' + str(t) + '_reg.tif')  # read tiffs
    ext_trace = np.zeros((np.shape(tiff_stack)[0], len(roi_list)))
    for c in range(len(coord_ext_480)):
        ext_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
        for f in range(np.shape(tiff_stack)[0]):
            ext_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                np.round(coord_ext_480[c][:, 1] * mscope.pixel_to_um)), np.int64(
                np.round(coord_ext_480[c][:, 0] * mscope.pixel_to_um))]) / np.shape(coord_ext_480[c])[0]
        ext_trace[:, c] = ext_trace_tiffmean
    ext_trace_trials.append(ext_trace)
ext_trace_arr = np.transpose(np.vstack(ext_trace_trials))  # trace as dataframe
trial_ext = []
frame_time_ext = []
for t in trials:
    idx_trial = np.where(trials == t)[0][0]
    trial_ext.extend(np.repeat(t, len(frame_time[idx_trial])))
    frame_time_ext.extend(frame_time[idx_trial])
dict_ext = {'trial': trial_ext, 'time': frame_time_ext}
df_ext1 = pd.DataFrame(dict_ext)
df_ext2 = pd.DataFrame(np.transpose(ext_trace_arr), columns=roi_list)
df_traces = pd.concat([df_ext1, df_ext2], axis=1)

# Get raw trace from EXTRACT ROIs
roi_list = list(df_traces.columns[2:])
df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext_480, df_traces, roi_list, trials, frame_time)

# #load 405 traces
# df_extract_rawtrace = pd.read_csv(
#     os.path.join(path, 'processed files', 'df_extract_raw.csv'))

# Find calcium events from 480 (to get amplitude threshold) and use that TrueStd in 405 calcium event detection
roi_trace = np.array(df_extract_rawtrace.iloc[:, 2:])
roi_list = list(df_extract_rawtrace.columns[2:])
trial_ext = list(df_extract_rawtrace['trial'])
trials = np.unique(trial_ext)
frame_time_ext = list(df_extract_rawtrace['time'])
data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
df_dFF1 = pd.DataFrame(data_dFF1)
df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
df_events_extract_rawtrace = pd.concat([df_dFF1, df_dFF2], axis=1)
roi_list = df_extract_rawtrace.columns[2:]
detrend_bool = 1
csv_name = 'df_events_extract_rawtrace'
for count_r, r in enumerate(roi_list):
    print('Processing events of ' + r)
    for count_t, t in enumerate(trials):
        data_480 = np.array(df_extract_rawtrace_detrended_480.loc[df_extract_rawtrace_detrended_480['trial'] == t, r])
        data = np.array(df_extract_rawtrace.loc[df_extract_rawtrace['trial'] == t, r])
        events_mat = np.zeros(len(data))
        if len(np.where(np.isnan(data_480))[0]) != len(data_480):
            # Get amplitude threshold for the same ROI and trial for 480 data
            [Ev_Onset_480, IncremSet_480, TrueStd_480] = mscope.compute_events_onset(data_480, mscope.sr, detrend_bool)
        if len(np.where(np.isnan(data))[0]) != len(data):
            [Ev_Onset, IncremSet] = mscope.compute_events_onset_405(data, mscope.sr, TrueStd_480, detrend_bool)
            if len(Ev_Onset) > 0:
                events = mscope.event_detection_calcium_trace(data, Ev_Onset, IncremSet, 3)
                if detrend_bool == 0:
                    events_new = []
                    for e in events:
                        if data[e] >= np.nanpercentile(data, 75):
                            events_new.append(e)
                    events = events_new
                events_mat[events] = 1
            else:
                print('No events for ' + r + ' trial ' + str(t))
            df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, r] = events_mat
        else:
            print('No events for ' + r + ' trial ' + str(t))
            df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, r] = events_mat
        count_t += 1
    count_r += 1
if len(csv_name) > 0:
    if not os.path.exists(os.path.join(mscope.path, 'processed files')):
        os.mkdir(os.path.join(mscope.path, 'processed files'))
    df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', csv_name + '.csv'), sep=',', index=False)

roi_plot = np.int64(np.random.choice(roi_list)[3:])
trial_plot = np.random.choice(trials)
mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace, traces_type, df_events_extract_rawtrace, trials, plot_data, print_plots)

# Detrend calcium trace
df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')

# Save data
if not os.path.exists(mscope.path + 'processed files'):
    os.mkdir(mscope.path + 'processed files')
df_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', 'df_extract_raw.csv'), sep=',',
                           index=False)
df_extract_rawtrace_detrended.to_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'), sep=',',
    index=False)
df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'),
                                  sep=',', index=False)
np.save(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), coord_ext_480, allow_pickle=True)
np.save(os.path.join(mscope.path, 'processed files', 'trials.npy'), trials)


