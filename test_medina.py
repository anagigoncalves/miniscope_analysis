# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib as mp

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
colors_session = mscope.colors_session(session_type, trials)

[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, trials, coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()

# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))

paw = 'FR'
align = 'stride'
sw_pts_trials = sw_strides_trials
df_events = df_events_extract_rawtrace
roi_plot = 2
sample_shuffle = 1000
shift_rand_array = np.random.randint(5*mscope.sr, 10*60*mscope.sr, size=sample_shuffle)
# #for i in range(len(shift_rand_array)):
i = 100
df_events_shuffle = pd.DataFrame()
for c in df_events.columns[2:]:
    df_events_shuffle[c] = np.roll(df_events[c], shift_rand_array[i])
df_events_shuffle.insert(loc=0, column='trial', value=df_events['trial'])
df_events_shuffle.insert(loc=0, column='time', value=df_events['time'])
# fig, ax = plt.subplots(2, 2, figsize=(30, 20), sharey=True, tight_layout=True)
# ax = ax.ravel()
idx_nocs = []
idx_cs = []
for p_idx, paw in enumerate(paws):
    if paw == 'FR':
        p = 0  # paw of tracking
    if paw == 'HR':
        p = 1
    if paw == 'FL':
        p = 2
    if paw == 'HL':
        p = 3
    nr_strides = np.zeros(len(st_strides_trials))
    for t in np.arange(1, len(st_strides_trials) + 1):
        nr_strides[t - 1] = np.shape(st_strides_trials[t - 1][0])[0]
    maximum_nrstrides = np.int64(np.max(nr_strides))
    cs_stride = np.zeros((maximum_nrstrides, len(st_strides_trials)))
    cs_stride[:] = np.nan
    cs_stride_shuffle = np.zeros((maximum_nrstrides, len(st_strides_trials)))
    cs_stride_shuffle[:] = np.nan
    for t in np.arange(1, len(st_strides_trials) + 1):
        if align == 'stride':
            excursion_beg = st_strides_trials[t - 1][p][:, 0, 4] / mscope.sr_loco
            excursion_end = st_strides_trials[t - 1][p][:, 1, 4] / mscope.sr_loco
        if align == 'stance':
            excursion_beg = st_strides_trials[t - 1][p][:, 0, 4] / mscope.sr_loco
            excursion_end = sw_pts_trials[t - 1][p][:, 0, 4] / mscope.sr_loco
        if align == 'swing':
            excursion_beg = sw_pts_trials[t - 1][p][:, 0, 4] / mscope.sr_loco
            excursion_end = st_strides_trials[t - 1][p][:, 1, 4] / mscope.sr_loco
        events = np.array(
            df_events.loc[(df_events['trial'] == t) & (df_events['ROI' + str(roi_plot)] == 1), 'time'])
        events_shuffle = np.array(
            df_events_shuffle.loc[(df_events_shuffle['trial'] == t) & (df_events_shuffle['ROI' + str(roi_plot)] == 1), 'time'])
        for s in range(len(excursion_beg)):
            cs_idx = np.where((events >= excursion_beg[s]) & (events <= excursion_end[s]))[0]
            cs_idx_shuffle = np.where((events_shuffle >= excursion_beg[s]) & (events_shuffle <= excursion_end[s]))[0]
            if len(cs_idx) > 0:
                cs_stride[s, t - 1] = len(cs_idx)
            if len(cs_idx) == 0:
                cs_stride[s, t - 1] = 0
            if len(cs_idx_shuffle) > 0:
                cs_stride_shuffle[s, t - 1] = len(cs_idx_shuffle)
            if len(cs_idx_shuffle) == 0:
                cs_stride_shuffle[s, t - 1] = 0
    df_cs_stride = pd.DataFrame(cs_stride, columns=np.arange(1, len(st_strides_trials) + 1))
    df_cs_stride_shuffle = pd.DataFrame(cs_stride_shuffle, columns=np.arange(1, len(st_strides_trials) + 1))
    t = 0
    df_cs_stride_trial = np.array(df_cs_stride.iloc[:, t])
    df_cs_stride_trial_nonan = df_cs_stride_trial[~np.isnan(df_cs_stride_trial)]
    df_cs_stride_trial_shuffle = np.array(df_cs_stride_shuffle.iloc[:, t])
    idx_nocs.append(np.where(df_cs_stride_trial_nonan==0)[0])
    idx_cs.append(np.where(df_cs_stride_trial_nonan)[0])

plt.figure()
plt.scatter(df_cs_stride.iloc[:, t].index, df_cs_stride_trial)
plt.scatter(df_cs_stride_shuffle.iloc[:, t].index, df_cs_stride_trial_shuffle)

plt.figure()
plt.scatter(param_trials[t][p], df_cs_stride_trial[~np.isnan(df_cs_stride_trial)])
plt.scatter(param_trials[t][p], df_cs_stride_trial_shuffle[~np.isnan(df_cs_stride_trial_shuffle)])

#TODO boxplot for the different values

#stride trajectory for the different values
fig, ax = plt.subplots(2, 2, tight_layout=True)
ax = ax.ravel()
for s in idx_nocs[2]:
    ax[0].plot(final_tracks_trials[t][0, 2, np.int64(st_strides_trials[t][2][s, 0, 4]):np.int64(st_strides_trials[t][2][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_nocs[0]:
    ax[1].plot(final_tracks_trials[t][0, 0, np.int64(st_strides_trials[t][0][s, 0, 4]):np.int64(st_strides_trials[t][0][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_nocs[3]:
    ax[2].plot(final_tracks_trials[t][0, 3, np.int64(st_strides_trials[t][3][s, 0, 4]):np.int64(st_strides_trials[t][3][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_nocs[1]:
    ax[3].plot(final_tracks_trials[t][0, 1, np.int64(st_strides_trials[t][1][s, 0, 4]):np.int64(st_strides_trials[t][1][s, 1, 4])], linewidth=0.5, color='black')
plt.suptitle('Strides no events')

fig, ax = plt.subplots(2, 2, tight_layout=True)
ax = ax.ravel()
for s in idx_cs[2]:
    ax[0].plot(final_tracks_trials[t][0, 2, np.int64(st_strides_trials[t][2][s, 0, 4]):np.int64(st_strides_trials[t][2][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[0]:
    ax[1].plot(final_tracks_trials[t][0, 0, np.int64(st_strides_trials[t][0][s, 0, 4]):np.int64(st_strides_trials[t][0][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[3]:
    ax[2].plot(final_tracks_trials[t][0, 3, np.int64(st_strides_trials[t][3][s, 0, 4]):np.int64(st_strides_trials[t][3][s, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[1]:
    ax[3].plot(final_tracks_trials[t][0, 1, np.int64(st_strides_trials[t][1][s, 0, 4]):np.int64(st_strides_trials[t][1][s, 1, 4])], linewidth=0.5, color='black')
plt.suptitle('Strides with events')

fig, ax = plt.subplots(2, 2, tight_layout=True)
ax = ax.ravel()
for s in idx_cs[2]:
    if s < np.shape(st_strides_trials[t][2])[0]-1:
        ax[0].plot(final_tracks_trials[t][0, 2, np.int64(st_strides_trials[t][2][s+1, 0, 4]):np.int64(st_strides_trials[t][2][s+1, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[0]:
    if s < np.shape(st_strides_trials[t][0])[0] - 1:
        ax[1].plot(final_tracks_trials[t][0, 0, np.int64(st_strides_trials[t][0][s+1, 0, 4]):np.int64(st_strides_trials[t][0][s+1, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[3]:
    if s < np.shape(st_strides_trials[t][3])[0] - 1:
        ax[2].plot(final_tracks_trials[t][0, 3, np.int64(st_strides_trials[t][3][s+1, 0, 4]):np.int64(st_strides_trials[t][3][s+1, 1, 4])], linewidth=0.5, color='black')
for s in idx_cs[1]:
    if s < np.shape(st_strides_trials[t][1])[0] - 1:
        ax[3].plot(final_tracks_trials[t][0, 1, np.int64(st_strides_trials[t][1][s+1, 0, 4]):np.int64(st_strides_trials[t][1][s+1, 1, 4])], linewidth=0.5, color='black')
plt.suptitle('Strides n+1 after event')
