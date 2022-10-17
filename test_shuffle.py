# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy

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
if session_type == 'tied' and animal == 'MC8855':
    trials_ses = np.array([3, 6])
    trials_ses_name = ['baseline speed', 'fast speed']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[0] + 1, trials_ses[1] + 1)]
    cond_plot = ['baseline', 'fast']
if session_type == 'tied' and animal != 'MC8855':
    trials_ses = np.array([6, 12, 18])
    trials_ses_name = ['baseline speed', 'fast speed']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[0] + 1, trials_ses[1] + 1),
                np.arange(trials_ses[1] + 1, trials_ses[2] + 1)]
    cond_plot = ['baseline', 'slow', 'fast']
if session_type == 'split' and animal == 'MC8855':
    trials_ses = np.array([3, 4, 13, 14])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[1], trials_ses[2] + 1),
                np.arange(trials_ses[3], len(trials) + 1)]
    cond_plot = ['baseline', 'split', 'washout']
if session_type == 'split' and animal != 'MC8855':
    trials_ses = np.array([6, 7, 16, 17])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
    idx_plot = [np.arange(1, trials_ses[0] + 1), np.arange(trials_ses[1], trials_ses[2] + 1),
                np.arange(trials_ses[3], len(trials) + 1)]
    cond_plot = ['baseline', 'split', 'washout']
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

# Load data
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, trials, coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()

# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
stride_duration_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
    stride_duration_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'stride_duration'))

traces_type = 'raw'
roi_list = df_extract_rawtrace_detrended.columns[2:]
roi_plot = []
for r in range(len(roi_list)):
    roi_plot.append(np.int64(roi_list[r][3:]))
count_r = 0
roi = 2
step_size = 2
trials_compute = idx_plot[0]
plot_condition = cond_plot[0]
p1 = 'FR'
p2 = 'FL'
nr_strides = 10
if p1 == 'FR':
    p1_idx = 0
if p1 == 'HR':
    p1_idx = 1
if p1 == 'FL':
    p1_idx = 2
if p1 == 'HL':
    p1_idx = 3
if p2 == 'FR':
    p2_idx = 0
if p2 == 'HR':
    p2_idx = 1
if p2 == 'FL':
    p2_idx = 2
if p2 == 'HL':
    p2_idx = 3
bin_vector = np.arange(-50, 50 + step_size, step_size)
sl_p1_events_trials = np.zeros((len(trials), len(bin_vector)))
sl_p1_events_trials[:] = np.nan
sl_sym_all = []
trial_id = []
sl_event_all = []
sl_event_all_shuffle = []
stride_duration_all = []
df_events_shuffle = df_events_extract_rawtrace
for i in range(1000):
    df_events_shuffle = df_events_shuffle.sample(frac=1, axis=0, random_state=42).reset_index(drop=True)
for count_t, t in enumerate(trials_compute):
    trial_index = np.where(trials == t)[0][0]
    df_events_trial = df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t].reset_index(drop=True)
    # shuffle events for all trials
    df_events_trial_shuffle = df_events_shuffle.loc[df_events_shuffle['trial'] == t].reset_index(drop=True)
    sl_p1 = param_trials[trial_index][p1_idx]
    sl_p2 = param_trials[trial_index][p2_idx]
    strides_p1 = st_strides_trials[trial_index][p1_idx]
    strides_p2 = st_strides_trials[trial_index][p2_idx]
    # get events between the strides
    sl_p1_events = np.zeros(np.shape(strides_p1)[0])
    sl_p1_events_shuffle = np.zeros(np.shape(strides_p1)[0])
    sl_sym = np.zeros(np.shape(strides_p1)[0])
    sl_sym[:] = np.nan
    for s in range(np.shape(strides_p1)[0]):
        events_stride = df_events_trial.loc[(df_events_trial['time'] >= strides_p1[s][0, 0] / 1000) & (
                df_events_trial['time'] <= strides_p1[s][1, 0] / 1000), 'ROI' + str(roi)]
        events_stride_shuffle = df_events_trial_shuffle.loc[(df_events_trial['time'] >= strides_p1[s][0, 0] / 1000) & (
                df_events_trial['time'] <= strides_p1[s][1, 0] / 1000), 'ROI' + str(roi)]
        stride_contra = \
            np.where((strides_p2[:, 0, 0] > strides_p1[s, 0, 0]) & (strides_p2[:, 0, 0] < strides_p1[s, 1, 0]))[
                0]
        if len(stride_contra) == 1:  # one event in the stride
            sl_sym[s] = sl_p1[s] - sl_p2[stride_contra]
        if len(stride_contra) > 1:  # more than two events in a stride
            sl_sym[s] = sl_p1[s] - np.nanmean(sl_p2[stride_contra])
        if len(np.where(events_stride)[0]) > 0:
            sl_p1_events[s] = len(np.where(events_stride)[0])
            sl_p1_events_shuffle[s] = len(np.where(events_stride_shuffle)[0])
    sl_sym_all.extend(sl_sym)
    trial_id.extend(np.repeat(t, len(sl_sym)))
    sl_event_all.extend(sl_p1_events)
    sl_event_all_shuffle.extend(sl_p1_events_shuffle)
    stride_duration_all.extend(stride_duration_trials[trial_index][p1_idx])

# shuffle sl_event_all
sl_event_all_array = np.array(sl_event_all)
sl_event_all_shuffle = np.array(sl_event_all_shuffle)
stride_duration_all_array = np.array(stride_duration_all)
bins_all = np.histogram(sl_sym_all, bin_vector)[1]
sl_p1_events_bin_count = np.zeros(len(bins_all))
sl_p1_events_bin_count[:] = np.nan
sl_p1_events_bin_all = np.zeros(len(bins_all))
sl_p1_events_bin_all[:] = np.nan
sl_p1_events_bin_all_shuffle = np.zeros(len(bins_all))
sl_p1_events_bin_all_shuffle[:] = np.nan
sl_p1_events_bin_count_shuffle = np.zeros(len(bins_all))
sl_p1_events_bin_count_shuffle[:] = np.nan
stride_duration_bin_all = np.zeros(len(bins_all))
stride_duration_bin_all[:] = np.nan
stride_duration_bin_avg = np.zeros(len(bins_all))
stride_duration_bin_avg[:] = np.nan
stride_duration_bin_sem = np.zeros(len(bins_all))
stride_duration_bin_sem[:] = np.nan
stride_nr_bin = np.zeros(len(bins_all))
stride_nr_bin[:] = np.nan
for count_b, b in enumerate(bins_all):
    if count_b == len(bins_all) - 1:
        idx_bin = np.where((sl_sym_all >= bins_all[count_b]) & (sl_sym_all < bins_all[-1]))[0]
    else:
        idx_bin = np.where((sl_sym_all >= bins_all[count_b]) & (sl_sym_all < bins_all[count_b + 1]))[0]
    if len(idx_bin) > nr_strides:
        sl_p1_events_bin_count[count_b] = np.sum(sl_event_all_array[idx_bin])
        sl_p1_events_bin_count_shuffle[count_b] = np.sum(sl_event_all_shuffle[idx_bin])
        stride_duration_bin_all[count_b] = np.cumsum(stride_duration_all_array[idx_bin])[-1]
        stride_duration_bin_avg[count_b] = np.mean(stride_duration_all_array[idx_bin])
        stride_duration_bin_sem[count_b] = np.std(stride_duration_all_array[idx_bin])/np.sqrt(len(idx_bin))
        sl_p1_events_bin_all[count_b] = sl_p1_events_bin_count[count_b] / stride_duration_bin_all[count_b]
        sl_p1_events_bin_all_shuffle[count_b] = sl_p1_events_bin_count_shuffle[count_b] / stride_duration_bin_all[count_b]
        stride_nr_bin[count_b] = len(idx_bin)

if not os.path.exists(os.path.join(mscope.path, 'images', 'events', traces_type)):
    os.mkdir(os.path.join(mscope.path, 'images', 'events', traces_type))
if not os.path.exists(os.path.join(mscope.path + 'images', 'events', traces_type, 'ROI' + str(roi))):
    os.mkdir(os.path.join(mscope.path + 'images', 'events', traces_type, 'ROI' + str(roi)))
fig, ax = plt.subplots(3, 2, figsize=(30, 20), tight_layout=True)
ax = ax.ravel()
ax[0].bar(bins_all[~np.isnan(sl_p1_events_bin_all)], sl_p1_events_bin_all[~np.isnan(sl_p1_events_bin_all)],
          width=step_size - (step_size * 0.1), color='black')
ax[0].bar(bins_all[~np.isnan(sl_p1_events_bin_all_shuffle)]+step_size/8, sl_p1_events_bin_all_shuffle[~np.isnan(sl_p1_events_bin_all_shuffle)],
          width=step_size - (step_size * 0.1), color='gray')
ax[0].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 4)
ax[0].set_ylabel('Event prob. norm. cum. stride duration',
                 fontsize=mscope.fsize - 4)
ax[0].set_title(
    'Event proportion for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
    fontsize=mscope.fsize - 4)
ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].bar(bins_all[~np.isnan(stride_duration_bin_all)],
          stride_duration_bin_all[~np.isnan(stride_duration_bin_all)], width=step_size - (step_size * 0.1),
          color='black')
ax[1].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 4)
ax[1].set_ylabel('Cumulative stride duration (ms)', fontsize=mscope.fsize - 4)
ax[1].set_title(
    'Stride duration for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
    fontsize=mscope.fsize - 4)
ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[2].bar(bins_all[~np.isnan(stride_nr_bin)], stride_nr_bin[~np.isnan(stride_nr_bin)],
          width=step_size - (step_size * 0.1), color='black')
ax[2].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 4)
ax[2].set_ylabel('Stride count', fontsize=mscope.fsize - 4)
ax[2].set_title(
    'Stride count for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
    fontsize=mscope.fsize - 4)
ax[2].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[3].bar(bins_all[~np.isnan(sl_p1_events_bin_count)],
          sl_p1_events_bin_count[~np.isnan(sl_p1_events_bin_count)], width=step_size - (step_size * 0.1),
          color='black')
ax[3].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 4)
ax[3].set_ylabel('Event count', fontsize=mscope.fsize - 4)
ax[3].set_title(
    'Event count for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
    fontsize=mscope.fsize - 4)
ax[3].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
ax[4].bar(bins_all[~np.isnan(stride_duration_bin_avg)],
          stride_duration_bin_avg[~np.isnan(stride_duration_bin_avg)], width=step_size - (step_size * 0.1),
          color='black')
ax[4].errorbar(bins_all[~np.isnan(stride_duration_bin_avg)],
               stride_duration_bin_avg[~np.isnan(stride_duration_bin_avg)],
               yerr=stride_duration_bin_sem[~np.isnan(stride_duration_bin_sem)], xerr=0, fmt='.',
               color='black')
ax[4].set_xlabel(param_name.replace('_', ' ') + ' symmetry', fontsize=mscope.fsize - 4)
ax[4].set_ylabel('Stride duration mean + sem (ms)', fontsize=mscope.fsize - 4)
ax[4].set_title(
    'Stride duration for ' + param_name.replace('_', ' ') + ' symmetry across trials ' + plot_condition,
    fontsize=mscope.fsize - 4)
ax[4].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[4].spines['right'].set_visible(False)
ax[4].spines['top'].set_visible(False)

dist_real = sl_p1_events_bin_all[~np.isnan(sl_p1_events_bin_all)]
dist_shuffle = sl_p1_events_bin_all_shuffle[~np.isnan(sl_p1_events_bin_all_shuffle)]
print(scipy.stats.kstest(dist_real, dist_shuffle, alternative='two-sided'))
scipy.stats.kstest(sl_event_all, sl_event_all_shuffle, alternative='two-sided')
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.bar(bins_all[~np.isnan(sl_p1_events_bin_all)], dist_real, width=step_size - (step_size * 0.1), color='black')
ax.bar(bins_all[~np.isnan(sl_p1_events_bin_all)], dist_shuffle, width=step_size - (step_size * 0.1), color='black')

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
plt.plot(np.cumsum(dist_real))
plt.plot(np.cumsum(dist_shuffle))