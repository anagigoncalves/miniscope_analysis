# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['red', 'magenta', 'blue', 'cyan']
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 1.01, 0.05)  # 5 deg
    align_event = 'st' #is always stance
    phase_paws = 'st-sw-st' #can be also 'st-st', need to write code for sw-sw
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
    phase_paws = 'st-sw-st'  # can be also 'st-st', need to write code for sw-sw
    bins_fr = bins*1000

s = 4 #MC9513

ses_info = session_data.iloc[s, :]
print(ses_info)
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                         ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                         date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
session_id = session_type + '_' + ses_info[2]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
 frames_dFF] = mscope.load_processed_files()
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, session_type, animal, session)
centroid_ext = mscope.get_roi_centroids(coord_ext)


# Load behavioral data
filelist = loco.get_track_files(animal, session)
final_tracks_trials = []
st_strides_trials = []
sw_strides_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
        frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                             phase_paws)

count_t = 0
p = 0
excursion_phase = np.zeros(len(final_tracks_trials[count_t][0, p, :]))
excursion_phase[:] = np.nan
s = 20
st_on = np.int64(st_strides_trials[count_t][p][s, 0, -1])
sw_on = np.int64(sw_strides_trials[count_t][p][s, 0, -1])
st_off = np.int64(st_strides_trials[count_t][p][s, 1, -1])
nr_st = len(final_tracks_trials[count_t][0, p, st_on:sw_on])
nr_sw = len(final_tracks_trials[count_t][0, p, sw_on:st_off])
excursion_phase[st_on:sw_on] = np.linspace(0, 0.5, nr_st + 1)[:-1]
excursion_phase[sw_on:st_off] = np.linspace(0.5, 1, nr_sw + 1)[:-1]
print(excursion_phase[st_on:st_off])

roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
roi = roi_list[20]
count_p = 0
paw = 'FR'
[cumulative_idx_paw, trial_id_paw, events_stride_trial_paw] = mscope.event_swst_stride(
    df_events_extract_rawtrace,
    st_strides_trials, sw_strides_trials, final_tracks_phase, bcam_time, align_dimension, align_event,
    trials, paw, roi, np.abs(bins[0]))

p1_idx = count_p
trial = 3
tr = trial-1
spikes_count = np.zeros(len(bins)-1)
dataset = events_stride_trial_paw[trial_id_paw == trial]
for value in dataset:
    if ~np.isnan(value):
        bin_value = np.digitize(value, bins)
        spikes_count[bin_value-1] += 1
phase_paw = final_tracks_phase[tr][0, p1_idx, :]
frames_bin, _ = np.histogram(phase_paw[~np.isnan(phase_paw)],
                             bins=len(bins)-1)  # Compute time spent in each bin
time_bin = frames_bin * (1 / mscope.sr_loco)
firing_rate = spikes_count / time_bin  # Compute firing rate
plt.figure()
plt.plot(firing_rate, color='black')

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
idx_nan = np.where(~np.isnan(events_stride_trial_paw))[0]
ax.scatter(events_stride_trial_paw[idx_nan], cumulative_idx_paw[idx_nan], s=1, color='black')
if align_dimension == 'phase':
    ax.axvline(x=0.5, color='black')
if align_dimension == 'time':
    ax.axvline(x=0, color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


