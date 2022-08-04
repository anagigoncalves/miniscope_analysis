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
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
plot_data = 0
save_data = 1
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

# Population plots  split ipsi fast
[df_extract, df_events_extract, df_extract_rawtrace, df_events_extract_rawtrace, trials, coord_ext, reg_th,
 reg_bad_frames] = mscope.load_processed_files()
df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')
df_extract_rawtrace_detrended_norm = mscope.norm_traces(df_extract_rawtrace_detrended, 'min_max', 'session')
df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'min_max', 'session')
df_extract_norm = mscope.norm_traces(df_extract, 'min_max', 'session')

data_1st_5s_list_raw = []
frames_len = []
trial_sum_raw = []
for t in trials:
    data_1st_5s_list_raw.append(np.transpose(df_extract_rawtrace_detrended_zscore.loc[df_extract_rawtrace_detrended_zscore['trial']==t].iloc[:5*mscope.sr,2:]))
    frames_len.append(np.shape(df_extract_rawtrace_detrended_zscore.loc[df_extract_rawtrace_detrended_zscore['trial']==t].iloc[:5*mscope.sr,2:])[0])
    trial_sum_raw.append(df_extract_rawtrace_detrended_zscore.loc[df_extract_rawtrace_detrended_zscore['trial'] == t].iloc[:5 * mscope.sr, 2:].sum().sum())
data_1st_5s_raw = np.concatenate(data_1st_5s_list_raw,axis=1)
cum_frames_len = np.cumsum(frames_len)
fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
sns.heatmap(data_1st_5s_raw, cbar='False')
for t in trials:
    ax.vlines(cum_frames_len[t-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_title('1st 5s of each trial raw trace', fontsize=mscope.fsize - 4)
ax.set_yticklabels(df_extract_rawtrace_detrended_zscore.columns[2::2], rotation=45, fontsize=mscope.fsize - 10)
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='x', length=0)
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
# ax.plot(np.arange(0,np.shape(data_1st_5s_raw)[1]),np.sum(data_1st_5s_raw,axis=0), color='black', linewidth=2)
ax.bar(np.arange(0,np.shape(data_1st_5s_raw)[1]),np.sum(data_1st_5s_raw,axis=0), color='black', linewidth=2)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 110, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_frames_bar', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.bar(np.arange(75,trials_ses[0]*150,150),trial_sum_raw[:trials_ses[0]], width = 150, color='darkgrey')
ax.bar(np.arange(trials_ses[0]*150+75,trials_ses[2]*150,150),trial_sum_raw[trials_ses[0]:trials_ses[2]], width = 150, color='crimson')
ax.bar(np.arange(trials_ses[2]*150+75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw[trials_ses[2]:], width = 150, color='blue')
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 6000, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_all_bar', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.plot(np.arange(75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw, color='black', marker='o', linewidth=2)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 1000, ymax = 7000, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_all_line', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
ax.plot(np.arange(0,np.shape(data_1st_5s_raw)[1]),np.sum(data_1st_5s_raw,axis=0), color='black', linewidth=2)
for t in np.array([3,13]):
    ax.vlines(cum_frames_len[t-1], ymin = -140, ymax = 400, color='black', linestyle='dashed')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_revyaxis', dpi=mscope.my_dpi)
fig, ax1 = plt.subplots(figsize=(5, 10), tight_layout=True)
ax1.plot(np.arange(75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw, color='black', marker='o', linewidth=2)
for t in np.array([3,13]):
    ax1.vlines(cum_frames_len[t-1], ymin = -5600, ymax = 12700, color='black', linestyle='dashed')
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_all_revyaxis', dpi=mscope.my_dpi)

data_1st_5s_list_dconv = []
frames_len = []
trial_sum_dconv = []
for t in trials:
    data_1st_5s_list_dconv.append(np.transpose(df_extract_norm.loc[df_extract_norm['trial']==t].iloc[:5*mscope.sr,2:]))
    frames_len.append(np.shape(df_extract_norm.loc[df_extract_norm['trial']==t].iloc[:5*mscope.sr,2:])[0])
    trial_sum_dconv.append(df_extract_norm.loc[df_extract_norm['trial'] == t].iloc[:5 * mscope.sr, 2:].sum().sum())
data_1st_5s_dconv = np.concatenate(data_1st_5s_list_dconv,axis=1)
cum_frames_len = np.cumsum(frames_len)
fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
sns.heatmap(data_1st_5s_dconv, cbar='False')
for t in trials:
    ax.vlines(cum_frames_len[t-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_title('1st 5s of each trial deconvolved trace', fontsize=mscope.fsize - 4)
ax.set_yticklabels(df_extract_norm[2::2], rotation=45, fontsize=mscope.fsize - 10)
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='x', length=0)
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.bar(np.arange(0,np.shape(data_1st_5s_dconv)[1]),np.sum(data_1st_5s_dconv,axis=0), color='black')
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 14, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_sum_frames_bar', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.plot(np.arange(0,np.shape(data_1st_5s_dconv)[1]),np.sum(data_1st_5s_dconv,axis=0), color='black', linewidth=2)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 20, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_sum_frames_line', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.bar(np.arange(75,np.shape(data_1st_5s_dconv)[1],150),trial_sum_dconv, color='black', width=140)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 400, ymax = 1200, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_sum_all_bar', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.plot(np.arange(75,np.shape(data_1st_5s_dconv)[1],150),trial_sum_dconv, color='black', marker='o', linewidth=2)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 800, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_sum_all_line', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
ax.plot(np.arange(0,np.shape(data_1st_5s_dconv)[1]),np.sum(data_1st_5s_dconv,axis=0), color='black', linewidth=2)
for t in np.array([3,13]):
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 14, color='black', linestyle='dashed')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_sum_revyaxis', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
ax.plot(np.arange(75,np.shape(data_1st_5s_dconv)[1],150),trial_sum_dconv, color='black', marker='o', linewidth=2)
for t in np.array([3,13]):
    ax.vlines(cum_frames_len[t-1], ymin = 400, ymax = 1200, color='black', linestyle='dashed')
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_all_revyaxis', dpi=mscope.my_dpi)


data_1st_5s_list_raw_events = []
frames_len = []
trial_sum_raw_events = []
for t in trials:
    data_1st_5s_list_raw_events.append(np.transpose(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial']==t].iloc[:5*mscope.sr,2:]))
    frames_len.append(np.shape(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial']==t].iloc[:5*mscope.sr,2:])[0])
    trial_sum_raw_events.append(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t].iloc[:5 * mscope.sr, 2:].sum().sum())
data_1st_5s_raw_events = np.concatenate(data_1st_5s_list_raw_events,axis=1)
cum_frames_len = np.cumsum(frames_len)
fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
sns.heatmap(data_1st_5s_raw_events, cbar='False')
for t in trials:
    ax.vlines(cum_frames_len[t-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_title('1st 5s of each trial raw events', fontsize=mscope.fsize - 4)
ax.set_yticklabels(df_events_extract_rawtrace[2::2], rotation=45, fontsize=mscope.fsize - 10)
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='x', length=0)
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_events', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
ax.bar(np.arange(75,trials_ses[0]*150,150),trial_sum_raw_events[:trials_ses[0]], width = 150, color='darkgrey')
ax.bar(np.arange(trials_ses[0]*150+75,trials_ses[2]*150,150),trial_sum_raw_events[trials_ses[0]:trials_ses[2]], width = 150, color='crimson')
ax.bar(np.arange(trials_ses[2]*150+75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw_events[trials_ses[2]:], width = 150, color='blue')
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = 0, ymax = 2300, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_events_sum_all_bar', dpi=mscope.my_dpi)

plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext)):
    plt.scatter(coord_ext[r][:, 0], coord_ext[r][:, 1], s=1, alpha=0.6)
    plt.text(coord_ext[r][0, 0], coord_ext[r][0, 1], str(r))
plt.imshow(ref_image, cmap='gray',
           extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um, 0])


centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.5
colormap_cluster = 'magma'
cmap = plt.get_cmap(colormap_cluster)
colors_new = [cmap(i) for i in np.linspace(0, 1, int(np.floor(20 + 1)))]
trial_plot = 3
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended_zscore, centroid_ext, distance_neurons, trial_plot, th_cluster, colormap_cluster, plot_data)
furthest_neuron = np.argmax(np.array(centroid_ext)[:, 0])
furthest_neuron = 183
neuron_order = np.argsort(distance_neurons[furthest_neuron, :])
roi_list = df_extract_rawtrace_detrended_zscore.columns[2:]
roi_list_ordered = roi_list[neuron_order]
idx_ordered = idx_roi_cluster[neuron_order]
dFF_trial = df_extract_rawtrace_detrended_zscore.loc[df_extract_rawtrace_detrended_zscore['trial'] == trial_plot]  # get dFF for the desired trial
colors = []
colors.append((0.390384, 0.100379, 0.501864, 1.0)) #purple dark
colors.append((0, 0, 0, 1.0)) #black
colors.append((0.716387, 0.214982, 0.47529, 1.0)) #purple light
colors.append((0.967671, 0.439703, 0.35981, 1.0)) #salmon
colors.append((0.994738, 0.62435, 0.427397, 1.0)) #orange center
colors.append((0.967671, 0.439703, 0.35981, 1.0))
fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
count_r = 0
for r in roi_list_ordered:
    plt.plot(frame_time[trial_plot - 1], dFF_trial[r] + (count_r/2), color='black')
    plt.plot(np.arange(-3,0), np.ones(3)*count_r/2, color=colors[idx_ordered[count_r] - 1], linewidth=4)
    count_r += 1
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='y', which='y', length=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
plt.savefig(os.path.join(mscope.path, 'images', 'cluster', 'traces_ncmposter'), dpi=mscope.my_dpi)