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
# after manual check replot the refined ROI selection
[df_extract, df_events_extract, df_extract_rawtrace, trials, coord_ext, reg_th, amp_arr, reg_bad_frames] = mscope.load_processed_files()
event_proportion_all = np.load(mscope.path + '\\processed files\\' + 'event_proportion_allrois.npy')
rois_names = np.array([2,4,7,9,12,19,24,39,44,60,67,68,73,74,87,100,104,105,114,123,128,129,131,135,142,161,172,3,5,8,10,11,13,14,15,16,18,20,21,23,25,27,28,34,37,40,42,45,50,51,53,61,62,63,70,72,75,76,77,78,80,83,88,94,97,109,116,117,118,121,124,127,150,151,159,166,170])
[df_extract_new, df_extract_new_raw, df_events_extract_new, coord_ext_new, keep_roi_idx] = mscope.refine_roi_list(rois_names, df_extract, df_extract_rawtrace, df_events_extract, coord_ext)
df_extract_new_norm = mscope.norm_traces(df_extract_new,'min_max', 'trial')
df_extract_new_raw_norm = mscope.norm_traces(df_extract_new_raw,'zscore', 'trial')
mscope.plot_stacked_traces(frame_time, df_extract_new_norm, trials_ses, plot_data)  # input can be one trial or trials_ses
mscope.plot_rois_ref_image(ref_image, coord_ext_new, plot_data)

data_1st_5s_list_raw = []
frames_len = []
trial_sum_raw = []
for t in trials:
    data_1st_5s_list_raw.append(np.transpose(df_extract_new_raw_norm.loc[df_extract_new_raw_norm['trial']==t].iloc[:5*mscope.sr,2:]))
    frames_len.append(np.shape(df_extract_new_raw_norm.loc[df_extract_new_raw_norm['trial']==t].iloc[:5*mscope.sr,2:])[0])
    trial_sum_raw.append(df_extract_new_raw_norm.loc[df_extract_new_raw_norm['trial'] == t].iloc[:5 * mscope.sr, 2:].sum().sum())
data_1st_5s_raw = np.concatenate(data_1st_5s_list_raw,axis=1)
cum_frames_len = np.cumsum(frames_len)
fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
sns.heatmap(data_1st_5s_raw, cbar='False')
for t in trials:
    ax.vlines(cum_frames_len[t-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_title('1st 5s of each trial raw trace', fontsize=mscope.fsize - 4)
ax.set_yticklabels(df_extract_new_norm.columns[2::2], rotation=45, fontsize=mscope.fsize - 10)
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='x', length=0)
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
# ax.plot(np.arange(0,np.shape(data_1st_5s_raw)[1]),np.sum(data_1st_5s_raw,axis=0), color='black', linewidth=2)
ax.bar(np.arange(0,np.shape(data_1st_5s_raw)[1]),np.sum(data_1st_5s_raw,axis=0), color='black', linewidth=2)
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = -140, ymax = 400, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_frames_bar', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
# ax.plot(np.arange(75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw, color='black', marker='o', linewidth=2)
ax.bar(np.arange(75,np.shape(data_1st_5s_raw)[1],150),trial_sum_raw, width = 140, color='black')
for t in trials:
    ax.vlines(cum_frames_len[t-1], ymin = -5600, ymax = 12700, color='black', linestyle='dashed')
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s_raw_sum_all_bar', dpi=mscope.my_dpi)
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
    data_1st_5s_list_dconv.append(np.transpose(df_extract_new_norm.loc[df_extract_new_norm['trial']==t].iloc[:5*mscope.sr,2:]))
    frames_len.append(np.shape(df_extract_new_norm.loc[df_extract_new_norm['trial']==t].iloc[:5*mscope.sr,2:])[0])
    trial_sum_dconv.append(df_extract_new_norm.loc[df_extract_new_norm['trial'] == t].iloc[:5 * mscope.sr, 2:].sum().sum())
data_1st_5s_dconv = np.concatenate(data_1st_5s_list_dconv,axis=1)
cum_frames_len = np.cumsum(frames_len)
fig, ax = plt.subplots(figsize=(25, 12), tight_layout=True)
sns.heatmap(data_1st_5s_dconv, cbar='False')
for t in trials:
    ax.vlines(cum_frames_len[t-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_title('1st 5s of each trial deconvolved trace', fontsize=mscope.fsize - 4)
ax.set_yticklabels(df_extract_new_norm.columns[2::2], rotation=45, fontsize=mscope.fsize - 10)
plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='x', length=0)
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
if save_data:
    plt.savefig(mscope.path+ '\\images\\'+ 'heatmap_1st_5s', dpi=mscope.my_dpi)
fig, ax = plt.subplots(figsize=(25, 5), tight_layout=True)
# ax.plot(np.arange(0,np.shape(data_1st_5s_dconv)[1]),np.sum(data_1st_5s_dconv,axis=0), color='black', linewidth=2)
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
# ax.plot(np.arange(75,np.shape(data_1st_5s_dconv)[1],150),trial_sum_dconv, color='black', marker='o', linewidth=2)
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
