import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import pandas as pd
import time
import seaborn as sns

# path inputs
path = 'I:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\'
path_loco = 'I:\\TM TRACKING FILES\\tied baseline\\tied baseline S1 040421\\'
session_type = 'tied'
delim = path[-1]
version_mscope = 'v4'
load_data = 1
improve_eventdet = 0
plot_figures = 0
plot_data = 1
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

# Population plots tied baseline
# after manual check replot the refined ROI selection
[df_extract, df_events_extract,  df_extract_rawtrace, trials, coord_ext, reg_th, amp_arr, reg_bad_frames] = mscope.load_processed_files()
event_proportion_all = np.load(mscope.path + '\\processed files\\' + 'event_proportion_allrois.npy')
rois_names = np.array([11,16,17,18,19,20,24,27,30,31,32,33,36,39,40,41,44,50,51,54,55,58,62,66,68,25,52])
[df_extract_new, df_extract_new_raw, df_events_extract_new, coord_ext_new, keep_roi_idx] = mscope.refine_roi_list(rois_names, df_extract,  df_extract_rawtrace, df_events_extract, coord_ext)
df_extract_new_norm = mscope.norm_traces(df_extract_new,'min_max', 'session')
df_extract_new_norm_raw = mscope.norm_traces(df_extract_new_raw,'zscore', 'trial')
mscope.plot_stacked_traces(frame_time, df_extract_new_norm, trials_ses, plot_data)  # input can be one trial or trials_ses
mscope.plot_rois_ref_image(ref_image, coord_ext_new, plot_data)
#tied baseline spatial maps
sp_up = np.array([16,18,24,25,33,36,39,40,50,58,62])
sp_down = np.array([17,19,20,31,44,52,54,55,66,68])
sp_others = np.array([11,27,30,32,41,51,52])
roi_list = df_extract_new.columns[2:]
plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext_new)):
    #plt.scatter(coord_ext_new[r][:, 0], coord_ext_new[r][:, 1], s=1, alpha=0.6, color='lightgray')
    if np.int64(roi_list[r][3:]) in sp_up:
        plt.scatter(coord_ext_new[r][:, 0], coord_ext_new[r][:, 1], s=1, alpha=0.6, color='red')
    if np.int64(roi_list[r][3:]) in sp_down:
        plt.scatter(coord_ext_new[r][:, 0], coord_ext_new[r][:, 1], s=1, alpha=0.6, color='blue')
plt.imshow(ref_image, cmap='gray',
           extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um, 0])
plt.title('ROIs grouped by activity regarding speed', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
#event proportion across trials and ROIs
event_proportion_all_new = event_proportion_all[keep_roi_idx,:]
fig, ax = plt.subplots(3,1,figsize=(8,20))
ax = ax.ravel()
for r in range(np.shape(event_proportion_all_new)[0]):
    ax[0].scatter(trials,event_proportion_all_new[r,:], color = 'black')
    ax[0].plot(trials, event_proportion_all_new[r, :],linewidth=0.5,color='lightgray')
    ax[0].set_title('All ROIs')
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    if np.int64(roi_list[r][3:]) in sp_up:
        ax[1].scatter(trials, event_proportion_all_new[r, :], color='red')
        ax[1].plot(trials, event_proportion_all_new[r, :],linewidth=0.5,color='lightgray')
        ax[1].set_title('Increase activity with speed')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
    if np.int64(roi_list[r][3:]) in sp_down:
        ax[2].scatter(trials, event_proportion_all_new[r, :], color='blue')
        ax[2].plot(trials, event_proportion_all_new[r, :],linewidth=0.5,color='lightgray')
        ax[2].set_title('Increase activity with speed')
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
#ROIs ordered all trials - events
rois_ordered_sp = np.concatenate((np.concatenate((sp_up,sp_down)),sp_others))
rois_ordered_sp_idx = np.zeros(len(rois_ordered_sp))
for r in range(len(rois_ordered_sp)):
    rois_ordered_sp_idx[r] = np.where('ROI' + str(rois_ordered_sp[r]) == roi_list)[0][0]
df_events_extract_data = df_events_extract_new.iloc[:,2:]
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.heatmap(np.transpose(df_events_extract_data.iloc[:,rois_ordered_sp_idx]), cmap='viridis', cbar=False)
ax.vlines(np.where(df_events_extract_new.iloc[:,1]==3)[0][-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.hlines([len(sp_up)], *ax.get_xlim(), color='white', linewidth=0.5)
ax.hlines([len(sp_up)+len(sp_down)], *ax.get_xlim(), color='white', linewidth=0.5)
ax.set_xticks(np.arange(df_events_extract_data.index[0], df_events_extract_data.index[-1], 500))
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.set_title('ROIs ordered by their response to speed', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)

#ROIs ordered all trials - deconv signal
rois_ordered_sp = np.concatenate((np.concatenate((sp_up,sp_down)),sp_others))
rois_ordered_sp_idx = np.zeros(len(rois_ordered_sp))
for r in range(len(rois_ordered_sp)):
    rois_ordered_sp_idx[r] = np.where('ROI' + str(rois_ordered_sp[r]) == roi_list)[0][0]
df_extract_data = df_extract_new_norm.iloc[:,2:]
fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.heatmap(np.transpose(df_extract_data.iloc[:,rois_ordered_sp_idx]), cmap='viridis', cbar=False)
ax.vlines(np.where(df_extract_new.iloc[:,1]==3)[0][-1], *ax.get_ylim(), color='white', linestyle='dashed')
ax.hlines([len(sp_up)], *ax.get_xlim(), color='white', linewidth=0.5)
ax.hlines([len(sp_up)+len(sp_down)], *ax.get_xlim(), color='white', linewidth=0.5)
ax.set_xticks(np.arange(df_extract_data.index[0], df_extract_data.index[-1], 500))
ax.set_xlabel('Frames', fontsize=mscope.fsize - 4)
ax.set_title('ROIs ordered by their response to speed', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=mscope.fsize - 8)
plt.yticks(fontsize=mscope.fsize - 8)