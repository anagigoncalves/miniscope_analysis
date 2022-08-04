# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import pandas as pd
import time
import seaborn as sns
import SlopeThreshold as ST

# path inputs
path = 'I:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'I:\\TM TRACKING FILES\\split ipsi fast\\split ipsi fast S1 050421\\'
# path = 'I:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\'
# path_loco = 'I:\\TM TRACKING FILES\\tied baseline\\tied baseline S1 040421\\'
session_type = 'split'
delim = path[-1]
version_mscope = 'v4'
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
if session_type == 'split' and animal == 'MC8855':
    trials_ses = np.array([3, 4, 13, 14])
    trials_ses_name = ['baseline', 'early split', 'late split', 'early washout']
if session_type == 'split' and animal != 'MC8855':
    trials_ses = np.array([6, 7, 16, 17])
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

[df_extract, df_events_extract, df_extract_rawtrace, df_events_extract_rawtrace, trials, coord_ext, reg_th, reg_bad_frames] = mscope.load_processed_files()
df_events_extract = mscope.get_events(df_extract, 0, 'df_events_extract') #find parameters for extract traces
# df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_raw_detrended')
trial_plot = 8
roi_plot = 42
plot_data = 0
mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract, df_events_extract, plot_data)


# #PCA ON EXTRACT AND DETRENDED TRACES
# df_extract_rawtrace_detrended_norm = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')
# df_extract_norm = mscope.norm_traces(df_extract, 'zscore', 'session')
# from sklearn.decomposition import PCA
# principalComponents_3CP_raw = PCA(n_components=3).fit_transform(np.array(df_extract_rawtrace_detrended_norm.iloc[:,2:]))
# principalComponents_3CP_extract = PCA(n_components=3).fit_transform(np.array(df_extract_rawtrace_detrended.iloc[:,2:]))
# trial_length = mscope.trial_length(df_extract)
# trial_length_cumsum = np.int64(np.cumsum(trial_length))
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],0], principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],1], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],0], principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],1], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],0], principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],1], s = 1, color='blue')
# ax[0].set_title('PC1 vs PC2')
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],0], principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],2], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],0], principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],2], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],0], principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],2], s = 1, color='blue')
# ax[0].set_title('PC1 vs PC3')
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],1], principalComponents_3CP_raw[trial_length_cumsum[1]:trial_length_cumsum[2],2], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],1], principalComponents_3CP_raw[trial_length_cumsum[2]:trial_length_cumsum[3],2], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],1], principalComponents_3CP_raw[trial_length_cumsum[11]:trial_length_cumsum[12],2], s = 1, color='blue')
# ax[0].set_title('PC2 vs PC3')
#
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],0], principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],1], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],0], principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],1], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],0], principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],1], s = 1, color='blue')
# ax[0].set_title('PC1 vs PC2')
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],0], principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],2], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],0], principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],2], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],0], principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],2], s = 1, color='blue')
# ax[0].set_title('PC1 vs PC3')
# fig, ax = plt.subplots(3, 1, figsize=(7,20), tight_layout=True)
# ax = ax.ravel()
# ax[0].scatter(principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],1], principalComponents_3CP_extract[trial_length_cumsum[1]:trial_length_cumsum[2],2], s = 1, color='darkgray')
# ax[1].scatter(principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],1], principalComponents_3CP_extract[trial_length_cumsum[2]:trial_length_cumsum[3],2], s = 1, color='crimson')
# ax[2].scatter(principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],1], principalComponents_3CP_extract[trial_length_cumsum[11]:trial_length_cumsum[12],2], s = 1, color='blue')
# ax[0].set_title('PC2 vs PC3')

