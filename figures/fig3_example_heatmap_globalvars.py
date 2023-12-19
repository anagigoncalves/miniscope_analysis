# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.signal as sp
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig sta\\'

session_data_idx = 0 #example is MC8855
cluster_plot = 0
ses_info = session_data.iloc[session_data_idx, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                         ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                         date.split('_')[-3][2:] + '\\')
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
session_type = path.split('\\')[-4].split(' ')[0]
animal = mscope.get_animal_id()
session = loco.get_session_id()
# Session data and inputs
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
 frames_dFF] = mscope.load_processed_files()
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, session_type, animal, session)
trials_ses_name.insert(len(trials_ses_name), 'late washout')
trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1] + 1), trials))[0]

#HEATMAP EXAMPLE BODYBARS
# Align dF/F and behavior (body position, speed, acceleration) for each trial and desired epoch
# Order ROIs by cluster
if len(clusters_rois) == 1:
    clusters_rois_flat = clusters_rois[0]
else:
    clusters_rois_flat = np.transpose(sum(clusters_rois, []))
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
df = mscope.norm_traces(df_extract_rawtrace_detrended[clusters_rois_flat], norm_name='zscore', axis='session') # Normalized dF/F traces for 'popul_heatmap'

# Load behavioral data and get acceleration
filelist = loco.get_track_files(animal, session)
bodyacc = []
bodycenter = []
bodyspeed = []
bodyjerk = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
        frames_loco[count_trial]))
    bodycenter_trial = sp.medfilt(loco.compute_bodycenter(final_tracks, 'X'), 25) #filter for tracking errors
    bodyspeed_trial = loco.compute_bodyspeed(bodycenter_trial)
    bodyacc_trial = loco.compute_bodyacc(bodycenter_trial)
    bodyjerk_trial = loco.compute_bodyjerk(bodycenter_trial)
    bodyacc.append(bodyacc_trial)
    bodycenter.append(bodycenter_trial)
    bodyspeed.append(bodyspeed_trial)
    bodyjerk.append(bodyjerk_trial)

[df_sorted, cluster_transition_idx] = mscope.sort_rois_clust(df, clusters_rois)

trial = 1
beg = 20
end = 45
fig, axs = plt.subplots(5, 1, figsize=(25, 15), tight_layout=True)
df_trial = df_sorted.loc[(df_sorted['trial'] == trial)&(df_sorted['time']>beg)&(df_sorted['time']<end)].iloc[:, 2:]  # Get df/f for the desired trial and interval
hm = sns.heatmap(df_trial.T, cmap='plasma', ax=axs[0], cbar=None)
axs[0].set_xticks([])
axs[0].set(xticklabels=[])
axs[0].set_ylabel('ROIs', fontsize=20)
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
for c in cluster_transition_idx:  # Lines to mark clusters in the heatmap
    axs[0].hlines(c + 1, *axs[0].get_xlim(), color='white', linestyle='dashed', linewidth=1)
# Behavior
t = np.linspace(beg, end, (end-beg)*loco.sr)  # Create x-axis time values
sns.lineplot(x=t, y=bodycenter[trial-1][beg*loco.sr:end*loco.sr], ax=axs[1], color='black', linewidth=2)
axs[1].set(xticklabels=[])
axs[1].set_xlim([t[0], t[-1]])
axs[1].set_ylabel('Body\nCenter (mm)', fontsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].tick_params(left=False, bottom=False)
sns.lineplot(x=t, y=bodyspeed[trial-1][beg*loco.sr:end*loco.sr], ax=axs[2], color='black', linewidth=2)
axs[2].set(xticklabels=[])
axs[2].set_xlim([t[0], t[-1]])
axs[2].set_ylabel('Body\nSpeed (mm/s)', fontsize=20)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)
sns.lineplot(x=t, y=bodyacc[trial-1][beg*loco.sr:end*loco.sr], ax=axs[3], color='black', linewidth=2)
axs[3].set_xlim([t[0], t[-1]])
axs[3].set_ylabel('Body\nAcceleration\n(mm/s\u00b2)', fontsize=20)
axs[3].set_xlabel('Time (s)', fontsize=20)
axs[3].tick_params(axis='both', which='major', labelsize=18)
axs[3].spines['right'].set_visible(False)
axs[3].spines['top'].set_visible(False)
sns.lineplot(x=t, y=bodyjerk[trial-1][beg*loco.sr:end*loco.sr], ax=axs[4], color='black', linewidth=2)
axs[4].set_xlim([t[0], t[-1]])
axs[4].set_ylabel('Body\nJerk\n(mm/s\u00b3)', fontsize=20)
axs[4].set_xlabel('Time (s)', fontsize=20)
axs[4].tick_params(axis='both', which='major', labelsize=18)
axs[4].spines['right'].set_visible(False)
axs[4].spines['top'].set_visible(False)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\fig sta\\', 'example_bodyvars_MC8855_splitipsifast_S1'), dpi=128)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\fig sta\\', 'example_bodyvars_MC8855_splitipsifast_S1.svg'), dpi=128)

fig, axs = plt.subplots(5, 1, figsize=(25, 15), tight_layout=True)
df_trial = df_sorted.loc[(df_sorted['trial'] == trial)&(df_sorted['time']>beg)&(df_sorted['time']<end)].iloc[:, 2:]  # Get df/f for the desired trial and interval
hm = sns.heatmap(df_trial.T, cmap='plasma', ax=axs[0])
cbar = hm.collections[0].colorbar
cbar.ax.set_label('\u0394F/F')
cbar.ax.tick_params(labelsize=16)
axs[0].set_xticks([])
axs[0].set(xticklabels=[])
axs[0].set_ylabel('ROIs', fontsize=20)
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
for c in cluster_transition_idx:  # Lines to mark clusters in the heatmap
    axs[0].hlines(c + 1, *axs[0].get_xlim(), color='white', linestyle='dashed', linewidth=1)
# Behavior
t = np.linspace(beg, end, (end-beg)*loco.sr)  # Create x-axis time values
sns.lineplot(x=t, y=bodycenter[trial-1][beg*loco.sr:end*loco.sr], ax=axs[1], color='black', linewidth=2)
axs[1].set(xticklabels=[])
axs[1].set_xlim([t[0], t[-1]])
axs[1].set_ylabel('Body\nCenter (mm)', fontsize=20)
axs[1].tick_params(axis='both', which='major', labelsize=18)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].tick_params(left=False, bottom=False)
sns.lineplot(x=t, y=bodyspeed[trial-1][beg*loco.sr:end*loco.sr], ax=axs[2], color='black', linewidth=2)
axs[2].set(xticklabels=[])
axs[2].set_xlim([t[0], t[-1]])
axs[2].set_ylabel('Body\nSpeed (mm/s)', fontsize=20)
axs[2].tick_params(axis='both', which='major', labelsize=18)
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)
sns.lineplot(x=t, y=bodyacc[trial-1][beg*loco.sr:end*loco.sr], ax=axs[3], color='black', linewidth=2)
axs[3].set_xlim([t[0], t[-1]])
axs[3].set_ylabel('Body\nAcceleration\n(mm/s\u00b2)', fontsize=20)
axs[3].set_xlabel('Time (s)', fontsize=20)
axs[3].tick_params(axis='both', which='major', labelsize=18)
axs[3].spines['right'].set_visible(False)
axs[3].spines['top'].set_visible(False)
sns.lineplot(x=t, y=bodyjerk[trial-1][beg*loco.sr:end*loco.sr], ax=axs[4], color='black', linewidth=2)
axs[4].set_xlim([t[0], t[-1]])
axs[4].set_ylabel('Body\nJerk\n(mm/s\u00b3)', fontsize=20)
axs[4].set_xlabel('Time (s)', fontsize=20)
axs[4].tick_params(axis='both', which='major', labelsize=18)
axs[4].spines['right'].set_visible(False)
axs[4].spines['top'].set_visible(False)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\fig sta\\', 'example_bodyvars_MC8855_splitipsifast_S1_legend'), dpi=128)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\fig sta\\', 'example_bodyvars_MC8855_splitipsifast_S1_legend.svg'), dpi=128)

