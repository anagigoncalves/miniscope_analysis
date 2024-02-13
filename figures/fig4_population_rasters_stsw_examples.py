import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_tied_S1.xlsx'))
protocol = 'tied baseline'
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 1.01, 0.05)  # 5 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
    bins_fr = bins*1000
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']

animal = 'MC9194'
firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_ro'
                                                                              'is.npy'))
# firing_rate_mean_trials_paw = np.nanmean(firing_rate_animal[:, p, :, :], axis=1)
s = np.where(session_data['animal'] == animal)[0][0]
ses_info = session_data.iloc[s, :]
print(ses_info)
date = ses_info[3]
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                         ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                         date.split('_')[-3][2:] + '\\')

import miniscope_session_class
import locomotion_class
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
animal = mscope.get_animal_id()
session = loco.get_session_id()
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
 frames_dFF] = mscope.load_processed_files()
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, protocol.split(' ')[0], animal, session)

roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
count_roi = roi_list.index('ROI12')
fig, ax = plt.subplots(1, 4, figsize=(25, 7), tight_layout=True)
for count_p, paw in enumerate(paws):
    sns.heatmap(firing_rate_animal[count_roi, count_p, :, :], cmap='viridis', cbar=None,
    vmin=np.nanmin(firing_rate_animal[count_roi, :, :, :]), vmax=np.nanmax(firing_rate_animal[count_roi, :, :, :]), ax=ax[count_p])
    ax[count_p].set_yticks(np.arange(0, len(trials)))
    ax[count_p].invert_yaxis()
    ax[count_p].set_xticks([0, 10, 20])
    ax[count_p].set_xticklabels(['0', '50', '100'], rotation=45)
    ax[count_p].set_yticklabels(list(map(str, trials)), rotation=0)
    ax[count_p].axvline(x=np.int64(len(bins[::-1])/2), color='white')
    ax[count_p].axhline(y=trials_ses[0, 1], color='white', linestyle='dashed')
    ax[count_p].axhline(y=trials_ses[1, 1], color='white', linestyle='dashed')
    ax[count_p].tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_heatmap'), dpi=256)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_heatmap.svg'), dpi=256)

fig, ax = plt.subplots(1, 4, figsize=(25, 7), tight_layout=True)
for count_p, paw in enumerate(paws):
    hm = sns.heatmap(firing_rate_animal[count_roi, count_p, :, :], cmap='viridis',
    vmin=np.nanmin(firing_rate_animal[count_roi, :, :, :]), vmax=np.nanmax(firing_rate_animal[count_roi, :, :, :]), ax=ax[count_p])
    hm.figure.axes[-1].set_ylabel('Event rate (Hz)', size=20)
    hm.figure.axes[-1].tick_params(labelsize=16)
    ax[count_p].set_yticks(np.arange(0, len(trials)))
    ax[count_p].invert_yaxis()
    ax[count_p].set_xticks([0, 10, 20])
    ax[count_p].set_xticklabels(['0', '50', '100'], rotation=45)
    ax[count_p].set_yticklabels(list(map(str, trials)), rotation=0)
    ax[count_p].axvline(x=np.int64(len(bins[::-1])/2), color='white')
    ax[count_p].axhline(y=trials_ses[0, 1], color='white', linestyle='dashed')
    ax[count_p].axhline(y=trials_ses[1, 1], color='white', linestyle='dashed')
    ax[count_p].tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_heatmap_legend'), dpi=256)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_heatmap_legend.svg'), dpi=256)

fig, ax = plt.subplots(1, 4, figsize=(25, 3), tight_layout=True)
for count_p, paw in enumerate(paws):
    ax[count_p].plot(bins[:-1]*100, np.nanmean(firing_rate_animal[count_roi, count_p, :, :], axis=0), color=paw_colors[count_p], linewidth=3)
    if align_dimension == 'phase':
        ax[count_p].axvline(x=50, color='black')
        ax[count_p].set_xlabel('Phase (%)', fontsize=20)
    if align_dimension == 'time':
        ax[count_p].axvline(x=0, color='black')
        ax[count_p].set_xlabel('Time (ms)', fontsize=20)
    # ax[count_p].set_ylim([0.8, 3.8])
    ax[count_p].set_ylim([0.8, 3.1])
    #ax[count_p].set_ylim([np.nanmin(np.nanmean(firing_rate_animal[count_roi, :, :, :], axis=0)), np.nanmax(np.nanmean(firing_rate_animal[count_roi, :, :, :], axis=0))])
    ax[count_p].spines['right'].set_visible(False)
    ax[count_p].spines['top'].set_visible(False)
    ax[count_p].tick_params(axis='both', which='major', labelsize=20)
    ax[count_p].set_ylabel('Event rate (Hz)', fontsize=20)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_paw_summary_legend'), dpi=256)
plt.savefig(os.path.join(save_path, animal + '_' + roi_list[count_roi] + '_paw_summary_legend.svg'), dpi=256)

