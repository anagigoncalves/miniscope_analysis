# # -*- coding: utf-8 -*-
# # %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import zscore

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\LocoCF\\head data analysis\\'
#define a save_path for STA processed data
protocol = 'split ipsi fast'

s=0
# for s in range(len(session_data)):
ses_info = session_data.iloc[s, :]
print(ses_info)
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
session_id = session_type + '_' + ses_info[2]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
head_angles = pd.read_csv(os.path.join(mscope.path, 'processed files', 'head_angles_time_adjusted.csv'))
head_angles_corr = mscope.correct_gimbal_lock(head_angles)

# Plot neural activity with head data
bins = np.arange(-2*np.pi, 2*np.pi, 0.1) # 0.5 radians
roi = 'ROI12'
def bin_trial_data(df_events_extract_rawtrace, roi, trials, bins, norm):
    spikes_per_bin_trial = np.zeros((len(trials), len(bins)))
    for count_t, trial in enumerate(trials):
        trial_data = df_events_extract_rawtrace.loc[
            (df_events_extract_rawtrace['trial'] == trial)]
        spk_data = trial_data.loc[trial_data[roi] == 1]
        head_data = head_angles_corr.loc[
            (head_angles_corr['trial'] == trial)]
        #Bin spikes and head data
        bin_indices = np.digitize(head_data['pitch'], bins) - 1
        spikes_per_bin_trial[count_t, :] = np.bincount(bin_indices, weights=trial_data[roi], minlength=len(bins))
    spikes_per_bin = np.sum(spikes_per_bin_trial, axis=0)
    if norm:
        spikes_per_bin = spikes_per_bin/np.max(spikes_per_bin)
    return spikes_per_bin

#Polar plot
fig = plt.figure(figsize=(5,5))
ax = plt.subplot(projection='polar')
spikes_per_bin_bs = bin_trial_data(df_events_extract_rawtrace, roi, trials_baseline, bins, 0)
ax.bar(bins, spikes_per_bin_bs, width=0.1, color=colors_session[1], bottom=0.0, alpha=0.5)
spikes_per_bin_es = bin_trial_data(df_events_extract_rawtrace, roi, trials_split[:2], bins, 0)
ax.bar(bins, spikes_per_bin_es, width=0.1, color=colors_session[trials_split[0]], bottom=0.0, alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#TODO compute for all ROIs the difference between histograms (bs and early split) for all animals and put in ROI map
#Histogram
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
spikes_per_bin_bs = bin_trial_data(df_events_extract_rawtrace, roi, trials_baseline, bins, 1)
spikes_per_bin_es = bin_trial_data(df_events_extract_rawtrace, roi, trials_split[:2], bins, 1)
ax.bar(bins, spikes_per_bin_bs, width=0.1, color=colors_session[1], alpha=0.5)
ax.bar(bins, spikes_per_bin_es, width=0.1, color=colors_session[trials_split[0]], alpha=0.5)
ax.set_xlim(-1, 1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
from scipy.spatial import distance
dst = distance.euclidean(spikes_per_bin_bs, spikes_per_bin_es)
print(dst)

# STA
var_name = 'yaw'
window = np.arange(-30, 30 + 1)
iter_n = 100
# Compute spike-triggered average (STA) on head data
def sta_head_data(df_events_extract_rawtrace, head_angles_corr, trials, window, var_name):
    signal_chunks_allrois = []
    sta_allrois = []
    for n in range(2, df_events_extract_rawtrace.shape[1]):
        sta = np.empty((0, len(window)))
        signal_chunks_tr = []
        for tr_idx, tr in enumerate(trials):
            head_angle_var = np.array(head_angles_corr.loc[head_angles_corr['trial'] == tr, var_name])
            signal_chunks = np.empty((0, len(window)))
            df_tr = df_events_extract_rawtrace[df_events_extract_rawtrace['trial'] == tr].reset_index(drop=True)
            events_idx = np.array(df_tr.index[df_tr.iloc[:, n] == 1])
            # Extract traces around each event for one ROI
            for i in events_idx:
                if i + window[0] >= 0 and i + window[-1] < len(head_angle_var):
                    extracted_signal = head_angle_var[i + window[0]:i + window[-1] + 1]
                    # List of raw traces for one ROI 'n' and trial 'tr'
                    signal_chunks = np.vstack((signal_chunks, extracted_signal))
            signal_chunks_tr.append(signal_chunks)  # Array of traces for one ROI all trials
        # Compute STA by trial for one ROI
        sta = np.vstack([np.nanmean(signal_chunks_tr[tr_idx], axis=0) for tr_idx, _ in enumerate(trials)])
        # List of raw traces for each ROI whole session
        signal_chunks_allrois.append(np.concatenate(signal_chunks_tr, axis=0))
        # STA by trial for all ROIs
        sta_allrois.append(sta)
    return sta_allrois, signal_chunks_allrois
sta_allrois, signal_chunks_allrois = sta_head_data(df_events_extract_rawtrace, head_angles_corr, trials, window, var_name)
# Standardize observed STA on STA computed with shuffled data
# Shuffle CS timestamps
shuffled_spikes_ts = mscope.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)
# Compute STA for shuffled data
def sta_head_data_shuffled(spikes_ts, head_angles_corr, trials, window, var_name):
    signal_chunks_allrois = []
    sta_allrois = []
    for n in range(len(spikes_ts)):
        sta = np.empty((0, len(window)))
        signal_chunks_tr = []
        for tr_idx, tr in enumerate(trials):
            head_angle_var = np.array(head_angles_corr.loc[head_angles_corr['trial'] == tr, var_name])
            head_angle_time = np.array(head_angles_corr.loc[head_angles_corr['trial'] == tr, 'time'])
            signal_chunks = np.empty((0, len(window)))
            events_ts = np.array(spikes_ts[n][tr_idx])  # Find timestamps of events for ROI 'n'
            matching_ts_idx = [np.abs(head_angle_time - ts).argmin() for ts in events_ts]
            for i in matching_ts_idx:
                if i + window[0] >= 0 and i + window[-1] < len(head_angle_var):
                    extracted_signal = head_angle_var[i + window[0]:i + window[-1] + 1]
                    # extracted_signal = (extracted_signal - np.mean(extracted_signal))/np.std(extracted_signal)
                    signal_chunks = np.vstack((signal_chunks, extracted_signal))
            signal_chunks_tr.append(signal_chunks)  # Array of traces for one ROI by trial
        # Compute STA by trial for one ROI
        for tr_idx, _ in enumerate(trials):
            sta_trial = np.nanmean(signal_chunks_tr[tr_idx], axis=0)
            sta = np.vstack((sta, sta_trial))
        # STA all rois
        signal_chunks_allrois.append(
            np.concatenate(signal_chunks_tr, axis=0))  # List of raw traces for each ROI whole session
        sta_allrois.append(sta)
    return sta_allrois
sta_shuffled_ts = np.array(sta_head_data_shuffled(shuffled_spikes_ts, head_angles_corr, trials, window, var_name))
# Standardize STA
mean_chance = np.nanmean(sta_shuffled_ts, axis=2)
sd_chance = np.nanstd(sta_shuffled_ts, axis=2)
sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
sta = np.zeros((len(sta_allrois), len(trials), len(window)))
sta_shuffled = np.zeros((len(sta_allrois), len(trials), len(window)))
for n in range(len(sta_allrois)):
    for tr in range(len(trials)):
        sta_zs[n, tr] = (sta_allrois[n][tr] - mean_chance[n][tr]) / sd_chance[n][tr]
        sta[n, tr] = sta_allrois[n][tr]
        sta_shuffled[n, tr] = mean_chance[n][tr]
if not os.path.exists(os.path.join(save_path, animal + ' ' + ses_info[0])):
    os.mkdir(os.path.join(save_path, animal + ' ' + ses_info[0]))
np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '_zscored.npy'), sta_zs)
np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '_shuffled.npy'), sta_shuffled)
np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'), sta)
np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses.npy'), trials_ses)
np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses_name.npy'), trials_ses_name)

# Plot the filtered data
trial = 4
trial_data = df_events_extract_rawtrace.loc[
    (df_events_extract_rawtrace['trial'] == trial)]
spk_data = trial_data.loc[trial_data[roi] == 1]
head_data = head_angles_corr.loc[
    (head_angles_corr['trial'] == trial)]
fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
plt.scatter(spk_data['time'], np.repeat(4, len(spk_data['time'])), s=10, color='black')
plt.plot(head_data['time'], zscore(head_data['yaw']), color='black', label='yaw')
plt.plot(head_data['time'], zscore(head_data['roll']), color='teal', label='roll')
plt.plot(head_data['time'], zscore(head_data['pitch']), color='darkblue', label='pitch')
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('Head angle value (a.u.)', fontsize=16)
ax.set_title(roi, fontsize=20)
ax.legend(frameon=False)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# PCA on head data
head_angles_arr = np.array(head_angles_corr.iloc[:, :3])
pca = PCA(n_components=3)
principalComponents_3CP = pca.fit_transform(head_angles_arr)
# Plot 2d
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
idx_bs = np.where(head_angles_corr['trial'] < trials_ses[0, 1])[0]
idx_es = np.where(head_angles_corr['trial'] == trials_ses[1, 0])[0]
idx_ls = np.where(head_angles_corr['trial'] == trials_ses[1, 1])[0]
idx_ae = np.where(head_angles_corr['trial'] == trials_ses[2, 0])[0]
plt.scatter(principalComponents_3CP[idx_bs, 0], principalComponents_3CP[idx_bs, 1], s=1, color='black')
plt.scatter(principalComponents_3CP[idx_es, 0], principalComponents_3CP[idx_es, 1], s=1, color='crimson')
plt.scatter(principalComponents_3CP[idx_ls, 0], principalComponents_3CP[idx_ls, 1], s=1, color='salmon')
plt.scatter(principalComponents_3CP[idx_ae, 0], principalComponents_3CP[idx_ae, 1], s=1, color='dodgerblue')
ax.set_title('First 2 PCs\nexplained variance of ' + str(
    np.round(np.cumsum(pca.explained_variance_ratio_)[1], decimals=3)), fontsize=20)
ax.set_xlabel('PC component 1', fontsize=16)
ax.set_ylabel('PC component 2', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)



