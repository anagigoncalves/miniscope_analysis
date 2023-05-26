# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
fsize = 24

plot_protocol = 'split contra fast'
plot_session = 'S2'

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data_all.xlsx')
computation_time = 60 #seconds
mean_data_animals = []
animal_in = []
sl_animals = []
ds_animals = []
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    if ses_info['protocol'] == plot_protocol and ses_info['session'] == plot_session:
        date = ses_info[3]
        # path inputs
        path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
        path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
        session_type = path.split('\\')[-4].split(' ')[0]
        mscope = miniscope_session_class.miniscope_session(path)
        loco = locomotion_class.loco_class(path_loco)

        # Session data and inputs
        animal = mscope.get_animal_id()
        session = loco.get_session_id()
        # [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
        #  clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
        [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
         clusters_rois, colors_cluster, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
        [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
        animal_in.append(animal)

        # raw signal clustered
        mean_data_trials_clusters = []
        for count_c, c in enumerate(clusters_rois):
            df_events_extract_rawtrace_rois = df_events_extract_rawtrace[c + ['time', 'trial']]
            mean_data_trials = np.zeros(len(trials))
            std_data_trials = np.zeros(len(trials))
            for count_t, t in enumerate(trials):
                df_events_extract_rawtrace_rois_time = df_events_extract_rawtrace_rois.loc[
                    (df_events_extract_rawtrace_rois['time'] <= computation_time) & (df_events_extract_rawtrace_rois['trial'] == t)]
                fr_roi_trials = np.zeros(len(df_events_extract_rawtrace_rois_time.columns[:-2]))
                for count_r, r in enumerate(df_events_extract_rawtrace_rois_time.columns[:-2]):
                    df_events_extract_rawtrace_roi = df_events_extract_rawtrace_rois_time[[r, 'time']]
                    fr_roi_trials[count_r] = 1/np.nanmean(df_events_extract_rawtrace_roi.loc[df_events_extract_rawtrace_roi[r] > 0, 'time'].diff())
                mean_data_trials[count_t] = np.nanmean(fr_roi_trials)
            mean_data_trials_clusters.append(mean_data_trials-np.nanmean(mean_data_trials[trials_baseline-1]))
        mean_data_animals.append(mean_data_trials_clusters)

        # Load behavioral data
        [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                                 frames_dFF)
        filelist = loco.get_track_files(animal, session)
        sl_trials = np.zeros(len(filelist))
        ds_trials = np.zeros(len(filelist))
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
                frames_loco[count_trial]))
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            sl_values = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
            sl_trials[count_trial] = np.nanmean(sl_values[0])-np.nanmean(sl_values[2])
            ds_values = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                                'double_support')
            ds_trials[count_trial] = np.nanmean(ds_values[0]) - np.nanmean(ds_values[2])
        sl_animals.append(sl_trials - np.nanmean(sl_trials[trials_baseline - 1]))
        ds_animals.append(ds_trials - np.nanmean(ds_trials[trials_baseline - 1]))

color_animals = ['black', 'indianred', 'maroon', 'darkorange', 'dodgerblue', 'mediumblue', 'dimgrey', 'darkorchid']
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.add_patch(plt.Rectangle((6.5, -0.5), 10, 1, fc='lightgrey', alpha=0.3))
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax.plot(np.arange(4, 27), mean_data_animals[a][c], marker='o', color=color_animals[a], markersize=5)
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 misisng trial 10 if split contra fast S1
            ax.plot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]), mean_data_animals[a][c], marker='o', color=color_animals[a], markersize=5)
        else:
            ax.plot(np.arange(1, 27), mean_data_animals[a][c], marker='o', color=color_animals[a], markersize=5)
ax.set_xlabel('Trials', fontsize=mscope.fsize - 8)
ax.set_ylabel('Firing rate (baseline subtracted)', fontsize=mscope.fsize - 8)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
plt.savefig(os.path.join(path_session_data, 'fr_allanimals_' + str(0) + 's_' + str(computation_time) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
ax = ax.ravel()
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[0].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3]), s=60,  color=color_animals[a])
        else:
            ax[0].scatter(mean_data_animals[a][c][6], np.abs(sl_animals[a][6]), s=60, color=color_animals[a])
ax[0].set_xlabel('FR (bs) for initial error trial', fontsize=mscope.fsize - 8)
ax[0].set_ylabel('Initial error absolute value (mm)', fontsize=mscope.fsize - 8)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[1].scatter(mean_data_animals[a][c][13], np.abs(sl_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[1].scatter(mean_data_animals[a][c][12], np.abs(sl_animals[a][12]), s=60, color=color_animals[a])
        else:
            ax[1].scatter(mean_data_animals[a][c][16], np.abs(sl_animals[a][16]), s=60, color=color_animals[a])
ax[1].set_xlabel('FR (bs) for after-effect trial', fontsize=mscope.fsize - 8)
ax[1].set_ylabel('After-effect absolute value (mm)', fontsize=mscope.fsize - 8)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3])-np.abs(sl_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3]) - np.abs(sl_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[2].scatter(mean_data_animals[a][c][6], np.abs(sl_animals[a][6])-np.abs(sl_animals[a][16]), s=60, color=color_animals[a])
ax[2].set_xlabel('FR (bs) for initial error trial', fontsize=mscope.fsize - 8)
ax[2].set_ylabel('Amount of adaptation during split (mm)', fontsize=mscope.fsize - 8)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[3].scatter(mean_data_animals[a][c][13], np.abs(sl_animals[a][3])-np.abs(sl_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[3].scatter(mean_data_animals[a][c][12], np.abs(sl_animals[a][3]) - np.abs(sl_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[3].scatter(mean_data_animals[a][c][16], np.abs(sl_animals[a][6])-np.abs(sl_animals[a][16]), s=60, color=color_animals[a])
ax[3].set_xlabel('FR (bs) for after-effect trial', fontsize=mscope.fsize - 8)
ax[3].set_ylabel('Amount of adaptation during split (mm)', fontsize=mscope.fsize - 8)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
plt.savefig(os.path.join(path_session_data, 'fr_slvalues_' + str(0) + 's_' + str(computation_time) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
ax = ax.ravel()
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[0].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3]), s=60,  color=color_animals[a])
        else:
            ax[0].scatter(mean_data_animals[a][c][6], np.abs(ds_animals[a][6]), s=60, color=color_animals[a])
ax[0].set_xlabel('FR (bs) for initial error trial', fontsize=mscope.fsize - 8)
ax[0].set_ylabel('Initial error absolute value (mm)', fontsize=mscope.fsize - 8)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[1].scatter(mean_data_animals[a][c][13], np.abs(ds_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[1].scatter(mean_data_animals[a][c][12], np.abs(sl_animals[a][12]), s=60, color=color_animals[a])
        else:
            ax[1].scatter(mean_data_animals[a][c][16], np.abs(ds_animals[a][16]), s=60, color=color_animals[a])
ax[1].set_xlabel('FR (bs) for after-effect trial', fontsize=mscope.fsize - 8)
ax[1].set_ylabel('After-effect absolute value (mm)', fontsize=mscope.fsize - 8)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3])-np.abs(ds_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3]) - np.abs(ds_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[2].scatter(mean_data_animals[a][c][6], np.abs(ds_animals[a][6])-np.abs(ds_animals[a][16]), s=60, color=color_animals[a])
ax[2].set_xlabel('FR (bs) for initial error trial', fontsize=mscope.fsize - 8)
ax[2].set_ylabel('Amount of adaptation during split (mm)', fontsize=mscope.fsize - 8)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if len(mean_data_animals[a][c]) == 23:
            ax[3].scatter(mean_data_animals[a][c][13], np.abs(ds_animals[a][3])-np.abs(ds_animals[a][13]), s=60,  color=color_animals[a])
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[3].scatter(mean_data_animals[a][c][12], np.abs(ds_animals[a][3]) - np.abs(ds_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[3].scatter(mean_data_animals[a][c][16], np.abs(ds_animals[a][6])-np.abs(ds_animals[a][16]), s=60, color=color_animals[a])
ax[3].set_xlabel('FR (bs) for after-effect trial', fontsize=mscope.fsize - 8)
ax[3].set_ylabel('Amount of adaptation during split (mm)', fontsize=mscope.fsize - 8)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
plt.savefig(os.path.join(path_session_data, 'fr_dsvalues_' + str(0) + 's_' + str(computation_time) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)