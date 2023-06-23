# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
fsize = 24

plot_protocol = 'split ipsi fast'
plot_session = 'S1'

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'E:\\Miniscope processed files'
session_data = pd.read_excel('E:\\Miniscope processed files\\session_data_split_S1.xlsx')
mean_data_animals = []
animal_in = []
sl_animals = []
ds_animals = []
coo_animals = []
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
        [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
         clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
        [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
        animal_in.append(animal)

        # Order ROIs by cluster
        if len(clusters_rois) == 1:
            clusters_rois_flat = clusters_rois[0]
        else:
            clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
        df_events_extract_zscore_clustered = df_events_extract_rawtrace[clusters_rois_flat]
        df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')

        # raw signal clustered
        time_beg_vec = np.arange(0, 60, 5)
        time_end_vec = np.arange(5, 60+5, 5)
        w = 0 # only from 0 to 5 seconds for each trial
        mean_data_trials_rois = []
        for count_c, c in enumerate(df_extract_rawtrace_detrended_zscore.columns[2:]):
            data_trials = np.zeros(len(trials))
            for count_t, t in enumerate(trials):
                data_trials[count_t] = df_extract_rawtrace_detrended_zscore.loc[
                                  df_extract_rawtrace_detrended_zscore['trial'] == t, c].iloc[time_beg_vec[w]*mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
            mean_data_trials_rois.append(data_trials-np.nanmean(data_trials[trials_baseline-1]))
        mean_data_animals.append(mean_data_trials_rois)

        # Load behavioral data
        [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                                 frames_dFF)
        filelist = loco.get_track_files(animal, session)
        sl_trials = np.zeros(len(filelist))
        ds_trials = np.zeros(len(filelist))
        coo_trials = np.zeros(len(filelist))
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
            coo_values = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                                'coo')
            coo_trials[count_trial] = np.nanmean(coo_values[0]) - np.nanmean(coo_values[2])
        sl_animals.append(sl_trials-np.nanmean(sl_trials[trials_baseline-1]))
        coo_animals.append(coo_trials - np.nanmean(coo_trials[trials_baseline - 1]))
        ds_animals.append(ds_trials-np.nanmean(ds_trials[trials_baseline-1]))

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
def get_colors_plot(animal_name, color_animals):
    if animal_name=='MC8855':
        color_plot = color_animals[0]
    if animal_name=='MC9194':
        color_plot = color_animals[1]
    if animal_name=='MC10221':
        color_plot = color_animals[2]
    if animal_name=='MC9513':
        color_plot = color_animals[3]
    if animal_name=='MC9226':
        color_plot = color_animals[4]
    return color_plot

fig, ax = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
ax = ax.ravel()
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[0].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[0].scatter(mean_data_animals[a][c][6], np.abs(sl_animals[a][6]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[0].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[0].set_ylabel('Initial error \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[1].scatter(mean_data_animals[a][c][13], np.abs(sl_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[1].scatter(mean_data_animals[a][c][12], np.abs(sl_animals[a][12]), s=60, color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[1].scatter(mean_data_animals[a][c][16], np.abs(sl_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[1].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[1].set_ylabel('After-effect \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3])-np.abs(sl_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(sl_animals[a][3]) - np.abs(sl_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[2].scatter(mean_data_animals[a][c][6], np.abs(sl_animals[a][6])-np.abs(sl_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[2].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[2].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[2].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[3].scatter(mean_data_animals[a][c][13], np.abs(sl_animals[a][3])-np.abs(sl_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[3].scatter(mean_data_animals[a][c][12], np.abs(sl_animals[a][3]) - np.abs(sl_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[3].scatter(mean_data_animals[a][c][16], np.abs(sl_animals[a][6])-np.abs(sl_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[3].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[3].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[3].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
plt.savefig(os.path.join(path_session_data, 'avg_activity_slvalues_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
ax = ax.ravel()
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[0].scatter(mean_data_animals[a][c][3], np.abs(coo_animals[a][3]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[0].scatter(mean_data_animals[a][c][6], np.abs(coo_animals[a][6]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[0].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[0].set_ylabel('Initial error \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[1].scatter(mean_data_animals[a][c][13], np.abs(coo_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[1].scatter(mean_data_animals[a][c][12], np.abs(coo_animals[a][12]), s=60, color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[1].scatter(mean_data_animals[a][c][16], np.abs(coo_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[1].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[1].set_ylabel('After-effect \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(coo_animals[a][3])-np.abs(coo_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(coo_animals[a][3]) - np.abs(coo_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[2].scatter(mean_data_animals[a][c][6], np.abs(coo_animals[a][6])-np.abs(coo_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[2].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[2].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[2].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[3].scatter(mean_data_animals[a][c][13], np.abs(coo_animals[a][3])-np.abs(coo_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[3].scatter(mean_data_animals[a][c][12], np.abs(coo_animals[a][3]) - np.abs(coo_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[3].scatter(mean_data_animals[a][c][16], np.abs(coo_animals[a][6])-np.abs(coo_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[3].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[3].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[3].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
plt.savefig(os.path.join(path_session_data, 'avg_activity_coovalues_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)

fig, ax = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
ax = ax.ravel()
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[0].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[0].scatter(mean_data_animals[a][c][6], np.abs(ds_animals[a][6]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[0].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[0].set_ylabel('Initial error \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[0].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[1].scatter(mean_data_animals[a][c][13], np.abs(ds_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[1].scatter(mean_data_animals[a][c][12], np.abs(ds_animals[a][12]), s=60, color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax[1].scatter(mean_data_animals[a][c][16], np.abs(ds_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[1].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[1].set_ylabel('After-effect \nabsolute value (mm)', fontsize=mscope.fsize - 4)
ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3])-np.abs(ds_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[2].scatter(mean_data_animals[a][c][3], np.abs(ds_animals[a][3]) - np.abs(ds_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[2].scatter(mean_data_animals[a][c][6], np.abs(ds_animals[a][6])-np.abs(ds_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[2].set_xlabel('Calcium signal \nfor initial error trial', fontsize=mscope.fsize - 4)
ax[2].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[1].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 0: #if MC8855
            ax[3].scatter(mean_data_animals[a][c][13], np.abs(ds_animals[a][3])-np.abs(ds_animals[a][13]), s=60,  color=get_colors_plot(animal_in[a], color_animals))
        elif len(mean_data_animals[a][c]) == 25: #if mc10221 missing trial 10 if split contra fast S1
            ax[3].scatter(mean_data_animals[a][c][12], np.abs(ds_animals[a][3]) - np.abs(ds_animals[a][12]), s=60,
                          color=color_animals[a])
        else:
            ax[3].scatter(mean_data_animals[a][c][16], np.abs(ds_animals[a][6])-np.abs(ds_animals[a][16]), s=60, color=get_colors_plot(animal_in[a], color_animals))
ax[3].set_xlabel('Calcium signal \nfor after-effect trial', fontsize=mscope.fsize - 4)
ax[3].set_ylabel('Amount of adaptation \nduring split (mm)', fontsize=mscope.fsize - 4)
ax[3].tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
plt.savefig(os.path.join(path_session_data, 'avg_activity_dsvalues_' + str(time_beg_vec[0]) + 's_' + str(time_end_vec[0]) + 's_' + plot_protocol + '_' + plot_session), dpi=mscope.my_dpi)