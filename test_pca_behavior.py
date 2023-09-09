# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')

f = 'MC9194'
session_data_idx = np.where(session_data['animal'] == f)[0][0]
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

# Load behavioral data
filelist = loco.get_track_files(animal, session)
final_tracks_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
        frames_dFF[count_trial]))
    final_tracks_trials.append(final_tracks)

# NORMAL PCA
chunk_size = 100
for count_t, trial in enumerate(trials):
    len_finaltracks = np.shape(final_tracks_trials[count_t])[-1]
    nr_chunks = np.int64(np.floor(len_finaltracks/chunk_size))
    for p in range(4):
        x_onepaw = mscope.z_score(loco.inpaint_nans(final_tracks_trials[count_t][0, p, :]-np.nanmean(final_tracks_trials[count_t][0, :4, :], axis=0)), 0)
        x_onepaw_reshaped = x_onepaw[:nr_chunks*chunk_size].reshape((nr_chunks, chunk_size))
        if p == 0:
            x_paws_reshaped = x_onepaw_reshaped
            trial_id = np.ones(np.shape(x_paws_reshaped)[0])*trial
        else:
            x_paws_reshaped = np.concatenate((x_paws_reshaped, x_onepaw_reshaped), axis=1)
    if count_t == 0:
        trial_id_all = trial_id
        x_paws_reshaped_all = x_paws_reshaped
    else:
        trial_id_all = np.concatenate((trial_id_all, trial_id), axis=0)
        x_paws_reshaped_all = np.concatenate((x_paws_reshaped_all, x_paws_reshaped), axis=0)

pca_behavior_all = PCA(n_components=3)
pcomp_behavior_all = pca_behavior_all.fit_transform(x_paws_reshaped_all)
exp_var = pca_behavior_all.explained_variance_ratio_

trial_id_baseline = np.where(trial_id_all<7)[0]
trial_id_esplit = np.where(trial_id_all==7)[0]
trial_id_ewashout = np.where(trial_id_all==17)[0]

fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(pcomp_behavior_all[:, 0], pcomp_behavior_all[:, 1], pcomp_behavior_all[:, 2], s=1)
ax.scatter(pcomp_behavior_all[trial_id_baseline, 0], pcomp_behavior_all[trial_id_baseline, 1], pcomp_behavior_all[trial_id_baseline, 2], s=5, color='black')
ax.scatter(pcomp_behavior_all[trial_id_esplit, 0], pcomp_behavior_all[trial_id_esplit, 1], pcomp_behavior_all[trial_id_esplit, 2], s=5, color='red')
ax.scatter(pcomp_behavior_all[trial_id_ewashout, 0], pcomp_behavior_all[trial_id_ewashout, 1], pcomp_behavior_all[trial_id_ewashout, 2], s=5, color='blue')

# CS-ALIGNED PCA
chunk_size = 100
roi = 'ROI1'
for count_t, trial in enumerate(trials):
    time_roi_events = df_events_extract_rawtrace.loc[df_events_extract_rawtrace.index[(df_events_extract_rawtrace[roi]==1)&(df_events_extract_rawtrace['trial']==trial)], 'time']
    idx_roi_events_behavior = [np.argmin(np.abs(bcam_time[count_t] - i)) for i in time_roi_events]
    for p in range(4):
        x_onepaw = mscope.z_score(loco.inpaint_nans(final_tracks_trials[count_t][0, p, :]-np.nanmean(final_tracks_trials[count_t][0, :4, :], axis=0)), 0)
        x_onepaw_reshaped_list = []
        for i in idx_roi_events_behavior:
            if i > 50 or i < len(x_onepaw)-50:
                x_onepaw_reshaped_list.append(x_onepaw[i-50:i+50])
        x_onepaw_reshaped_list_clean = [x for x in x_onepaw_reshaped_list if len(x)>0]
        x_onepaw_reshaped_cs = np.array(x_onepaw_reshaped_list_clean)
        if p == 0:
            x_paws_reshaped_cs_single = x_onepaw_reshaped_cs
            trial_id_cs_single = np.ones(np.shape(x_paws_reshaped_cs_single)[0])*trial
        else:
            x_paws_reshaped_cs_single = np.concatenate((x_paws_reshaped_cs_single, x_onepaw_reshaped_cs), axis=1)
    if count_t == 0:
        trial_id_cs = trial_id_cs_single
        x_paws_reshaped_cs = x_paws_reshaped_cs_single
    else:
        trial_id_cs = np.concatenate((trial_id_cs, trial_id_cs_single), axis=0)
        x_paws_reshaped_cs = np.concatenate((x_paws_reshaped_cs, x_paws_reshaped_cs_single), axis=0)

pca_behavior_cs = PCA(n_components=3)
pcomp_behavior_cs = pca_behavior_cs.fit_transform(x_paws_reshaped_cs)
exp_var_cs = pca_behavior_cs.explained_variance_ratio_

trial_id_baseline_cs = np.where(trial_id_cs<7)[0]
trial_id_esplit_cs = np.where(trial_id_cs==7)[0]
trial_id_ewashout_cs = np.where(trial_id_cs==17)[0]

fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(pcomp_behavior_cs[:, 0], pcomp_behavior_cs[:, 1], pcomp_behavior_cs[:, 2], s=1, color='black')
ax.scatter(pcomp_behavior_cs[trial_id_baseline_cs, 0], pcomp_behavior_cs[trial_id_baseline_cs, 1], pcomp_behavior_cs[trial_id_baseline_cs, 2], s=5, color='black')
ax.scatter(pcomp_behavior_cs[trial_id_esplit_cs, 0], pcomp_behavior_cs[trial_id_esplit_cs, 1], pcomp_behavior_cs[trial_id_esplit_cs, 2], s=5, color='red')
ax.scatter(pcomp_behavior_cs[trial_id_ewashout_cs, 0], pcomp_behavior_cs[trial_id_ewashout_cs, 1], pcomp_behavior_cs[trial_id_ewashout_cs, 2], s=5, color='blue')



