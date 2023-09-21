# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\split ipsi fast S1\\'
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\Phase difference behavior\\'
N_trials = 26

paws_2 = ['FR', 'HR', 'FL', 'HL']
phase_diff_fr = np.zeros((np.shape(session_data)[0], N_trials, 4))
phase_diff_hl = np.zeros((np.shape(session_data)[0], N_trials, 4))
phase_diff_fr[:] = np.nan
phase_diff_hl[:] = np.nan
for session_data_idx in range(np.shape(session_data)[0]):
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
    [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
        mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

    # PLOT STEP-LENGTH SYMMETRY
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    st_strides_trials = []
    sw_strides_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, np.int64(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')

    if animal == 'MC8855':
        trials_new = trials + 3
        trials_idx = np.where(np.in1d(np.arange(N_trials+1), trials_new))[0]
    else:
        trials_idx = np.where(np.in1d(np.arange(N_trials+1), trials))[0]

    for p in range(len(paws_2)):
        paw_diff_fr_trials = loco.phase_diff(final_tracks_phase, paws_2[p], 'FR', 'X')
        paw_diff_hl_trials = loco.phase_diff(final_tracks_phase, paws_2[p], 'HL', 'X')
        for count_t, t in enumerate(trials_idx):
            phase_diff_fr[session_data_idx, t-1, p] = np.rad2deg(np.nanmean(paw_diff_fr_trials[count_t]))
            phase_diff_hl[session_data_idx, t-1, p] = np.rad2deg(np.nanmean(paw_diff_hl_trials[count_t]))

ae_idx = 16
colors_paws = ['red', 'magenta', 'blue', 'cyan']
fig, ax = plt.subplots(figsize=(5, 10), tight_layout = True)
for p in range(4):
    ax.plot(np.arange(1, N_trials+1), np.nanmean(phase_diff_fr[:, :, p], axis=0), color=colors_paws[p], linewidth=2)
    ax.fill_between(np.arange(1, N_trials + 1), np.nanmean(phase_diff_fr[:, :, p], axis=0)-np.nanstd(phase_diff_fr[:, :, p], axis=0),
    np.nanmean(phase_diff_fr[:, :, p], axis=0)+np.nanstd(phase_diff_fr[:, :, p], axis=0), color=colors_paws[p], alpha=0.3)
ax.axvline(x=6.5, linestyle='dashed', color='black')
ax.axvline(x=16.5, linestyle='dashed', color='black')
ax.set_xlabel('Trials', fontsize=20)
ax.set_ylabel('Phase difference (deg)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'paw_diff_fr_split_ipsi_fast_S1'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout = True)
for p in range(4):
    ax.scatter(np.ones(len(phase_diff_fr[:, ae_idx, p]))*p, phase_diff_fr[:, ae_idx, p]-np.nanmean(phase_diff_fr[:, :6, p], axis=1), color=colors_paws[p])
ax.set_xlabel('Paws', fontsize=20)
ax.set_ylabel('Phase difference (bs)\nfor after-effect trial', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'paw_diff_fr_ae_split_ipsi_fast_S1'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(5, 10), tight_layout = True)
for p in range(4):
    ax.plot(np.arange(1, N_trials+1), np.nanmean(phase_diff_hl[:, :, p], axis=0), color=colors_paws[p], linewidth=2)
    ax.fill_between(np.arange(1, N_trials + 1), np.nanmean(phase_diff_hl[:, :, p], axis=0)-np.nanstd(phase_diff_hl[:, :, p], axis=0),
    np.nanmean(phase_diff_hl[:, :, p], axis=0)+np.nanstd(phase_diff_hl[:, :, p], axis=0), color=colors_paws[p], alpha=0.3)
ax.axvline(x=6.5, linestyle='dashed', color='black')
ax.axvline(x=16.5, linestyle='dashed', color='black')
ax.set_xlabel('Trials', fontsize=20)
ax.set_ylabel('Phase difference (deg)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'paw_diff_hl_split_ipsi_fast_S1'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout = True)
for p in range(4):
    ax.scatter(np.ones(len(phase_diff_hl[:, ae_idx, p]))*p, phase_diff_hl[:, ae_idx, p]-np.nanmean(phase_diff_hl[:, :6, p], axis=1), color=colors_paws[p])
ax.set_xlabel('Paws', fontsize=20)
ax.set_ylabel('Phase difference (bs)\nfor after-effect trial', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'paw_diff_hl_ae_split_ipsi_fast_S1'), dpi=mscope.my_dpi)

#phase_diff_hl[:, ae_idx, 0]-np.nanmean(phase_diff_hl[:, :6, 0], axis=1)
#no ae for hl phase for MC10221 and MC9513