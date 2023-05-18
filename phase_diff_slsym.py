# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt

# path inputs
path = 'J:\\Miniscope processed files\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split ipsi fast S1 050421\\'
session_type = path.split('\\')[-4].split(' ')[0]

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()

frames_dFF = np.load(os.path.join(path, 'processed files', 'black_frames.npy'))
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials, clusters_rois, colors_cluster, idx_roi_cluster_ordered,
 ref_image, frames_dFF] = mscope.load_processed_files()
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
param_trials_fr_mean = np.zeros(len(trials))
stride_duration_trials = []
final_tracks_forwadloco_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
        frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_forwadloco = loco.final_tracks_forwardlocomotion(final_tracks, st_strides_mat)
    final_tracks_forwadloco_trials.append(final_tracks_forwadloco)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(
        loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
    param_trials_fr_mean[count_trial] = np.nanmean(param_trials[-1][0]) - np.nanmean(param_trials[-1][2])
final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                    'st-st')
time_continuous = []
phase_diff_FR_FL_stst = []
phase_diff_FR_HR_stst = []
phase_diff_FR_HL_stst = []
paw_diff_FR_FL = []
paw_diff_FR_HR = []
paw_diff_FR_HL = []
for count_trial, trial in enumerate(trials):
    phase_diff_FR_FL_stst.extend(final_tracks_trials_phase[count_trial][0, 0, :]-final_tracks_trials_phase[count_trial][0, 2, :])
    phase_diff_FR_HR_stst.extend(
        final_tracks_trials_phase[count_trial][0, 0, :]-final_tracks_trials_phase[count_trial][0, 1, :])
    phase_diff_FR_HL_stst.extend(
        final_tracks_trials_phase[count_trial][0, 0, :] - final_tracks_trials_phase[count_trial][0, 3, :])
    paw_diff_FR_FL.extend(final_tracks_trials[count_trial][0, 0, :]-final_tracks_trials[count_trial][0, 2, :])
    paw_diff_FR_HR.extend(
        final_tracks_trials[count_trial][0, 0, :]-final_tracks_trials[count_trial][0, 1, :])
    paw_diff_FR_HL.extend(
        final_tracks_trials[count_trial][0, 0, :] - final_tracks_trials[count_trial][0, 3, :])
    time_continuous.extend(bcam_time[count_trial]+(loco.trial_time*count_trial))

[param_all_idx, param_all_time, param_all] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, 'FR', 'FL', 1, 1)
window = 100
fig, ax = plt.subplots(3, 2, figsize=(10, 10), tight_layout=True)
ax = ax.ravel()
rectangle11 = plt.Rectangle((3,-50), 10, 90, fc='dimgrey',alpha=0.3)
ax[0].add_patch(rectangle11)
rectangle12 = plt.Rectangle((3,-1000), 10, 1300, fc='dimgrey',alpha=0.3)
ax[1].add_patch(rectangle12)
rectangle21 = plt.Rectangle((3,-1), 10, 2, fc='dimgrey',alpha=0.3)
rectangle22 = plt.Rectangle((3,-30), 10, 60, fc='dimgrey',alpha=0.3)
ax[2].add_patch(rectangle21)
ax[3].add_patch(rectangle22)
rectangle31 = plt.Rectangle((3,-80), 10, 160, fc='dimgrey',alpha=0.3)
rectangle32 = plt.Rectangle((3,-6000), 10, 10000, fc='dimgrey',alpha=0.3)
ax[4].add_patch(rectangle31)
ax[5].add_patch(rectangle32)
ax[0].plot(param_all_time / 60, param_all, color='black')
ax[0].set_xlabel('Time (min)')
ax[0].set_title('step length symmetry', fontsize=mscope.fsize - 4)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[2].plot(np.array(time_continuous)/ 60, np.array(phase_diff_FR_FL_stst), color='blue')
ax[2].set_xlabel('Time (min)')
ax[2].set_title('FR-FL phase difference', fontsize=mscope.fsize - 4)
ax[2].spines['right'].set_visible(False)
ax[2].spines['top'].set_visible(False)
ax[4].plot(np.array(time_continuous)/ 60, np.array(paw_diff_FR_FL), color='red')
ax[4].set_xlabel('Time (min)')
ax[4].set_title('FR-FL difference', fontsize=mscope.fsize - 4)
ax[4].spines['right'].set_visible(False)
ax[4].spines['top'].set_visible(False)
ax[1].plot(param_all_time / 60, np.convolve(param_all, np.ones(window), 'same'), color='black')
ax[1].set_xlabel('Time (min)')
ax[1].set_title('step length symmetry', fontsize=mscope.fsize - 4)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[3].plot(np.array(time_continuous)/ 60, np.convolve(np.array(phase_diff_FR_FL_stst), np.ones(window), 'same'), color='blue')
ax[3].set_xlabel('Time (min)')
ax[3].set_title('FR-FL phase difference', fontsize=mscope.fsize - 4)
ax[3].spines['right'].set_visible(False)
ax[3].spines['top'].set_visible(False)
ax[5].plot(np.array(time_continuous)/ 60, np.convolve(np.array(paw_diff_FR_FL), np.ones(window), 'same'), color='red')
ax[5].set_xlabel('Time (min)')
ax[5].set_title('FR-FL difference', fontsize=mscope.fsize - 4)
ax[5].spines['right'].set_visible(False)
ax[5].spines['top'].set_visible(False)
plt.savefig('J:\\Miniscope processed files\\step_length_phasediff_pawdiff_window_'+str(window), dpi=mscope.my_dpi)