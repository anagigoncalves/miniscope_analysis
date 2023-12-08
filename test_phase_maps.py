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
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\phase maps st-sw-st\\'
N_trials = 26

for session_data_idx in range(np.shape(session_data)[0]):
# session_data_idx = 1
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
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-sw-st')

# plt.figure()
# plt.scatter(np.arange(0, 100), final_tracks_phase[0][0, 0, 1000:1100])
# plt.scatter(np.arange(0, 100), final_tracks_phase[0][0, 2, 1000:1100])

    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    colors_cluster_rois = []
    for i in idx_roi_cluster_ordered:
        colors_cluster_rois.append(colors_cluster[i-1])

    def get_cs_index_behavior_index(df_events, trials, roi_list):
        cs_idx_bcamtime_trials = []
        for trial in trials:
            trial_idx = np.where(trials == trial)[0][0]
            cs_idx_bcamtime = []
            for count_r, r in enumerate(roi_list):
                data_roi = df_events.loc[df_events['trial'] == trial, r]
                data_time = np.array(df_events.loc[df_events['trial'] == trial, 'time'])
                cs_time = data_time[np.where(data_roi)[0]]
                bcam_idx_match = [np.argmin(np.abs(i-bcam_time[trial_idx])) for i in cs_time]
                cs_idx_bcamtime.append(bcam_idx_match)
            cs_idx_bcamtime_trials.append(cs_idx_bcamtime)
        return cs_idx_bcamtime_trials

    cs_idx_bcamtime_trials = get_cs_index_behavior_index(df_events_extract_rawtrace, trials, roi_list)

    trial_list = np.array([trials_baseline[-1], trials_split[0], trials_split[-1], trials_washout[0], trials_washout[-1]])
    trial_list_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
    phase_list = np.array([[0, 1], [0, 2], [0, 3]])
    phase_list_name = ['FR-HR', 'FR-FL', 'FR-HL']
    for phase in range(np.shape(phase_list)[0]):
        fig, ax = plt.subplots(1, 5, figsize=(25,5), tight_layout=True)
        ax = ax.ravel()
        for count_t, t in enumerate(trial_list):
            trial_idx = np.where(trials == t)[0][0]
            ax[count_t].scatter(final_tracks_phase[trial_idx][0, phase_list[phase, 0], :], final_tracks_phase[trial_idx][0, phase_list[phase, 1], :], s=1, color='darkgray')
            for count_r, r in enumerate(roi_list):
                ax[count_t].scatter(final_tracks_phase[trial_idx][0, phase_list[phase, 0], cs_idx_bcamtime_trials[trial_idx][count_r]], final_tracks_phase[trial_idx][0, phase_list[phase, 1], cs_idx_bcamtime_trials[trial_idx][count_r]], s=5,
                       color=colors_cluster_rois[count_r])
            ax[count_t].set_xlabel(phase_list_name[phase][:phase_list_name[phase].find('-')] + ' st-sw-st', fontsize=20)
            ax[count_t].set_ylabel(phase_list_name[phase][phase_list_name[phase].find('-')+1:] + ' st-sw-st', fontsize=20)
            ax[count_t].spines['right'].set_visible(False)
            ax[count_t].spines['top'].set_visible(False)
            ax[count_t].tick_params(axis='both', which='major', labelsize=18)
            ax[count_t].set_title(trial_list_name[count_t], fontsize=20)
        plt.savefig(os.path.join(save_path, animal + '_split_ipsi_fast_S1_' + phase_list_name[phase] + '_st-sw-st'), dpi=mscope.my_dpi)
    plt.close('all')

