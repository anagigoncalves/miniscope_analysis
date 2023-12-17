# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
align_event = 'stance'
align_dimension = 'phase'
protocol = 'split_ipsi_fast'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['red', 'magenta', 'blue', 'cyan']
phase_bins = np.arange(0, 1, 0.05)

for s in range(len(session_data)):
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
    colors_session = mscope.colors_session(animal, session_type, trials, 1)
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    st_strides_trials = []
    sw_strides_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-sw-st')

    #Analysis for each cluster
    for c in range(len(clusters_rois)):
        if len(clusters_rois) == 1:
            clusters_plot = np.append(clusters_rois[c], ['trial', 'time'])
        else:
            clusters_plot = clusters_rois[c]
            clusters_plot.append('trial')
            clusters_plot.append('time')
        df_cluster = df_events_extract_rawtrace[clusters_plot]
        #Compute fraction of co-active ROIs
        data_strides = st_strides_trials
        frac_active_rois_stride_phase_paw = []
        for p1_idx in range(4):
            frac_active_rois_stride_phase = []
            for t, trial in enumerate(trials):
                df_time = np.array(df_cluster.loc[df_cluster['trial'] == trial, 'time'])*1000
                df_trial = df_cluster.loc[df_cluster['trial'] == trial].iloc[:, :-2]
                frac_active_rois = np.array(df_trial.sum(axis=1)/np.shape(df_trial)[1])
                #See how fraction of co-active ROIs changes across the stride cycle
                phase_paw = final_tracks_phase[t][0, p1_idx, :]
                frames_bin, _ = np.histogram(phase_paw[~np.isnan(phase_paw)], bins=len(phase_bins))  # Compute time spent in each bin
                time_bin = frames_bin * (1 / mscope.sr_loco)
                frac_active_rois_stride_phase_trial = np.zeros((np.shape(data_strides[t][p1_idx])[0], len(phase_bins)))
                for s in range(np.shape(data_strides[t][p1_idx])[0]):
                    st_on_time = data_strides[t][p1_idx][s, 0, 0]
                    st_off_time = data_strides[t][p1_idx][s, 1, 0]
                    #frames of neural data corresponding to the stride
                    data_idx_stride = np.where((df_time > st_on_time) & (df_time < st_off_time))[0]
                    #timestamps of neural data corresponding to the stride
                    data_time_stride = df_time[np.where((df_time > st_on_time) & (df_time < st_off_time))[0]]/1000
                    #get closest behavioral frame for each neural data frame within the stride
                    data_frame_stride = np.zeros(len(data_time_stride))
                    for count_i, i in enumerate(data_time_stride):
                        data_frame_stride[count_i] = np.argmin(np.abs(i-bcam_time[t]))
                    phase_data = final_tracks_phase[t][0, p1_idx, np.int64(data_frame_stride)]
                    phase_data_bins = np.digitize(phase_data, phase_bins, right=True)
                    frac_active_rois_stride_phase_trial[s, phase_data_bins-1] = frac_active_rois[data_idx_stride]/time_bin[phase_data_bins-1]
                frac_active_rois_stride_phase.append(frac_active_rois_stride_phase_trial)
            frac_active_rois_stride_phase_paw.append(frac_active_rois_stride_phase)

        fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 10), sharey=True, sharex=True)
        ax = ax.ravel()
        for count_p, p in enumerate(np.array([0, 2, 3, 1])):
            for t, trial in enumerate(trials_ses.flatten()):
                ax[count_p].plot(phase_bins, np.nanmean(frac_active_rois_stride_phase_paw[p][t], axis=0), color=colors_session[trial], linewidth=3)
            ax[count_p].spines['right'].set_visible(False)
            ax[count_p].spines['top'].set_visible(False)
            ax[count_p].tick_params(axis='both', which='major', labelsize=20)
            ax[count_p].set_ylabel('Fraction of co-active\nROIs norm.', fontsize=20)
            ax[count_p].set_title(paws[p], fontsize=20)
            ax[count_p].set_xlabel('% Phase', fontsize=20)
            ax[count_p].axvline(x=0.5, color='black')
        plt.savefig(os.path.join(save_path, 'fraction_co_active_rois_' + align_event + '_' + align_dimension + '_' + protocol + '_' + animal+'cluster'+str(c+1)), dpi=256)
        plt.savefig(os.path.join(save_path, 'fraction_co_active_rois_' + align_event + '_' + align_dimension + '_' + protocol + '_' + animal+'cluster'+str(c+1) + '.svg'), dpi=256)
    plt.close('all')