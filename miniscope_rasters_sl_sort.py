# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'D:\\Miniscope processed files'
session_data = pd.read_excel('D:\\session_data.xlsx')
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
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
    traces_type = 'raw'
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    colors_session = mscope.colors_session(session_type, trials, 1)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
    if session_type == 'split':
        colors_phases = ['black', 'crimson', 'teal']
    if session_type == 'tied':
        colors_phases = ['black', 'orange', 'purple']
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
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
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        final_tracks_forwadloco = loco.final_tracks_forwardlocomotion(final_tracks, st_strides_mat)
        final_tracks_forwadloco_trials.append(final_tracks_forwadloco)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
        param_trials_fr_mean[count_trial] = np.nanmean(param_trials[-1][0])-np.nanmean(param_trials[-1][2])
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')

    traj = 'time'
    time_window = 0.2
    sym = 1
    remove_nan = 0
    for cluster_plot in np.arange(1, len(clusters_rois)+1):
        if plot_data:
            align_str = ['st', 'sw']
            for align in align_str:
                # raster
                fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
                ax = ax.ravel()
                for count_p, p in enumerate(paws):
                    [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
                    ax[count_p].scatter(events_stride_trial, cumulative_idx, s=1, color='black')
                    ax[count_p].axvline(x=0, color='black')
                    ax[count_p].axhline(y=np.where(trial_id == trials_ses[0, 1])[0][-1], color='black', linestyle='dashed')
                    ax[count_p].axhline(y=np.where(trial_id == trials_ses[1, 1])[0][-1], color='black', linestyle='dashed')
                    ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
                    ax[count_p].set_ylabel('Aligned to ' + str(align), fontsize=mscope.fsize - 8)
                    ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
                    ax[count_p].spines['right'].set_visible(False)
                    ax[count_p].spines['top'].set_visible(False)
                    ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
                if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type,)):
                    os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type))
                if not os.path.exists(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot))):
                    os.mkdir(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot)))
                if print_plots:
                    plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_raster_trial_order_' + align + '_' + traces_type), dpi=mscope.my_dpi)
                # histogram for the different session phases
                fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
                ax = ax.ravel()
                for count_p, p in enumerate(paws):
                    [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters,
                                                                                               st_strides_trials, sw_strides_trials,
                                                                                               align, trials, p, cluster_plot,
                                                                                               time_window, traj)
                    for t in range(np.shape(trials_ses)[0]):
                        [hist_result, xaxis] = np.histogram(events_stride_trial[np.where(trial_id == trials_ses[t, 0])[0][0]:
                                                                                np.where(trial_id == trials_ses[t, 1])[0][-1]],
                                                            range=(-time_window * 1000, time_window * 1000), bins=20)
                        ax[count_p].plot(xaxis[:-1], hist_result, color=colors_phases[t], linewidth=2)
                        # ax[count_p].plot(xaxis[:-1], hist_result / np.nanmax(hist_result), color=colors_phases[t], linewidth=2)
                    ax[count_p].axvline(x=0, color='black')
                    ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
                    ax[count_p].set_ylabel('Aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
                    ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
                    ax[count_p].spines['right'].set_visible(False)
                    ax[count_p].spines['top'].set_visible(False)
                    ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
                if print_plots:
                    plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_hist_trial_order_' + align + '_' + traces_type), dpi=mscope.my_dpi)
                # raster sorted by sl sym directional
                fig, ax = plt.subplots(1, 4, figsize=(20, 5), tight_layout=True)
                ax = ax.ravel()
                for count_p, p in enumerate(paws):
                    [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
                    if p == 'FR':
                        p2 = 'FL'
                    if p == 'HR':
                        p2 = 'HL'
                    if p == 'FL':
                        p2 = 'FR'
                    if p == 'HL':
                        p2 = 'HR'
                    [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, p, p2, sym, remove_nan)  # SL symmetry for each stride
                    sl_idx_all_sorted = np.argsort(sl_sym_all_array)
                    ax[count_p].scatter(events_stride_trial[sl_idx_all_sorted], sl_sym_all_array, s=1, color='black')
                    ax[count_p].axvline(x=0, color='black')
                    ax[count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
                    ax[count_p].set_ylabel('Aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
                    ax[count_p].set_title(p + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
                    ax[count_p].spines['right'].set_visible(False)
                    ax[count_p].spines['top'].set_visible(False)
                    ax[count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
                if print_plots:
                    plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_raster_sl_sym_' + align + '_' + traces_type), dpi=mscope.my_dpi)
                # raster sorted by sl sym directional -
                window = 200
                fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
                for count_p, p in enumerate(paws):
                    [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, align, trials, p, cluster_plot, time_window, traj)
                    if p == 'FR':
                        p2 = 'FL'
                    if p == 'HR':
                        p2 = 'HL'
                    if p == 'FL':
                        p2 = 'FR'
                    if p == 'HL':
                        p2 = 'HR'
                    [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, p, p2, sym, remove_nan)  # SL symmetry for each stride
                    sl_idx_all_sorted = np.argsort(sl_sym_all_array)
                    events_near_stsw_idx = np.where((events_stride_trial[sl_idx_all_sorted] > -50) & (events_stride_trial[sl_idx_all_sorted] < 50))[0]
                    sl_sym_stsw = sl_sym_all_array[sl_idx_all_sorted][events_near_stsw_idx]
                    sl_sym_stsw_notnan = sl_sym_stsw[~np.isnan(sl_sym_stsw)]
                    events_stride_trial_binary = np.zeros(len(events_stride_trial))
                    events_stride_trial_binary[events_near_stsw_idx] = 1
                    notnan_sl_sym = np.where(~np.isnan(sl_sym_stsw))[0]
                    events_ma = np.convolve(events_stride_trial_binary[notnan_sl_sym], np.ones(window) / window, 'same')
                    events_ma_notnan = events_ma[np.where(~np.isnan(sl_sym_stsw))[0]]
                    # [hist_result, xaxis] = np.histogram(sl_sym_stsw_notnan, bins=20)
                    # ax.plot(xaxis[:-1], hist_result / np.nanmax(hist_result), color=paw_colors[count_p], linewidth=2)
                    ax.plot(sl_sym_stsw_notnan, events_ma_notnan, color=paw_colors[count_p], linewidth=2)
                ax.axvline(x=0, color='black')
                ax.set_xlabel('Step length symmetry', fontsize=mscope.fsize - 8)
                ax.set_ylabel('M. A. of events aligned to ' + str(align) + ' sorted by sl symmetry', fontsize=mscope.fsize - 8)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
                if print_plots:
                    plt.savefig(os.path.join(mscope.path, 'images', 'cluster', traces_type, 'Cluster' + str(cluster_plot), 'Cluster' + str(cluster_plot) + '_summary_sl_sym_' + align + '_' + traces_type), dpi=mscope.my_dpi)
                plt.close('all')