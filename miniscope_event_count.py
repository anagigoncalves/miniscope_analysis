# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import auc

version_mscope = 'v4'
plot_data = 1
print_plots = 1
save_data = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'E:\\Miniscope processed files'
session_data = pd.read_excel('E:\\session_data.xlsx')
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

    # FOV coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

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

    for cluster_plot in np.arange(1, len(clusters_rois) + 1):
        # Event count
        mscope.get_event_count_wholetrial(df_events_trace_clusters, traces_type, colors_session, trials, cluster_plot, plot_data, print_plots)
        event_count_loco = mscope.get_event_count_locomotion(df_events_trace_clusters, traces_type, colors_session, trials, bcam_time, st_strides_trials, cluster_plot, plot_data, print_plots)
        # Proportion of events in strides
        align = 'stride'
        for p in paws:
            df_events_stride_all = mscope.events_stride(df_events_trace_clusters, st_strides_trials, sw_strides_trials, p, cluster_plot, align, save_data)
            event_probability = mscope.event_probability_plot(df_events_stride_all, df_events_trace_clusters, traces_type, colors_session, p, cluster_plot, plot_data, print_plots)
        print(bam)