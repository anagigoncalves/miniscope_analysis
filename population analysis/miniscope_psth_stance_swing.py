# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files\\'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data_all.xlsx')
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
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
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
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)

    traj = 'time'
    time_window = 1 #default 0.2
    bin_number = 50 #default 20
    align_str = ['st', 'sw']
    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    for align in align_str:
        psth_arr_single = []
        animal_id_single = []
        roi_id_single = []
        paw_id_single = []
        trial_id_single = []
        time_align_single = []
        for roi_plot in roi_list:
            for count_p, p in enumerate(paws):
                [cumulative_idx, trial_id, events_stride_trial] = mscope.event_swst_stride(df_events_extract_rawtrace,
                                                                                           st_strides_trials, sw_strides_trials,
                                                                                           align, trials, p, np.int64(roi_plot[3:]),
                                                                                           time_window, traj)
                idx_nan = np.where(~np.isnan(events_stride_trial))[0]
                trial_id_notnan = trial_id[idx_nan]
                for t in range(np.shape(trials_ses)[0]):
                    nr_strides = np.where(trial_id == trials_ses[t, 1])[0][-1]-np.where(trial_id_notnan == trials_ses[t, 0])[0][0]
                    [hist_result, xaxis] = np.histogram(events_stride_trial[np.where(trial_id_notnan == trials_ses[t, 0])[0][0]:
                                                                            np.where(trial_id_notnan == trials_ses[t, 1])[0][-1]], range=(-time_window*1000, time_window*1000), bins=bin_number)
                    time_align_single.extend(xaxis[:-1])
                    psth_arr_single.extend(hist_result/nr_strides)
                    animal_id_single.extend(np.repeat(animal, len(hist_result)))
                    roi_id_single.extend(np.repeat(np.int64(roi_plot[3:]), len(hist_result)))
                    paw_id_single.extend(np.repeat(p, len(hist_result)))
                    trial_id_single.extend(np.repeat(cond_plot[t], len(hist_result)))
        dict_psth_single = {'psth': psth_arr_single, 'time': time_align_single, 'animal': animal_id_single, 'roi': roi_id_single,
                        'paw': paw_id_single, 'trial_type': trial_id_single}
        if align == 'st':
            df_psth_st = pd.DataFrame(dict_psth_single)
        if align == 'sw':
            df_psth_sw = pd.DataFrame(dict_psth_single)

distance_neurons_ordered = np.argsort(distance_neurons[:, 0])
rois_ordered_distance = []
rois_ordered_distance_str = []
for i in distance_neurons_ordered:
    rois_ordered_distance.append(np.int64(roi_list[i][3:]))
    rois_ordered_distance_str.append(roi_list[i])
df_psth_plot = df_psth_st.loc[(df_psth_st['paw'] == 'FR') & (df_psth_st['trial_type'] == 'baseline')]
df_psth_plot_pivot = df_psth_plot.pivot(index='roi', column='time', values='psth')
df_psth_plot_pivot_ordered = df_psth_plot_pivot.loc[rois_ordered_distance]

fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
sns.heatmap(df_psth_plot_pivot_ordered, cmap='viridis')
ax.vlines(bin_number/2, *ax.get_ylim(), color='white', linestyle='dashed')
ax.set_yticks(np.arange(0, len(rois_ordered_distance), 4))
ax.set_yticklabels(rois_ordered_distance_str[::4], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xticks(np.arange(0, len(df_psth_plot_pivot_ordered.columns), 4))
ax.set_xticklabels(df_psth_plot_pivot_ordered.columns[::4], rotation=45, fontsize=mscope.fsize - 12)
ax.set_xlabel('Time (ms)', fontsize=mscope.fsize - 4)

