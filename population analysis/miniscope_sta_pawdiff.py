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

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
if not os.path.exists(path_session_data + 'STA difference between paws'):
    os.mkdir(path_session_data + 'STA difference between paws')
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
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    # Order ROIs mediolateral
    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    centroids_mediolateral = []
    for c in range(len(centroid_ext)):
        centroids_mediolateral.append(centroid_ext[c][0])
    distance_neurons_ordered = np.argsort(centroids_mediolateral)
    rois_ordered_distance_str = []
    for i in distance_neurons_ordered:
        rois_ordered_distance_str.append(roi_list[i])

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    paws_rel_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        paws_rel_trials.append(paws_rel)

    window = 0.2
    window_ave = 0.05 #window for averaging the CS
    paw_sta_mean_cstime_rois = np.zeros((len(rois_ordered_distance_str), len(trials)))
    paw_sta_mean_cswindow_rois = np.zeros((len(rois_ordered_distance_str), len(trials)))
    paw_sta_mean_cstime_windowave_rois = np.zeros((len(rois_ordered_distance_str), len(trials)))
    paw_sta_mean_cstime_wholewindowave_rois = np.zeros((len(rois_ordered_distance_str), len(trials)))
    for count_roi, roi in enumerate(rois_ordered_distance_str):
        paw_diff_sta_mean = np.zeros((len(trials), np.int64(window * 2 * mscope.sr_loco)))
        paw_diff_sta_std = np.zeros((len(trials), np.int64(window * 2 * mscope.sr_loco)))
        for count_trial, trial_sta in enumerate(trials):
            trial_idx = np.where(trials == trial_sta)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (df_events_extract_rawtrace['trial'] == trial_sta), 'time'])
            paw_diff_trial = paws_rel_trials[trial_idx][0]-paws_rel_trials[trial_idx][2]
            paw_diff_sta = np.zeros((len(events_trial), np.int64(window*2*mscope.sr_loco)))
            paw_diff_sta[:] = np.nan
            for count_e, e in enumerate(events_trial):
                if (e >= window) and e <= (bcam_trial[-1]-0.2):
                    window_sta = np.array([e-window, e+window])
                    window_sta_idx = np.array([np.argmin(np.abs(window_sta[0]-bcam_trial)), np.argmin(np.abs(window_sta[1]-bcam_trial))])
                    if len(paw_diff_trial[window_sta_idx[0]:window_sta_idx[1]]) == np.int64(window*2*mscope.sr_loco): #sometimes is one more or one less....
                        paw_diff_sta[count_e, :] = paw_diff_trial[window_sta_idx[0]:window_sta_idx[1]]
            paw_diff_sta_mean[count_trial, :] = np.nanmean(paw_diff_sta, axis=0)
            paw_diff_sta_std[count_trial, :] = np.nanstd(paw_diff_sta, axis=0)
            # fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
            # sns.heatmap(paw_diff_sta, cmap='viridis')
            # ax.vlines(np.int64(window*2*mscope.sr_loco) / 2, *ax.get_ylim(), color='white', linestyle='dashed')
            # ax.set_yticks(np.arange(0, len(events_trial), 8))
            # ax.set_yticklabels(list(map(str, np.arange(0, len(events_trial), 8))))
            # ax.set_xticks(np.array([0, 66, 132]))
            # ax.set_xticklabels(['-200', '0', '200'], rotation=45, fontsize=mscope.fsize - 10)
            # ax.set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            # ax.set_ylabel('Event #', fontsize=mscope.fsize - 8)
            # ax.set_title('STA of FR-FL', fontsize=mscope.fsize - 8)
        paw_sta_mean_cstime = np.zeros(len(trials))
        paw_sta_mean_cstime_windowave = np.zeros(len(trials))
        paw_sta_mean_cstime_wholewindowave = np.zeros(len(trials))
        paw_sta_std_cstime = np.zeros(len(trials))
        for count_t, t in enumerate(trials):
            paw_sta_mean_cstime[count_t] = paw_diff_sta_mean[count_t, np.int64(window*mscope.sr_loco)]
            paw_sta_std_cstime[count_t] = paw_diff_sta_std[count_t, np.int64(window*mscope.sr_loco)]
            paw_sta_mean_cstime_windowave[count_t] = np.nanmean(paw_diff_sta_mean[count_t, np.int64(window*mscope.sr_loco)-np.int64(window_ave*mscope.sr_loco):np.int64(window*mscope.sr_loco)+np.int64(window_ave*mscope.sr_loco)])
            paw_sta_mean_cstime_wholewindowave[count_t] = np.nanmean(paw_diff_sta_mean[count_t, :])
        paw_sta_mean_cstime_rois[count_roi, :] = paw_sta_mean_cstime
        paw_sta_mean_cstime_windowave_rois[count_roi, :] = paw_sta_mean_cstime_windowave
        paw_sta_mean_cstime_wholewindowave_rois[count_roi, :] = paw_sta_mean_cstime_wholewindowave

        # fig, ax = plt.subplots(figsize=(7, 3), tight_layout=True)
        # for count_t, t in enumerate(trials):
        #     ax.plot(np.linspace(-window, window, np.shape(paw_diff_sta_mean)[1]), paw_diff_sta_mean[count_t, :], color=colors_session[t], linewidth=2)
        # ax.vlines(0, *ax.get_ylim(), color='black', linestyle='dashed')
        # ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
        # plt.xticks(fontsize=mscope.fsize - 4)
        # plt.yticks(fontsize=mscope.fsize - 4)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # if not os.path.exists(os.path.join(path_session_data + 'STA difference between paws', animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session))):
        #     os.mkdir(os.path.join(path_session_data + 'STA difference between paws', animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session)))
        # plt.savefig(os.path.join(path_session_data, 'STA difference between paws',  animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session), roi + '_FR-FL'), dpi=mscope.my_dpi)
        # plt.close('all')

    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    sns.heatmap(paw_sta_mean_cstime_rois, cmap='viridis')
    ax.set_yticks(np.arange(0, len(rois_ordered_distance_str), 8))
    ax.set_yticklabels(list(map(str, rois_ordered_distance_str[::8])), rotation=45, fontsize=mscope.fsize - 10)
    ax.set_xticks(trials)
    ax.set_xticklabels(list(map(str,trials)), fontsize=mscope.fsize - 10, rotation=45)
    ax.set_xlabel('Trials', fontsize=mscope.fsize - 8)
    ax.set_title('average FR-FL around calcium event', fontsize=mscope.fsize - 8)
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws', animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    sns.heatmap(paw_sta_mean_cstime_windowave_rois, cmap='viridis')
    ax.set_yticks(np.arange(0, len(rois_ordered_distance_str), 8))
    ax.set_yticklabels(list(map(str, rois_ordered_distance_str[::8])), rotation=45, fontsize=mscope.fsize - 10)
    ax.set_xticks(trials)
    ax.set_xticklabels(list(map(str,trials)), fontsize=mscope.fsize - 10, rotation=45)
    ax.set_xlabel('Trials', fontsize=mscope.fsize - 8)
    ax.set_title('average FR-FL 50ms around calcium event', fontsize=mscope.fsize - 8)
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws', animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_windowave50msbefore'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    sns.heatmap(paw_sta_mean_cstime_wholewindowave_rois, cmap='viridis')
    ax.set_yticks(np.arange(0, len(rois_ordered_distance_str), 8))
    ax.set_yticklabels(list(map(str, rois_ordered_distance_str[::8])), rotation=45, fontsize=mscope.fsize - 10)
    ax.set_xticks(trials)
    ax.set_xticklabels(list(map(str,trials)), fontsize=mscope.fsize - 10, rotation=45)
    ax.set_xlabel('Trials', fontsize=mscope.fsize - 8)
    ax.set_title('average FR-FL 200ms around calcium event', fontsize=mscope.fsize - 8)
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws', animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_windowave200msbefore'), dpi=mscope.my_dpi)
    plt.close('all')


# fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
# if session_type == 'split':
#     # rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(paw_sta_mean_cstime-paw_sta_std_cstime)), 10,
#     # max(paw_sta_mean_cstime+paw_sta_std_cstime) - min(paw_sta_mean_cstime-paw_sta_std_cstime), fc='grey', alpha=0.3)
#     rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(paw_sta_mean_cstime)), 10,
#     max(paw_sta_mean_cstime) - min(paw_sta_mean_cstime), fc='grey', alpha=0.3)
#     ax.add_patch(rectangle)
# ax.plot(trials, paw_sta_mean_cstime, color='black')
# # ax.fill_between(trials, paw_sta_mean_cstime-paw_sta_std_cstime, paw_sta_mean_cstime+paw_sta_std_cstime, color='black',alpha=0.3)
# for count_t, t in enumerate(trials):
#     idx_trial = np.where(trials == t)[0][0]
#     ax.scatter(t, paw_sta_mean_cstime[idx_trial], s=80, color=colors_session[t])
# ax.set_xlabel('Trials', fontsize=16)
# ax.set_ylabel('average FR-FL around calcium event', fontsize=16)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)





