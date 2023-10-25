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
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\tied baseline S1\\'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['red', 'magenta', 'blue', 'cyan']
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 1.01, 0.05)  # 5 deg
    align_event = 'st' #is always stance
    phase_paws = 'st-sw-st' #can be also 'st-st', need to write code for sw-sw
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
    phase_paws = 'st-sw-st'  # can be also 'st-st', need to write code for sw-sw
    bins_fr = bins*1000

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

    if not os.path.exists(os.path.join(save_path, animal + ' ' + ses_info[0])):
        os.mkdir(os.path.join(save_path, animal + ' ' + ses_info[0]))

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
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, phase_paws)

    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    cumulative_idx_rois = []
    trial_id_rois = []
    events_stride_trial_rois = []
    firing_rate_rois = []
    spike_prob_rois = []
    for roi in roi_list:
        cumulative_idx_all_paws = []
        trial_id_all_paws = []
        events_stride_trial_all_paws = []
        firing_rate_all_paws = []
        spike_prob_all_paws = []
        for count_p, paw in enumerate(paws):
            [cumulative_idx_paw, trial_id_paw, events_stride_trial_paw] = mscope.event_swst_stride(df_events_extract_rawtrace,
            st_strides_trials, sw_strides_trials, final_tracks_phase, bcam_time, align_dimension, align_event,
            trials, paw, roi, np.abs(bins[0]))
            firing_rate_paw, spike_prob_paw = mscope.firing_rate_swst(events_stride_trial_paw, trial_id_paw, final_tracks_phase, trials, bins_fr, align_dimension)
            cumulative_idx_all_paws.append(cumulative_idx_paw)
            trial_id_all_paws.append(trial_id_paw)
            events_stride_trial_all_paws.append(events_stride_trial_paw)
            firing_rate_all_paws.append(firing_rate_paw)
            spike_prob_all_paws.append(spike_prob_paw)
        cumulative_idx_rois.append(cumulative_idx_all_paws)
        trial_id_rois.append(trial_id_all_paws)
        events_stride_trial_rois.append(events_stride_trial_all_paws)
        firing_rate_rois.append(firing_rate_all_paws)
        spike_prob_rois.append(spike_prob_all_paws)

    np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                         'raster_events_stride_trial_rois'), events_stride_trial_rois, allow_pickle=True)
    np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                         'raster_trial_id_rois'), trial_id_rois, allow_pickle=True)
    np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                         'raster_cumulative_idx_rois'), cumulative_idx_rois, allow_pickle=True)
    np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                         'raster_firing_rate_rois'), firing_rate_rois, allow_pickle=True)
    np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                         'raster_spike_prob_rois'), spike_prob_rois, allow_pickle=True)

    # Rasterplot divided by block and sorted by time for each ROI
    for count_roi, roi in enumerate(roi_list):
        fig, ax = plt.subplots(3, 4, figsize=(20, 15), tight_layout=True)
        for count_p, paw in enumerate(paws):
            idx_nan = np.where(~np.isnan(events_stride_trial_rois[count_roi][count_p]))[0]
            ax[0, count_p].scatter(events_stride_trial_rois[count_roi][count_p][idx_nan],
                                   cumulative_idx_rois[count_roi][count_p][idx_nan], s=1, color='black')
            if align_dimension == 'phase':
                ax[0, count_p].axvline(x=0.5, color='black')
            if align_dimension == 'time':
                ax[0, count_p].axvline(x=0, color='black')
            ax[0, count_p].axhline(y=np.where(trial_id_rois[count_roi][count_p] == trials_ses[0, 1])[0][-1], color='black', linestyle='dashed')
            ax[0, count_p].axhline(y=np.where(trial_id_rois[count_roi][count_p] == trials_ses[1, 1])[0][-1], color='black', linestyle='dashed')
            ax[0, count_p].set_title(paw + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
            ax[0, count_p].spines['right'].set_visible(False)
            ax[0, count_p].spines['top'].set_visible(False)
            ax[0, count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            sns.heatmap(firing_rate_rois[count_roi][count_p], cmap='viridis', cbar=None,
            vmin=np.nanmin(firing_rate_rois[count_roi]), vmax=np.nanmax(firing_rate_rois[count_roi]), ax=ax[1, count_p])
            ax[1, count_p].invert_yaxis()
            ax[1, count_p].set_yticks(np.arange(0, len(trials)))
            ax[1, count_p].set_xticklabels(list(map(str, np.round(bins[:-1], 2))), rotation=45)
            ax[1, count_p].set_yticklabels(list(map(str, trials)), rotation=45)
            ax[1, count_p].axvline(x=np.int64(len(bins[::-1])/2), color='white')
            ax[1, count_p].axhline(y=trials_ses[0, 1], color='white', linestyle='dashed')
            ax[1, count_p].axhline(y=trials_ses[1, 1], color='white', linestyle='dashed')
            for count_t, t in enumerate(trials):
                ax[2, count_p].plot(bins[:-1], firing_rate_rois[count_roi][count_p][count_t, :], color=colors_session[t])
            if align_dimension == 'phase':
                ax[2, count_p].axvline(x=0.5, color='black')
            if align_dimension == 'time':
                ax[2, count_p].axvline(x=0, color='black')
            ax[2, count_p].set_ylim([np.nanmin(firing_rate_rois[count_roi]), np.nanmax(firing_rate_rois[count_roi])])
            ax[2, count_p].spines['right'].set_visible(False)
            ax[2, count_p].spines['top'].set_visible(False)
            ax[2, count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            ax[2, count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            ax[2, count_p].set_ylabel('Aligned to ' + str(align_event), fontsize=mscope.fsize - 8)
        plt.savefig(os.path.join(save_path, animal + ' ' + ses_info[0], 'raster_' + align_dimension + '_' +
                                 align_event + '_' + roi), dpi=mscope.my_dpi)

        fig, ax = plt.subplots(3, 4, figsize=(20, 15), tight_layout=True)
        for count_p, paw in enumerate(paws):
            idx_nan = np.where(~np.isnan(events_stride_trial_rois[count_roi][count_p]))[0]
            ax[0, count_p].scatter(events_stride_trial_rois[count_roi][count_p][idx_nan],
                                   cumulative_idx_rois[count_roi][count_p][idx_nan], s=1, color='black')
            if align_dimension == 'phase':
                ax[0, count_p].axvline(x=0.5, color='black')
            if align_dimension == 'time':
                ax[0, count_p].axvline(x=0, color='black')
            ax[0, count_p].axhline(y=np.where(trial_id_rois[count_roi][count_p] == trials_ses[0, 1])[0][-1], color='black', linestyle='dashed')
            ax[0, count_p].axhline(y=np.where(trial_id_rois[count_roi][count_p] == trials_ses[1, 1])[0][-1], color='black', linestyle='dashed')
            ax[0, count_p].set_title(paw + ' paw', color=paw_colors[count_p], fontsize=mscope.fsize - 6)
            ax[0, count_p].spines['right'].set_visible(False)
            ax[0, count_p].spines['top'].set_visible(False)
            ax[0, count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            sns.heatmap(firing_rate_rois[count_roi][count_p], cmap='viridis', cbar=None,
            vmin=np.nanmin(firing_rate_rois[count_roi]), vmax=np.nanmax(firing_rate_rois[count_roi]), ax=ax[1, count_p])
            ax[1, count_p].invert_yaxis()
            ax[1, count_p].set_yticks(np.arange(0, len(trials)))
            ax[1, count_p].set_xticklabels(list(map(str, np.round(bins[:-1], 2))), rotation=45)
            ax[1, count_p].set_yticklabels(list(map(str, trials)), rotation=45)
            ax[1, count_p].axvline(x=np.int64(len(bins[::-1])/2), color='white')
            ax[1, count_p].axhline(y=trials_ses[0, 1], color='white', linestyle='dashed')
            ax[1, count_p].axhline(y=trials_ses[1, 1], color='white', linestyle='dashed')
            for count_t, t in enumerate(trials):
                ax[2, count_p].plot(bins[:-1], spike_prob_rois[count_roi][count_p][count_t, :], color=colors_session[t])
            if align_dimension == 'phase':
                ax[2, count_p].axvline(x=0.5, color='black')
            if align_dimension == 'time':
                ax[2, count_p].axvline(x=0, color='black')
            ax[2, count_p].set_ylim([np.nanmin(spike_prob_rois[count_roi]), np.nanmax(spike_prob_rois[count_roi])])
            ax[2, count_p].spines['right'].set_visible(False)
            ax[2, count_p].spines['top'].set_visible(False)
            ax[2, count_p].tick_params(axis='both', which='major', labelsize=mscope.fsize - 10)
            ax[2, count_p].set_xlabel('Time (ms)', fontsize=mscope.fsize - 8)
            ax[2, count_p].set_ylabel('Aligned to ' + str(align_event), fontsize=mscope.fsize - 8)
        plt.savefig(os.path.join(save_path, animal + ' ' + ses_info[0], 'raster_prob_' + align_dimension + '_' +
                                 align_event + '_' + roi), dpi=mscope.my_dpi)
        plt.close('all')

