import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters sw time\\split ipsi fast S1\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\DS sorted rasters\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
paws = ['FR', 'HR', 'FL', 'HL']
param = 'double_support'
bins = 20
plot_raster = 0

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

def sort_stride_roi(cum_idx_roi, param_all):
    # Create a dictionary for quick lookup of param_all values
    param_dict = {i: param_all[i - 1] for i in cum_idx_roi}
    # Update param_idx_roi based on cum_idx_roi values
    param_idx_roi = np.zeros(len(cum_idx_roi))
    for idx, j in enumerate(cum_idx_roi):
        if j in param_dict:
            param_idx_roi[idx] = param_dict[j]
    # Get indices of sorted param values
    param_idx_roi_sorted = cum_idx_roi[np.argsort(param_idx_roi)]
    return param_idx_roi, param_idx_roi_sorted

param_fr_prob_sum_st = []
param_fr_prob_sum_sw = []
param_fr_value = []
param_fr_roi_id = []
param_fr_animal_id = []
param_fl_prob_sum_st = []
param_fl_prob_sum_sw = []
param_fl_value = []
param_fl_roi_id = []
param_fl_animal_id = []
for animal in animals:
    # Session data and inputs
    session_data_idx = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    session = loco.get_session_id()
    trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
        trials, protocol.split(' ')[0], animal, session)
    [_, _, _, df_extract_rawtrace_detrended, _, _, _, _, trials, _, _, _, _, _, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)

    #Compute continuous gait parameters
    filelist = loco.get_track_files(animal, session)
    st_strides_trials = []
    param_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        st_strides_trials.append(st_strides_mat)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                            param))
    cumulative_idx_array, param_all_time, param_all = loco.param_continuous_sym(param_trials, st_strides_trials, trials, 'FR', 'FL', sym=1, remove_nan=0)

    # Load raster in time
    cumulative_idx_rois = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_cumulative_idx_rois.npy'), allow_pickle=True)
    events_stride_trial_rois = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_events_stride_trial_rois.npy'), allow_pickle=True)
    trial_id_rois = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_trial_id_rois.npy'), allow_pickle=True)

    # Create folders to save plots
    if not os.path.exists(os.path.join(save_path, protocol)):
        os.mkdir(os.path.join(save_path, protocol))
    if not os.path.exists(os.path.join(save_path, protocol, animal)):
        os.mkdir(os.path.join(save_path, protocol, animal))

    # Rasters sorted by param values and get binned values - FR
    for count_roi, roi in enumerate(roi_list):
        # Sometimes cumulative idx rois has one more stride
        if cumulative_idx_rois[count_roi][0][-1]>len(param_all):
            idx_to_del = np.where(cumulative_idx_rois[count_roi][0] > len(param_all))[0][0]
            cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][0][:idx_to_del]
            events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][0][:idx_to_del]
        else:
            cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][0]
            events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][0]
        [param_idx_roi, param_idx_roi_sorted] = sort_stride_roi(cumulative_idx_rois_corr, param_all)
        idx_nan = np.where(~np.isnan(events_stride_trial_rois_corr[param_idx_roi_sorted]))[0]
        param_idx_roi_bins = np.histogram(param_idx_roi[~np.isnan(param_idx_roi)], bins=bins)
        heatmap, xedges, yedges = \
            np.histogram2d(events_stride_trial_rois_corr[param_idx_roi_sorted][idx_nan],
                           cumulative_idx_rois_corr[idx_nan], bins=bins)
        neg_bins = np.where(xedges < 0)[0][:-1]
        pos_bins = np.where(xedges >= 0)[0]-1
        param_fr_prob_sum_st.extend(np.nansum(heatmap.T[neg_bins, :], axis=0))
        param_fr_prob_sum_sw.extend(np.nansum(heatmap.T[pos_bins, :], axis=0))
        param_fr_value.extend(param_idx_roi_bins[1][:-1])
        param_fr_roi_id.extend(np.repeat(roi, len(np.nansum(heatmap.T[neg_bins, :], axis=0))))
        param_fr_animal_id.extend(
            np.repeat(animal, len(np.nansum(heatmap.T[neg_bins, :], axis=0))))

    # Rasters sorted by param values and get binned values - FL
    for count_roi, roi in enumerate(roi_list):
        # Sometimes cumulative idx rois has one more stride
        if cumulative_idx_rois[count_roi][2][-1]>len(param_all):
            idx_to_del = np.where(cumulative_idx_rois[count_roi][2] > len(param_all))[0][0]
            cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][2][:idx_to_del]
            events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][2][:idx_to_del]
        else:
            cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][2]
            events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][2]
        [param_idx_roi, param_idx_roi_sorted] = sort_stride_roi(cumulative_idx_rois_corr, param_all)
        idx_nan = np.where(~np.isnan(events_stride_trial_rois_corr[param_idx_roi_sorted]))[0]
        param_idx_roi_bins = np.histogram(param_idx_roi[~np.isnan(param_idx_roi)], bins=bins)
        heatmap, xedges, yedges = \
            np.histogram2d(events_stride_trial_rois_corr[param_idx_roi_sorted][idx_nan],
                           cumulative_idx_rois_corr[idx_nan], bins=bins)
        neg_bins = np.where(xedges < 0)[0][:-1]
        pos_bins = np.where(xedges >= 0)[0]-1
        param_fl_prob_sum_st.extend(np.nansum(heatmap.T[neg_bins, :], axis=0))
        param_fl_prob_sum_sw.extend(np.nansum(heatmap.T[pos_bins, :], axis=0))
        param_fl_value.extend(param_idx_roi_bins[1][:-1])
        param_fl_roi_id.extend(np.repeat(roi, len(np.nansum(heatmap.T[neg_bins, :], axis=0))))
        param_fl_animal_id.extend(np.repeat(animal, len(np.nansum(heatmap.T[neg_bins, :], axis=0))))

    if plot_raster:
        # Plot raster sorted by param values for all paws
        for count_roi, roi in enumerate(roi_list):
            fig, ax = plt.subplots(1, 4, figsize=(15, 5), tight_layout=True)
            for count_p, paw in enumerate(paws):
                # Sometimes cumulative idx rois has one more stride
                if cumulative_idx_rois[count_roi][count_p][-1] > len(param_all):
                    idx_to_del = np.where(cumulative_idx_rois[count_roi][count_p] > len(param_all))[0][0]
                    cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][count_p][:idx_to_del]
                    events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][count_p][:idx_to_del]
                else:
                    cumulative_idx_rois_corr = cumulative_idx_rois[count_roi][count_p]
                    events_stride_trial_rois_corr = events_stride_trial_rois[count_roi][count_p]
                [param_idx_roi, param_idx_roi_sorted] = sort_stride_roi(cumulative_idx_rois_corr, param_all)
                idx_nan = np.where(~np.isnan(events_stride_trial_rois_corr[param_idx_roi_sorted]))[0]
                param_idx_roi_bins = np.histogram(param_idx_roi[~np.isnan(param_idx_roi)], bins=bins)
                heatmap, xedges, yedges = \
                    np.histogram2d(events_stride_trial_rois_corr[param_idx_roi_sorted][idx_nan],
                                   cumulative_idx_rois_corr[idx_nan], bins=bins)
                neg_bins = np.where(xedges < 0)[0][::-1]
                pos_bins = np.where(xedges >= 0)[0]-1
                heatmap_totalcount_xbin = np.nansum(heatmap.T, axis=1)
                im = ax[count_p].imshow(heatmap.T/heatmap_totalcount_xbin, cmap='viridis', vmin=0, vmax=0.2)
                ax[count_p].set_yticks(np.arange(0, bins)[::4])
                ax[count_p].set_yticklabels(np.round(param_idx_roi_bins[1][:-1:4]))
                ax[count_p].set_xticks(np.arange(0, bins)[::4])
                ax[count_p].set_xticklabels(np.round(xedges[:-1:4]), rotation=45)
                ax[count_p].axvline(x=bins/2, color='white')
                ax[count_p].tick_params(axis='both', which='major', labelsize=10)
                ax[count_p].set_xlabel('Time from swing onset (ms)', fontsize=14)
                ax[count_p].set_ylabel('DS sym.', fontsize=14)
                ax[count_p].set_title(paw, fontsize=14)
            plt.savefig(os.path.join(save_path, protocol, animal, roi_list[count_roi] + '_heatmap_ds_sorted'), dpi=256)
            plt.close('all')

prob_sum_fr = {'animal': param_fr_animal_id, 'roi': param_fr_roi_id, 'param_val': param_fr_value, 'prob_st': param_fr_prob_sum_st, 'prob_sw': param_fr_prob_sum_sw}
prob_sum_fr_df = pd.DataFrame(prob_sum_fr)
prob_sum_fl = {'animal': param_fl_animal_id, 'roi': param_fl_roi_id, 'param_val': param_fl_value, 'prob_st': param_fl_prob_sum_st, 'prob_sw': param_fl_prob_sum_sw}
prob_sum_fl_df = pd.DataFrame(prob_sum_fl)
prob_sum_fr_df.to_csv(os.path.join(save_path, protocol, 'prob_sum_fr_df.csv'), index=False)
prob_sum_fl_df.to_csv(os.path.join(save_path, protocol, 'prob_sum_fl_df.csv'), index=False)

# #cbar
# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# for count_p, paw in enumerate(paws):
#     [param_idx_roi, param_idx_roi_sorted] = sort_stride_roi(cumulative_idx_rois[count_roi][count_p], param_all)
#     idx_nan = np.where(~np.isnan(events_stride_trial_rois[count_roi][count_p][param_idx_roi_sorted]))[0]
#     param_idx_roi_bins = np.histogram(param_idx_roi[~np.isnan(param_idx_roi)], bins=bins)
#     heatmap, xedges, yedges = \
#         np.histogram2d(events_stride_trial_rois[count_roi][count_p][param_idx_roi_sorted][idx_nan],
#                        cumulative_idx_rois[count_roi][count_p][idx_nan], bins=bins)
#     heatmap_totalcount_xbin = np.nansum(heatmap.T, axis=1)
#     im = ax[count_p].imshow(heatmap.T/heatmap_totalcount_xbin, cmap='viridis', vmin=0, vmax=0.2)
#     fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.5)
# plt.savefig(os.path.join(save_path, 'cbar'), dpi=mscope.my_dpi)
#

