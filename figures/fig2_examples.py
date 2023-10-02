# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_tied_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
var_name = 'Body_speed'
protocol_type = 'tied'
if protocol_type == 'tied':
    cond_name = ['slow', 'baseline', 'fast']
    colors_cond = ['purple', 'black', 'orange']
if protocol_type == 'split':
    cond_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
    colors_cond = ['black', (0.403921568627451, 0.0, 0.05098039215686274, 1.0),
                   (0.9896613190730839, 0.7597147950089126, 0.6663101604278074, 1.0),
                   (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0),
                   (0.7935828877005348, 0.8702317290552584, 0.9429590017825312, 1.0)]
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.5, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]

session_data_idx = 1 #example is MC9194
cluster_plot = 0
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

sta_zs = np.load(
    os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))

if protocol_type == 'tied':
    sta_zs_zoom = np.zeros((np.shape(sta_zs)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zs_zoom[:] = np.nan
    for count_c, c in enumerate(trials_ses_name):
        if trials_ses_name[count_c] == 'baseline speed':
            bs_idx = trials_ses_name.index('baseline speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zs_zoom[:, 1, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name[count_c] == 'slow speed':
            bs_idx = trials_ses_name.index('slow speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zs_zoom[:, 0, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name[count_c] == 'fast speed':
            bs_idx = trials_ses_name.index('fast speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zs_zoom[:, 2, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
if protocol_type == 'split':
    sta_zs_zoom = np.zeros((np.shape(sta_zs)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zs_zoom[:] = np.nan
    trials_ses = trials_ses.flatten()[1:]
    for count_c, c in enumerate(trials_ses_name):  # if odd is -1, if even is the next
        if count_c % 2 == 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c] - 1)[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
        if count_c % 2 != 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c] + 1)[0][0]]
        sta_zs_zoom[:, count_c, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)

clusters_in_session_all = np.unique(coord_ext_overlap)
if len(np.where(clusters_in_session_all == 0)[0]) > 0:
    clusters_in_session = np.delete(clusters_in_session_all, np.where(clusters_in_session_all == 0)[0][0])
else:
    clusters_in_session = np.delete(clusters_in_session_all, np.where(clusters_in_session_all == 0)[0])
sta_zs_zoom_cluster = np.zeros((len(clusters_in_session), len(cond_name), xaxis_end - xaxis_start))
sta_zs_zoom_cluster[:] = np.nan
for count_c, c in enumerate(clusters_in_session):
    sta_zs_zoom_cluster[count_c, :, :] = np.nanmean(sta_zs_zoom[coord_ext_overlap == c, :, :],
                                                    axis=0)
# CLUSTER SUMMARY
fig, ax = plt.subplots(figsize=(5, 5), tight_layout='True')
for t in range(len(cond_name)):
    ax.plot(xaxis[xaxis_start:xaxis_end], sta_zs_zoom_cluster[cluster_plot, t, :],
            color=colors_cond[t], label=cond_name[t], linewidth=3)
ax.axvline(x=0, linestyle='dashed', color='black')
# ax.legend(cond_name, fontsize=16, frameon=False)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel(var_name.replace('_', ' '), fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal + '_' + ses_info[
                             0].replace(' ', '_') + '_cluster' + str(cluster_plot + 1) + '_summary'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(10, 10))
for t in range(len(cond_name)):
    ax.plot(xaxis[xaxis_start:xaxis_end], sta_zs_zoom_cluster[cluster_plot, t, :],
            color=colors_cond[t], label=cond_name[t], linewidth=3)
ax.axvline(x=0, linestyle='dashed', color='black')
ax.legend(cond_name, fontsize=16, frameon=False)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel(var_name.replace('_', ' '), fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal + '_' + ses_info[
                             0].replace(' ', '_') + '_cluster' + str(cluster_plot + 1) + '_summary_legend'), dpi=mscope.my_dpi)

# #HEATMAP EXAMPLE BODYBARS
# # Align dF/F and behavior (body position, speed, acceleration) for each trial and desired epoch
# # Order ROIs by cluster
# if len(clusters_rois) == 1:
#     clusters_rois_flat = clusters_rois[0]
# else:
#     clusters_rois_flat = np.transpose(sum(clusters_rois, []))
# clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
# clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
# df = mscope.norm_traces(df_extract_rawtrace_detrended[clusters_rois_flat], norm_name='zscore', axis='session') # Normalized dF/F traces for 'popul_heatmap'
#
# # Load behavioral data and get acceleration
# filelist = loco.get_track_files(animal, session)
# bodyacc = []
# bodycenter = []
# bodyspeed = []
# for count_trial, f in enumerate(filelist):
#     [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
#         frames_loco[count_trial]))
#     bodycenter_trial = loco.compute_bodycenter(final_tracks, 'X')
#     bodyspeed_trial = loco.compute_bodyspeed(bodycenter_trial)
#     bodyacc_trial = loco.compute_bodyacc(bodycenter_trial)
#     bodyacc.append(bodyacc_trial)
#     bodycenter.append(bodycenter_trial)
#     bodyspeed.append(bodyspeed_trial)
#
# [df_sorted, cluster_transition_idx] = mscope.sort_rois_clust(df, clusters_rois)
#
# trial = 2
# beg = 20
# end = 45
# fig, axs = plt.subplots(4, 1, figsize=(25, 10))
# df_trial = df_sorted.loc[(df_sorted['trial'] == trial)&(df_sorted['time']>beg)&(df_sorted['time']<end)].iloc[:, 2:]  # Get df/f for the desired trial and interval
# hm = sns.heatmap(df_trial.T, cmap='plasma', ax=axs[0])
# cbar = hm.collections[0].colorbar
# cbar.ax.set_label('\u0394F/F')
# cbar.ax.tick_params(labelsize=16)
# axs[0].set_xticks([])
# axs[0].set(xticklabels=[])
# axs[0].set_ylabel('ROIs', fontsize=20)
# axs[0].spines['right'].set_visible(False)
# axs[0].spines['top'].set_visible(False)
# axs[0].spines['bottom'].set_visible(False)
# for c in cluster_transition_idx:  # Lines to mark clusters in the heatmap
#     axs[0].hlines(c + 1, *axs[0].get_xlim(), color='white', linestyle='dashed', linewidth=1)
# # Behavior
# t = np.linspace(beg, end, (end-beg)*loco.sr)  # Create x-axis time values
# sns.lineplot(x=t, y=bodycenter[trial-1][beg*loco.sr:end*loco.sr], ax=axs[1], color='black', linewidth=2)
# axs[1].set(xticklabels=[])
# axs[1].set_xlim([t[0], t[-1]])
# axs[1].set_ylabel('Body\nCenter (mm)', fontsize=20)
# axs[1].tick_params(axis='both', which='major', labelsize=18)
# axs[1].spines['right'].set_visible(False)
# axs[1].spines['top'].set_visible(False)
# axs[1].tick_params(left=False, bottom=False)
# sns.lineplot(x=t, y=bodyspeed[trial-1][beg*loco.sr:end*loco.sr], ax=axs[2], color='black', linewidth=2)
# axs[2].set(xticklabels=[])
# axs[2].set_xlim([t[0], t[-1]])
# axs[2].set_ylabel('Body\nSpeed (mm/s)', fontsize=20)
# axs[2].tick_params(axis='both', which='major', labelsize=18)
# axs[2].spines['right'].set_visible(False)
# axs[2].spines['top'].set_visible(False)
# sns.lineplot(x=t, y=bodyacc[trial-1][beg*loco.sr:end*loco.sr], ax=axs[3], color='black', linewidth=2)
# axs[3].set_xlim([t[0], t[-1]])
# axs[3].set_ylabel('Body\nAcceleration\n(mm/s\u00b2)', fontsize=16)
# axs[3].set_xlabel('Time (s)', fontsize=20)
# axs[3].tick_params(axis='both', which='major', labelsize=18)
# axs[3].spines['right'].set_visible(False)
# axs[3].spines['top'].set_visible(False)
# plt.savefig(os.path.join('J:\\Thesis\\for figures\\fig2\\', 'example_bodyvars_MC9194_splitipsifast_S1_cbar'), dpi=128)
