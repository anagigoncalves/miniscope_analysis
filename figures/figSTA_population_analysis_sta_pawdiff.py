# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mp
from scipy import signal as sig
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA paw spatial diff\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
protocol_type = 'split'
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
zoom_in = np.array([-1, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89],
                     [6.80, 1.75],
                     [6.98, 1.47],
                     [6.39, 1.62]]) #AP, ML
sort_type = 'ML'
var_name = 'FL-FR-phase'

sta_zoom_all = []
animal_list = []
sta_animal_id = []
sta_cluster_size = []
sta_ap = []
sta_ml = []
peaks_cluster_all = []
sta_zoom_all_notzscored = []
sta_zoom_all_shuffled = []
for count_f, f in enumerate(animal_order):
    session_data_idx = np.where(session_data['animal'] == f)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    # Session data and inputs
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                             frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    trials_ses_name.insert(len(trials_ses_name), 'late washout')
    trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]

    sta_zs = np.load(
        os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '_zscored.npy'))

    sta = np.load(
        os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))

    sta_shuffled = np.load(
        os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '_shuffled.npy'))

    # Get cluster global coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = fov_coords[count_f]
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

    # SEPARATE BY TRIALS YOU WANT TO PLOT AND THE OVERLAPPING CLUSTERS
    xaxis_short = xaxis[xaxis_start:xaxis_end]
    idx_time0 = np.where(xaxis_short == 0)[0][0]
    xaxis_short_crop = xaxis_short[:idx_time0]
    peaks_cluster = np.zeros((len(clusters_rois), len(cond_name)))
    for count_c in range(len(clusters_rois)):  # 0 are ROIs that don't overlap with reference session
        sta_zs_zoom = np.zeros((len(clusters_rois[count_c]), len(cond_name), xaxis_end - xaxis_start))
        sta_zs_zoom[:] = np.nan
        sta_zoom = np.zeros((len(clusters_rois[count_c]), len(cond_name), xaxis_end - xaxis_start))
        sta_zoom[:] = np.nan
        sta_zoom_shuffled = np.zeros((len(clusters_rois[count_c]), len(cond_name), xaxis_end - xaxis_start))
        sta_zoom_shuffled[:] = np.nan
        if protocol_type == 'tied':
            for count_t, t in enumerate(trials_ses_name):
                if trials_ses_name[count_t] == 'baseline speed':
                    bs_idx = trials_ses_name.index('baseline speed')
                    trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                    trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                    sta_zs_zoom[:, 1, :] = np.nanmean(sta_zs[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom[:, 1, :] = np.nanmean(
                        sta[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom_shuffled[:, 1, :] = np.nanmean(
                        sta_shuffled[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                if trials_ses_name[count_t] == 'slow speed':
                    bs_idx = trials_ses_name.index('slow speed')
                    trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                    trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                    sta_zs_zoom[:, 0, :] = np.nanmean(sta_zs[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom[:, 0, :] = np.nanmean(
                        sta[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom_shuffled[:, 0, :] = np.nanmean(
                        sta_shuffled[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                if trials_ses_name[count_t] == 'fast speed':
                    bs_idx = trials_ses_name.index('fast speed')
                    trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                    trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                    sta_zs_zoom[:, 2, :] = np.nanmean(sta_zs[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom[:, 2, :] = np.nanmean(
                        sta[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                    sta_zoom_shuffled[:, 2, :] = np.nanmean(
                        sta_shuffled[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if protocol_type == 'split':
            trials_ses_split = trials_ses.flatten()[1:]
            for count_t, t in enumerate(trials_ses_name): #if odd is -1, if even is the next
                if count_t % 2 == 0:
                    trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t]-1)[0][0]]
                    trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
                if count_t % 2 != 0:
                    trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
                    trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t]+1)[0][0]]
                sta_zs_zoom[:, count_t, :] = np.nanmean(sta_zs[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                sta_zoom[:, count_t, :] = np.nanmean(
                    sta[idx_roi_cluster_ordered == count_c+1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
                sta_zoom_shuffled[:, count_t, :] = np.nanmean(
                    sta_shuffled[idx_roi_cluster_ordered == count_c + 1, trial_start_idx:trial_end_idx,
                    xaxis_start:xaxis_end], axis=1)
        # save also cluster, animal id, AP and ML global coordinates
        sta_zoom_all.append(sta_zs_zoom)
        sta_animal_id.append(animal)
        sta_cluster_size.append(np.shape(sta_zs_zoom)[0])
        sta_ap.append(cluster_coord[count_c, 0])
        sta_ml.append(cluster_coord[count_c, 1])
        sta_zoom_all_notzscored.append(sta_zoom)
        sta_zoom_all_shuffled.append(sta_zoom_shuffled)
        # QUANTIFY LAGS IN CROSS CORRELATION ACROSS TRIALS
        sta_zs_zoom_c = np.nanmean(sta_zs_zoom, axis=0)
        for t in range(len(cond_name)):
            if session_type == 'tied' and animal == 'MC8855' and t == 0: #no slow speed
                amp = np.nan
                latency = np.nan
            else:
                amp, latency = mscope.get_peakamp_latency(sta_zs_zoom_c[t, :], xaxis_short)
            peaks_cluster[count_c, t] = latency
    peaks_cluster_all.append(peaks_cluster)

sort_ml = np.argsort(sta_ml)
sort_ap = np.argsort(sta_ap)
if sort_type == 'ML':
    sta_zs_zoom_all_sort = []
    sta_zoom_all_sort = []
    sta_shuffled_zoom_all_sort = []
    for i in sort_ml:
        sta_zs_zoom_all_sort.append(sta_zoom_all[i])
        sta_zoom_all_sort.append(sta_zoom_all_notzscored[i])
        sta_shuffled_zoom_all_sort.append(sta_zoom_all_shuffled[i])
if sort_type == 'AP':
    sta_zs_zoom_all_sort = []
    sta_zoom_all_sort = []
    sta_shuffled_zoom_all_sort = []
    for i in sort_ap:
        sta_zs_zoom_all_sort.append(sta_zoom_all[i])
        sta_zoom_all_sort.append(sta_zoom_all_notzscored[i])
        sta_shuffled_zoom_all_sort.append(sta_zoom_all_shuffled[i])
if sort_type == 'none':
    sta_zs_zoom_all_sort = sta_zoom_all
    sta_zoom_all_sort = sta_zoom_all_notzscored
    sta_shuffled_zoom_all_sort = sta_zoom_all_shuffled
sta_zoom_all_concat = np.concatenate(sta_zs_zoom_all_sort)
sta_zoom_all_concat_notzscored = np.concatenate(sta_zoom_all_sort)
sta_zoom_all_concat_shuffled = np.concatenate(sta_shuffled_zoom_all_sort)

#ANIMALS SUMMARY
fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    # hm = sns.heatmap(sta_zoom_all_concat[:, t, :], vmax=np.nanpercentile(sta_zoom_all_concat,99.5),
    #             vmin=np.nanpercentile(sta_zoom_all_concat,0.5), cmap='coolwarm', ax=ax[t])
    hm = sns.heatmap(sta_zoom_all_concat[:, t, :], vmax=4.5,
                vmin=-4.5, cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis==0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat)[2]]))
    ax[t].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end],2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white')
    ax[t].set_yticks(np.cumsum(np.array(sta_cluster_size)))
    ax[t].tick_params(left=False)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    for a in np.cumsum(sta_cluster_size)[:-1]:
        ax[t].axhline(y=a, c='white', linestyle='--')
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=6)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_' + var_name + '_animal_summary_zscored_sort_'+sort_type+'_animal_lines'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    hm = sns.heatmap(sta_zoom_all_concat_shuffled[:, t, :], vmax=np.nanpercentile(sta_zoom_all_concat_shuffled[:, t, :],99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_shuffled[:, t, :],0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis==0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat)[2]]))
    ax[t].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end],2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white')
    ax[t].set_yticks(np.cumsum(np.array(sta_cluster_size)))
    ax[t].tick_params(left=False)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # for a in np.cumsum(sta_cluster_size)[:-1]:
    #     ax[t].axhline(y=a, c='k', linestyle='--')
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=6)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_' + var_name + '_animal_summary_sort_shuffled_'+sort_type), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    hm = sns.heatmap(sta_zoom_all_concat_notzscored[:, t, :], vmax=np.nanpercentile(sta_zoom_all_concat_notzscored[:, 1, :],99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_notzscored[:, 1, :],0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis==0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat)[2]]))
    ax[t].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end],2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white')
    ax[t].set_yticks(np.cumsum(np.array(sta_cluster_size)))
    ax[t].tick_params(left=False)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # for a in np.cumsum(sta_cluster_size)[:-1]:
    #     ax[t].axhline(y=a, c='k', linestyle='--')
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=6)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_' + var_name + '_animal_summary_sort_notzscored_'+sort_type), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    idx_notsig = np.where((sta_zoom_all_concat[:, t, :] < 2) & (sta_zoom_all_concat[:, t, :] > -2))
    sta_zoom_notzscore_sig = np.rad2deg(sta_zoom_all_concat_notzscored[:, t, :]).copy()
    sta_zoom_notzscore_sig[idx_notsig] = np.nan
    hm = sns.heatmap(sta_zoom_notzscore_sig, vmax=np.nanpercentile(sta_zoom_notzscore_sig,99.5),
                vmin=np.nanpercentile(sta_zoom_notzscore_sig,0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis==0)[0][0]-xaxis_start, np.shape(sta_zoom_notzscore_sig)[1]]))
    ax[t].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end],2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white')
    ax[t].set_yticks(np.cumsum(np.array(sta_cluster_size)))
    ax[t].tick_params(left=False)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # for a in np.cumsum(sta_cluster_size)[:-1]:
    #     ax[t].axhline(y=a, c='k', linestyle='--')
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=6)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_' + var_name + '_animal_summary_sort_notzscored_sig_'+sort_type), dpi=mscope.my_dpi)

peaks_cluster_arr = np.concatenate(peaks_cluster_all)
cmap_plot = mp.cm.get_cmap('plasma', np.shape(peaks_cluster_arr)[0])
color_list = [mp.colors.rgb2hex(cmap_plot(i)[:3]) for i in range(cmap_plot.N)]
fig, ax = plt.subplots(len(session_data['animal']), 1, figsize=(5, 10), tight_layout=True, sharex=True)
ax = ax.ravel()
for i in range(np.shape(peaks_cluster_arr)[0]):
    animal_id = sta_animal_id[i]
    if animal_id == 'MC8855':
        j = 0
    if animal_id == 'MC9194':
        j = 1
    if animal_id == 'MC9226':
        j = 2
    if animal_id == 'MC9513':
        j = 3
    if animal_id == 'MC10221':
        j = 4
    ax[j].plot(range(len(cond_name)), peaks_cluster_arr[i, :], c=color_list[i], linewidth=2, label=str(np.round(sta_ml[i],4)))
    ax[j].legend(frameon=False, fontsize=12)
    ax[j].set_xticks(range(len(cond_name)))
    ax[j].set_xticklabels(cond_name, fontsize=16, rotation=45)
    ax[j].set_ylabel(var_name + '\npeak latency', fontsize=14)
    ax[j].spines['right'].set_visible(False)
    ax[j].spines['top'].set_visible(False)
    ax[j].tick_params(axis='both', which='major', labelsize=14)
    ax[j].set_ylim([-0.3, 0])
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_' + var_name + '_animal_summary_sort' + sort_type + '_notzscored_quantification'), dpi=mscope.my_dpi)

