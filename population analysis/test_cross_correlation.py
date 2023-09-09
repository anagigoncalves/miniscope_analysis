# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal as sig
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\STA paw spatial diff\\split ipsi fast S1\\'
if not os.path.exists(os.path.join(save_path, 'Plots')):
    os.mkdir(os.path.join(save_path, 'Plots'))
protocol_type = save_path.split('\\')[-2].split(' ')[0]
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
zoom_in = np.array([-0.25, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
idx_minus100 = np.where(xaxis == -0.1)[0][0] - xaxis_start
idx_0 = np.where(xaxis == 0)[0][0] - xaxis_start
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
vars = ['FR-FL displacement difference', 'FR-HL displacement difference', 'FR-HR displacement difference']

var_name = 'FR-FL displacement difference'
sta_zoom_all = []
animal_list = []
sta_animal_transition = []
sta_animal_minus100 = []
sta_animal_latency = []
sta_animal_0 = []
sta_animal_colors_cluster = []
f = 'MC9194'
session_data_idx = np.where(session_data['animal'] == f)[0][0]
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
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, session_type, animal, session)
trials_ses_name.insert(len(trials_ses_name), 'late washout')
trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1] + 1), trials))[0]

sta_zs = np.load(os.path.join(save_path, f + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))
sta_zs_clusterid = np.load(os.path.join(save_path, f + ' ' + ses_info[0],
                                        'sta_bodyvars_' + var_name.replace(' ', '_') + '_cluster_transition_idx.npy'))
cluster_beg = np.insert(sta_zs_clusterid[:-1], 0, 0)
cluster_end = np.append(sta_zs_clusterid[1:], np.shape(sta_zs)[0])
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
cluster_beg = np.insert(sta_zs_clusterid[:-1], 0, 0)
cluster_end = np.append(sta_zs_clusterid[1:], np.shape(sta_zs)[0])
sta_zs_zoom_cluster = np.zeros((len(sta_zs_clusterid), len(cond_name), xaxis_end - xaxis_start))
sta_zs_zoom_cluster[:] = np.nan
for count_r, r in enumerate(sta_zs_clusterid):
    sta_zs_zoom_cluster[count_r, :, :] = np.nanmean(sta_zs_zoom[cluster_beg[count_r]:cluster_end[count_r], :, :],
                                                    axis=0)
sta_animal_minus100.append(sta_zs_zoom_cluster[:, :, idx_minus100])
sta_animal_0.append(sta_zs_zoom_cluster[:, :, idx_0])
sta_animal_latency.append(xaxis[np.argmax(sta_zs_zoom_cluster[:, :, :idx_0], axis=2) + xaxis_start])
sta_zoom_all.append(sta_zs_zoom)
sta_animal_transition.append(np.shape(sta_zs_zoom)[0])
animal_list.append(animal)
sta_animal_colors_cluster.append(colors_cluster)

fig, ax = plt.subplots(1, len(sta_zs_clusterid), figsize=(15, 5), tight_layout=True, sharey=True)
ax = ax.ravel()
for c in range(len(sta_zs_clusterid)):
    for t in range(len(cond_name)):
        ax[c].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_zoom[cluster_beg[c]:cluster_end[c], t, :], axis=0),
                   color=colors_cond[t], linewidth=2)
    ax[c].axvline(x=0, linestyle='dashed', color='black')
    ax[c].set_xlabel('Time (s)', fontsize=18)
    ax[c].set_ylabel(var_name, fontsize=18)
    ax[c].set_title('Cluster ' + str(c + 1), color=colors_cluster[c], fontsize=20)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=16)

xaxis_crosscorr = xaxis[xaxis_start:xaxis_end]
idx_time0 = np.where(xaxis_crosscorr == 0)[0][0]
xaxis_crosscorr_crop = xaxis_crosscorr[:idx_time0]
lags_trials = np.zeros((np.shape(sta_zs_zoom)[0],len(cond_name)))
lags_cluster = np.zeros((len(cluster_beg),len(cond_name)))
lags_cluster_after0 = np.zeros((len(cluster_beg),len(cond_name)))
for r in range(np.shape(sta_zs_zoom)[0]):
    for t in range(len(cond_name[:-1])):
        correlation = sig.correlate(sta_zs_zoom[r, 0, :idx_time0], sta_zs_zoom[r, t+1, :idx_time0], mode='full')
        lags = sig.correlation_lags(sta_zs_zoom[r, 0, :idx_time0].size, sta_zs_zoom[r, t+1, :idx_time0].size, mode='full')
        lags_trials[r, t+1] = xaxis_crosscorr_crop[lags[np.argmax(correlation)]]
for c in range(len(cluster_beg)):
    sta_zs_zoom_c = np.nanmean(sta_zs_zoom[cluster_beg[c]:cluster_end[c], :, :], axis=0)
    for t in range(len(cond_name[:-1])):
        correlation = sig.correlate(sta_zs_zoom_c[0, :idx_time0], sta_zs_zoom_c[t + 1, :idx_time0], mode='full')
        lags = sig.correlation_lags(sta_zs_zoom_c[0, :idx_time0].size, sta_zs_zoom_c[t + 1, :idx_time0].size, mode='full')
        lags_cluster[c, t + 1] = xaxis_crosscorr_crop[lags[np.argmax(correlation)]]
        correlation_after0 = sig.correlate(sta_zs_zoom_c[0, :idx_time0], sta_zs_zoom_c[t + 1, idx_time0:], mode='full')
        lags_after0 = sig.correlation_lags(sta_zs_zoom_c[0, :idx_time0].size, sta_zs_zoom_c[t + 1, idx_time0:].size, mode='full')
        lags_cluster_after0[c, t + 1] = xaxis_crosscorr_crop[lags_after0[np.argmax(correlation_after0)]]

fig, ax = plt.subplots(1, len(sta_zs_clusterid), figsize=(15, 5), tight_layout=True, sharey=True)
ax = ax.ravel()
for c in range(len(sta_zs_clusterid)):
    lags_cluster_plot = lags_trials[cluster_beg[c]:cluster_end[c], :]
    for r in range(np.shape(lags_cluster_plot)[0]):
        ax[c].plot(range(len(cond_name)), lags_cluster_plot[r, :], color='black', linewidth=0.5, alpha=0.5)
    ax[c].plot(range(len(cond_name)), lags_cluster[c, :], color=colors_cluster[c], linewidth=2)
    ax[c].plot(range(len(cond_name)), lags_cluster_after0[c, :], color=colors_cluster[c], linewidth=2, linestyle='dashed')
    ax[c].set_xticks(range(len(cond_name)))
    ax[c].set_xticklabels(['baseline', 'early split', 'late split', 'early washout', 'late washout'], fontsize=16, rotation=45)
    ax[c].set_xlabel('Time (s)', fontsize=18)
    ax[c].set_ylabel(var_name, fontsize=18)
    ax[c].set_title('Cluster ' + str(c + 1), color=colors_cluster[c], fontsize=20)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=16)




