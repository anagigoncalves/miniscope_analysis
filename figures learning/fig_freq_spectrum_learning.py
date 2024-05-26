#Do this for that cluster than changes

# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch, windows, butter, filtfilt
import seaborn as sns
import matplotlib.ticker as tkr

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
load_pc_path = 'J:\\LocoCF\\miniscopes learning\\PCA validation and clusters (only baseline trials)\\'
protocol = 'split ipsi fast'
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
save_path = 'J:\\LocoCF\\miniscopes learning\\power spectrum\\'
window_size = 256
f_low = 2
f_high = 10
f_order = 2

def powerspectrum(data, window_size, sr):
    window_filt = windows.hamming(window_size, sym=True)
    noverlap = window_size//2
    freq, Pxx = welch(data, fs=sr, window=window_filt, noverlap=noverlap, scaling='density', nfft=10000)
    weight = np.floor(len(data)-window_size/(window_size-noverlap))+1
    return freq, Pxx, weight

def butter_bandpass(data, lowcut, highcut, sr, order):
    b, a = butter(order, [lowcut, highcut], btype='band', fs=sr, analog=False)
    y = filtfilt(b, a, data)
    return y

def mi_index(a, b):
    return (a-b)/(a+b)

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

# Load PC coefficients data
pc_coeff = pd.read_csv(os.path.join(load_pc_path, 'pc_coeff_df_clusters_' + '_'.join(protocol.split(' ')) + '.csv'))

roi_animal = np.zeros(len(animals))
for count_a, animal in enumerate(animals):
    session_data_idx = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    loco = locomotion_class.loco_class(path_loco)
    session = loco.get_session_id()
    loco = locomotion_class.loco_class(path_loco)
    # Session data and inputs
    df_extract_rawtrace_detrended = pd.read_csv(
        os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
    roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
    trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
    colors_session = np.load(os.path.join(mscope .path, 'processed files', 'colors_session.npy'), allow_pickle=True)[()]
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, protocol.split(' ')[0], animal, session)

    ps_rois = np.zeros((5001, len(roi_list), len(trials_ses.flatten())))
    for count_t, t in enumerate(trials_ses.flatten()):
        for count_r, roi in enumerate(roi_list):
            data = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial']==t, roi])
            data_filt = butter_bandpass(data, f_low, f_high, mscope.sr, f_order)
            f, ps, w = powerspectrum(data_filt, window_size, mscope.sr)
            ps_rois[:, count_r, count_t] = ps

    ps_mean = np.nanmean(ps_rois, axis=1)
    ps_std = np.nanstd(ps_rois, axis=1)

    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(18)
    for count_t, t in enumerate(trials_ses.flatten()):
        ax.plot(f, ps_mean[:, count_t], color=colors_session[t], linewidth=2)
        ax.fill_between(f, ps_mean[:, count_t]-ps_std[:, count_t], ps_mean[:, count_t]+ps_std[:, count_t], color=colors_session[t], alpha=0.2)
    ax.set_xlim([0, 10])
    ax.set_xlabel('Frequency (Hz)', fontsize=20)
    ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
    ax.set_title(animal + ' ' + protocol)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'ps_' + animal + '_' + protocol.replace(' ', '_') + '_rois_average'), dpi=mscope.my_dpi)
    plt.savefig(os.path.join(save_path, 'ps_' + animal + '_' + protocol.replace(' ', '_')+'_rois_average.svg'), dpi=mscope.my_dpi)
    if count_a > 0:
        ps_rois_all = np.concatenate((ps_rois_all, ps_rois), axis=1)
    else:
        ps_rois_all = ps_rois
    roi_animal[count_a] = len(roi_list)
    plt.close('all')

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
bs_es_data = mi_index(np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 2], axis=0), np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 1], axis=0))
bs_ae_data = mi_index(np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 4], axis=0), np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 1], axis=0))
ax.plot(np.arange(0, np.shape(ps_rois_all)[1]), bs_es_data, color=colors_session[trials_ses[1, 0]], marker='o', linestyle='none', markersize=2)
ax.plot(np.arange(0, np.shape(ps_rois_all)[1]), bs_ae_data, color=colors_session[trials_ses[2, 0]], marker='o', linestyle='none', markersize=2)
plt.vlines(np.arange(0, np.shape(ps_rois_all)[1]), ymin=bs_es_data, ymax=bs_ae_data, linewidth=0.4,
           color='darkgrey')
for a in range(len(roi_animal)):
    plt.axvline(x=np.cumsum(roi_animal)[a], color='black', linewidth=0.5, linestyle='dashed')
plt.ylim([-1, 1])
ax.set_xlabel('ROI ID', fontsize=20)
ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'ps_ratio_max_' + protocol.replace(' ', '_') + '_rois_bs_es_bs_ae'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'ps_ratio_max_' + protocol.replace(' ', '_')+'_rois_bs_es_bs_ae.svg'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
bs_max = np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 1], axis=0)
es_max = np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 2], axis=0)
ae_max = np.nanmax(ps_rois_all[(f > 2) & (f < 6), :, 4], axis=0)
ax.plot(np.arange(0, np.shape(ps_rois_all)[1]), es_max-bs_max, color=colors_session[trials_ses[1, 0]], marker='o', linestyle='none', markersize=2)
ax.plot(np.arange(0, np.shape(ps_rois_all)[1]), ae_max-bs_max, color=colors_session[trials_ses[2, 0]], marker='o', linestyle='none', markersize=2)
plt.vlines(np.arange(0, np.shape(ps_rois_all)[1]), ymin=es_max-bs_max, ymax=ae_max-bs_max, linewidth=0.4,
           color='darkgrey')
for a in range(len(roi_animal)):
    plt.axvline(x=np.cumsum(roi_animal)[a], color='black', linewidth=0.5, linestyle='dashed')
ax.set_xlabel('ROI ID', fontsize=20)
ax.set_ylabel('Power spectral density (a.u.)\ndelta on maximum values', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.ylim([-0.0015, 0.0015])
plt.savefig(os.path.join(save_path, 'ps_delta_max_' + protocol.replace(' ', '_') + '_rois_bs_es_bs_ae'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'ps_delta_max_' + protocol.replace(' ', '_')+'_rois_bs_es_bs_ae.svg'), dpi=mscope.my_dpi)

if protocol == 'split ipsi fast':
    # Plot comparisons for cluster 1 (the one that changes the most for split ipsi fast)
    cluster_idx = np.where(pc_coeff['cluster_pca'] == 1)[0]
    ps_mean = np.nanmean(ps_rois_all[:, cluster_idx, :], axis=1)
    ps_std = np.nanstd(ps_rois_all[:, cluster_idx, :], axis=1)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(18)
    for count_t, t in enumerate(trials_ses.flatten()):
        ax.plot(f, ps_mean[:, count_t], color=colors_session[t], linewidth=2)
        ax.fill_between(f, ps_mean[:, count_t] - ps_std[:, count_t], ps_mean[:, count_t] + ps_std[:, count_t],
                        color=colors_session[t], alpha=0.2)
    ax.set_xlim([0, 10])
    ax.set_xlabel('Frequency (Hz)', fontsize=20)
    ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
    ax.set_title('Cluster 1 ' + protocol, fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'ps_cluster1_' + protocol.replace(' ', '_') + '_rois_average'),
                dpi=mscope.my_dpi)
    plt.savefig(os.path.join(save_path, 'ps_cluster1_' + protocol.replace(' ', '_') + '_rois_average.svg'),
                dpi=mscope.my_dpi)

if protocol == 'split contra fast':
    # Plot comparisons for cluster 2 (the one that changes the most for split contra fast)
    cluster_idx = np.where(pc_coeff['cluster_pca']==2)[0]
    ps_mean = np.nanmean(ps_rois_all[:, cluster_idx, :], axis=1)
    ps_std = np.nanstd(ps_rois_all[:, cluster_idx, :], axis=1)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(18)
    for count_t, t in enumerate(trials_ses.flatten()):
        ax.plot(f, ps_mean[:, count_t], color=colors_session[t], linewidth=2)
        ax.fill_between(f, ps_mean[:, count_t] - ps_std[:, count_t], ps_mean[:, count_t] + ps_std[:, count_t],
                        color=colors_session[t], alpha=0.2)
    ax.set_xlim([0, 10])
    ax.set_xlabel('Frequency (Hz)', fontsize=20)
    ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
    ax.set_title('Cluster 2 ' + protocol, fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'ps_cluster2_' + protocol.replace(' ', '_') + '_rois_average'),
                dpi=mscope.my_dpi)
    plt.savefig(os.path.join(save_path, 'ps_cluster2_' + protocol.replace(' ', '_') + '_rois_average.svg'),
                dpi=mscope.my_dpi)