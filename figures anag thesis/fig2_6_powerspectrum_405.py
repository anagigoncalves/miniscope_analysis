# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import welch, windows, butter, filtfilt
import matplotlib.ticker as tkr

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class

path_session_data = 'J:\\Miniscope processed files'
animal = 'MC13420' #MC13419 has very few ROIs
protocol = 'split_contra_fast'
window_size = 256
save_path = 'J:\\Thesis\\for figures\\405 control\\'

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

#load 480 session
path = os.path.join(path_session_data, 'TM RAW FILES', 'split contra fast', animal, '2022_05_31\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', 'split contra fast S1 310522\\')
mscope = miniscope_session_class.miniscope_session(path)
session = 1
# Session data and inputs
df_extract_rawtrace_detrended = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
trials_480 = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))

#load 405 session
path_405 = os.path.join(path_session_data, 'TM RAW FILES', 'split contra fast 405', animal, '2022_05_31\\')
path_loco_405 = os.path.join(path_session_data, 'TM TRACKING FILES', 'split contra fast S2 310522\\')
mscope_405 = miniscope_session_class.miniscope_session(path_405)
session_405 = 2
# Session data and inputs
df_extract_rawtrace_detrended_405 = pd.read_csv(
    os.path.join(mscope_405.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))

ps_480_rois = []
ps_405_rois = []
for roi in roi_list:
    data_480 = np.array(df_extract_rawtrace_detrended[roi])
    data_filt_480 = butter_bandpass(data_480, 2, 10, mscope.sr, 2)
    f, ps_480, w_480 = powerspectrum(data_filt_480, window_size, mscope.sr)
    ps_480_rois.append(ps_480)
    data_405 = np.array(df_extract_rawtrace_detrended_405[roi])
    data_filt_405 = butter_bandpass(data_405, 2, 10, mscope.sr, 2)
    f, ps_405, w_405 = powerspectrum(data_filt_405, window_size, mscope.sr)
    ps_405_rois.append(ps_405)

ps_480_mean = np.nanmean(np.array(ps_480_rois), axis=0)
ps_480_std = np.nanstd(np.array(ps_480_rois), axis=0)
ps_405_mean = np.nanmean(np.array(ps_405_rois), axis=0)
ps_405_std = np.nanstd(np.array(ps_405_rois), axis=0)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
ax.yaxis.get_offset_text().set_fontsize(18)
for r in range(len(ps_480_rois)):
    ax.scatter(0, np.nanmean(ps_480_rois[r][(f > 3.5) & (f < 4.5)]), s=20, color='purple')
    ax.scatter(1, np.nanmean(ps_405_rois[r][(f > 3.5) & (f < 4.5)]), s=20, color='black')
    ax.plot(np.arange(2), [np.nanmean(ps_480_rois[r][(f > 3.5) & (f < 4.5)]), np.nanmean(ps_405_rois[r][(f > 3.5) & (f < 4.5)])], color='darkgray', linewidth=0.5)
ax.set_ylim([0, 0.00025])
ax.set_xticks([0, 1])
ax.set_xticklabels(['480 nm', '405 nm'], rotation=45)
ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_') + '_rois_comparison'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_')+'_rois_comparison.svg'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(18)
ax.plot(f, ps_480_mean, color='purple')
ax.fill_between(f, ps_480_mean-ps_480_std, ps_480_mean+ps_480_std, color='purple', alpha=0.5)
ax.plot(f, ps_405_mean, color='black')
ax.fill_between(f, ps_405_mean-ps_405_std, ps_405_mean+ps_405_std, color='black', alpha=0.5)
ax.set_xlim([0, 10])
ax.set_xlabel('Frequency (Hz)', fontsize=20)
ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_') + '_rois_average'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_')+'_rois_average.svg'), dpi=mscope.my_dpi)

roi = 'ROI1'
ps_480_roi_trials = []
for trial in trials_480:
    data_480 = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial, roi])
    data_filt_480 = butter_bandpass(data_480, 2, 10, mscope.sr, 2)
    f, ps_480, w_480 = powerspectrum(data_filt_480, window_size, mscope.sr)
    ps_480_roi_trials.append(ps_480)

formatter = tkr.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
hm = sns.heatmap(np.array(ps_480_roi_trials), cmap='plasma', ax=ax, cbar_kws={'format': formatter})
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=26)
ax.set_xticks(np.arange(0, len(ps_480_roi_trials[0]), 500))
ax.set_xticklabels(np.round(f[::500], 1))
ax.set_yticks(trials_480[::2])
ax.set_yticklabels(trials_480[::2])
ax.set_xlim([0, 3250])
ax.set_ylabel('Trials', fontsize=20)
ax.set_xlabel('Frequency (Hz)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_') + '_roi1_example'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_')+'_roi1_example.svg'), dpi=mscope.my_dpi)


