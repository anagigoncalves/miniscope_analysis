# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import welch, windows, butter, filtfilt
import matplotlib.ticker as tkr

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
save_path = 'J:\\LocoCF\\miniscopes learning\\'
window_size = 256

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

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

animal = 'MC8855'

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
# Session data and inputs
df_extract_rawtrace_detrended = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
trials_480 = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))

ps_480_rois = []
for roi in roi_list:
    data_480 = np.array(df_extract_rawtrace_detrended[roi])
    data_filt_480 = butter_bandpass(data_480, 2, 10, mscope.sr, 2)
    f, ps_480, w_480 = powerspectrum(data_filt_480, window_size, mscope.sr)
    ps_480_rois.append(ps_480)

ps_480_mean = np.nanmean(np.array(ps_480_rois), axis=0)
ps_480_std = np.nanstd(np.array(ps_480_rois), axis=0)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(18)
ax.plot(f, ps_480_mean, color='purple')
ax.fill_between(f, ps_480_mean-ps_480_std, ps_480_mean+ps_480_std, color='purple', alpha=0.5)
ax.set_xlim([0, 10])
ax.set_xlabel('Frequency (Hz)', fontsize=20)
ax.set_ylabel('Power spectral density (a.u.)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)

roi = roi_list[0]
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


