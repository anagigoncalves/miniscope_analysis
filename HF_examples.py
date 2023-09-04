# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

day = '14'
dils = ['dilution_1_to_10', 'dilution_1_to_50', 'dilution_1_to_100']
paths = ['C:\\Users\\Ana\\Desktop\\', 'D:\\Titer analysis\\TM RAW FILES\\', 'D:\\Titer analysis\\TM RAW FILES\\']
fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
for count_p, dil in enumerate(dils):
    path = paths[count_p] + dil.replace('_', ' ') + '\\2020_01_' + day + '\\'
    session_type = path.split('\\')[-4].split(' ')[0]
    # import classes
    os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
    import miniscope_session_class
    mscope = miniscope_session_class.miniscope_session(path)
    df_extract_rawtrace_detrended = pd.read_csv(
        os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
    trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
    roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
    roi = np.random.choice(roi_list, size=1)[0]
    trial = np.random.choice(trials, size=1)[0]
    F = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial, roi])[:30*mscope.sr]
    F_time = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial, 'time'])[:30*mscope.sr]
    ax.plot(F_time, F + (count_p/2), linewidth=2, label=dil.replace('_', ' '))
#ax.legend(frameon=False, fontsize=16)
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('\u0394F/F', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('D:\\Titer analysis\\example_14days', dpi=256)

day = '14'
dil = 'dilution_1_to_100'
path = 'D:\\Titer analysis\\TM RAW FILES\\' + dil.replace('_', ' ') + '\\2020_01_' + day + '\\'
session_type = path.split('\\')[-4].split(' ')[0]
# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
df_extract_rawtrace_detrended = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
roi = np.random.choice(roi_list, size=1)[0]
roi = 'ROI18'
trial = np.random.choice(trials, size=1)[0]
fig, ax = plt.subplots(figsize=(10,5), tight_layout=True)
for count_t, t in enumerate(trials):
    F = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == t, roi])
    F_time = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == t, 'time'])
    ax.plot(F_time, F + (count_t/4), linewidth=2, color='black')
ax.set_xlabel('Time (s)', fontsize=16)
ax.set_ylabel('\u0394F/F', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('D:\\Titer analysis\\example_14days_dil1to100_roi18_alltrials_pauses_bursts', dpi=256)

#np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == 3, 'ROI3])[:30*mscope.sr] #bursty day 17 dil 100
#np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == 4, 'ROI18])[:30*mscope.sr] #bursty day 14 dil 100