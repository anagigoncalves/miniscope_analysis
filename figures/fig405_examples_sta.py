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
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\split contra fast S1\\'
save_path = 'J:\\Thesis\\for figures\\405 control\\'
var_name = 'Body acceleration'
window = np.arange(-330, 330 + 1)  # Samples
iter_n = 100 # Number of iterations of CS timestamps random shuffling
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

roi_plot = 5

# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', 'split contra fast', 'MC13420', '2022_05_31\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', 'split contra fast S1 310522')
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
session_type = path.split('\\')[-4].split(' ')[0]
animal = mscope.get_animal_id()
session = 1
# Session data and inputs
df_extract_rawtrace_detrended = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
df_events_extract_rawtrace = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'))
coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
ref_image = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)
frames_dFF = np.load(os.path.join(mscope.path, 'processed files', 'black_frames.npy'), allow_pickle=True)
colors_session = np.load(os.path.join(mscope.path, 'processed files', 'colors_session.npy'), allow_pickle=True)
colors_session = colors_session[()]
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, session_type, animal, session)
trials_ses_name.insert(len(trials_ses_name), 'late washout')
trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1] + 1), trials))[0]

sta = np.load(
    os.path.join(load_path, animal + ' split contra fast', 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))

if protocol_type == 'tied':
    sta_zoom = np.zeros((np.shape(sta)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zoom[:] = np.nan
    for count_c, c in enumerate(trials_ses_name):
        if trials_ses_name[count_c] == 'baseline speed':
            bs_idx = trials_ses_name.index('baseline speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zoom[:, 1, :] = np.nanmean(sta[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name[count_c] == 'slow speed':
            bs_idx = trials_ses_name.index('slow speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zoom[:, 0, :] = np.nanmean(sta[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name[count_c] == 'fast speed':
            bs_idx = trials_ses_name.index('fast speed')
            trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
            sta_zoom[:, 2, :] = np.nanmean(sta[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
if protocol_type == 'split':
    sta_zoom = np.zeros((np.shape(sta)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zoom[:] = np.nan
    trials_ses = trials_ses.flatten()[1:]
    for count_c, c in enumerate(trials_ses_name):  # if odd is -1, if even is the next
        if count_c % 2 == 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c] - 1)[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
        if count_c % 2 != 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c] + 1)[0][0]]
        sta_zoom[:, count_c, :] = np.nanmean(sta[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)

# ROI SUMMARY
fig, ax = plt.subplots(figsize=(5, 5), tight_layout='True')
for t in range(len(cond_name)):
    ax.plot(xaxis[xaxis_start:xaxis_end], sta_zoom[roi_plot, t, :],
            color=colors_cond[t], label=cond_name[t], linewidth=3)
ax.axvline(x=0, linestyle='dashed', color='black')
# ax.legend(cond_name, fontsize=16, frameon=False)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel(var_name.replace('_', ' '), fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal +
                         '_' + 'split_contra_fast' + '_roi' + str(roi_plot + 1) + '_summary'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(10, 10))
for t in range(len(cond_name)):
    ax.plot(xaxis[xaxis_start:xaxis_end], sta_zoom[roi_plot, t, :],
            color=colors_cond[t], label=cond_name[t], linewidth=3)
ax.axvline(x=0, linestyle='dashed', color='black')
ax.legend(cond_name, fontsize=16, frameon=False)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel(var_name.replace('_', ' '), fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal +
                         '_' + 'split_contra_fast' + '_roi' + str(roi_plot + 1) + '_summary_legend'), dpi=mscope.my_dpi)

sta_405 = np.load(
    os.path.join(load_path, animal + ' split contra fast 405', 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))
trials_405 = np.arange(1, 27)
trials_ses_name_405 = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
trials_idx_405 = np.where(np.in1d(np.arange(trials_405[0], trials_405[-1] + 1), trials_405))[0]
trials_ses_405 = np.array([[ 1,  6],
                            [ 7, 16],
                            [17, 26]])
if protocol_type == 'tied':
    sta_zoom_405 = np.zeros((np.shape(sta_405)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zoom_405[:] = np.nan
    for count_c, c in enumerate(trials_ses_name):
        if trials_ses_name_405[count_c] == 'baseline speed':
            bs_idx = trials_ses_name_405.index('baseline speed')
            trial_start_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 1])[0][0]]
            sta_zoom_405[:, 1, :] = np.nanmean(sta_405[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name_405[count_c] == 'slow speed':
            bs_idx = trials_ses_name_405.index('slow speed')
            trial_start_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 1])[0][0]]
            sta_zs_zoom_405[:, 0, :] = np.nanmean(sta_405[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        if trials_ses_name_405[count_c] == 'fast speed':
            bs_idx = trials_ses_name_405.index('fast speed')
            trial_start_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 0])[0][0]]
            trial_end_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[bs_idx, 1])[0][0]]
            sta_zoom_405[:, 2, :] = np.nanmean(sta_405[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
if protocol_type == 'split':
    sta_zs_zoom_405 = np.zeros((np.shape(sta_405)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zs_zoom_405[:] = np.nan
    trials_ses_405 = trials_ses_405.flatten()[1:]
    for count_c, c in enumerate(trials_ses_name_405):  # if odd is -1, if even is the next
        if count_c % 2 == 0:
            trial_start_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[count_c] - 1)[0][0]]
            trial_end_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[count_c])[0][0]]
        if count_c % 2 != 0:
            trial_start_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[count_c])[0][0]]
            trial_end_idx = trials_idx_405[np.where(trials_405 == trials_ses_405[count_c] + 1)[0][0]]
        sta_zs_zoom_405[:, count_c, :] = np.nanmean(sta_405[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout='True')
for t in range(len(cond_name)):
    ax.plot(xaxis[xaxis_start:xaxis_end], sta_zs_zoom_405[roi_plot, t, :],
            color=colors_cond[t], label=cond_name[t], linewidth=3)
ax.axvline(x=0, linestyle='dashed', color='black')
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel(var_name.replace('_', ' '), fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + animal +
                         '_' + 'split_contra_fast' + '_roi' + str(roi_plot + 1) + '_summary_405'), dpi=mscope.my_dpi)