# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
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
protocol_type = 'split'
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-1, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
var_names = ['Body position', 'Body speed', 'Body acceleration']

sta_zs_vars = []
sta_zs_405_vars = []
for var in var_names:
    path = os.path.join(path_session_data, 'TM RAW FILES', 'split contra fast', 'MC13419', '2022_05_31\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', 'split contra fast S1 310522')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
    animal = mscope.get_animal_id()
    session = 1
    # Session data and inputs
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    trials_ses_name.insert(len(trials_ses_name), 'late washout')
    trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]
    sta_zs = np.load(
        os.path.join(load_path, animal + ' split contra fast',
                     'sta_bodyvars_' + var.replace(' ', '_') + '_zscored.npy'))
    sta_zs_zoom = np.nanmean(sta_zs[:, :, xaxis_start:xaxis_end], axis=1)
    sta_zs_405 = np.load(
        os.path.join(load_path, animal + ' split contra fast 405', 'sta_bodyvars_' + var.replace(' ', '_') + '_zscored.npy'))
    sta_zs_zoom_405 = np.nanmean(sta_zs_405[:, :, xaxis_start:xaxis_end], axis=1)
    sta_zs_vars.append(sta_zs_zoom)
    sta_zs_405_vars.append(sta_zs_zoom_405)

#ANIMALS SUMMARY HEATMAP
fig, ax = plt.subplots(1, len(var_names), figsize=(20, 10), tight_layout='True')
for count_v, var in enumerate(var_names):
    hm = sns.heatmap(sta_zs_vars[count_v], vmax=np.nanpercentile(sta_zs_vars[count_v], 99.5),
                vmin=np.nanpercentile(sta_zs_vars[count_v], 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zs_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(xaxis[xaxis_start]), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[count_v].set_title(var, fontsize=16)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, len(var_names), figsize=(20, 10), tight_layout='True')
for count_v, var in enumerate(var_names):
    hm = sns.heatmap(sta_zs_405_vars[count_v], vmax=np.nanpercentile(sta_zs_405_vars[count_v], 99.5),
                vmin=np.nanpercentile(sta_zs_405_vars[count_v], 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zs_405_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(xaxis[xaxis_start]), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[count_v].set_title(var, fontsize=16)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_405'), dpi=mscope.my_dpi)
