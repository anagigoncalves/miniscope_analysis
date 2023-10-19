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
load_path = path_session_data + '\\Analysis on population data\\STA paw spatial diff\\split contra fast S1\\'
save_path = 'J:\\Thesis\\for figures\\405 control\\'
protocol_type = 'split'
cond_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-1, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
var_name = 'FR-FL'
animals = ['MC13420'] #MC13419 has very few ROIs

fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89]]) #AP, ML

sort_type = 'ML'

sta_ap = []
sta_ml = []
sta_480_animals = []
sta_405_animals = []
sta_animal_id = []
for animal in animals:
    #load 480 session
    path = os.path.join(path_session_data, 'TM RAW FILES', 'split contra fast', animal, '2022_05_31\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', 'split contra fast S1 310522')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
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
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    trials_ses_name.insert(len(trials_ses_name), 'late washout')
    trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]

    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = fov_coords[0]
    fov_corner = np.array([fov_coord[0] + 0.5, fov_coord[1] - 0.5])
    centroid_dist_corner = (np.array(centroid_ext) * 0.001) + fov_corner

    sta = np.load(
        os.path.join(load_path, animal + ' split contra fast',
                     'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))
    # load 405 STA data
    sta_405 = np.load(
        os.path.join(load_path, animal + ' split contra fast 405', 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))
    trials_ses_split = trials_ses.flatten()[1:]
    sta_zoom_480 = np.zeros((np.shape(sta)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zoom_480[:] = np.nan
    sta_zoom_405 = np.zeros((np.shape(sta)[0], len(cond_name), xaxis_end - xaxis_start))
    sta_zoom_405[:] = np.nan
    for count_t, t in enumerate(trials_ses_name): #if odd is -1, if even is the next
        if count_t % 2 == 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t]-1)[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
        if count_t % 2 != 0:
            trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
            trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t]+1)[0][0]]
        sta_zoom_480[:, count_t, :] = np.nanmean(
            sta[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
        sta_zoom_405[:, count_t, :] = np.nanmean(
            sta_405[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
    sta_480_animals.append(sta_zoom_480)
    sta_405_animals.append(sta_zoom_405)
    sta_animal_id.append(np.repeat(animal, np.shape(sta_zoom_480)[0]))
    sta_ap.extend(centroid_dist_corner[:, 0])
    sta_ml.extend(centroid_dist_corner[:, 1])
#sort ML or AP values
sort_ml = np.argsort(sta_ml)
sort_ap = np.argsort(sta_ap)
#concatenate ROIs across animals
sta_zoom_all_480_concat = np.concatenate(sta_480_animals)
sta_zoom_all_405_concat = np.concatenate(sta_405_animals)
sta_animal_id_concat = np.concatenate(sta_animal_id)
if sort_type == 'ML':
    sta_zoom_all_480_sort = sta_zoom_all_480_concat[sort_ml]
    sta_zoom_all_405_sort = sta_zoom_all_405_concat[sort_ml]
    sta_animal_id_sort = sta_animal_id_concat[sort_ml]
if sort_type == 'AP':
    sta_zoom_all_480_sort = sta_zoom_all_480_concat[sort_ap]
    sta_zoom_all_405_sort = sta_zoom_all_405_concat[sort_ap]
    sta_animal_id_sort = sta_animal_id_concat[sort_ap]
if sort_type == 'none':
    sta_zoom_all_480_sort = sta_zoom_all_480_concat
    sta_zoom_all_405_sort = sta_zoom_all_405_concat
    sta_animal_id_sort = sta_animal_id_concat

#ANIMALS SUMMARY HEATMAP
fig, ax = plt.subplots(1, np.shape(sta_zoom_all_480_sort)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_480_sort)[1]):
    hm = sns.heatmap(sta_zoom_all_480_sort[:, t, :], vmax=np.nanpercentile(sta_zoom_all_480_sort[:, t, :], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_480_sort[:, t, :], 0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_480_sort[:, t, :])[1]]))
    ax[t].set_xticklabels([str(xaxis[xaxis_start]), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[t].set_yticks(np.arange(0, np.shape(sta_zoom_all_480_sort[t])[0], 50))
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml[::50]), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap[::50]), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id_sort[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_pawdiff_FR-FL_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, np.shape(sta_zoom_all_405_sort)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_405_sort)[1]):
    hm = sns.heatmap(sta_zoom_all_405_sort[:, t, :], vmax=np.nanpercentile(sta_zoom_all_405_sort[:, t, :], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_405_sort[:, t, :], 0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_405_sort[:, t, :])[1]]))
    ax[t].set_xticklabels([str(xaxis[xaxis_start]), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[t].set_yticks(np.arange(0, np.shape(sta_zoom_all_480_sort[t])[0], 50))
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ml[::50]), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[t].set_yticklabels(list(map(str, np.round(np.sort(sta_ap[::50]), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[t].set_ylabel('   '.join(sta_animal_id_sort[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_pawdiff_FR-FL_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_405'), dpi=mscope.my_dpi)
