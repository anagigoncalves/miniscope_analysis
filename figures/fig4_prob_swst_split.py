import matplotlib.collections
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
Ntrials = 26
trials = np.arange(1, Ntrials+1)
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
bins = np.arange(0, 1.01, 0.05)  # 5 deg
protocol = 'split ipsi fast'
paws = ['FR', 'HR', 'FL', 'HL']

# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.5]]) #AP, ML
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

animal_id = []
trial_id = []
roi_id = []
phase_id = []
prob_val = []
paw_id = []
coord_AP = []
coord_ML = []
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
    # Compute ROI coordinates
    coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    centroid_ext_swap = np.array(centroid_ext)[:, [1, 0]]
    fov_coord = fov_coords[count_a]
    fov_corner = np.array([fov_coord[0] - 0.5, fov_coord[1] - 0.5])
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    if protocol == 'split ipsi fast' and animal == 'MC8855':
        event_stride = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_cumulative_idx_rois.npy'), allow_pickle=True)
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, 3:, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split ipsi fast' and animal == 'MC9226':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, :-3, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC8855':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, 3:, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        trial_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25])
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, trial_idx, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    else:
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_amp_st = np.nanmax(firing_rate_animal[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_animal[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_animal[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_animal[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    for p, paw in enumerate(paws):
        for t, trial in enumerate(np.arange(1, Ntrials + 1)):
            animal_id.extend(np.repeat(animal, np.shape(firing_rate_amp_st)[0]))
            trial_id.extend(np.repeat(trial, np.shape(firing_rate_amp_st)[0]))
            phase_id.extend(np.repeat('st', np.shape(firing_rate_amp_st)[0]))
            paw_id.extend(np.repeat(paw, np.shape(firing_rate_amp_st)[0]))
            roi_id.extend(np.arange(0, np.shape(firing_rate_amp_st)[0]))
            prob_val.extend(firing_rate_amp_st[:, p, t])
            coord_AP.extend(centroid_dist_corner[:, 0])
            coord_ML.extend(centroid_dist_corner[:, 1])
            animal_id.extend(np.repeat(animal, np.shape(firing_rate_amp_sw)[0]))
            trial_id.extend(np.repeat(trial, np.shape(firing_rate_amp_sw)[0]))
            phase_id.extend(np.repeat('sw', np.shape(firing_rate_amp_sw)[0]))
            paw_id.extend(np.repeat(paw, np.shape(firing_rate_amp_sw)[0]))
            roi_id.extend(np.arange(0, np.shape(firing_rate_amp_sw)[0]))
            prob_val.extend(firing_rate_amp_sw[:, p, t])
            coord_AP.extend(centroid_dist_corner[:, 0])
            coord_ML.extend(centroid_dist_corner[:, 1])

prob_dict = {'animal': animal_id, 'trial': trial_id, 'roi': roi_id, 'phase': phase_id, 'prob': prob_val, 'paw': paw_id, 'coord_AP': coord_AP,
        'coord_ML': coord_ML}
df_amp = pd.DataFrame(prob_dict)