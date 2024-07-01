import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA

# Input data
plot_data = 1
protocol = 'split ipsi fast'
protocol_id = 'split_S1'
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\' + protocol + ' S1\\'
tied_input_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\tied baseline S1\\'
save_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\' + protocol + ' S1\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_' + protocol_id + '.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 105, 10)  # 5 deg
    align_event = 'st'  # is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.025)  # 25 ms
    bins_fr = bins * 1000
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

paws = ['FR', 'HR', 'FL', 'HL']
# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                       [6.24, 1],
                       [6.64, 1],
                       [6.48, 1.5],
                       [6.48, 1.7]])  # AP, ML

tied_idx = [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]

def zscoring(data, axis_value):
    if axis_value == 1:
        data_mean = np.repeat(np.nanmean(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
        data_std = np.repeat(np.nanstd(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
    if axis_value == 0:
        data_mean = np.tile(np.nanmean(data, axis=axis_value), (np.shape(data)[0], 1))
        data_std = np.tile(np.nanstd(data, axis=axis_value), (np.shape(data)[0], 1))
    data_zscore = (data - data_mean) / data_std
    return data_zscore

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

# Loop across animals for trial average
firing_rate_mean_trials_paws_list = []
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    firing_rate_mean_trials_paw_animalid = []
    roi_nr = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        roi_nr.append(np.shape(firing_rate_animal)[0])
        if protocol.split(' ')[0] == 'split':
            firing_rate_mean_trials_paw.append(
                zscoring(np.nanmean(firing_rate_animal[:, p, tied_idx[count_a], :], axis=1).T, 0))
        if protocol.split(' ')[0] == 'tied':
            firing_rate_mean_trials_paw.append(
                zscoring(np.nanmean(firing_rate_animal[:, p, :, :], axis=1).T, 0))
        firing_rate_mean_trials_paw_animalid.append(np.repeat(count_a, np.shape(firing_rate_animal)[1]))
    firing_rate_mean_trials_paw_concat = np.hstack(firing_rate_mean_trials_paw)
    # list of array of FR x time for each paw
    firing_rate_mean_trials_paws_list.append(firing_rate_mean_trials_paw_concat)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws = firing_rate_mean_trials_paw_concat
    else:
        firing_rate_animal_trials_concat_paws = np.concatenate(
            (firing_rate_animal_trials_concat_paws, firing_rate_mean_trials_paw_concat), axis=0)

# Get coordinates for all ROIs
roi_coordinates = []
animal_list = []
roi_name_list = []
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
    df_extract_rawtrace_detrended = pd.read_csv(
        os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
    roi_name_list.extend(df_extract_rawtrace_detrended.columns[2:])
    animal_list.extend(np.repeat(animal, len(df_extract_rawtrace_detrended.columns[2:])))
    # Compute ROI coordinates
    coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    centroid_ext_arr = np.array(centroid_ext)
    # Flip coords horizontally and vertically because image in miniscope is flipped
    centroid_ext_flip = np.zeros(np.shape(centroid_ext_arr))
    centroid_ext_flip[:, 1] = 1000 - centroid_ext_arr[:, 0]
    centroid_ext_flip[:, 0] = 1000 - centroid_ext_arr[:, 1]
    # Need to swap again, because now ML and AP are swapped
    # Adjust for the FOV coordinates to get global coordinates
    centroid_ext_swap = np.array(centroid_ext_flip)[:, [1, 0]]
    fov_coord = fov_coords[count_a]
    fov_corner = np.array(
        [fov_coord[1] - 0.5, fov_coord[0] - 0.5])  # ML is the centroid[:, 0] and AP the centroid[:, 1]
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    roi_coordinates.extend(centroid_dist_corner)
roi_coordinates_arr = np.array(roi_coordinates)

# PCA on concatenated space FR x (time x paws)
comp = 20
pca_fr_paws = PCA(n_components=comp)
firing_rate_animal_trials_concat_paws_tied = np.load(os.path.join(tied_input_path, 'pca_input_data_tied_baseline.npy'))
pca_fit_fr_paws_tied = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws_tied)
pca_fit_fr_paws_fit_transform_tied = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws_tied)
# Project session data onto tied PC space
# Can't do this as long as the number of features is different
# Run PCA on new dataset
pca_fr_paws_new = PCA(n_components=comp)
pca_fit_fr_paws_new = pca_fr_paws_new.fit(firing_rate_animal_trials_concat_paws)
pca_fit_fr_paws_fit_transform_new = pca_fr_paws_new.fit_transform(firing_rate_animal_trials_concat_paws)

# Check if PCs are that different
pca_fit_fr_single_paws_tied = np.reshape(pca_fit_fr_paws_fit_transform_tied.T, ((20, 4, len(bins) - 1)))
pca_fit_fr_single_paws_new = np.reshape(pca_fit_fr_paws_fit_transform_new.T, ((20, 4, len(bins) - 1)))
fig, ax = plt.subplots(2, 3, tight_layout=True, figsize=(10, 5))
for c in range(3):
    for count_p in range(len(paws)):
        ax[0, c].plot(bins_fr[:-1], pca_fit_fr_single_paws_tied[c, count_p, :], color=paw_colors[count_p], linewidth=3)
        ax[0, c].spines['right'].set_visible(False)
        ax[0, c].spines['top'].set_visible(False)
        ax[0, c].tick_params(axis='both', which='major', labelsize=20)
        ax[0, c].set_ylabel('arbitrary units', fontsize=20)
        ax[0, c].set_title('PC' + str(c + 1), fontsize=20)
        ax[0, c].set_xlabel('% Phase', fontsize=20)
        ax[0, c].axvline(x=50, color='black')
        ax[1, c].plot(bins_fr[:-1], pca_fit_fr_single_paws_new[c, count_p, :], color=paw_colors[count_p], linewidth=3)
        ax[1, c].spines['right'].set_visible(False)
        ax[1, c].spines['top'].set_visible(False)
        ax[1, c].tick_params(axis='both', which='major', labelsize=20)
        ax[1, c].set_ylabel('arbitrary units', fontsize=20)
        ax[1, c].set_title('PC' + str(c + 1), fontsize=20)
        ax[1, c].set_xlabel('% Phase', fontsize=20)
        ax[1, c].axvline(x=50, color='black')

# # Put coefficients into dataframe
# pc_coeff_df = pd.DataFrame({'animal': animal_list, 'roi': roi_name_list, 'coord_x': roi_coordinates_arr[:, 0],
#                             'coord_y': roi_coordinates_arr[:, 1]})
# for pc in range(5):
#     pc_coeff_df['PC'+str(pc+1)] = pca_fit_fr_paws_fit_transform[:, pc]
# # Save dataframe as csv
# pc_coeff_df.to_csv(
#     os.path.join(save_path, 'pc_coeff_df_' + protocol.replace(' ', '_') + '.csv'), sep=',', index=False)