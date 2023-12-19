import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import sklearn.metrics as sm

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters sw time\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'sw'
align_dimension = 'time'
if align_dimension == 'phase':
    bins = np.arange(0, 1.01, 0.05)  # 5 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
    bins_fr = bins*1000
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']
# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML

tied_idx = [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
split_idx = [np.arange(3, 12+1), np.arange(6, 15+1), np.arange(6, 15+1), np.arange(6, 15+1), np.arange(6, 15+1)]
washout_idx = [np.arange(13, 22+1), np.arange(16, 25+1), np.arange(16, 22+1), np.arange(16, 25+1), np.arange(16, 25+1)]
def zscoring(data, axis_value):
    if axis_value == 1:
        data_mean = np.repeat(np.nanmean(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
        data_std = np.repeat(np.nanstd(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
    if axis_value == 0:
        data_mean = np.tile(np.nanmean(data, axis=axis_value), (np.shape(data)[0], 1))
        data_std = np.tile(np.nanstd(data, axis=axis_value), (np.shape(data)[0], 1))
    data_zscore = (data - data_mean)/data_std
    return data_zscore
def minmax(data, axis_value):
    if axis_value == 1:
        data_mean = np.repeat(np.nanmean(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
        data_centered = data-data_mean
        data_min = np.repeat(np.nanmin(data_centered, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
        data_max = np.repeat(np.nanmax(data_centered, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
    if axis_value == 0:
        data_mean = np.tile(np.nanmean(data, axis=axis_value), (np.shape(data)[0], 1))
        data_centered = data-data_mean
        data_min = np.tile(np.nanmin(data, axis=axis_value), (np.shape(data)[0], 1))
        data_max = np.tile(np.nanmax(data, axis=axis_value), (np.shape(data)[0], 1))
    data_minmax = (data_centered-data_min)/(data_max-data_min)
    return data_minmax

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

# Loop across animals for trial average
firing_rate_mean_trials_paws_list = []
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    firing_rate_mean_trials_paw_animalid = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, :, :], axis=1),0))
        firing_rate_mean_trials_paw_animalid.append(np.repeat(count_a, np.shape(firing_rate_animal)[0]))
    firing_rate_mean_trials_paw_concat = np.vstack(firing_rate_mean_trials_paw)
    # list of array of FR x time for each paw
    firing_rate_mean_trials_paws_list.append(firing_rate_mean_trials_paw_concat)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws = firing_rate_mean_trials_paw_concat
    else:
        firing_rate_animal_trials_concat_paws = np.concatenate(
            (firing_rate_animal_trials_concat_paws, firing_rate_mean_trials_paw_concat), axis=1)
# # Zscore each paw firing rate using the concatenated mean and std (reference space)
# data = firing_rate_animal_trials_concat_paws
# col_len = np.shape(firing_rate_mean_trials_paws_list[0])[1]
# data_fr_mean = np.tile(np.nanmean(data, axis=0), (np.shape(data)[0], 1))[:, :col_len]
# data_fr_std = np.tile(np.nanstd(data, axis=0), (np.shape(data)[0], 1))[:, :col_len]
# firing_rate_mean_trials_paws_list_zscored = []
# for p in range(4):
#     data_paw_zscored = (firing_rate_mean_trials_paws_list[p] - data_fr_mean) / data_fr_std
#     firing_rate_mean_trials_paws_list_zscored.append(data_paw_zscored)

#Get coordinates for all ROIs
roi_coordinates = []
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
    centroid_ext_arr = np.array(centroid_ext)
    #Flip coords horizontally and vertically because image in miniscope is flipped
    centroid_ext_flip = np.zeros(np.shape(centroid_ext_arr))
    centroid_ext_flip[:, 1] = 1000-centroid_ext_arr[:, 0]
    centroid_ext_flip[:, 0] = 1000-centroid_ext_arr[:, 1]
    #Need to swap again, because now ML and AP are swapped
    #Adjust for the FOV coordinates to get global coordinates
    centroid_ext_swap = np.array(centroid_ext_flip)[:, [1, 0]] 
    fov_coord = fov_coords[count_a]
    fov_corner = np.array([fov_coord[1] - 0.5, fov_coord[0] - 0.5]) #ML is the centroid[:, 0] and AP the centroid[:, 1]
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    roi_coordinates.extend(centroid_dist_corner)
roi_coordinates_arr = np.array(roi_coordinates)

### Population activity cluster around sw or st - mean activity across trials
# PCA on concatenated space FR x (time x paws)
comp = 9
pca_fr_paws = PCA(n_components=comp)
pca_fit_fr_paws = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
pca_fit_fr_paws_fit_transform = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws)
# # OPTION 1 - PCA AVERAGED CONCATENATED FORM - MATRIX MULTIPLICATION
# pca_fit_fr_paws_scores = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws_zscored)
# # Project mean firing rate activity for each paw in the reference PC space
# pca_fit_fr_single_paws = []
# for count_p in range(len(paws)):
#     pca_fit_fr_single_paws.append(np.dot(firing_rate_mean_trials_paws_list_zscored[count_p].T, pca_fit_fr_paws_scores).T)
# OPTION 2 - PCA TRIAL AVERAGED FORM
pca_fit_fr_single_paws = np.reshape(pca_fit_fr_paws.components_, ((comp, 4, firing_rate_mean_trials_paw_concat.shape[1])))

# Explained variance for each component
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_paws.explained_variance_ratio_)*100,
    color='black', s=20)
# ax.set_title('Explained variance\nratio of components', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 100])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Cumulative explained\nvariance ratio (%)', fontsize=20)
ax.set_xlabel('Component number', fontsize=20)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance.svg'), dpi=256)

# First PCs over time
fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(15, 4), sharey=True)
ax = ax.ravel()
for c in range(3):
    for count_p in range(len(paws)):
        #ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[count_p][c, :], color=paw_colors[count_p], linewidth=2)
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[c, count_p, :], color=paw_colors[count_p], linewidth=3)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=20)
        ax[c].set_ylabel('Event rate\nz-scored', fontsize=20)
        ax[c].set_title('PC'+str(c+1), fontsize=20)
        if align_dimension == 'time':
            ax[c].set_xlabel('Time (ms)', fontsize=20)
            ax[c].axvline(x=0, color='black')
        if align_dimension == 'phase':
            ax[c].set_xlabel('% Phase', fontsize=20)
            ax[c].axvline(x=0.5, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_components'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_components.svg'), dpi=256)

# Trajectories in PC space - 3 dimensions (in 2 circular manifold)
fig = plt.figure(tight_layout=True)
ax = plt.axes(projection='3d')
for count_p in range(len(paws)):
    # ax.plot3D(pca_fit_fr_single_paws[count_p][0, :], pca_fit_fr_single_paws[count_p][1, :],
    #          pca_fit_fr_single_paws[count_p][2, :], color=paw_colors[count_p], linewidth=2)
    # ax.scatter(pca_fit_fr_single_paws[count_p][0, 10], pca_fit_fr_single_paws[count_p][1, 10],
    #          pca_fit_fr_single_paws[count_p][2, 10], color=paw_colors[count_p], s=60)
    ax.plot3D(pca_fit_fr_single_paws[0, count_p, :], pca_fit_fr_single_paws[1, count_p, :],
           pca_fit_fr_single_paws[2, count_p, :], color=paw_colors[count_p], linewidth=2)
    ax.scatter(pca_fit_fr_single_paws[0, count_p, 10], pca_fit_fr_single_paws[1, count_p, 10],
            pca_fit_fr_single_paws[2, count_p, 10], color=paw_colors[count_p], s=60)
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_ticklabels([])
    axis._axinfo['axisline']['linewidth'] = 2
    axis._axinfo['grid']['color'] = [1, 1, 1]
    axis.set_pane_color((1, 1, 1))
ax.set_xlabel('PC1 (' + str(np.round(pca_fit_fr_paws.explained_variance_ratio_[0]*100, 1)) + '%)', fontsize=20)
ax.set_ylabel('PC2 (' + str(np.round(pca_fit_fr_paws.explained_variance_ratio_[1]*100, 1)) + '%)', fontsize=20)
ax.set_zlabel('PC3 (' + str(np.round(pca_fit_fr_paws.explained_variance_ratio_[2]*100, 1)) + '%)', fontsize=20)
ax.view_init(10, 30)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories.svg'), dpi=256)

for c in range(3):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=pca_fit_fr_paws_fit_transform[:, c], cmap='coolwarm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    ax.set_title('PC' + str(c + 1), fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    cbar.mappable.set_clim([-15, 15])
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation.svg'),
                dpi=256)

# Reconstruction error
nrmse = np.zeros(20)
data_X = firing_rate_animal_trials_concat_paws
for c in range(20):
    pca_fr_paws = PCA(n_components=c+1)
    pca_fit_fr_paws_models = pca_fr_paws.fit(data_X)
    data_PCA = pca_fit_fr_paws_models.transform(data_X)
    data_X_rPCA = pca_fit_fr_paws_models.inverse_transform(data_PCA)
    r2 = sm.r2_score(data_X, data_X_rPCA)
    rmse = np.sqrt(sm.mean_squared_error(data_X, data_X_rPCA))
    nrmse[c] = rmse/np.sqrt(np.mean(data_X**2))
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(1, 21), nrmse,
    color='black', s=20)
ax.scatter(comp, nrmse[comp-1], s=20, color='darkgrey')
# ax.set_title('Reconstruction error\nof components', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 1])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Normalized RMSE', fontsize=20)
ax.set_xlabel('Number of components', fontsize=20)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_reconstruction_error'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_reconstruction_error.svg'), dpi=256)
