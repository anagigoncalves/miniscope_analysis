import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split contra fast'
align_dimension = 'phase'
bins = np.arange(0, 105, 10)  # 5 deg
bins_fr = bins
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

paws = ['FR', 'HR', 'FL', 'HL']
# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML

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

# Loop across animals for trial average baseline concatenation
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    firing_rate_mean_trials_paw_animalid = []
    roi_nr = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        roi_nr.append(np.shape(firing_rate_animal)[0])
        if animal == 'MC8855':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, :3, :], axis=1).T, 0))
        else:
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, :6, :], axis=1).T, 0))
        firing_rate_mean_trials_paw_animalid.append(np.repeat(count_a, np.shape(firing_rate_animal)[1]))
    firing_rate_mean_trials_paw_concat_bs = np.hstack(firing_rate_mean_trials_paw)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws_bs = firing_rate_mean_trials_paw_concat_bs
    else:
        firing_rate_animal_trials_concat_paws_bs = np.concatenate(
            (firing_rate_animal_trials_concat_paws_bs, firing_rate_mean_trials_paw_concat_bs), axis=0)
# Loop across animals for trial average early split concatenation
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if animal == 'MC8855':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [3, 4], :], axis=1).T, 0))
        else:
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [6, 7], :], axis=1).T, 0))
    firing_rate_mean_trials_paw_concat_es = np.hstack(firing_rate_mean_trials_paw)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws_es = firing_rate_mean_trials_paw_concat_es
    else:
        firing_rate_animal_trials_concat_paws_es = np.concatenate(
            (firing_rate_animal_trials_concat_paws_es, firing_rate_mean_trials_paw_concat_es), axis=0)
# Loop across animals for trial average late split concatenation
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if animal == 'MC8855':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [11, 12], :], axis=1).T, 0))
        elif animal == 'MC10221' and protocol == 'split contra fast':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [13, 14], :], axis=1).T, 0))
        else:
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [14, 15], :], axis=1).T, 0))
    firing_rate_mean_trials_paw_concat_ls = np.hstack(firing_rate_mean_trials_paw)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws_ls = firing_rate_mean_trials_paw_concat_ls
    else:
        firing_rate_animal_trials_concat_paws_ls = np.concatenate(
            (firing_rate_animal_trials_concat_paws_ls, firing_rate_mean_trials_paw_concat_ls), axis=0)
# Loop across animals for trial average after-effect concatenation
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if animal == 'MC8855':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [13, 14], :], axis=1).T, 0))
        elif animal == 'MC10221' and protocol == 'split contra fast':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [15, 16], :], axis=1).T, 0))
        else:
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [16, 17], :], axis=1).T, 0))
    firing_rate_mean_trials_paw_concat_ae = np.hstack(firing_rate_mean_trials_paw)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws_ae = firing_rate_mean_trials_paw_concat_ae
    else:
        firing_rate_animal_trials_concat_paws_ae = np.concatenate(
            (firing_rate_animal_trials_concat_paws_ae, firing_rate_mean_trials_paw_concat_ae), axis=0)
# Loop across animals for trial average late washout concatenation
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if animal == 'MC8855':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [21, 22], :], axis=1).T, 0))
        elif animal == 'MC10221' and protocol == 'split contra fast':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [23, 24], :], axis=1).T, 0))
        elif animal == 'MC9226' and protocol == 'split ipsi fast':
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [21, 22], :], axis=1).T, 0))
        else:
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, [24, 25], :], axis=1).T, 0))
    firing_rate_mean_trials_paw_concat_lw = np.hstack(firing_rate_mean_trials_paw)
    # Paw concatenation FR x (timexpaws)
    if p == 0:
        firing_rate_animal_trials_concat_paws_lw = firing_rate_mean_trials_paw_concat_lw
    else:
        firing_rate_animal_trials_concat_paws_lw = np.concatenate(
            (firing_rate_animal_trials_concat_paws_lw, firing_rate_mean_trials_paw_concat_lw), axis=0)

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

### Population activity cluster around sw and st - mean activity across trials
# PCA on concatenated space FR x (time x paws)
comp = 9
pca_fr_paws_bs = PCA(n_components=comp)
pca_fit_fr_paws_bs = pca_fr_paws_bs.fit(firing_rate_animal_trials_concat_paws_bs)
pca_fit_fr_paws_fit_transform_bs = pca_fr_paws_bs.fit_transform(firing_rate_animal_trials_concat_paws_bs)

# Explained variance for each component
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_paws_bs.explained_variance_ratio_)*100,
    color='black', s=20)
ax.set_xticks([0, 3, 6, 9])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 100])
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Cumulative explained\nvariance ratio (%)', fontsize=20)
ax.set_xlabel('Component number', fontsize=20)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_dimension + '_explained_variance'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_dimension + '_explained_variance.svg'), dpi=256)

# Reconstruction of PC - baseline
pca_fr_paws = PCA(n_components=4)
pca_fit_fr_paws_models_bs = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws_bs)
data_PCA_bs = pca_fit_fr_paws_models_bs.transform(firing_rate_animal_trials_concat_paws_bs)
pca_fit_fr_single_paws_bs = np.reshape(data_PCA_bs.T, ((4, 4, len(bins)-1)))
fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(15, 5), sharey=True)
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws_bs[c, count_p, :], color=paw_colors[count_p], linewidth=3)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=20)
    ax[c].set_ylabel('arbitrary units', fontsize=20)
    ax[c].set_title('PC' + str(c + 1), fontsize=20)
    ax[c].set_xlabel('Phase (%)', fontsize=20)
    ax[c].axvline(x=50, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_bs_' + align_dimension + '_temporaldimension'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_bs_' + align_dimension + '_temporaldimension.svg'), dpi=256)

# First PCs contribution to each ROI
for c in range(4):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=pca_fit_fr_paws_models_bs.components_[c, :], cmap='coolwarm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    ax.set_title('PC' + str(c + 1), fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    cbar.mappable.set_clim([-0.1, 0.1])
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_baseline_' + align_dimension + '_pc' + str(c+1) + '_roilocation'), dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_baseline_' + align_dimension + '_pc' + str(c+1) + '_roilocation.svg'), dpi=256)

# Reconstruction of PC - early split
data_PCA_es = pca_fit_fr_paws_models_bs.transform(firing_rate_animal_trials_concat_paws_es)
pca_fit_fr_single_paws_es = np.reshape(data_PCA_es.T, ((4, 4, len(bins)-1)))
fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(15, 5), sharey=True)
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws_es[c, count_p, :], color=paw_colors[count_p], linewidth=3)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=20)
    ax[c].set_ylabel('arbitrary units', fontsize=20)
    ax[c].set_title('PC' + str(c + 1), fontsize=20)
    ax[c].set_xlabel('Phase (%)', fontsize=20)
    ax[c].axvline(x=50, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_earlysplit_' + align_dimension + '_temporaldimension'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_earlysplit_' + align_dimension + '_temporaldimension.svg'), dpi=256)

# Reconstruction of PC - late split
data_PCA_ls = pca_fit_fr_paws_models_bs.transform(firing_rate_animal_trials_concat_paws_ls)
pca_fit_fr_single_paws_ls = np.reshape(data_PCA_ls.T, ((4, 4, len(bins)-1)))
fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(15, 5), sharey=True)
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws_ls[c, count_p, :], color=paw_colors[count_p], linewidth=3)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=20)
    ax[c].set_ylabel('arbitrary units', fontsize=20)
    ax[c].set_title('PC' + str(c + 1), fontsize=20)
    ax[c].set_xlabel('Phase (%)', fontsize=20)
    ax[c].axvline(x=50, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_latesplit_' + align_dimension + '_temporaldimension'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_latesplit_' + align_dimension + '_temporaldimension.svg'), dpi=256)

# Reconstruction of PC - after-effect
data_PCA_ae = pca_fit_fr_paws_models_bs.transform(firing_rate_animal_trials_concat_paws_ae)
pca_fit_fr_single_paws_ae = np.reshape(data_PCA_ae.T, ((4, 4, len(bins)-1)))
fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(15, 5), sharey=True)
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws_ae[c, count_p, :], color=paw_colors[count_p], linewidth=3)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=20)
    ax[c].set_ylabel('arbitrary units', fontsize=20)
    ax[c].set_title('PC' + str(c + 1), fontsize=20)
    ax[c].set_xlabel('Phase (%)', fontsize=20)
    ax[c].axvline(x=50, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aftereffect_' + align_dimension + '_temporaldimension'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aftereffect_' + align_dimension + '_temporaldimension.svg'), dpi=256)

# Reconstruction of PC - late washout
data_PCA_lw = pca_fit_fr_paws_models_bs.transform(firing_rate_animal_trials_concat_paws_lw)
pca_fit_fr_single_paws_lw = np.reshape(data_PCA_lw.T, ((4, 4, len(bins)-1)))
fig, ax = plt.subplots(1, 4, tight_layout=True, figsize=(15, 5), sharey=True)
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws_lw[c, count_p, :], color=paw_colors[count_p], linewidth=3)
    ax[c].spines['right'].set_visible(False)
    ax[c].spines['top'].set_visible(False)
    ax[c].tick_params(axis='both', which='major', labelsize=20)
    ax[c].set_ylabel('arbitrary units', fontsize=20)
    ax[c].set_title('PC' + str(c + 1), fontsize=20)
    ax[c].set_xlabel('Phase (%)', fontsize=20)
    ax[c].axvline(x=50, color='black')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_latewashout_' + align_dimension + '_temporaldimension'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_latewashout_' + align_dimension + '_temporaldimension.svg'), dpi=256)

count_p = 0 #FR paws
for pc in range(4):
    fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
    ax.plot(bins_fr[:-1], pca_fit_fr_single_paws_bs[pc, count_p, :], color='black', linewidth=3)
    ax.plot(bins_fr[:-1], pca_fit_fr_single_paws_es[pc, count_p, :]+(1*20), color='crimson', linewidth=3)
    ax.plot(bins_fr[:-1], pca_fit_fr_single_paws_ls[pc, count_p, :]+(2*20), color='salmon', linewidth=3)
    ax.plot(bins_fr[:-1], pca_fit_fr_single_paws_ae[pc, count_p, :]+(3*20), color='blue', linewidth=3)
    ax.plot(bins_fr[:-1], pca_fit_fr_single_paws_lw[pc, count_p, :]+(4*20), color='dodgerblue', linewidth=3)
    ax.axvline(x=50, color='black')
    ax.set_title('PC' + str(pc + 1), fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.set_ylabel('Arbitrary units', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='y', which='y', length=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_split_stages_FR_pc' + str(pc+1) + '_' + align_dimension + '_temporaldimension'),
                dpi=256)
    plt.savefig(
        os.path.join(save_path, 'pca_mean_firingrate_split_stages_FR_pc' + str(pc+1) + '_' + align_dimension + '_temporaldimension.svg'),
        dpi=256)

