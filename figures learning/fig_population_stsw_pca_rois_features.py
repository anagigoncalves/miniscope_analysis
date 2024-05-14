import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns

# Input data
plot_data = 0
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 105, 10)  # 5 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.025) # 25 ms
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

def zscoring(data, axis_value):
    if axis_value == 1:
        data_mean = np.repeat(np.nanmean(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
        data_std = np.repeat(np.nanstd(data, axis=axis_value).T, [np.shape(data)[1]], axis=0).reshape(np.shape(data))
    if axis_value == 0:
        data_mean = np.tile(np.nanmean(data, axis=axis_value), (np.shape(data)[0], 1))
        data_std = np.tile(np.nanstd(data, axis=axis_value), (np.shape(data)[0], 1))
    data_zscore = (data - data_mean)/data_std
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
            firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, tied_idx[count_a], :], axis=1).T, 0))
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

#Get coordinates for all ROIs
roi_coordinates = []
animal_list = []
roi_name_list = []
cluster_id_rois = []
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
    #get cluster ID per ROI
    idx_roi_cluster_ordered = np.load(os.path.join(mscope.path, 'processed files', 'clusters_rois_idx_order.npy'),
                                      allow_pickle=True)
    if count_a == 0:
        max_prev_animal = 0
    else:
        max_prev_animal = np.max(cluster_id)
    cluster_id = idx_roi_cluster_ordered+max_prev_animal
    cluster_id_rois.extend(cluster_id)
roi_coordinates_arr = np.array(roi_coordinates)

### Population activity cluster around sw or st - mean activity across trials
# PCA on concatenated space FR x (time x paws)
comp = 9
pca_fr_paws = PCA(n_components=comp)
pca_fit_fr_paws = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
pca_fit_fr_paws_fit_transform = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws)

pc_coeff_df = pd.DataFrame({'animal': animal_list, 'roi': roi_name_list, 'coord_x': roi_coordinates_arr[:, 0],
            'coord_y': roi_coordinates_arr[:, 1], 'cluster_id': np.array(cluster_id_rois)})
for pc in range(comp):
    pc_coeff_df['PC'+str(pc+1)] =  pca_fit_fr_paws.components_[pc, :]
pc_coeff_df.to_csv(
    os.path.join(save_path, 'pc_coeff_df_' + protocol.replace(' ','_') + '.csv'), sep=',', index=False)

if plot_data:
    # Check input data not zscored
    for p in range(len(paws)):
        firing_rate_mean_trials_paw = []
        for count_a, animal in enumerate(animals):
            firing_rate_animal = np.load(
                os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
            firing_rate_mean_trials_paw.append(np.nanmean(firing_rate_animal[:, p, :, :], axis=1))
        firing_rate_mean_trials_paw_concat = np.vstack(firing_rate_mean_trials_paw)
        # Paw concatenation FR x (timexpaws)
        if p == 0:
            firing_rate_animal_trials_concat_paws_nozscore = firing_rate_mean_trials_paw_concat
        else:
            firing_rate_animal_trials_concat_paws_nozscore = np.concatenate(
                (firing_rate_animal_trials_concat_paws_nozscore, firing_rate_mean_trials_paw_concat), axis=1)
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
    hm = sns.heatmap(firing_rate_animal_trials_concat_paws_nozscore[
                     np.argsort(np.argmax(firing_rate_animal_trials_concat_paws_nozscore[:, :len(bins) - 1], axis=1)),
                     :20],
                     ax=ax, cmap='viridis')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.set_xticks(np.linspace(0, 20, 10))
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
    ax.set_ylabel('ROI #', fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'input_data_sorted_peak_nozscoring_' + align_event + '_' + align_dimension),
                dpi=256)
    plt.savefig(
        os.path.join(save_path, 'input_data_sorted_peak_nozscoring_' + align_event + '_' + align_dimension + '.svg'),
        dpi=256)

    # Explained variance for each component
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_paws.explained_variance_ratio_)*100,
        color='black', s=20)
    # ax.set_title('Explained variance\nratio of components', fontsize=20)
    ax.set_xticks([0, 3, 6, 9])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([0, 100])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('Cumulative explained\nvariance ratio (%)', fontsize=20)
    ax.set_xlabel('Component number', fontsize=20)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance.svg'), dpi=256)

    # First PCs contribution to each ROI
    for c in range(4):
        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
        sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=pca_fit_fr_paws.components_[c, :], cmap='coolwarm')
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
        plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation'), dpi=256)
        plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation.svg'), dpi=256)

    # Trajectories
    sw_idx = np.int64(((len(bins)-1)/2)-1)
    pca_fit_fr_components_paws = np.reshape(pca_fit_fr_paws_fit_transform.T, ((9, 4, len(bins)-1)))
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    for count_p in range(len(paws)):
        ax.plot(pca_fit_fr_components_paws[0, count_p, :], pca_fit_fr_components_paws[1, count_p, :],
                color=paw_colors[count_p], linewidth=2)
        ax.scatter(pca_fit_fr_components_paws[0, count_p, sw_idx], pca_fit_fr_components_paws[1, count_p, sw_idx],
                   color=paw_colors[count_p], s=60)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('PC2', fontsize=20)
    ax.set_xlabel('PC3', fontsize=20)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories.svg'), dpi=256)

    # Reconstruction of PC
    pca_fr_paws = PCA(n_components=3)
    pca_fit_fr_paws_models = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
    data_PCA = pca_fit_fr_paws_models.transform(firing_rate_animal_trials_concat_paws)
    pca_fit_fr_single_paws = np.reshape(data_PCA.T, ((3, 4, len(bins)-1)))
    fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=(15, 5), sharey=True)
    ax = ax.ravel()
    for c in range(3):
        for count_p in range(len(paws)):
            ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[c, count_p, :], color=paw_colors[count_p], linewidth=3)
            ax[c].spines['right'].set_visible(False)
            ax[c].spines['top'].set_visible(False)
            ax[c].tick_params(axis='both', which='major', labelsize=20)
            ax[c].set_ylabel('arbitrary units', fontsize=20)
            ax[c].set_title('PC'+str(c+1), fontsize=20)
            if align_dimension == 'time':
                ax[c].set_xlabel('Time (ms)', fontsize=20)
                ax[c].axvline(x=0, color='black')
            if align_dimension == 'phase':
                ax[c].set_xlabel('% Phase', fontsize=20)
                ax[c].axvline(x=50, color='black')
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_temporaldimension'), dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_temporaldimension.svg'), dpi=256)

    # Input data sorted by peak location
    firing_rate_animal_trials_concat_paws_transpose = firing_rate_animal_trials_concat_paws.T
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
    hm = sns.heatmap(firing_rate_animal_trials_concat_paws_transpose[np.argsort(np.argmax(firing_rate_animal_trials_concat_paws_nozscore[:, :len(bins)-1], axis=1)), :len(bins)-1],
            ax=ax, cmap='viridis')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.set_xticks(np.linspace(0, len(bins)-1, 10))
    ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
    ax.set_ylabel('ROI #', fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'input_data_sorted_peak_' + align_event + '_' + align_dimension), dpi=256)
    plt.savefig(os.path.join(save_path, 'input_data_sorted_peak_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

    # Reconstructed data sorted by peak location (first 3 PCs)
    pca_fr_paws = PCA(n_components=4)
    pca_fit_fr_paws_models = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
    data_PCA = pca_fit_fr_paws_models.transform(firing_rate_animal_trials_concat_paws)
    data_X_rPCA = pca_fit_fr_paws_models.inverse_transform(data_PCA)
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
    hm = sns.heatmap(data_X_rPCA[:20, np.argsort(np.argmax(firing_rate_animal_trials_concat_paws_nozscore[:, :len(bins)-1], axis=1))].T,
            ax=ax, cmap='viridis')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.set_xticks(np.linspace(0, 20, 10))
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
    ax.set_ylabel('ROI #', fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'pca_3comp_data_sorted_peak_' + align_event + '_' + align_dimension), dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_3comp_data_sorted_peak_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

