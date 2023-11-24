import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import sklearn.metrics as sm

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig3\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'st'
align_dimension = 'phase'
trials = np.arange(1, 21)
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
fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89],
                     [6.98, 1.47],
                     [6.39, 1.62],
                     [6.80, 1.75]]) #AP, ML

tied_idx = [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
split_idx = [np.arange(3, 12+1), np.arange(6, 15+1), np.arange(6, 15+1), np.arange(6, 15+1), np.arange(6, 15+1)]
washout_idx = [np.arange(13, 22+1), np.arange(16, 25+1), np.arange(16, 22+1), np.arange(16, 25+1), np.arange(16, 25+1)]
def zscoring(data):
    data_zscore = (data - np.nanmean(data, axis=0))/np.nanstd(data, axis=0)
    return data_zscore

def min_max(data):
    data_minmax = (data-np.nanmin(data, axis=0))/(np.nanmax(data, axis=0)-np.nanmin(data, axis=0))
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
        firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, :, :], axis=1)))
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

#Get coordinates for all ROIs
roi_coordinates = []
sl_animals = []
coo_animals = []
ds_animals = []
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
    fov_coord = fov_coords[0]
    fov_corner = np.array([fov_coord[0] + 0.5, fov_coord[1] - 0.5])
    centroid_dist_corner = (np.array(centroid_ext) * 0.001) + fov_corner
    roi_coordinates.extend(centroid_dist_corner)
    # Compute learning
    trials_ses = np.load(os.path.join(path, 'processed files', 'trials.npy'))
    frames_dFF = np.load(os.path.join(path, 'processed files', 'black_frames.npy'), allow_pickle=True)
    filelist = loco.get_track_files(animal, session)
    sl_sym_mean = np.zeros(len(trials_ses))
    coo_sym_mean = np.zeros(len(trials_ses))
    ds_sym_mean = np.zeros(len(trials_ses))
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_dFF[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        sl_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
        sl_sym_mean[count_trial] = np.nanmean(sl_trials[0])-np.nanmean(sl_trials[2])
        ds_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'double_support')
        ds_sym_mean[count_trial] = np.nanmean(ds_trials[0])-np.nanmean(ds_trials[2])
        coo_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'coo')
        coo_sym_mean[count_trial] = np.nanmean(coo_trials[0])-np.nanmean(coo_trials[2])
    sl_bs_mean = (sl_sym_mean - np.nanmean(sl_sym_mean[tied_idx[count_a]]))
    coo_bs_mean = (coo_sym_mean - np.nanmean(coo_sym_mean[tied_idx[count_a]]))
    ds_bs_mean = (ds_sym_mean - np.nanmean(ds_sym_mean[tied_idx[count_a]]))
    # after-effect
    sl_animals.extend(np.repeat(sl_bs_mean[washout_idx[count_a][0]], len(centroid_dist_corner)))
    ds_animals.extend(np.repeat(ds_bs_mean[washout_idx[count_a][0]], len(centroid_dist_corner)))
    coo_animals.extend(np.repeat(coo_bs_mean[washout_idx[count_a][0]], len(centroid_dist_corner)))
    # change over split
    # sl_animals.extend(np.repeat(sl_bs_mean[split_idx[count_a][-1]]-sl_bs_mean[split_idx[count_a][0]], len(centroid_dist_corner)))
    # ds_animals.extend(np.repeat(ds_bs_mean[split_idx[count_a][-1]]-ds_bs_mean[split_idx[count_a][0]], len(centroid_dist_corner)))
    # coo_animals.extend(np.repeat(coo_bs_mean[split_idx[count_a][-1]]-coo_bs_mean[split_idx[count_a][0]], len(centroid_dist_corner)))
roi_coordinates_arr = np.array(roi_coordinates)
sl_animals_arr = np.array(sl_animals)
ds_animals_arr = np.array(ds_animals)
coo_animals_arr = np.array(coo_animals)

### Population activity cluster around sw or st - mean activity across trials
# PCA on concatenated space FR x (time x paws)
comp = 9
pca_fr_paws = PCA(n_components=comp)
pca_fit_fr_paws = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
pca_fit_fr_paws_fit_transform = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws)
# OPTION 1 - PCA AVERAGED CONCATENATED FORM - MATRIX MULTIPLICATION
# pca_fit_fr_paws_scores = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws)
# # Project mean firing rate activity for each paw in the reference PC space
# pca_fit_fr_single_paws = []
# for count_p in range(len(paws)):
#     pca_fit_fr_single_paws.append(np.dot(firing_rate_mean_trials_paws_list[count_p].T, pca_fit_fr_paws_scores).T)
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
        # ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[count_p][c, :], color=paw_colors[count_p], linewidth=2)
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[c, count_p, :], color=paw_colors[count_p], linewidth=3)
        # ax[c].set_title('Component ' + str(c+1), fontsize=16)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=20)
        ax[c].set_ylabel('Firing rate\nnorm. (Hz)', fontsize=20)
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
    #         pca_fit_fr_single_paws[count_p][2, :], color=paw_colors[count_p], linewidth=2)
    # ax.scatter(pca_fit_fr_single_paws[count_p][0, 10], pca_fit_fr_single_paws[count_p][1, 10],
    #         pca_fit_fr_single_paws[count_p][2, 10], color=paw_colors[count_p], s=60)
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
# ax.view_init(20, 80)
ax.view_init(10, 60)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories.svg'), dpi=256)

for c in range(3):
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    sc = ax.scatter(roi_coordinates_arr[:, 1], roi_coordinates_arr[:, 0], s=5, c=pca_fit_fr_paws_fit_transform[:, c], cmap='coolwarm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_roilocation.svg'),
                dpi=256)

for c in range(3):
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    ax.scatter(pca_fit_fr_paws_fit_transform[:, c], sl_animals_arr, color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('PC ' + str(c+1) + ' score\nfor each ROI', fontsize=20)
    ax.set_ylabel('Step length\n after-effect (mm)', fontsize=20)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_learning'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_pc' + str(c+1) + '_learning.svg'),
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
