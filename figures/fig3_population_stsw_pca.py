import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig3\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
path_data = 'J:\\Miniscope processed files\\TM RAW FILES\\'
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'tied baseline'
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
# paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']
greys = mp.cm.get_cmap('Greys', 14)
reds = mp.cm.get_cmap('Reds', 23)
blues = mp.cm.get_cmap('Blues', 23)
colors_session = {1: greys(14), 2: greys(10), 3: greys(7), 4: reds(23), 5: reds(21), 6: reds(19), 7: reds(17),
                  8: reds(15), 9: reds(13), 10: reds(11), 11: reds(9), 12: reds(7), 13: reds(5), 14: blues(23),
                  15: blues(21), 16: blues(19),  17: blues(17), 18: blues(15), 19: blues(13), 20: blues(11)}

def zscoring(data):
    data_zscore = (data - np.nanmean(data, axis=0))/np.nanstd(data, axis=0)
    return data_zscore

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

### Population activity cluster around sw or st - mean activity across trials
# PCA on concatenated space FR x (time x paws)
comp = 9
pca_fr_paws = PCA(n_components=comp)
pca_fit_fr_paws = pca_fr_paws.fit(firing_rate_animal_trials_concat_paws)
pca_fit_fr_paws_scores = pca_fr_paws.fit_transform(firing_rate_animal_trials_concat_paws)

# Project mean firing rate activity for each paw in the reference PC space
pca_fit_fr_single_paws = []
for count_p in range(len(paws)):
    pca_fit_fr_single_paws.append(np.dot(firing_rate_mean_trials_paws_list[count_p].T, pca_fit_fr_paws_scores).T)

# Explained variance for each component
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_paws.explained_variance_ratio_),
    color='black', s=20)
ax.set_title('Explained variance\nratio of components', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([0, 1])
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_ylabel('Cumulative explained\nvariance ratio', fontsize=16)
ax.set_xlabel('Component number', fontsize=16)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)

# First PCs over time
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7), sharey=True)
ax = ax.ravel()
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_single_paws[count_p][c, :], color=paw_colors[count_p], linewidth=2)
        ax[c].set_title('Component ' + str(c+1), fontsize=16)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=14)
        # ax[c].set_ylabel('Firing rate (Hz) z-scored', fontsize=14)
        if align_dimension == 'time':
            ax[c].set_xlabel('Time (ms)', fontsize=14)
            ax[c].axvline(x=0, color='black')
        if align_dimension == 'phase':
            ax[c].set_xlabel('% Phase', fontsize=14)
            ax[c].axvline(x=0.5, color='black')
        # ax[c].set_ylim([-0.45, 0.65])
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_components'), dpi=256)

# Trajectories in PC space
fig = plt.figure()
ax = plt.axes(projection ='3d')
for count_p in range(len(paws)):
    ax.plot3D(pca_fit_fr_single_paws[count_p][0, :], pca_fit_fr_single_paws[count_p][1, :],
            pca_fit_fr_single_paws[count_p][2, :], color=paw_colors[count_p], linewidth=2)
    ax.scatter(pca_fit_fr_single_paws[count_p][0, 10], pca_fit_fr_single_paws[count_p][1, 10],
            pca_fit_fr_single_paws[count_p][2, 10], color=paw_colors[count_p], s=60)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_zlabel('PC3', fontsize=14)
ax.view_init(10, 0)
ax.grid(False)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)

# ### Population activity cluster around sw or st - mean activity for each trial

# # Loop across animals for trial concatenation
# firing_rate_trials_separated_all_paws = []
# firing_rate_trials_together_all_paws = []
# firing_rate_cluster_id_paws = []
# firing_rate_animal_id_paws = []
# for p in range(len(paws)):
#     firing_rate_cluster_id = []
#     firing_rate_animal_id = []
#     for count_a, animal in enumerate(animals):
#         firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
#         ses_animals = session_data['animal']
#         animal_idx = np.where(ses_animals == animal)[0][0]
#         ses_info = session_data.iloc[animal_idx, :]
#         idx_roi_cluster_ordered = np.load(os.path.join(path_data, ses_info[0],
#                 ses_info[1], ses_info[3], 'processed files', 'clusters_rois_idx_order.npy'), allow_pickle=True)
#         animal_idx_list = np.repeat(animal, len(idx_roi_cluster_ordered))
#         if animal == 'MC8855':
#             firing_rate_animal_crop = zscoring(firing_rate_animal[:, p, :20, :]) #remove last 3 washout
#             firing_rate_animal_crop_trial_2nd_dim = np.reshape(firing_rate_animal_crop,
#         (np.shape(firing_rate_animal_crop)[0], (np.shape(firing_rate_animal_crop)[1]*np.shape(firing_rate_animal_crop)[2])))
#         elif animal == 'MC9226' and protocol == 'split ipsi fast':
#             firing_rate_animal_crop = zscoring(firing_rate_animal[:, p, 3:, :])
#             firing_rate_animal_crop_trial_2nd_dim = np.reshape(firing_rate_animal_crop,
#        (np.shape(firing_rate_animal_crop)[0], (np.shape(firing_rate_animal_crop)[1]*np.shape(firing_rate_animal_crop)[2])))
#         else:
#             firing_rate_animal_crop = zscoring(firing_rate_animal[:, p, 3:23, :]) #remove first 3 baseline
#             firing_rate_animal_crop_trial_2nd_dim = np.reshape(firing_rate_animal_crop, (np.shape(firing_rate_animal_crop)[0],
#                 (np.shape(firing_rate_animal_crop)[1] * np.shape(firing_rate_animal_crop)[2])))
#         if count_a == 0:
#             # FR x trials x Time 3d array
#             firing_rate_animal_trials_roi_concat = firing_rate_animal_crop
#             # FR x (trials x Time) 2d array
#             firing_rate_animal_trials_roi_concat_reshape = firing_rate_animal_crop_trial_2nd_dim
#         else:
#             firing_rate_animal_trials_roi_concat = np.concatenate((firing_rate_animal_trials_roi_concat, firing_rate_animal_crop), axis=0)
#             firing_rate_animal_trials_roi_concat_reshape = np.concatenate((firing_rate_animal_trials_roi_concat_reshape, firing_rate_animal_crop_trial_2nd_dim), axis=0)
#         firing_rate_cluster_id.extend(idx_roi_cluster_ordered)
#         firing_rate_animal_id.extend(list(animal_idx_list))
#     firing_rate_trials_together_all_paws.append(firing_rate_animal_trials_roi_concat_reshape)
#     firing_rate_trials_separated_all_paws.append(firing_rate_animal_trials_roi_concat)
#     firing_rate_cluster_id_paws.append(firing_rate_cluster_id)
#     firing_rate_animal_id_paws.append(firing_rate_animal_id)

# p = 0 # do for FR paw
# comp = 9
# sel_trials = np.array([2, 3, 12, 13]) #for plotting
# pca_fr_trials = PCA(n_components=comp)
# pca_fit_fr_trials = pca_fr_trials.fit(firing_rate_trials_together_all_paws[p])
# pca_fit_fr_trials_scores = pca_fr_trials.fit_transform(firing_rate_trials_together_all_paws[p])
#
# # Project mean firing rate activity for each trial in the reference PC space
# pca_fit_fr_single_trials = []
# for t in range(20):
#     pca_fit_fr_single_trials.append(np.dot(firing_rate_trials_separated_all_paws[p][:, t, :].T, pca_fit_fr_trials_scores).T)
#
# # Explained variance for each component
# fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
# ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_trials.explained_variance_ratio_),
#     color='black', s=20)
# ax.set_title('Explained variance\nratio of components', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim([0, 1])
# ax.tick_params(axis='both', which='major', labelsize=16)
# ax.set_ylabel('Cumulative explained\nvariance ratio', fontsize=16)
# ax.set_xlabel('Component number', fontsize=16)
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_trialPC_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)
#
# # First PCs over time
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7), sharey=True)
# ax = ax.ravel()
# for c in range(4):
#     for count_t in sel_trials:
#         ax[c].plot(bins_fr[:-1], pca_fit_fr_single_trials[count_t][c, :], color=colors_session[count_t+1], linewidth=2)
#         ax[c].set_title('Component ' + str(c+1), fontsize=16)
#         ax[c].spines['right'].set_visible(False)
#         ax[c].spines['top'].set_visible(False)
#         ax[c].tick_params(axis='both', which='major', labelsize=14)
#         # ax[c].set_ylabel('Firing rate (Hz) z-scored', fontsize=14)
#         if align_dimension == 'time':
#             ax[c].set_xlabel('Time (ms)', fontsize=14)
#             ax[c].axvline(x=0, color='black')
#         if align_dimension == 'phase':
#             ax[c].set_xlabel('% Phase', fontsize=14)
#             ax[c].axvline(x=0.5, color='black')
#         # ax[c].set_ylim([-0.45, 0.65])
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_trialPC_' + align_event + '_' + align_dimension + '_components'), dpi=256)
#
# # Trajectories in PC space
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# for count_t in sel_trials:
#     ax.plot3D(pca_fit_fr_single_trials[count_t][0, :], pca_fit_fr_single_trials[count_t][1, :],
#             pca_fit_fr_single_trials[count_t][2, :], color=colors_session[count_t+1], linewidth=2)
#     ax.scatter(pca_fit_fr_single_trials[count_t][0, 10], pca_fit_fr_single_trials[count_t][1, 10],
#             pca_fit_fr_single_trials[count_t][2, 10], color=colors_session[count_t+1], s=60)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.set_xlabel('PC1', fontsize=14)
# ax.set_ylabel('PC2', fontsize=14)
# ax.set_zlabel('PC3', fontsize=14)
# ax.view_init(10, 60)
# ax.grid(False)
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_trialPC_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)

# ### Population activity cluster around sw or st - mean activity for each trial for one cluster
# p = 0 # do for FR paw
# comp = 9
# sel_trials = np.array([2, 3, 12, 13]) #for plotting
# animal = 'MC9513'
# cluster_plot = 1
# animal_id = np.array(firing_rate_animal_id_paws[p])
# cluster_id = np.array(firing_rate_cluster_id_paws[p])
# animal_id_idx = np.where(animal_id == animal)[0]
# rois_plot_cluster = animal_id_idx[np.where(cluster_id[animal_id_idx]==cluster_plot)[0]]
# pca_fr_trials_animal = PCA(n_components=comp)
# pca_fit_fr_trials_animal = pca_fr_trials.fit(firing_rate_trials_together_all_paws[p][rois_plot_cluster, :])
# pca_fit_fr_trials_animal_scores = pca_fr_trials.fit_transform(firing_rate_trials_together_all_paws[p][rois_plot_cluster, :])
#
# # Project mean firing rate activity for each trial in the reference PC space
# pca_fit_fr_single_trials_animal = []
# for t in range(20):
#     pca_fit_fr_single_trials_animal.append(np.dot(firing_rate_trials_separated_all_paws[p][rois_plot_cluster, t, :].T, pca_fit_fr_trials_animal_scores).T)
#
# # Explained variance for each component
# fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
# ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_trials_animal.explained_variance_ratio_),
#     color='black', s=20)
# ax.set_title('Explained variance\nratio of components', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylim([0, 1])
# ax.tick_params(axis='both', which='major', labelsize=16)
# ax.set_ylabel('Cumulative explained\nvariance ratio', fontsize=16)
# ax.set_xlabel('Component number', fontsize=16)
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_trialPC_' + animal + 'cluster' + str(cluster_plot) + '_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)
#
# # First PCs over time
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7), sharey=True)
# ax = ax.ravel()
# for c in range(4):
#     for count_t in sel_trials:
#         ax[c].plot(bins_fr[:-1], pca_fit_fr_single_trials_animal[count_t][c, :], color=colors_session[count_t+1], linewidth=2)
#         ax[c].set_title('Component ' + str(c+1), fontsize=16)
#         ax[c].spines['right'].set_visible(False)
#         ax[c].spines['top'].set_visible(False)
#         ax[c].tick_params(axis='both', which='major', labelsize=14)
#         # ax[c].set_ylabel('Firing rate (Hz) z-scored', fontsize=14)
#         if align_dimension == 'time':
#             ax[c].set_xlabel('Time (ms)', fontsize=14)
#             ax[c].axvline(x=0, color='black')
#         if align_dimension == 'phase':
#             ax[c].set_xlabel('% Phase', fontsize=14)
#             ax[c].axvline(x=0.5, color='black')
#         # ax[c].set_ylim([-0.45, 0.65])
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_trialPC_' + animal + 'cluster' + str(cluster_plot) + '_' + align_event + '_' + align_dimension + '_components'), dpi=256)
#
# # Trajectories in PC space
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# for count_t in sel_trials:
#     ax.plot3D(pca_fit_fr_single_trials_animal[count_t][0, :], pca_fit_fr_single_trials_animal[count_t][1, :],
#             pca_fit_fr_single_trials_animal[count_t][2, :], color=colors_session[count_t+1], linewidth=2)
#     ax.scatter(pca_fit_fr_single_trials_animal[count_t][0, 10], pca_fit_fr_single_trials_animal[count_t][1, 10],
#             pca_fit_fr_single_trials_animal[count_t][2, 10], color=colors_session[count_t+1], s=60)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.tick_params(axis='both', which='major', labelsize=14)
# ax.set_xlabel('PC1', fontsize=14)
# ax.set_ylabel('PC2', fontsize=14)
# ax.set_zlabel('PC3', fontsize=14)
# ax.view_init(10, 60)
# ax.grid(False)
# plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_trialPC_' + animal + 'cluster' + str(cluster_plot) + '_' + align_event + '_' + align_dimension + '_trajectories'), dpi=256)
#
#
#
