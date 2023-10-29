import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import pandas as pd
from sklearn.decomposition import PCA

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
paw_colors = ['red', 'magenta', 'blue', 'cyan']
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
firing_rate_mean_trials_concat_paws = []
firing_rate_mean_trials_concat_paws_animalid = []
for p in range(len(paws)):
    firing_rate_mean_trials_paw = []
    firing_rate_mean_trials_paw_animalid = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        #firing_rate_mean_trials_paw.append(zscoring(np.nanmean(firing_rate_animal[:, p, :, :], axis=1)))
        firing_rate_mean_trials_paw.append(np.nanmean(firing_rate_animal[:, p, :, :], axis=1))
        firing_rate_mean_trials_paw_animalid.append(np.repeat(count_a, np.shape(firing_rate_animal)[0]))
    firing_rate_mean_trials_paw_concat = np.vstack(firing_rate_mean_trials_paw)
    firing_rate_mean_trials_paw_animalid_concat = np.concatenate(firing_rate_mean_trials_paw_animalid)
    firing_rate_mean_trials_concat_paws.append(firing_rate_mean_trials_paw_concat)
    firing_rate_mean_trials_concat_paws_animalid.append(firing_rate_mean_trials_paw_animalid_concat)

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
#             firing_rate_animal_trials_roi_concat = firing_rate_animal_crop
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
#
# Population activity cluster around sw or st
comp = 9
pca_fit_fr_paws = []
pcomp_fr_paws = []
for p in range(4):
    pca_fr = PCA(n_components=comp)
    pca_fit_fr = pca_fr.fit(firing_rate_mean_trials_concat_paws[p])
    pcomp_fr = pca_fr.fit_transform(firing_rate_mean_trials_concat_paws[p])
    pca_fit_fr_paws.append(pca_fit_fr)
    pcomp_fr_paws.append(pcomp_fr)

# explained variance for each component
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
for count_p in range(len(paws)):
    ax.scatter(np.arange(1, comp+1), np.cumsum(pca_fit_fr_paws[count_p].explained_variance_ratio_),
    color=paw_colors[count_p], label=paws[count_p], s=20)
    ax.set_title('Explained variance\nratio of components', fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel('Cumulative explained\nvariance ratio', fontsize=16)
    ax.set_xlabel('Component number', fontsize=16)
ax.legend(frameon=False, fontsize=16, loc='lower right')
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_aligned_' + align_event + '_' + align_dimension + '_explained_variance'), dpi=256)

# First PCs over time
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 10), sharey=True)
ax = ax.ravel()
for count_p in range(len(paws)):
    for c in range(4):
        ax[c].plot(bins_fr[:-1], pca_fit_fr_paws[count_p].components_[c, :], color=paw_colors[count_p], linewidth=2)
        ax[c].set_title('Component ' + str(c+1), fontsize=16)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=14)
        #ax[c].set_ylabel('Firing rate (Hz) z-scored', fontsize=14)
        ax[c].set_ylabel('Firing rate (Hz)', fontsize=14)
        if align_dimension == 'time':
            ax[c].set_xlabel('Time (ms)', fontsize=14)
            ax[c].axvline(x=0, color='black')
        if align_dimension == 'phase':
            ax[c].set_xlabel('% Phase', fontsize=14)
            ax[c].axvline(x=0.5, color='black')
        ax[c].set_ylim([-0.45, 0.65])
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_components'), dpi=256)

# Explained variance ratio for each component across the four paws
fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7), sharey=True)
ax = ax.ravel()
for c in range(4):
    for count_p in range(len(paws)):
        ax[c].bar(count_p, pca_fit_fr_paws[count_p].explained_variance_ratio_[c], color=paw_colors[count_p])
        ax[c].set_xticks(np.arange(4))
        ax[c].set_xticklabels(paws)
        ax[c].set_title('Component ' + str(c+1), fontsize=14)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].tick_params(axis='both', which='major', labelsize=12)
        ax[c].set_ylabel('Explained variance\nratio', fontsize=12)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_explained_variance_component'), dpi=256)

# clusters in two dimensions colorcoded by animals/clusters of animals for FR paw PCA
paw = 2
dim1 = np.array([0, 0, 0, 1, 1, 2])
dim2 = np.array([1, 2, 3, 2, 3, 3])
dim1_name = ['PC1', 'PC1', 'PC1', 'PC2', 'PC2', 'PC3']
dim2_name = ['PC2', 'PC3', 'PC4', 'PC3', 'PC4', 'PC4']
fig, ax = plt.subplots(3, 2, tight_layout=True, figsize=(25, 15), sharey=True, sharex = True)
ax = ax.ravel()
for count_p in range(len(dim1)):
    scatter = ax[count_p].scatter(pcomp_fr_paws[paw][:, dim1[count_p]], pcomp_fr_paws[paw][:, dim2[count_p]], s=20,
        c=firing_rate_mean_trials_concat_paws_animalid[paw], cmap='tab10')
    ax[count_p].spines['right'].set_visible(False)
    ax[count_p].spines['top'].set_visible(False)
    ax[count_p].tick_params(axis='both', which='major', labelsize=14)
    ax[count_p].set_ylabel(dim1_name[count_p], fontsize=14)
    ax[count_p].set_xlabel(dim2_name[count_p], fontsize=14)
    ax[count_p].legend(handles = scatter.legend_elements()[0], labels=animals, frameon=False, fontsize=12)
fig.suptitle('Clustering of ROIs across first 2 PCs\n', fontsize=20)
plt.savefig(os.path.join(save_path, 'pca_mean_firingrate_' + align_event + '_' + align_dimension + '_scores_FRpaw'), dpi=256)

# # # Population activity between trials
# p = 0
# comp = 9
# animal = 'MC10221'
# cluster = 3
# animal_id = np.array(firing_rate_animal_id_paws[p])
# cluster_id = np.array(firing_rate_cluster_id_paws[p])
# animal_id_idx = np.where(animal_id==animal)[0]
# rois_plot_cluster = animal_id_idx[np.where(cluster_id[animal_id_idx]==cluster)[0]]
# pca_fr_trial_concat = PCA(n_components=comp)
# pca_fit_all_trials_concat_fr = pca_fr_trial_concat.fit(firing_rate_trials_together_all_paws[p][rois_plot_cluster, :])
# pcomp_all_trials_concat_fr = pca_fr_trial_concat.fit_transform(firing_rate_trials_together_all_paws[p][rois_plot_cluster, :])
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 7))
# ax = ax.ravel()
# for c in range(4):
#     data = np.reshape(pca_fit_all_trials_concat_fr.components_[c, :], (len(trials), len(bins)-1))
#     for count_t, t in enumerate(np.array([3, 4, 13, 14, 20])):
#         ax[c].plot(bins[:-1], data[count_t, :], color=colors_session[t], linewidth=2)
#     ax[c].set_title('Component ' + str(c + 1), fontsize=16)
#     ax[c].spines['right'].set_visible(False)
#     ax[c].spines['top'].set_visible(False)
#     ax[c].tick_params(axis='both', which='major', labelsize=14)
#     ax[c].set_ylabel('Firing rate (Hz)', fontsize=14)
#     if align_dimension == 'time':
#         ax[c].set_xlabel('Time (ms)', fontsize=14)
#         ax[c].axvline(x=0, color='black')
#     if align_dimension == 'phase':
#         ax[c].set_xlabel('% Phase', fontsize=14)
#         ax[c].axvline(x=0.5, color='black')
#
# # Explained variance ratio for each component across the four paws
# fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(10, 5), sharey=True)
# ax = ax.ravel()
# for c in range(4):
#     for count_p in range(len(paws)):
#         ax[c].bar(count_p, pca_fit_all_trials_concat_fr.explained_variance_ratio_[c], color=paw_colors[count_p])
#         ax[c].set_xticklabels(paws)
#         ax[c].set_title('Component ' + str(c+1), fontsize=14)
#         ax[c].spines['right'].set_visible(False)
#         ax[c].spines['top'].set_visible(False)
#         ax[c].tick_params(axis='both', which='major', labelsize=12)
#         ax[c].set_ylabel('Explained variance\nratio', fontsize=12)