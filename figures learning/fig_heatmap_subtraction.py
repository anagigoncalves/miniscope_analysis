import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split contra fast'
align_event = 'st'
align_dimension = 'phase'
bins = np.arange(0, 105, 10)  # 10 deg

# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML

def cart2pol(mat):
    rho = np.sqrt(mat[:, 0]**2+mat[:, 1]**2)
    theta = np.arctan2(mat[:, 1], mat[:, 0])
    return rho, theta

def sort_activity(data, phase_bool, plot_data):
    pca_fr_paws = PCA(n_components=3)
    data_mean = np.tile(np.nanmean(data, axis=0), (np.shape(data)[0], 1))
    data_std = np.tile(np.nanstd(data, axis=0), (np.shape(data)[0], 1))
    data_zscore = (data - data_mean) / data_std
    pca_fit_fr_paws = pca_fr_paws.fit(data_zscore)
    pca_fit_fr_paws_fit_transform = pca_fit_fr_paws.components_.T
    pca_coef = pca_fit_fr_paws_fit_transform[:, [0, 1]]
    [rho_bs, theta] = cart2pol(pca_coef)
    theta_sort = np.argsort(theta)[::-1]
    if plot_data:
        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
        plt.scatter(pca_coef[:, 0], pca_coef[:, 1])
    return theta, theta_sort

# Loop across animals for trial average - baseline
firing_rate_mean_trials_paw_bs = []
firing_rate_max_bs = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
        firing_rate_mean = np.nanmean(firing_rate_animal[:, 0, :3, :], axis=1)
        firing_rate_mean_trials_paw_bs.append(firing_rate_mean)
        firing_rate_max_bs.extend(bins[np.argmax(firing_rate_mean, axis=1)])
    else:
        firing_rate_mean = np.nanmean(firing_rate_animal[:, 0, :6, :], axis=1)
        firing_rate_mean_trials_paw_bs.append(firing_rate_mean)
        firing_rate_max_bs.extend(bins[np.argmax(firing_rate_mean, axis=1)])
firing_rate_mean_trials_paw_concat_bs = np.vstack(firing_rate_mean_trials_paw_bs)

# Loop across animals for trial average - early split
firing_rate_mean_trials_paw_es = []
firing_rate_max_es = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
        firing_rate_mean_trials_paw_es.append(np.nanmean(firing_rate_animal[:, 0, [3, 4], :], axis=1))
        firing_rate_max_es.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [3, 4], :], axis=1), axis=1)])
    else:
        firing_rate_mean_trials_paw_es.append(np.nanmean(firing_rate_animal[:, 0, [6, 7], :], axis=1))
        firing_rate_max_es.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [6, 7], :], axis=1), axis=1)])
firing_rate_mean_trials_paw_concat_es = np.vstack(firing_rate_mean_trials_paw_es)

# Loop across animals for trial average - late split
firing_rate_mean_trials_paw_ls = []
firing_rate_max_ls = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
        firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, 0, [11, 12], :], axis=1))
        firing_rate_max_ls.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [11, 12], :], axis=1), axis=1)])
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, 0, [13, 14], :], axis=1))
        firing_rate_max_ls.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [13, 14], :], axis=1), axis=1)])
    else:
        firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, 0, [14, 15], :], axis=1))
        firing_rate_max_ls.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [14, 15], :], axis=1), axis=1)])
firing_rate_mean_trials_paw_concat_ls = np.vstack(firing_rate_mean_trials_paw_ls)

# Loop across animals for trial average - after-effect
firing_rate_mean_trials_paw_ae = []
firing_rate_max_ae = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
        firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, 0, [13, 14], :], axis=1))
        firing_rate_max_ae.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [13, 14], :], axis=1), axis=1)])
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, 0, [15, 16], :], axis=1))
        firing_rate_max_ae.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [15, 16], :], axis=1), axis=1)])
    else:
        firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, 0, [16, 17], :], axis=1))
        firing_rate_max_ae.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [16, 17], :], axis=1), axis=1)])
firing_rate_mean_trials_paw_concat_ae = np.vstack(firing_rate_mean_trials_paw_ae)

# Loop across animals for trial average - late washout
firing_rate_mean_trials_paw_lw = []
firing_rate_max_lw = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
        firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1))
        firing_rate_max_lw.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1), axis=1)])
    elif protocol == 'split ipsi fast' and animal == 'MC9226':
        firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1))
        firing_rate_max_lw.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1), axis=1)])
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1))
        firing_rate_max_lw.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [21, 22], :], axis=1), axis=1)])
    else:
        firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, 0, [24, 25], :], axis=1))
        firing_rate_max_lw.extend(bins[np.argmax(np.nanmean(firing_rate_animal[:, 0, [24, 25], :], axis=1), axis=1)])
firing_rate_mean_trials_paw_concat_lw = np.vstack(firing_rate_mean_trials_paw_lw)

roi_list = np.arange(1, np.shape(firing_rate_mean_trials_paw_concat_bs)[0] + 1)
[theta_bs, theta_bs_sort] = sort_activity(firing_rate_mean_trials_paw_concat_bs.T, 1, 1)
fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=0, vmax=3.5)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_ylabel('ROI #', fontsize=20)
if align_dimension == 'phase':
    ax.set_xlabel('Phase (%)', fontsize=20)
if align_dimension == 'time':
    ax.set_xlabel('Time (s)', fontsize=20)
plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_es[theta_bs_sort]-firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='coolwarm', vmin=-3, vmax=3)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_es)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
ax.set_ylabel('ROI #', fontsize=20)
if align_dimension == 'phase':
    ax.set_xlabel('Phase (%)', fontsize=20)
if align_dimension == 'time':
    ax.set_xlabel('Time (s)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_earlysplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_earlysplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ls[theta_bs_sort]-firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='coolwarm', vmin=-3, vmax=3)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_ls)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
ax.set_ylabel('ROI #', fontsize=20)
if align_dimension == 'phase':
    ax.set_xlabel('Phase (%)', fontsize=20)
if align_dimension == 'time':
    ax.set_xlabel('Time (s)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_latesplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_latesplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ae[theta_bs_sort]-firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='coolwarm', vmin=-3, vmax=3)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_ae)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
ax.set_ylabel('ROI #', fontsize=20)
if align_dimension == 'phase':
    ax.set_xlabel('Phase (%)', fontsize=20)
if align_dimension == 'time':
    ax.set_xlabel('Time (s)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_aftereffect_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_aftereffect_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_lw[theta_bs_sort]-firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='coolwarm', vmin=-3, vmax=3)
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_lw)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
ax.set_ylabel('ROI #', fontsize=20)
if align_dimension == 'phase':
    ax.set_xlabel('Phase (%)', fontsize=20)
if align_dimension == 'time':
    ax.set_xlabel('Time (s)', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_latewashout_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_bs_minus_latewashout_' + align_event + '_' + align_dimension + '.svg'), dpi=256)