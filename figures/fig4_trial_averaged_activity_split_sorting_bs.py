import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\Front right baseline activity sorted\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split contra fast'
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    phase_bool = 1
    bins = np.arange(0, 105, 10)  # 10 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    phase_bool = 0
    bins = np.arange(-0.125, 0.126, 0.25) # 25 ms
    bins_fr = bins*1000
paws = ['FR', 'HR', 'FL', 'HL']

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

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

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
[theta_bs, theta_bs_sort] = sort_activity(firing_rate_mean_trials_paw_concat_bs.T, phase_bool, 1)
fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort], 99))
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
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_es[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_es[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_es[theta_bs_sort], 99))
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
plt.savefig(os.path.join(save_path, 'firing_rate_earlysplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_earlysplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ls[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_ls[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_ls[theta_bs_sort], 99))
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
plt.savefig(os.path.join(save_path, 'firing_rate_latesplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_latesplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ae[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_ae[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_ae[theta_bs_sort], 99))
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
plt.savefig(os.path.join(save_path, 'firing_rate_aftereffect_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_aftereffect_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_lw[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_lw[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_lw[theta_bs_sort], 99))
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
plt.savefig(os.path.join(save_path, 'firing_rate_latewashout_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_latewashout_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

# Get coordinates for all ROIs
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

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_bs, cmap='viridis')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_bs_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_bs_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_es, cmap='viridis')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_es_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_es_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_ls, cmap='viridis')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_ls_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_ls_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_ae, cmap='viridis')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_ae_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_ae_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_lw, cmap='viridis')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_lw_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode_lw_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

bin_transition = np.where(bins>=50)[0][0]
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_es[:, bin_transition:], axis=1), color='green', s=10)
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_es[:, :5], axis=1), color='orange', s=10)
ax.set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax.set_ylabel('Calcium event\nrate early split trials (Hz)', fontsize=20)
ax.legend(['Stance phase', 'Swing phase'], fontsize=16, frameon=False)
ax.plot([1, 2.5], [1, 2.5], color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_earlysplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_earlysplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_ls[:, bin_transition:], axis=1), color='green', s=10)
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_ls[:, :5], axis=1), color='orange', s=10)
ax.set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax.set_ylabel('Calcium event\nrate late split trials (Hz)', fontsize=20)
ax.legend(['Stance phase', 'Swing phase'], fontsize=16, frameon=False)
ax.plot([1, 2.5], [1, 2.5], color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_latesplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_latesplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_ae[:, bin_transition:], axis=1), color='green', s=10)
ax.scatter(np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_ae[:, :5], axis=1), color='orange', s=10)
ax.set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax.set_ylabel('Calcium event\nrate after-effect trials (Hz)', fontsize=20)
ax.legend(['Stance phase', 'Swing phase'], fontsize=16, frameon=False)
ax.plot([1, 2.5], [1, 2.5], color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_aftereffect_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mean_stride_phases_baseline_aftereffect_' + align_event + '_' + align_dimension + '.svg'), dpi=256)