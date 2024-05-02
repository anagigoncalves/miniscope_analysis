import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\Front right baseline activity sorted\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_tied_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'tied baseline'
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    phase_bool = 1
    bins = np.arange(0, 105, 10)  # 10 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    phase_bool = 0
    bins = np.arange(-0.125, 0.126, 0.025) # 25 ms
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
    theta_sort = np.argsort(theta)
    if plot_data:
        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
        plt.scatter(pca_coef[:, 0], pca_coef[:, 1])
    return theta, theta_sort

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

# Loop across animals for trial average
firing_rate_mean_trials_paw_bs = []
firing_rate_max_bs = []
for count_a, animal in enumerate(animals):
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
    firing_rate_mean = np.nanmean(firing_rate_animal[:, 0, :, :], axis=1)
    firing_rate_mean_trials_paw_bs.append(firing_rate_mean)
    firing_rate_max_bs.extend(bins[np.argmax(firing_rate_mean, axis=1)])
firing_rate_mean_trials_paw_concat_bs = np.vstack(firing_rate_mean_trials_paw_bs)

roi_list = np.arange(1, np.shape(firing_rate_mean_trials_paw_concat_bs)[0] + 1)
[theta_bs, theta_bs_sort] = sort_activity(firing_rate_mean_trials_paw_concat_bs.T, phase_bool, 1)
fig, ax = plt.subplots(tight_layout=True, figsize=(9, 7))
hm = sns.heatmap(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort],
        ax=ax, cmap='viridis', vmin=np.nanpercentile(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort], 1),
        vmax=np.nanpercentile(firing_rate_mean_trials_paw_concat_bs[theta_bs_sort], 99))
cbar = hm.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
ax.set_xticks(np.linspace(0, 10, 10))
ax.set_xticklabels(np.int64(np.linspace(0, bins[-1], 10)))
ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
ax.set_yticklabels(list(map(str, roi_list[theta_bs_sort][::20])))
if align_dimension == 'time':
    ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
ax.set_ylabel('ROI #', fontsize=24)
ax.set_xlabel('Stride phase (%)', fontsize=24)
ax.tick_params(axis='both', which='major', labelsize=18)
plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

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
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], s=15, c=firing_rate_max_bs)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
# ax.set_title('Step cycle baseline tuning', fontsize=20)
plt.gca().invert_yaxis()
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim([0, 100])
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode'), dpi=256)
plt.savefig(os.path.join(save_path, 'roi_map_phase_max_colorcode.svg'), dpi=256)
