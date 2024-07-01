import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
protocol = 'tied baseline'
protocol_id = 'tied_S1'
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\' + protocol + ' S1\\'
load_path_acc = 'J:\\Miniscope processed files\\Analysis on population data\\STA bodyvars X\\' + protocol + ' S1\\'
load_pc_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\cluster pc space\\' + protocol + ' S1\\'
save_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\averaged cluster activity\\' + protocol + ' S1\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_' + protocol_id + '.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
bins = np.arange(0, 105, 10)  # 10 deg
align_event = 'st'
align_dimension = 'phase'
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-1, 0.25])
xaxis = window/330

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')

# Load PC coefficients data
pc_coeff = pd.read_csv(os.path.join(load_pc_path, 'pc_coeff_df_clusters_' + '_'.join(protocol.split(' ')) + '.csv'))

for cluster_id in np.sort(pc_coeff['cluster_pca'].unique()):
    pc_coeff_cluster = pc_coeff.loc[(pc_coeff['cluster_pca'] == cluster_id)]
    firing_rate_animal_all = []
    sta_zs_all = []
    for count_animal, animal in enumerate(animals):
        # get activity aligned to stride cycle
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        # get acceleration profile aligned to neural activity
        sta_zs = np.load(os.path.join(load_path_acc, animal + ' ' + protocol, 'sta_bodyvars_Body_acceleration_zscored.npy'))
        roi_idx = np.int64(pc_coeff_cluster.loc[(pc_coeff_cluster['animal'] == animal)].reset_index().index)
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            firing_rate_animal_roi = np.nanmean(firing_rate_animal[roi_idx, :, :3, :], axis=2)
            sta_zs_roi = np.nanmean(sta_zs[roi_idx, :3, :], axis=1)
        elif (protocol == 'split ipsi fast' and animal != 'MC8855') or (protocol == 'split contra fast' and animal != 'MC8855'):
            firing_rate_animal_roi = np.nanmean(firing_rate_animal[roi_idx, :, :6, :], axis=2)
            sta_zs_roi = np.nanmean(sta_zs[roi_idx, :6, :], axis=1)
        elif protocol == 'tied baseline':
            firing_rate_animal_roi = np.nanmean(firing_rate_animal[roi_idx, :, :, :], axis=2)
            sta_zs_roi = np.nanmean(sta_zs[roi_idx, :, :], axis=1)
        if count_animal == 0:
            firing_rate_animal_all = firing_rate_animal_roi
            sta_zs_all = sta_zs_roi
        else:
            firing_rate_animal_all = np.concatenate((firing_rate_animal_all, firing_rate_animal_roi), axis=0)
            sta_zs_all = np.concatenate((sta_zs_all, sta_zs_roi), axis=0)
    firing_rate_mean = np.nanmean(firing_rate_animal_all, axis=0)
    firing_rate_std = np.nanstd(firing_rate_animal_all, axis=0)
    sta_zs_mean = np.nanmean(sta_zs_all, axis=0)
    sta_zs_std = np.nanstd(sta_zs_all, axis=0)

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(14, 5))
    ax = ax.ravel()
    for count_p in range(len(paw_colors)):
        ax[0].plot(bins[:-1], firing_rate_mean[count_p, :], color=paw_colors[count_p], linewidth=2)
        ax[0].fill_between(bins[:-1], firing_rate_mean[count_p, :]-firing_rate_std[count_p, :],
                        firing_rate_mean[count_p, :]+firing_rate_std[count_p, :],
                        color=paw_colors[count_p], alpha=0.3)
    ax[0].set_xticks([0, 50, 90])
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[0].set_ylabel('Calcium event rate (Hz)', fontsize=20)
    ax[0].set_xlabel('Phase bins (%)', fontsize=20)
    ax[1].plot(xaxis, sta_zs_mean, color='black', linewidth=2)
    ax[1].fill_between(xaxis, sta_zs_mean-sta_zs_std, sta_zs_mean+sta_zs_std, color='black', alpha=0.3)
    ax[1].axvline(x=0, color='darkgray', linestyle='dashed')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].set_ylabel('Body acceleration z-scored', fontsize=20)
    ax[1].set_xlabel('Time (s)', fontsize=20)
    plt.savefig(os.path.join(save_path, 'cluster_summary_clusterid_' + str(cluster_id)), dpi=256)

    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    sc = ax.scatter(pc_coeff['coord_x'], pc_coeff['coord_y'], s=15, color='darkgray')
    sc = ax.scatter(pc_coeff_cluster['coord_x'], pc_coeff_cluster['coord_y'], s=15, color='blue')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    ax.set_title('Cluster' + str(cluster_id + 1), fontsize=20)
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(save_path, 'cluster_location_clusterid_' + str(cluster_id)), dpi=256)
    plt.close('all')
