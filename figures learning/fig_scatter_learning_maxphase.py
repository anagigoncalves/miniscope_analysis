import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
bins = np.arange(0, 105, 10)  # 10 deg
align_event = 'st'
align_dimension = 'phase'

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')

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

bin_transition = np.where(bins>=50)[0][0]
fig, ax = plt.subplots(1,2, tight_layout=True, figsize=(10, 5))
ax = ax.ravel()
ax[0].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_es[:, bin_transition:], axis=1), color='green', s=10)
ax[1].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_es[:, :5], axis=1), color='orange', s=10)
ax[0].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[0].set_ylabel('Calcium event\nrate early split trials (Hz)', fontsize=20)
ax[0].plot([1, 4], [1, 4], color='black')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[1].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[1].set_ylabel('Calcium event\nrate early split trials (Hz)', fontsize=20)
ax[1].plot([1, 4], [1, 4], color='black')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_earlysplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_earlysplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(1,2, tight_layout=True, figsize=(10, 5))
ax = ax.ravel()
ax[0].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_ls[:, bin_transition:], axis=1), color='green', s=10)
ax[1].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_ls[:, :5], axis=1), color='orange', s=10)
ax[0].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[0].set_ylabel('Calcium event\nrate late split trials (Hz)', fontsize=20)
ax[0].plot([1, 4], [1, 4], color='black')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[1].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[1].set_ylabel('Calcium event\nrate late split trials (Hz)', fontsize=20)
ax[1].plot([1, 4], [1, 4], color='black')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_latesplit_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_latesplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(1,2, tight_layout=True, figsize=(10, 5))
ax = ax.ravel()
ax[0].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_ae[:, bin_transition:], axis=1), color='green', s=10)
ax[1].scatter(np.nanmax(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1), np.nanmax(firing_rate_mean_trials_paw_concat_ae[:, :5], axis=1), color='orange', s=10)
ax[0].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[0].set_ylabel('Calcium event\nrate after-effect trials (Hz)', fontsize=20)
ax[0].plot([1, 4], [1, 4], color='black')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].tick_params(axis='both', which='major', labelsize=20)
ax[1].set_xlabel('Calcium event\nrate baseline trials (Hz)', fontsize=20)
ax[1].set_ylabel('Calcium event\nrate after-effect trials (Hz)', fontsize=20)
ax[1].plot([1, 4], [1, 4], color='black')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_aftereffect_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'max_stride_phases_baseline_aftereffect_' + align_event + '_' + align_dimension + '.svg'), dpi=256)