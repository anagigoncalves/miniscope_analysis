import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
load_pc_path = 'J:\\LocoCF\\miniscopes learning\\PCA validation and clusters\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
bins = np.arange(0, 105, 10)  # 10 deg
align_event = 'st'
align_dimension = 'phase'

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Projects\\Dev\\miniscope_analysis\\')

# Load PC coefficients data
pc_coeff = pd.read_csv(os.path.join(load_pc_path, 'pc_coeff_df_clusters_' + '_'.join(protocol.split(' ')) + '.csv'))

def mi_index(a, b):
    return (a-b)/(a+b)

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

bin_transition = np.where(bins >= 50)[0][0]
# Order clusters ID and get count for each id and each color
idx_order = np.argsort(pc_coeff['cluster_pca'])
clusters_ordered = pc_coeff['cluster_pca'][idx_order]
clusters, counts_clusters = np.unique(clusters_ordered, return_counts=True)
cmap_cluster = plt.get_cmap('jet')
colors_cluster  = [cmap_cluster(i) for i in np.linspace(0, 1, len(clusters))]

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
init_val = 0
for c in clusters:
    rectangle = plt.Rectangle((init_val, -0.5), counts_clusters[c], 1, fc=colors_cluster[c], alpha=0.1)
    plt.gca().add_patch(rectangle)
    init_val += counts_clusters[c]
bs_es_data_sw = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_es[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1))
bs_es_data_st = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_es[:, :5], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1))
ax.scatter(np.arange(0, len(bs_es_data_st)), bs_es_data_st[idx_order], color='orange', s=10)
ax.scatter(np.arange(0, len(bs_es_data_st)), bs_es_data_sw[idx_order], color='green', s=10)
plt.vlines(np.arange(0, len(bs_es_data_st)), ymin=bs_es_data_sw[idx_order], ymax=bs_es_data_st[idx_order], linewidth=0.4, color='darkgrey')
ax.set_xlabel('ROI ID', fontsize=20)
ax.set_ylabel('Early split vs baseline ratio', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_earlysplit_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_earlysplit_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
init_val = 0
for c in clusters:
    rectangle = plt.Rectangle((init_val, -0.5), counts_clusters[c], 1, fc=colors_cluster[c], alpha=0.1)
    plt.gca().add_patch(rectangle)
    init_val += counts_clusters[c]
bs_ls_data_sw = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_ls[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1))
bs_ls_data_st = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_ls[:, :5], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1))
ax.scatter(np.arange(0, len(bs_ls_data_st)), bs_ls_data_st[idx_order], color='orange', s=10)
ax.scatter(np.arange(0, len(bs_ls_data_st)), bs_ls_data_sw[idx_order], color='green', s=10)
plt.vlines(np.arange(0, len(bs_ls_data_st)), ymin=bs_ls_data_sw[idx_order], ymax=bs_ls_data_st[idx_order], linewidth=0.4, color='darkgrey')
ax.set_xlabel('ROI ID', fontsize=20)
ax.set_ylabel('Late split vs baseline ratio', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_latesplit_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_latesplit_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

fig, ax = plt.subplots(tight_layout=True, figsize=(10, 5))
init_val = 0
for c in clusters:
    rectangle = plt.Rectangle((init_val, -0.5), counts_clusters[c], 1, fc=colors_cluster[c], alpha=0.1)
    plt.gca().add_patch(rectangle)
    init_val += counts_clusters[c]
bs_ae_data_sw = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_ae[:, bin_transition:], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, bin_transition:], axis=1))
bs_ae_data_st = mi_index(np.nanmean(firing_rate_mean_trials_paw_concat_ae[:, :5], axis=1), np.nanmean(firing_rate_mean_trials_paw_concat_bs[:, :bin_transition], axis=1))
ax.scatter(np.arange(0, len(bs_ae_data_st)), bs_ae_data_st[idx_order], color='orange', s=10)
ax.scatter(np.arange(0, len(bs_ae_data_st)), bs_ae_data_sw[idx_order], color='green', s=10)
plt.vlines(np.arange(0, len(bs_ae_data_st)), ymin=bs_ae_data_sw[idx_order], ymax=bs_ae_data_st[idx_order], linewidth=0.4, color='darkgrey')
ax.set_xlabel('ROI ID', fontsize=20)
ax.set_ylabel('After-effect vs baseline ratio', fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_aftereffect_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension), dpi=256)
plt.savefig(os.path.join(save_path, 'mi_stride_phases_baseline_aftereffect_' + protocol.replace(' ', '_') + '_' + align_event + '_' + align_dimension + '.svg'), dpi=256)