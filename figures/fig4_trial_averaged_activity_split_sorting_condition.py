import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\Activity sorted\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'st'
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 105, 5)  # 5 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
    bins_fr = bins*1000
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')

for p in range(4):
    # Loop across animals for trial average - baseline
    firing_rate_mean_trials_paw_bs = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            firing_rate_mean_trials_paw_bs.append(np.nanmean(firing_rate_animal[:, p, :3, :], axis=1))
        else:
            firing_rate_mean_trials_paw_bs.append(np.nanmean(firing_rate_animal[:, p, :6, :], axis=1))
    firing_rate_mean_trials_paw_concat_bs = np.vstack(firing_rate_mean_trials_paw_bs)

    # Loop across animals for trial average - early split
    firing_rate_mean_trials_paw_es = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            #firing_rate_mean_trials_paw_es.append(firing_rate_animal[:, p, 3, :])
            firing_rate_mean_trials_paw_es.append(np.nanmean(firing_rate_animal[:, p, [3, 4], :], axis=1))
        else:
            # firing_rate_mean_trials_paw_es.append(firing_rate_animal[:, p, 6, :])
            firing_rate_mean_trials_paw_es.append(np.nanmean(firing_rate_animal[:, p, [6, 7], :], axis=1))
    firing_rate_mean_trials_paw_concat_es = np.vstack(firing_rate_mean_trials_paw_es)

    # Loop across animals for trial average - late split
    firing_rate_mean_trials_paw_ls = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            # firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 12, :])
            firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, p, [11, 12], :], axis=1))
        elif protocol == 'split contra fast' and animal == 'MC10221':
            # firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 13, :])
            firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, p, [12, 13], :], axis=1))
        else:
            #firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 14, :])
            firing_rate_mean_trials_paw_ls.append(np.nanmean(firing_rate_animal[:, p, [13, 14], :], axis=1))
    firing_rate_mean_trials_paw_concat_ls = np.vstack(firing_rate_mean_trials_paw_ls)

    # Loop across animals for trial average - after-effect
    firing_rate_mean_trials_paw_ae = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            # firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 13, :])
            firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, p, [13, 14], :], axis=1))
        elif protocol == 'split contra fast' and animal == 'MC10221':
            # firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 15, :])
            firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, p, [15, 16], :], axis=1))
        else:
            # firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 16, :])
            firing_rate_mean_trials_paw_ae.append(np.nanmean(firing_rate_animal[:, p, [16, 17], :], axis=1))
    firing_rate_mean_trials_paw_concat_ae = np.vstack(firing_rate_mean_trials_paw_ae)

    # Loop across animals for trial average - late washout
    firing_rate_mean_trials_paw_lw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            # firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 22, :])
            firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, p, [21, 22], :], axis=1))
        elif protocol == 'split ipsi fast' and animal == 'MC9226':
            # firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 22, :])
            firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, p, [21, 22], :], axis=1))
        elif protocol == 'split contra fast' and animal == 'MC10221':
            # firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 24, :])
            firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, p, [23, 24], :], axis=1))
        else:
            # firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 25, :])
            firing_rate_mean_trials_paw_lw.append(np.nanmean(firing_rate_animal[:, p, [24, 25], :], axis=1))
    firing_rate_mean_trials_paw_concat_lw = np.vstack(firing_rate_mean_trials_paw_lw)

    roi_list = np.arange(1, np.shape(firing_rate_mean_trials_paw_concat_bs)[0]+1)

    data_bs = firing_rate_mean_trials_paw_concat_bs[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_bs, axis=1))]
    data_es = firing_rate_mean_trials_paw_concat_es[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_es, axis=1))]
    data_ls = firing_rate_mean_trials_paw_concat_ls[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ls, axis=1))]
    data_ae = firing_rate_mean_trials_paw_concat_ae[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ae, axis=1))]
    data_lw = firing_rate_mean_trials_paw_concat_lw[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_lw, axis=1))]
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
    plt.plot(bins[np.argmax(data_bs, axis=1)], np.arange(len(bins[np.argmax(data_bs, axis=1)])), color='black', linewidth=2)
    plt.plot(bins[np.argmax(data_es, axis=1)], np.arange(len(bins[np.argmax(data_es, axis=1)])), color='crimson', linewidth=2)
    plt.plot(bins[np.argmax(data_ls, axis=1)], np.arange(len(bins[np.argmax(data_ls, axis=1)])), color='salmon', linewidth=2)
    plt.plot(bins[np.argmax(data_ae, axis=1)], np.arange(len(bins[np.argmax(data_ae, axis=1)])), color='blue', linewidth=2)
    plt.plot(bins[np.argmax(data_lw, axis=1)], np.arange(len(bins[np.argmax(data_lw, axis=1)])), color='dodgerblue', linewidth=2)
    plt.legend(['baseline', 'early split', 'late split', 'early washout', 'late washout'], frameon=False, fontsize=16)
    ax.set_ylabel('ROI #', fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.axvline(50, color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_sorted_' + paws[p] + '_' + align_event + '_' + align_dimension), dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_sorted_' + paws[p] + '_' + align_event + '_' + align_dimension + '.svg'), dpi=256)

# fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
# hm = sns.heatmap(firing_rate_mean_trials_paw_concat_bs[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_bs, axis=1))],
#         ax=ax, cmap='viridis')
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
# ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
# ax.set_yticklabels(list(map(str, roi_list[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_bs, axis=1))][::20])))
# ax.set_xticks(np.linspace(0, 20, 10))
# ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
# if align_dimension == 'time':
#     ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
# ax.set_ylabel('ROI #', fontsize=20)
# ax.set_xlabel('Phase (%)', fontsize=20)
# plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension), dpi=256)
# plt.savefig(os.path.join(save_path, 'firing_rate_baseline_' + align_event + '_' + align_dimension + '.svg'), dpi=256)
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
# hm = sns.heatmap(firing_rate_mean_trials_paw_concat_es[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_es, axis=1))],
#         ax=ax, cmap='viridis')
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
# ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
# ax.set_yticklabels(list(map(str, roi_list[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_es, axis=1))][::20])))
# ax.set_xticks(np.linspace(0, 20, 10))
# ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
# if align_dimension == 'time':
#     ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
# ax.set_ylabel('ROI #', fontsize=20)
# ax.set_xlabel('Phase (%)', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(save_path, 'firing_rate_earlysplit_' + align_event + '_' + align_dimension), dpi=256)
# plt.savefig(os.path.join(save_path, 'firing_rate_earlysplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
# hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ls[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ls, axis=1))],
#         ax=ax, cmap='viridis')
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
# ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
# ax.set_yticklabels(list(map(str, roi_list[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ls, axis=1))][::20])))
# ax.set_xticks(np.linspace(0, 20, 10))
# ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
# if align_dimension == 'time':
#     ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
# ax.set_ylabel('ROI #', fontsize=20)
# ax.set_xlabel('Phase (%)', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(save_path, 'firing_rate_latesplit_' + align_event + '_' + align_dimension), dpi=256)
# plt.savefig(os.path.join(save_path, 'firing_rate_latesplit_' + align_event + '_' + align_dimension + '.svg'), dpi=256)
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
# hm = sns.heatmap(firing_rate_mean_trials_paw_concat_ae[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ae, axis=1))],
#         ax=ax, cmap='viridis')
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
# ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
# ax.set_yticklabels(list(map(str, roi_list[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_ae, axis=1))][::20])))
# ax.set_xticks(np.linspace(0, 20, 10))
# ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
# if align_dimension == 'time':
#     ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
# ax.set_ylabel('ROI #', fontsize=20)
# ax.set_xlabel('Phase (%)', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(save_path, 'firing_rate_aftereffect_' + align_event + '_' + align_dimension), dpi=256)
# plt.savefig(os.path.join(save_path, 'firing_rate_aftereffect_' + align_event + '_' + align_dimension + '.svg'), dpi=256)
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
# hm = sns.heatmap(firing_rate_mean_trials_paw_concat_lw[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_lw, axis=1))],
#         ax=ax, cmap='viridis')
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=18)
# ax.set_yticks(np.linspace(0, np.shape(firing_rate_mean_trials_paw_concat_bs)[0], 20))
# ax.set_yticklabels(list(map(str, roi_list[np.argsort(np.argmax(firing_rate_mean_trials_paw_concat_lw, axis=1))][::20])))
# ax.set_xticks(np.linspace(0, 20, 10))
# ax.set_xticklabels(np.round(np.linspace(0, bins[-1], 10), 1))
# if align_dimension == 'time':
#     ax.set_xticklabels(np.round(np.linspace(bins[0], bins[-1], 10), 2))
# ax.set_ylabel('ROI #', fontsize=20)
# ax.set_xlabel('Phase (%)', fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(save_path, 'firing_rate_latewashout_' + align_event + '_' + align_dimension), dpi=256)
# plt.savefig(os.path.join(save_path, 'firing_rate_latewashout_' + align_event + '_' + align_dimension + '.svg'), dpi=256)