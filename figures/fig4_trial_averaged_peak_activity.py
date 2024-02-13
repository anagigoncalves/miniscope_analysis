import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\Peak activity\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
bins = np.arange(0, 105, 5)  # 5 deg
paws = ['FR', 'HR', 'FL', 'HL']

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')

def norm_hist(data):
    hist_data = np.histogram(data, range=(
        np.min(data), np.max(data)))
    weights_data = np.ones_like(data)/np.max(hist_data[0])
    return weights_data

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
            firing_rate_mean_trials_paw_es.append(firing_rate_animal[:, p, 3, :])
        else:
            firing_rate_mean_trials_paw_es.append(firing_rate_animal[:, p, 6, :])
    firing_rate_mean_trials_paw_concat_es = np.vstack(firing_rate_mean_trials_paw_es)

    # Loop across animals for trial average - late split
    firing_rate_mean_trials_paw_ls = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 12, :])
        elif protocol == 'split contra fast' and animal == 'MC10221':
            firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 13, :])
        else:
            firing_rate_mean_trials_paw_ls.append(firing_rate_animal[:, p, 14, :])
    firing_rate_mean_trials_paw_concat_ls = np.vstack(firing_rate_mean_trials_paw_ls)

    # Loop across animals for trial average - after-effect
    firing_rate_mean_trials_paw_ae = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 13, :])
        elif protocol == 'split contra fast' and animal == 'MC10221':
            firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 15, :])
        else:
            firing_rate_mean_trials_paw_ae.append(firing_rate_animal[:, p, 16, :])
    firing_rate_mean_trials_paw_concat_ae = np.vstack(firing_rate_mean_trials_paw_ae)

    # Loop across animals for trial average - late washout
    firing_rate_mean_trials_paw_lw = []
    for count_a, animal in enumerate(animals):
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        if (protocol == 'split ipsi fast' and animal == 'MC8855') or (protocol == 'split contra fast' and animal == 'MC8855'):
            firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 22, :])
        elif protocol == 'split ipsi fast' and animal == 'MC9226':
            firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 22, :])
        elif protocol == 'split contra fast' and animal == 'MC10221':
            firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 24, :])
        else:
            firing_rate_mean_trials_paw_lw.append(firing_rate_animal[:, p, 25, :])
    firing_rate_mean_trials_paw_concat_lw = np.vstack(firing_rate_mean_trials_paw_lw)

    data_bs = bins[np.argmax(firing_rate_mean_trials_paw_concat_bs, axis=1)]
    data_es = bins[np.argmax(firing_rate_mean_trials_paw_concat_es, axis=1)]
    data_ls = bins[np.argmax(firing_rate_mean_trials_paw_concat_ls, axis=1)]
    data_ae = bins[np.argmax(firing_rate_mean_trials_paw_concat_ae, axis=1)]
    data_lw = bins[np.argmax(firing_rate_mean_trials_paw_concat_lw, axis=1)]
    fig, ax = plt.subplots(tight_layout=True, figsize=(6, 5))
    plt.hist(data_bs, histtype='step', color='black', linewidth=2)
    plt.hist(data_es, histtype='step', color='crimson', linewidth=2)
    plt.hist(data_ls, histtype='step', color='salmon', linewidth=2)
    plt.hist(data_ae, histtype='step', color='blue', linewidth=2)
    plt.hist(data_lw, histtype='step', color='dodgerblue', linewidth=2)
    ax.set_ylabel('Counts', fontsize=20)
    ax.set_xlabel('Phase (%)', fontsize=20)
    ax.axvline(50, color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_peak_' + paws[p] + '_' + protocol.replace(' ', '_')), dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_peak_' + paws[p] + '_' + protocol.replace(' ', '_') + '.svg'), dpi=256)