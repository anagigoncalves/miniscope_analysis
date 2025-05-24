import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\User\Desktop\neural data analysis')
from utils import normalize, load_data, image_saver


def plot_modul_idx(data, ax=None, tick_labels=['baseline', 'early split', 'late split', 'early washout', 'late washout']):
    if ax is None:
        fig, ax = plt.subplots()
    x = range(len(data[0]))
    
    ax.plot(x, data[0], color='darkorange', marker='.', linewidth=3, markersize=18, label='Stance')
    ax.plot(x, data[1], color='green', marker='.', linewidth=3, markersize=18, label='Swing')

    ax.set_ylabel('Firing rate change (%)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xticks(range(len(tick_labels))); ax.set_xticklabels(tick_labels, fontsize=20, rotation=90)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.axhline(0, linestyle='--', color='dimgray')
    plt.tight_layout()
    plt.show()
    return ax


def plot_peak_firing(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    x = np.arange(1, len(data[0])+1)
    
    ax.plot(x, data[0], color='darkorange', marker='.', linewidth=3, markersize=18, label='Stance')
    ax.plot(x, data[1], color='green', marker='.', linewidth=3, markersize=18, label='Swing')

    ax.set_ylabel('Peak firing rate (z-score)', fontsize=20)
    ax.set_xlabel('Trials', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()
    ax.margins(0)
    return ax


session = 'split ipsi fast'
data_dir = f'C:\\Users\\User\\Desktop\\mscope\\{session}'
save_plot = False
subjects = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
peak_phase_block_all = []
gain_sw_block_all = []
gain_st_block_all = []
for subject in subjects:
    
    # if subject in ['MC9194', 'MC9513', 'MC10221']:
    #     blocks = [[4, 5], [6, 7], [14, 15], [16, 17], [24, 25]]
    # elif subject == 'MC9226':
    #     blocks = [[4, 5], [6, 7], [14, 15], [16, 17], [21, 22]]
    # elif subject == 'MC8855':
    #     blocks = [[1, 2], [4, 5], [12, 13], [14, 15], [21, 22]]
    
    if subject in ['MC9194', 'MC9513', 'MC10221']:
        blocks = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    elif subject == 'MC9226':
        blocks = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18], [19, 20, 21, 22]]
    elif subject == 'MC8855':
        blocks = [[0, 1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15, 16, 17], [18, 19, 20, 21, 22]]
        
    data = load_data(data_dir, f'tuning_curves_{subject}', folder='data')
    peth = data[:, :, 0]
    peth_norm = normalize(peth, ax=-1)
    
    n_rois = peth.shape[0]; n_trials = peth.shape[1]; n_bins = peth.shape[2]; n_blocks = len(blocks)
    
    
    ################## PEAK FIRING ##################
    peak_st = np.zeros((n_rois, n_trials))
    peak_sw = np.zeros((n_rois, n_trials))   
    for tr in range(n_trials):
        peak_st[:, tr] = np.max(peth_norm[:, tr, :n_bins//2], axis=-1)
        peak_sw[:, tr] = np.max(peth_norm[:, tr, n_bins//2:], axis=-1)
            
    # Stance and swing peak firing for each trial by animal
    fig, ax = plt.subplots()
    plot_peak_firing([np.median(peak_st, axis=0), np.median(peak_sw, axis=0)], ax=ax)
    ax.axvspan(blocks[1][1], blocks[3][0], alpha=0.5, color='lightgrey')
    if save_plot:
        image_saver(data_dir, 'modulation index', f'Peak firing {subject}')
        plt.close()
    
    
    ################## MODULATION INDEX ##################
    peak_st_bsl = np.median(peak_st[:, blocks[0]], axis=-1, keepdims=True)
    peak_sw_bsl = np.median(peak_sw[:, blocks[0]], axis=-1, keepdims=True)
    
    gain_st = ((peak_st - peak_st_bsl)/peak_st_bsl) * 100
    gain_sw = ((peak_sw - peak_sw_bsl)/peak_sw_bsl) * 100

    gain_st_block = np.array([np.median(gain_st[:, b], axis=-1) for b in blocks]).T
    gain_st_block[:, 0] = 0
    gain_sw_block = np.array([np.median(gain_sw[:, b], axis=-1) for b in blocks]).T
    gain_sw_block[:, 0] = 0

    # # Modulation index by ROI
    # for i in range(0, n_rois, 10):
    #     fig, ax = plt.subplots()
    #     plot_modul_idx([gain_st_block[i], gain_sw_block[i]], ax=None)
    #     if save_plot:
    #         image_saver(data_dir, 'Modulation index', f'Modulation index ROI{i} {subject}')
    #         plt.close()
    
    # Modulation index by animal
    fig, ax = plt.subplots(figsize=(5,6))
    plot_modul_idx([np.median(gain_st_block, axis=0), np.median(gain_sw_block, axis=0)], ax=ax)
    [ax.plot(gain_st_block[i], c='darkorange', alpha=0.05) for i in range(n_rois)]
    [ax.plot(gain_sw_block[i], c='darkgreen', alpha=0.05) for i in range(n_rois)]
    ax.set_ylim(-60, 60)
    if save_plot:
        image_saver(data_dir, 'modulation index', f'Modulation index {subject}')
        plt.close()
    
    gain_st_block_all.append(gain_st_block)
    gain_sw_block_all.append(gain_sw_block)