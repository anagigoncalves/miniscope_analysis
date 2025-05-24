import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(r'C:\Users\User\Desktop\neural data analysis')
from neuro_visual import plot_peth_popul, plot_peth, plot_peth_trials
from utils import image_saver, load_data


############### ASSIGN VARIABLES ###############
session = 'split ipsi fast'
path = f'C:\\Users\\User\\Desktop\\mscope\\{session}'
save_plot = False
color_paw = ['red', 'magenta', 'blue', 'cyan']
color_blocks = ['black', 'crimson', 'blue']
threshold = 3
fr_lim = (0.5, 3.5)
animals = ['MC9513', 'MC9226', 'MC10221', 'MC8855', 'MC9194']


############### LOAD DATA & PRE-PROCESS ###############
for animal in animals:
    print(f'Processing animal {animal}...')
    
    tuning_curves = load_data(path, f'tuning_curves_{animal}', folder='data')
    tuning_curves_chance = load_data(path, f'tuning_curves_chance_{animal}', folder='data')
    rois = pd.read_csv(os.path.join(path, 'data', f'rois_{animal}.csv'), header=None)

    # Assign some variables
    n_rois = tuning_curves.shape[0]
    n_trials = tuning_curves.shape[1]
    n_bins = tuning_curves.shape[-1]
    
    # Define experimental blocks
    if session == 'tied baseline':
        blocks = [[0, n_trials]]
    else:
        # if animal in ['MC9513', 'MC10221', 'MC9194']:
        #     blocks = [[0, 6], [6, 11], [11, 16], [16, 21], [21, 25]]
        # elif animal == 'MC9226':
        #     blocks = [[0, 6], [6, 11], [11, 16], [16, 19], [19, 23]]
        # elif animal == 'MC8855':
        #     blocks = [[0, 3], [3, 8], [8, 13], [13, 18], [18, 23]]
        if animal in ['MC9513', 'MC10221', 'MC9194']:
            blocks = [[4, 6], [6, 8], [14, 16], [16, 18], [24, 26]]
        elif animal == 'MC9226':
            blocks = [[4, 6], [6, 8], [14, 16], [16, 18], [21, 23]]
        elif animal == 'MC8855':
            blocks = [[1, 3], [3, 5], [11, 13], [13, 15], [21, 23]]
    n_blocks = len(blocks)
    
    # Get average tuning curves for shuffled distribution and their variability
    tuning_curves_chance_mean = np.mean(tuning_curves_chance, axis=0)
    tuning_curves_chance_std = np.std(tuning_curves_chance, axis=0)
    
    # Get average tuning curves by block
    tuning_curves_block = np.zeros((n_rois, n_blocks, 4, n_bins))
    tuning_curves_chance_mean_block = np.zeros((n_rois, n_blocks, 4, n_bins))
    tuning_curves_chance_std_block = np.zeros((n_rois, n_blocks, 4, n_bins))
    for b, (start, end) in enumerate(blocks):
        tuning_curves_block[:, b, :] = np.median(tuning_curves[:, start:end, :], axis=1)
        tuning_curves_chance_mean_block[:, b, :] = np.median(tuning_curves_chance_mean[:, start:end, :], axis=1)
        tuning_curves_chance_std_block[:, b, :] = np.median(tuning_curves_chance_std[:, start:end, :], axis=1)
    
    # Get preferred phase
    peak_phase = np.argmax(tuning_curves_block, axis=-1)

    # Standardize tuning_curves on shuffled data
    # tuning_curves_norm = (tuning_curves - tuning_curves_chance_mean)/tuning_curves_chance_std
    tuning_curves_norm = (tuning_curves_block - tuning_curves_chance_mean_block)/tuning_curves_chance_std_block


    ############## PLOT DATA ##############
    # Plot ROIs trial by trial
    for i in range(n_rois):
        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        for p in range(4):
            plot_peth_trials(tuning_curves[i, :, p], xlabel='FR phase', ylabel='Trials',  
                          xticks=[0, n_bins/2, n_bins], xticklabels=['St', 'Sw', 'St'], lim=fr_lim, ax=ax[0, p])
            plot_peth(tuning_curves_block[i, [0, 1, 3], p], xlabel='FR phase', ylabel='Firing rate (Hz)',  
                      xticks=[0, (n_bins-1)/2, n_bins-1], xticklabels=['St', 'Sw', 'St'], 
                      ax=ax[1, p], color_trials=color_blocks, alpha=1, mean_off=True)
            if session != 'tied baseline':
                ax[0, p].axhline(blocks[0][1], linestyle='--', color='white')
                ax[0, p].axhline(blocks[2][1], linestyle='--', color='white')
            ax[0, p].axvline(n_bins/2, linestyle='--', color='white')
            ax[1, p].axvline((n_bins-1)/2, linestyle='--', color='k')
            ax[1, p].set_ylim(fr_lim)                
        if save_plot:
            image_saver(path, 'figures', f'{rois[i]}_trials_{animal}')
            plt.close()

    # Plot ROIs baseline with confidence interval
    for i in range(n_rois):
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        for p in range(4):
            plot_peth(tuning_curves_norm[i, :, p], xlabel='FR phase', ylabel='Firing rate (Hz)',  
                      xticks=[0, (n_bins-1)/2, n_bins-1], xticklabels=['St', 'Sw', 'St'], 
                      ax=ax[p], mean_off=True, color_trials='lightgray')
            plot_peth(tuning_curves_norm[i, 0, p], xlabel='FR phase', ylabel='Firing rate (Hz)',  
                      xticks=[0, (n_bins-1)/2, n_bins-1], xticklabels=['St', 'Sw', 'St'], 
                      color_mean=color_paw[p], ax=ax[p], linewidth=3, alpha=1)
            # upper_bound = tuning_curves_chance_mean_block[i, 0, p] + threshold*tuning_curves_chance_std_block[i, 0, p]
            # lower_bound = tuning_curves_chance_mean_block[i, 0, p] - threshold*tuning_curves_chance_std_block[i, 0, p]
            # ax[p].fill_between(range(n_bins), upper_bound, lower_bound, color = 'lightgray', alpha = 0.5)
            ax[p].axhline(threshold, linestyle='--', color='k') 
            ax[p].axhline(-threshold, linestyle='--', color='k') 
            ax[p].axvline((n_bins-1)/2, linestyle='--', color='k') 
            ax[p].set_ylim(-3, 3)
        if save_plot:
            image_saver(path, 'figures', f'{rois[i]}_confidence_{animal}')
            plt.close()

    # Plot population
    fig, axs = plt.subplots(4, n_blocks, figsize=(n_blocks*4, 16))
    for b, _ in enumerate(blocks):
        for p in range(4):
            if n_blocks > 1:
                ax = axs[p, b]
            else:
                ax = axs[p]
            plot_peth_popul(tuning_curves_block[:, b, p], ylabel='ROIs',
                            xticks=[0, n_bins/2, n_bins], xticklabels=['St', 'Sw', 'St'], ax=ax, 
                            sort_var=peak_phase[:, 0, 0], cbar_label='Firing rate (Hz)', lim=fr_lim)
    if save_plot:
        image_saver(path, 'figures', f'Population_{animal}')
        plt.close()
