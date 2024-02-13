import matplotlib.collections
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st time\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
protocol = 'split ipsi fast'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
Ntrials = 26
trials = np.arange(1, Ntrials+1)
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
paws = ['FR', 'HR', 'FL', 'HL']
save_fig = 1

# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

animal_id = []
trial_id = []
roi_id = []
phase_id = []
amp_val = []
paw_id = []
coord_AP = []
coord_ML = []
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
    #Flip coords horizontally and vertically because image in miniscope is flipped
    centroid_ext_flip = np.zeros(np.shape(centroid_ext_arr))
    centroid_ext_flip[:, 1] = 1000-centroid_ext_arr[:, 0]
    centroid_ext_flip[:, 0] = 1000-centroid_ext_arr[:, 1]
    #Need to swap again, because now ML and AP are swapped
    #Adjust for the FOV coordinates to get global coordinates
    centroid_ext_swap = np.array(centroid_ext_flip)[:, [1, 0]]
    fov_coord = fov_coords[count_a]
    fov_corner = np.array([fov_coord[1] - 0.5, fov_coord[0] - 0.5]) #ML is the centroid[:, 0] and AP the centroid[:, 1]
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    if protocol == 'split ipsi fast' and animal == 'MC8855':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, 3:, :] = firing_rate_animal
        firing_rate_stride_median = np.nanmedian(firing_rate_match_arr, axis=3)
        firing_rate_stride_median_tile = np.repeat(firing_rate_stride_median[:, :, :, None],
                                                   np.shape(firing_rate_match_arr)[3], axis=3)
        firing_rate_match_arr_centered = firing_rate_match_arr-firing_rate_stride_median_tile
        firing_rate_amp = np.nanmax(firing_rate_match_arr_centered, axis=-1)
    elif protocol == 'split ipsi fast' and animal == 'MC9226':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, :-3, :] = firing_rate_animal
        firing_rate_stride_median = np.nanmedian(firing_rate_match_arr, axis=3)
        firing_rate_stride_median_tile = np.repeat(firing_rate_stride_median[:, :, :, None],
                                                   np.shape(firing_rate_match_arr)[3], axis=3)
        firing_rate_match_arr_centered = firing_rate_match_arr-firing_rate_stride_median_tile
        firing_rate_amp = np.nanmax(firing_rate_match_arr_centered, axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC8855':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, 3:, :] = firing_rate_animal
        firing_rate_stride_median = np.nanmedian(firing_rate_match_arr, axis=3)
        firing_rate_stride_median_tile = np.repeat(firing_rate_stride_median[:, :, :, None],
                                                   np.shape(firing_rate_match_arr)[3], axis=3)
        firing_rate_match_arr_centered = firing_rate_match_arr-firing_rate_stride_median_tile
        firing_rate_amp = np.nanmax(firing_rate_match_arr_centered, axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        trial_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25])
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, trial_idx, :] = firing_rate_animal
        firing_rate_stride_median = np.nanmedian(firing_rate_match_arr, axis=3)
        firing_rate_stride_median_tile = np.repeat(firing_rate_stride_median[:, :, :, None],
                                                   np.shape(firing_rate_match_arr)[3], axis=3)
        firing_rate_match_arr_centered = firing_rate_match_arr-firing_rate_stride_median_tile
        firing_rate_amp = np.nanmax(firing_rate_match_arr_centered, axis=-1)
    else:
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_stride_median = np.nanmedian(firing_rate_animal, axis=3)
        firing_rate_stride_median_tile = np.repeat(firing_rate_stride_median[:, :, :, None],
                                                   np.shape(firing_rate_animal)[3], axis=3)
        firing_rate_animal_centered = firing_rate_animal-firing_rate_stride_median_tile
        firing_rate_amp = np.nanmax(firing_rate_animal_centered, axis=-1)
    for p, paw in enumerate(paws):
        for t, trial in enumerate(np.arange(1, Ntrials + 1)):
            animal_id.extend(np.repeat(animal, np.shape(firing_rate_amp)[0]))
            trial_id.extend(np.repeat(trial, np.shape(firing_rate_amp)[0]))
            phase_id.extend(np.repeat('cycle', np.shape(firing_rate_amp)[0]))
            paw_id.extend(np.repeat(paw, np.shape(firing_rate_amp)[0]))
            roi_id.extend(np.arange(0, np.shape(firing_rate_amp)[0]))
            amp_val.extend(firing_rate_amp[:, p, t])
            coord_AP.extend(centroid_dist_corner[:, 0])
            coord_ML.extend(centroid_dist_corner[:, 1])

amp_dict = {'animal': animal_id, 'trial': trial_id, 'roi': roi_id, 'phase': phase_id, 'amp': amp_val, 'paw': paw_id, 'coord_AP': coord_AP,
        'coord_ML': coord_ML}
df_amp = pd.DataFrame(amp_dict)

# Do also line plots of mean and std - learning style - step cycle modulation
for paw in paws:
    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    data_plot = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='cycle'), ['trial', 'amp']]
    rectangle = plt.Rectangle((6.5, 0), 10, 1, fc='lightgrey', alpha=0.3, zorder=-1)
    plt.gca().add_patch(rectangle)
    ax = sns.lineplot(data=data_plot, x='trial', y='amp', estimator='mean', ci='sd', color='black', marker='o')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Event rate amplitude', fontsize=20)
    if save_fig:
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_step_cycle' + paw + '_trials'),
                    dpi=256)
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_step_cycle' + paw + '_trials.svg'),
                    dpi=256)


# Do also line plots of single rois per animal - sw and st modulation
for paw in paws:
    data_plot = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='cycle')]
    fig, ax = plt.subplots(5, 1, figsize=(7, 15), tight_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for count_a, animal in enumerate(animals):
        data_plot_animal = data_plot.loc[data_plot['animal'] == animal]
        for roi in data_plot_animal.roi.unique():
            data_plot_animal_roi = data_plot_animal.loc[data_plot_animal['roi'] == roi]
            ax[count_a].plot(np.arange(1, Ntrials+1), data_plot_animal_roi['amp'], color='black', linewidth=0.1)
        ax[count_a].axvline(x=6.5, color='black')
        ax[count_a].axvline(x=16.5, color='black')
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='both', which='major', labelsize=20)
        ax[count_a].set_xlabel('Trial', fontsize=20)
        ax[count_a].set_ylabel('Event\nrate\namplitude', fontsize=20)
        if save_fig:
            plt.savefig(os.path.join(save_path, 'firing_rate_amp_' + paw + '_singlerois_trials'),
                        dpi=256)
            plt.savefig(os.path.join(save_path, 'firing_rate_amp_' + paw + '_singlerois_trials.svg'),
                        dpi=256)