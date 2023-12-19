import matplotlib.collections
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
protocol = 'split contra fast'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
Ntrials = 26
trials = np.arange(1, Ntrials+1)
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
bins = np.arange(0, 1.01, 0.05)  # 5 deg
paws = ['FR', 'HR', 'FL', 'HL']

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
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split ipsi fast' and animal == 'MC9226':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, :-3, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC8855':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, 3:, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    elif protocol == 'split contra fast' and animal == 'MC10221':
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        trial_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25])
        firing_rate_match_arr = np.zeros((np.shape(firing_rate_animal)[0], 4, Ntrials, 20))
        firing_rate_match_arr[:] = np.nan
        firing_rate_match_arr[:, :, trial_idx, :] = firing_rate_animal
        firing_rate_amp_st = np.nanmax(firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_match_arr[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    else:
        firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_rois.npy'))
        firing_rate_amp_st = np.nanmax(firing_rate_animal[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1) - np.nanmin(
            firing_rate_animal[:, :, :, :np.where(bins == 0.5)[0][0]], axis=-1)
        firing_rate_amp_sw = np.nanmax(firing_rate_animal[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1) - np.nanmin(
            firing_rate_animal[:, :, :, np.where(bins == 0.5)[0][0]:], axis=-1)
    for p, paw in enumerate(paws):
        for t, trial in enumerate(np.arange(1, Ntrials + 1)):
            animal_id.extend(np.repeat(animal, np.shape(firing_rate_amp_st)[0]))
            trial_id.extend(np.repeat(trial, np.shape(firing_rate_amp_st)[0]))
            phase_id.extend(np.repeat('st', np.shape(firing_rate_amp_st)[0]))
            paw_id.extend(np.repeat(paw, np.shape(firing_rate_amp_st)[0]))
            roi_id.extend(np.arange(0, np.shape(firing_rate_amp_st)[0]))
            amp_val.extend(firing_rate_amp_st[:, p, t])
            coord_AP.extend(centroid_dist_corner[:, 0])
            coord_ML.extend(centroid_dist_corner[:, 1])
            animal_id.extend(np.repeat(animal, np.shape(firing_rate_amp_sw)[0]))
            trial_id.extend(np.repeat(trial, np.shape(firing_rate_amp_sw)[0]))
            phase_id.extend(np.repeat('sw', np.shape(firing_rate_amp_sw)[0]))
            paw_id.extend(np.repeat(paw, np.shape(firing_rate_amp_sw)[0]))
            roi_id.extend(np.arange(0, np.shape(firing_rate_amp_sw)[0]))
            amp_val.extend(firing_rate_amp_sw[:, p, t])
            coord_AP.extend(centroid_dist_corner[:, 0])
            coord_ML.extend(centroid_dist_corner[:, 1])

amp_dict = {'animal': animal_id, 'trial': trial_id, 'roi': roi_id, 'phase': phase_id, 'amp': amp_val, 'paw': paw_id, 'coord_AP': coord_AP,
        'coord_ML': coord_ML}
df_amp = pd.DataFrame(amp_dict)

for paw in paws:
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    rectangle = plt.Rectangle((1.5, 0), 2, 10, fc='lightgrey', alpha=0.3, zorder=-1)
    plt.gca().add_patch(rectangle)
    data_plot = df_amp.loc[(df_amp['paw']==paw)&(df_amp['trial'].isin(np.array([1, 6, 7, 16, 17, 26])))]
    ax = sns.violinplot(data=data_plot,
        x='trial', y='amp', saturate=1, hue='phase', split=True, inner=None, palette=['orange', 'green'], legend=False)
    for collection in ax.collections:
        if isinstance(collection, mp.collections.PolyCollection):
            collection.set_edgecolor(collection.get_facecolor())
            collection.set_facecolor('none')
            collection.set_linewidth(4)
    ax.scatter(np.arange(0, 6)-0.1, data_plot.loc[data_plot['phase']=='st'].groupby('trial').mean()['amp'],
             s=20, color='orange')
    ax.scatter(np.arange(0, 6)+0.1, data_plot.loc[data_plot['phase']=='sw'].groupby('trial').mean()['amp'],
             s=20, color='green')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Event rate amplitude', fontsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_dist_' + paw + '_trials'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_dist_' + paw + '_trials.svg'),
                dpi=256)
    plt.close('all')

# Do also line plots of single rois per animal
for paw in paws:
    data_plot_sw = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='sw')]
    fig, ax = plt.subplots(5, 1, figsize=(7, 15), tight_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for count_a, animal in enumerate(animals):
        data_plot_sw_animal = data_plot_sw.loc[data_plot_sw['animal'] == animal]
        for roi in data_plot_sw_animal.roi.unique():
            data_plot_sw_animal_roi = data_plot_sw_animal.loc[data_plot_sw_animal['roi'] == roi]
            ax[count_a].plot(np.arange(1, Ntrials+1), data_plot_sw_animal_roi['amp'], color='green', linewidth=0.1)
        ax[count_a].axvline(x=6.5, color='black')
        ax[count_a].axvline(x=16.5, color='black')
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='both', which='major', labelsize=20)
        ax[count_a].set_xlabel('Trial', fontsize=20)
        ax[count_a].set_ylabel('Event\nrate\namplitude', fontsize=20)
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_sw_' + paw + '_singlerois_trials'),
                    dpi=256)
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_sw_' + paw + '_singlerois_trials.svg'),
                    dpi=256)
    data_plot_st = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='st')]
    fig, ax = plt.subplots(5, 1, figsize=(7, 15), tight_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()
    for count_a, animal in enumerate(animals):
        data_plot_st_animal = data_plot_st.loc[data_plot_st['animal'] == animal]
        for roi in data_plot_st_animal.roi.unique():
            data_plot_st_animal_roi = data_plot_st_animal.loc[data_plot_st_animal['roi'] == roi]
            ax[count_a].plot(np.arange(1, Ntrials+1), data_plot_st_animal_roi['amp'], color='orange', linewidth=0.1)
        ax[count_a].axvline(x=6.5, color='black')
        ax[count_a].axvline(x=16.5, color='black')
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='both', which='major', labelsize=20)
        ax[count_a].set_xlabel('Trial', fontsize=20)
        ax[count_a].set_ylabel('Event\nrate\namplitude', fontsize=20)
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_st_' + paw + '_singlerois_trials'),
                    dpi=256)
        plt.savefig(os.path.join(save_path, 'firing_rate_amp_st_' + paw + '_singlerois_trials.svg'),
                    dpi=256)
plt.close('all')

# Do also line plots of mean and std - learning style
for paw in paws:
    fig, ax = plt.subplots(figsize=(5, 7), tight_layout=True)
    data_plot_sw = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='sw'), ['trial', 'amp']]
    data_plot_st = df_amp.loc[(df_amp['paw']==paw)&(df_amp['phase']=='st'), ['trial', 'amp']]
    rectangle = plt.Rectangle((6.5, 0), 10,6, fc='lightgrey', alpha=0.3, zorder=-1)
    plt.gca().add_patch(rectangle)
    ax = sns.lineplot(data=data_plot_st, x='trial', y='amp', estimator='mean', ci='sd', color='orange', marker='o')
    ax = sns.lineplot(data=data_plot_sw, x='trial', y='amp', estimator='mean', ci='sd', color='green', marker='o')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Event rate amplitude', fontsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_' + paw + '_trials'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_' + paw + '_trials.svg'),
                dpi=256)
    plt.close('all')

for paw in paws:
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    data_plot_st_tf = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 7) & (df_amp['phase'] == 'st')]
    sc = ax.scatter(data_plot_st_tf['coord_AP'], data_plot_st_tf['coord_ML'], s=15, c=data_plot_st_tf['amp'],
                    cmap='viridis')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    cbar.mappable.set_clim(0.5, 7.5)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_earlysplit_st_' + paw + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_earlysplit_st_' + paw + '_roilocation.svg'),
                dpi=256)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    data_plot_sw_tf = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 7) & (df_amp['phase'] == 'sw')]
    sc = ax.scatter(data_plot_sw_tf['coord_AP'], data_plot_sw_tf['coord_ML'], s=15, c=data_plot_sw_tf['amp'],
                    cmap='viridis')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.mappable.set_clim(0.5, 7.5)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_earlysplit_sw_' + paw + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_earlysplit_sw_' + paw + '_roilocation.svg'),
                dpi=256)
    plt.close('all')
for paw in paws:
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    data_plot_st_ti = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 7) & (df_amp['phase'] == 'st')]
    data_plot_st_tf = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 16) & (df_amp['phase'] == 'st')]
    delta_amp_st = np.array(data_plot_st_tf['amp'])-np.array(data_plot_st_ti['amp'])
    sc = ax.scatter(data_plot_st_tf['coord_AP'], data_plot_st_tf['coord_ML'], s=15, c=delta_amp_st,
                    cmap='coolwarm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    cbar.mappable.set_clim(-4, 4)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_deltasplit_st_' + paw + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_deltasplit_st_' + paw + '_roilocation.svg'),
                dpi=256)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
    data_plot_sw_ti = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 7) & (df_amp['phase'] == 'sw')]
    data_plot_sw_tf = df_amp.loc[(df_amp['paw'] == paw) & (df_amp['trial'] == 16) & (df_amp['phase'] == 'sw')]
    delta_amp_sw = np.array(data_plot_sw_tf['amp']) - np.array(data_plot_sw_ti['amp'])
    sc = ax.scatter(data_plot_sw_tf['coord_AP'], data_plot_sw_tf['coord_ML'], s=15, c=delta_amp_sw,
                    cmap='coolwarm')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('AP coordinate (mm)', fontsize=20)
    ax.set_xlabel('ML coordinate (mm)', fontsize=20)
    plt.gca().invert_yaxis()
    cbar = plt.colorbar(sc)
    cbar.mappable.set_clim(-4, 4)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_deltasplit_sw_' + paw + '_roilocation'),
                dpi=256)
    plt.savefig(os.path.join(save_path, 'firing_rate_amp_deltasplit_sw_' + paw + '_roilocation.svg'),
                dpi=256)
    plt.close('all')