import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
path_session_data = 'J:\\Miniscope processed files'
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\Calcium modulation\\split contra fast S1\\Histograms\\'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split contra fast'
paw = 'FR'
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']
align_dimension = 'phase'
if align_dimension == 'phase':
    bins = np.arange(0, 105, 10)  # 10 deg
    align_event = 'st' #is always stance
    bins_fr = bins
if align_dimension == 'time':
    bins = np.arange(-0.125, 0.126, 0.25) # 25 ms
    bins_fr = bins*1000

for animal in animals:

    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_ro'
                                                                                  'is.npy'))
    s = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')

    import miniscope_session_class
    import locomotion_class
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
     frames_dFF] = mscope.load_processed_files()
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
        trials, protocol.split(' ')[0], animal, session)

    if paw == 'FR':
        p = 0
    if paw == 'HR':
        p = 1
    if paw == 'FL':
        p = 2
    if paw == 'HL':
        p = 3
    firing_rate_animal = np.nanmean(firing_rate_animal[:, p, :, :], axis=0)

    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    sns.heatmap(firing_rate_animal, cmap='viridis', cbar=None,
        vmin=np.nanmin(firing_rate_animal), vmax=np.nanmax(firing_rate_animal))
    ax.set_yticks(np.arange(0, len(trials)))
    ax.invert_yaxis()
    ax.set_xticks([0, 5, 10])
    ax.set_xticklabels(['0', '50', '100'], rotation=45)
    ax.set_yticklabels(list(map(str, trials)), rotation=0)
    ax.axvline(x=np.int64(len(bins[::-1])/2), color='white')
    ax.axhline(y=trials_ses[0, 1], color='white', linestyle='dashed')
    ax.axhline(y=trials_ses[1, 1], color='white', linestyle='dashed')
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(os.path.join(save_path, animal + '_FR_heatmap'), dpi=256)
    plt.savefig(os.path.join(save_path, animal + '_FR_heatmap.svg'), dpi=256)
    plt.close('all')