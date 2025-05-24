import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'st'
align_dimension = 'phase'
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']

animal = 'MC9194'
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

def calcium_rate_modulation_baseline(firing_rate_animal, paw, phase_plot, bs_trial):
    if paw == 'FR':
        p = 0
    if paw == 'HR':
        p = 1
    if paw == 'FL':
        p = 2
    if paw == 'HL':
        p = 3
    if phase_plot == 'st_full':
        phase_idx = np.array([0, 1, 2, 3, 4])
    if phase_plot == 'sw_full':
        phase_idx = np.array([5, 6, 7, 8, 9])
    if phase_plot == 'st_transition': #80% to 20% phase
        phase_idx = np.array([0, 1, 8, 9])
    if phase_plot == 'sw_transition': #30% to 60% phase
        phase_idx = np.array([3, 4, 5])
    if bs_trial == 'first_trial':
        firing_rate_FR_bs = np.nanmean(firing_rate_animal[:, p, 0, phase_idx], axis=-1)
    if bs_trial == 'block':
        firing_rate_FR_bs = np.nanmean(np.nanmean(firing_rate_animal[:, p, trials_baseline-1, phase_idx], axis=-1), axis=-1)
    firing_rate_FR = np.nanmean(firing_rate_animal[:, p, :, phase_idx], axis=0)
    firing_rate_FR_bs_rep = np.swapaxes(np.tile(firing_rate_FR_bs, (np.shape(firing_rate_FR)[1], 1)), 0, 1)
    firing_rate_FR_mod = (firing_rate_FR-firing_rate_FR_bs_rep)*100
    #TODO plot by block or by single trial
    return firing_rate_FR_mod

#TODO same plot separate by animal
#TODO with first trial of each block or do by blocks
#TODO pool all animals together
#parameters to plot
paw = 'FR'
bs_trial = 'first_trial' #options: first_trial, block
firing_rate_FR_st_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'st_full', bs_trial)
firing_rate_FR_sw_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'sw_full', bs_trial)
plt.plot(np.nanmean(firing_rate_FR_st_mod[:, [0, 6, 15, 16, 25]], axis=0), color='orange')
plt.plot(np.nanmean(firing_rate_FR_sw_mod[:, [0, 6, 15, 16, 25]], axis=0), color='green')