import os
import pynapple as nap
import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm, rayleigh
import copy

os.chdir(r'C:\Users\User\Desktop\neural data analysis')
from utils import map_timestamps


def get_trial_ts(timestamps, trial_id):
    idx = np.where(np.diff(trial_id) == 1)[0]
    ts_on = np.insert(timestamps[idx+1], 0, 0)
    ts_off = np.insert(timestamps[idx], len(timestamps[idx+1]), timestamps[-1])
    return (ts_on, ts_off)


session = 'split ipsi fast'
animals = ['MC9513', 'MC9226', 'MC10221', 'MC8855', 'MC9194']
paws = ['FR', 'HR', 'FL', 'HL']
cols = [paw + ' phase' for paw in paws]
path = f'C:\\Users\\User\\Desktop\\mscope\\{session}'
n_bins = 10
n_iters = 500
save_data = True


for animal in animals:
    
    print(f'Processing animal {animal}...')
    
    ############## LOAD & PREPROCESS DATA ##############
    # MC9226 doesn't have session 1 in tied experiment
    if session == 'tied baseline' and animal == 'MC9226':
        n_session = 2
    else:
        n_session = 1
    
    # Load data
    spike_train = pd.read_csv(os.path.join(path, 'data', f'neural_data_{animal}_S{n_session}.csv'))
    behavior = pd.read_csv(os.path.join(path, 'data', f'behavioral_data_{animal}_S{n_session}.csv'))
    
    # Downsample behavior
    idx = map_timestamps(spike_train['session time'].values, behavior['session time'].values)
    behavior = behavior.loc[idx]
    
    # Get trial timestamps
    trial_ts = get_trial_ts(spike_train['session time'].values, spike_train['trial'].values)
    n_trials = spike_train['trial'].iloc[-1]
    
    # Get ROI IDs
    rois = pd.Series(spike_train.iloc[:, 3:].columns)
    n_rois = len(rois)
    
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
    

    ############## CREATE PYNAPPLE OBJECTS ##############
    # Create Pynapple object for spike trains
    ts = {idx: nap.Ts(t=spike_train.loc[spike_train[column] == 1, 'session time'].values) for idx, column in enumerate(spike_train.iloc[:, 3:].columns)}
    spikes_ts = nap.TsGroup(data=ts, group=rois)
    
    # Create Pynapple object for behavior
    stride_phase = nap.TsdFrame(t=spike_train['session time'].values, d=behavior.loc[:, cols].values, columns=cols)
    
    # Create Pynapple object for trials
    trials = nap.IntervalSet(start=trial_ts[0], end=trial_ts[1])
    

    ############## COMPUTE TUNING CURVES ##############
    # Estimate tuning curves for each trial
    tuning_curves = np.zeros((n_rois, n_trials, 4, n_bins))
    for tr_idx, tr in enumerate(trials):
        for p in range(4):
            tuning_curves[:, tr_idx, p, :] = nap.compute_1d_tuning_curves(spikes_ts, stride_phase[:, p], n_bins, ep=tr, minmax=(0, 1)).values.T
    
    # Estimate tuning curves for shuffled spike time distribution
    tuning_curves_chance = np.zeros((n_iters, n_rois, n_trials, 4, n_bins))
    for tr_idx, tr in enumerate(trials):
        spikes_ts_block = copy.copy(spikes_ts)
        spikes_ts_block.time_support=tr
        for i in range(n_iters):
            spikes_ts_shuffled = nap.shuffle_ts_intervals(spikes_ts_block)
            for p in range(4):
                tuning_curves_chance[i, :, tr_idx, p, :] = nap.compute_1d_tuning_curves(spikes_ts_shuffled, stride_phase[:, p], n_bins, ep=tr, minmax=(0, 1)).T.values
    
    # Save data
    if save_data:
        with open(os.path.join(path, 'data', f'tuning_curves_chance_{animal}.npy'), 'wb') as file:
            pickle.dump(tuning_curves_chance, file)
        with open(os.path.join(path, 'data', f'tuning_curves_{animal}.npy'), 'wb') as file:
            pickle.dump(tuning_curves, file)
        rois.to_csv(os.path.join(path, 'data', f'rois_{animal}.csv'), index=False, header=False)