import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters sw time\\split ipsi fast S1\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\DS sorted rasters\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

param_all_animals = []
for animal in animals:
    # Session data and inputs
    session_data_idx = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')
    session = loco.get_session_id()
    trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
        trials, protocol.split(' ')[0], animal, session)
    [_, _, _, df_extract_rawtrace_detrended, _, _, _, _, trials, _, _, _, _, _, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)

    #Compute continuous gait parameters
    filelist = loco.get_track_files(animal, session)
    st_strides_trials = []
    ds_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        st_strides_trials.append(st_strides_mat)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        ds_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                            'double_support'))
    cumulative_idx_array, param_all_time, param_all = loco.param_continuous_sym(ds_trials, st_strides_trials, trials, 'FR', 'FL', sym=1, remove_nan=0)
    param_all_animals.extend(param_all)


plt.figure()
plt.hist(param_all_animals, bins=100)