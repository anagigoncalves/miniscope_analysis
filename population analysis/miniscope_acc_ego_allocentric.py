# -*- coding: utf-8 -*-

# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import pandas as pd
from scipy.stats import norm

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Projects\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\\Vestibular vs postural information\\split ipsi fast S1 2STD threshold\\'
window = np.arange(-330, 330 + 1)  # Samples
iter_n = 100 # Number of iterations of CS timestamps random shuffling
th = 1
def subset_bodyvars(arr_speed, std, arr_subset, th):
    frames_in = np.where((arr_speed<std*th)&(arr_speed>-std*th))[0]
    frames_out = np.where((arr_speed>=std*th)|(arr_speed<=-std*th))[0]
    arr_in = np.zeros(len(arr_subset))
    arr_in[:] = np.nan
    arr_in[frames_in] = arr_subset[frames_in]
    arr_out = np.zeros(len(arr_subset))
    arr_out[:] = np.nan
    arr_out[frames_out] = arr_subset[frames_out]
    return arr_in, arr_out

for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = session_type + '_' + ses_info[2]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    # Load behavioral data and get acceleration
    filelist = loco.get_track_files(animal, session)
    bodyacc_in = []
    bodycenter_in = []
    bodyspeed_in = []
    bodyacc_out = []
    bodycenter_out = []
    bodyspeed_out = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        bodycenter_trial = sp.medfilt(loco.compute_bodycenter(final_tracks, 'X'), 25) #filter for tracking errors
        bodyspeed_trial = loco.compute_bodyspeed(bodycenter_trial)
        bodyacc_trial = loco.compute_bodyacc(bodycenter_trial)
        mu, std = norm.fit(bodyspeed_trial) #get mean and std of bodyspeed
        [bodycenter_trial_in, bodycenter_trial_out] = subset_bodyvars(bodyspeed_trial, std, bodycenter_trial, th)
        [bodyspeed_trial_in, bodyspeed_trial_out] = subset_bodyvars(bodyspeed_trial, std, bodyspeed_trial, th)
        [bodyacc_trial_in, bodyacc_trial_out] = subset_bodyvars(bodyspeed_trial, std, bodyacc_trial, th)
        bodycenter_in.append(bodycenter_trial_in)
        bodyspeed_in.append(bodyspeed_trial_in)
        bodyacc_in.append(bodyacc_trial_in)
        bodycenter_out.append(bodycenter_trial_out)
        bodyspeed_out.append(bodyspeed_trial_out)
        bodyacc_out.append(bodyacc_trial_out)
        
    # Dictionary of all the independent variables on which computing the STA
    ind_vars = {'Body position in': bodycenter_in, 'Body speed in': bodyspeed_in, 'Body acceleration in': bodyacc_in,
                'Body position out': bodycenter_out, 'Body speed out': bodyspeed_out, 'Body acceleration out': bodyacc_out}
    keys=list(ind_vars.keys())

    # Loop through independent variables to compute and plot STAs of each one
    for var in range(len(ind_vars)):
        var_name = keys[var]
        variable = ind_vars[var_name]
        # Compute spike-triggered average (STA) of kinematic variables
        sta_allrois, signal_chunks_allrois = mscope.sta(df_events_extract_rawtrace, variable, bcam_time, window, trials)
        # Standardize observed STA on STA computed with shuffled data
        # Shuffle CS timestamps
        shuffled_spikes_ts = mscope.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)
        # Compute STA for shuffled data
        sta_shuffled_ts = np.array(mscope.sta_shuffled(shuffled_spikes_ts, variable, bcam_time, window, trials))
        # Standardize STA
        mean_chance = np.nanmean(sta_shuffled_ts, axis=2)
        sd_chance = np.nanstd(sta_shuffled_ts, axis=2)
        sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
        sta = np.zeros((len(sta_allrois), len(trials), len(window)))
        sta_shuffled = np.zeros((len(sta_allrois), len(trials), len(window)))
        for n in range(len(sta_allrois)):
            for tr in range(len(trials)):
                sta_zs[n, tr] = (sta_allrois[n][tr] - mean_chance[n][tr]) / sd_chance[n][tr]
                sta[n, tr] = sta_allrois[n][tr]
                sta_shuffled[n, tr] = mean_chance[n][tr]
        if not os.path.exists(os.path.join(save_path, animal + ' ' + ses_info[0])):
            os.mkdir(os.path.join(save_path, animal + ' ' + ses_info[0]))
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                             'sta_bodyvars_' + var_name.replace(' ', '_') + '_zscored.npy'), sta_zs)
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                             'sta_bodyvars_' + var_name.replace(' ', '_') + '_shuffled.npy'), sta_shuffled)
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
                             'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'), sta)
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
            'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses.npy'), trials_ses)
        np.save(os.path.join(save_path, animal + ' ' + ses_info[0],
            'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses_name.npy'), trials_ses_name)













