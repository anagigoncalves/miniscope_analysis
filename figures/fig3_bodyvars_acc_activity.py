# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\STA bodyvars\\tied baseline S1\\'
sta_type = 'bodyvars'
window = np.arange(-330, 330 + 1)  # Samples
iter_n = 100 # Number of iterations of CS timestamps random shuffling

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
    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)

    # Load behavioral data and get acceleration
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    bodyacc = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
            frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        bodycenter_trial = sp.medfilt(loco.compute_bodycenter(final_tracks, 'X'), 25) #filter for tracking errors
        bodyacc_trial = loco.compute_bodyacc(bodycenter_trial)
        bodyacc.append(bodyacc_trial)

window = 0.1 #100ms
window_len = window*loco.sr
overlap = 0.5
trial = 1
t = 0
data_events_trial = df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial']==trial]
acc_trial = bodyacc[t]
frame_beg = np.int64(np.arange(0, len(bcam_time[t])-window_len, np.int64(window_len*overlap)))
frame_end = np.int64(np.arange(window_len, len(bcam_time[t]), np.int64(window_len*overlap)))
acc_windows = np.zeros(len(frame_beg))
data_neuro_windows = np.zeros((len(roi_list), len(frame_beg)))
for i in range(len(frame_beg)):
    acc_windows[i] = np.nanmean(acc_trial[frame_beg[i]:frame_end[i]])
    time_neuro_beg = np.argmin(np.abs(bcam_time[t][frame_beg[i]]-data_events_trial['time']))
    time_neuro_end = np.argmin(np.abs(bcam_time[t][frame_end[i]]-data_events_trial['time']))
    for count_r, r in enumerate(roi_list):
        data_neuro = data_events_trial[r][time_neuro_beg:time_neuro_end]
        if len(np.where(data_events_trial[r][time_neuro_beg:time_neuro_end])[0])>0:
            data_neuro_windows[count_r, i] = 1