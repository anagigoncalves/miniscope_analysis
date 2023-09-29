import os
import numpy as np
import pandas as pd
import pickle
import warnings
import random
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
import behav_locked_neural_activity_class
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
blna = behav_locked_neural_activity_class.behav_locked_neural_activity('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')

path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'

seed_value = 42
random.seed(seed_value)

align = 'stride'
t_dim = 'phase'
p1 = 0
p2 = 2
paws = ['FR', 'HR', 'FL', 'HL']
bins = np.arange(0, 1.01, 0.05) # 5 deg
save_plot = True

# for s in range(len(session_data)): 
for s in [1]: # MC9513
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = session_type + '_' + ses_info[2]
    animal = mscope.get_animal_id()
    session = loco.get_session_id()

    # Load session data
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [_, _, _, df_extract_rawtrace_detrended, df_events_extract_rawtrace, _, _, _, trials,
     clusters_rois, colors_cluster, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
    [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)

    # flatten the list of ROIs 'clusters_rois'
    rois_sorted = []
    for i in range(len(clusters_rois)):
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    # Compute gait params & kinematics
    st_strides_trials = []
    sw_strides_trials = []
    final_tracks_trials = []
    param_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, _, _, _, _, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length'))
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')
    paw_difference = nxb.paw_diff(final_tracks_trials, p1, p2)
    _, paw_speed, paw_acc = nxb.paw_kinematic(final_tracks_trials, p1, 0, win_len = 81, polyorder = 3)
    _, body_speed, body_acc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 81, polyorder = 3)
    
    sorting_vars = {f'{p1}-{p2} difference' : paw_difference, 'paw speed': paw_speed, 'paw acceleration': paw_acc, 'body speed': body_speed, 'body acceleration': body_acc}

    # Load spikes timing data 
    file_path = os.path.join(path_session_data, 'Stride-locked neural activity', f'{animal}_{session_id}_{align}-locked_spikes_{t_dim}.npy')
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)

    # random_rois = [random.randint(0, len(dataset['spikes timing'][paws[p1]])-1) for _ in range(10)]

    for n in [64, 65, 70]:   
        # Retrieve st/sw indices and timestamps and spikes timing from data structure
        spikes_timing = dataset['spikes timing'][paws[p1]][n]
        sw_on_idx = dataset['stsw idx spike'][paws[p1]]['sw onset idx'][n]
        st_on_idx = dataset['stsw idx spike'][paws[p1]]['st onset idx'][n]
        st_off_idx = dataset['stsw idx spike'][paws[p1]]['st offset idx'][n]
    
        # Retrieve sw phase:
        sw_phase = [final_tracks_trials_phase[tr_idx][0, p1, sw_on_idx[tr_idx].astype(int)] for tr_idx, _ in enumerate(trials)]
        
        # Find st of the secondary paw p2
        p2_st_on_idx = dataset['stsw idx spike'][paws[p2]]['st onset idx'][n]
        p2_st_idx = []
        for tr_idx, _ in enumerate(trials):
            p2_st_idx_tr = []
            for i in range(len(st_on_idx[tr_idx])):
                start = st_on_idx[tr_idx][i].astype(int)
                end = st_off_idx[tr_idx][i].astype(int)
                idx = (p2_st_on_idx[tr_idx][np.where((p2_st_on_idx[tr_idx].astype(int) > start) & (p2_st_on_idx[tr_idx] < end))]).astype(int)
                p2_st_idx_tr.append(idx)
            p2_st_idx.append(p2_st_idx_tr)
        
        p2_st_phase = []
        for tr_idx, _ in enumerate(trials):
            p2_st_phase_tr = []
            for i in p2_st_idx[tr_idx]:
                if i:
                    phase = final_tracks_trials_phase[tr_idx][0, p2, i] 
                else:
                    phase = np.nan 
                p2_st_phase_tr.append(phase)
            p2_st_phase.append(p2_st_phase_tr)
            
        # Create a dictionary of all the sorting variables
        sorting_vars = {f'{paws[p2]}-st phase': p2_st_phase, f'{paws[p1]}-{paws[p2]} difference': paw_difference, 'paw speed': paw_speed, 
                        'paw acceleration': paw_acc, 'body speed': body_speed, 'body acceleration': body_acc, 'sw phase': sw_phase}
        
        # Transform arrays with spike timing for each stride in tuples to preserve structure when concatenating for sorting   
        spikes_timing_tuple = blna.sublist_arrays2tuples(spikes_timing)
        
        # Plot sorted rasters
        for sorted_by, behavior in sorting_vars.items():
            if sorted_by not in [f'{paws[p2]}-st phase', 'sl symmetry', 'sw phase']:
                behavior_stride = blna.get_mean_behav_stride(behavior, st_on_idx, sw_on_idx, trials) # Average of a behavioral variable (e.b.: paw acceleration) for each stride
            else:
                behavior_stride = behavior
            sorted_spikes_timing, sorted_behavior_stride = blna.sort_variable(np.concatenate(spikes_timing), np.concatenate(behavior_stride))
            print(f'Plotting rasters sorted by {sorted_by}')
            blna.plot_behav_locked_activity_sorted_rois(spikes_timing, sorted_spikes_timing, bins, trials, sorted_by, animal, session_id, paws[p1], rois_sorted[n], align, save_plot, t_dim)  
        
        # PLOT TRACES & BEHAVIOR DISTRIBUTION