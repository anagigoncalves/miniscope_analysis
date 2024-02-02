import os
import numpy as np
import pandas as pd
import warnings
from scipy.ndimage import median_filter
import pickle
warnings.filterwarnings('ignore')

os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
import behav_locked_neural_activity_class
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
blna = behav_locked_neural_activity_class.behav_locked_neural_activity('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')

path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S2.xlsx')

paws = ['FR', 'HR', 'FL', 'HL']

def flatten_list(my_list):
    return [item for array in my_list for item in array]

data = {}
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = ses_info[0]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [_, _, _, _, _, _, _, _, trials,
      _, _, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
    [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, _, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)
    if session_id == 'tied':
        split_blocks = trials_ses
    else:
        split_blocks = blna.split_expblocks(trials_ses)
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    paws_rel = []
    st_strides_trials = []
    sw_strides_trials = []      
    for count_trial, f in enumerate(filelist):
        [final_tracks, _, _, _, _, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        final_tracks_trials.append(final_tracks)
    bodycenter, bodyspeed, bodyacc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 81, polyorder = 3)
    # cumul_idx = np.insert(np.cumsum([len(bodyspeed[tr_idx]) for tr_idx, _ in enumerate(trials)]), 0, 0)
    st_idx = {}; sw_idx = {};  
    for p, paw in enumerate(paws):
        st_idx[paw] = [st_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        sw_idx[paw] = [sw_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        
    trial_id = np.array(flatten_list([np.repeat(tr+1, trial.shape[-1]) for tr, trial in enumerate(final_tracks_trials)]))
    paw_displ = pd.DataFrame()
    for p in range(4):
        for ax in [0, 1, 3]:
            if ax == 0:
                axis = 'X'
            elif ax == 1:
                axis = 'Y'
            elif ax == 3:
                axis = 'Z'
            paw_displ_tr = []
            for tr in range(len(trials)):
                paws_rel = loco.get_paws_rel(final_tracks_trials[tr], axis)
                filt_paws_rel = median_filter(paws_rel[p], 10)
                paw_displ_tr.append(filt_paws_rel - np.nanmean(filt_paws_rel))
            paw_displ[f'{paws[p]} {axis}'] = flatten_list(paw_displ_tr)
            
    data[animal] = {'paws positions': paw_displ, 'trial id': trial_id, 'color trials': colors_session, 
                    'experimental blocks': split_blocks, 'body speed': bodyspeed,
                    'body acceleration': bodyacc, 'stance idx': st_idx, 'swing idx': sw_idx}

# Save data
file_path = os.path.join(path_session_data,'Behavior', f'Paw position all animals {session_id}.npy')
with open(file_path, 'wb') as file:
    pickle.dump(data, file)