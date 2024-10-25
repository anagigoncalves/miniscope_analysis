import os
import numpy as np
import pandas as pd

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
protocol = 'split ipsi fast'
reg_frames_clear = np.int64(np.ones(len(session_data)))
print(session_data)
#reg_frames_clear[0] = 7 #tied baseline S1
reg_frames_clear[0] = 3 #split ipsi fast S1
reg_frames_clear[3] = 2 #split ipsi fast S1

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
    head_angles = pd.read_csv(os.path.join(mscope.path, 'processed files', 'head_angles.csv'))

    def time_adjustment(head_angles, frames_dFF, reg_bad_frames, reg_frames_nr):
        trials_vec = np.int64(head_angles['trial'].unique())
        roll_list = []
        pitch_list = []
        yaw_list = []
        time_list = []
        trial_list = []
        for count_t, t in enumerate(trials_vec):
            time_trial = np.array(head_angles.loc[head_angles['trial'] == t, 'time'])
            time_list.extend(time_trial[frames_dFF[count_t] - 1:])
            roll_trial = np.array(head_angles.loc[head_angles['trial'] == t, 'roll'])
            roll_list.extend(roll_trial[frames_dFF[count_t] - 1:])
            pitch_trial = np.array(head_angles.loc[head_angles['trial'] == t, 'pitch'])
            pitch_list.extend(pitch_trial[frames_dFF[count_t] - 1:])
            yaw_trial = np.array(head_angles.loc[head_angles['trial'] == t, 'yaw'])
            yaw_list.extend(yaw_trial[frames_dFF[count_t] - 1:])
            trial_trial = np.array(head_angles.loc[head_angles['trial'] == t, 'trial'])
            trial_list.extend(trial_trial[frames_dFF[count_t] - 1:])
        dict_ori = {'roll': roll_list, 'pitch': pitch_list, 'yaw': yaw_list, 'time': time_list, 'trial': trial_list}
        head_orientation = pd.DataFrame(dict_ori)  # create dataframe with dFF, roi id and trial id
        if protocol == 'tied baseline' and animal == 'MC8855':
            print('a')
            head_orientation1 = head_orientation.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation2 = head_orientation1.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation3 = head_orientation2.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation4 = head_orientation3.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation5 = head_orientation4.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation6 = head_orientation5.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation_corr = head_orientation6.drop(reg_bad_frames).reset_index(drop=True)
        elif protocol == 'split ipsi fast' and animal == 'MC8855':
            head_orientation1 = head_angles_time_corr.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation_corr = head_orientation1.drop(reg_bad_frames).reset_index(drop=True)
        elif protocol == 'split ipsi fast' and animal == 'MC9226':
            print('c')
            head_orientation1 = head_orientation.drop(reg_bad_frames).reset_index(drop=True)
            head_orientation_corr = head_orientation1.drop(reg_bad_frames).reset_index(drop=True)
        else:
            print('d')
            head_orientation_corr = head_orientation.drop(reg_bad_frames).reset_index(drop=True)
        return head_orientation_corr

    head_angles_time_corr = time_adjustment(head_angles, frames_dFF, reg_bad_frames, reg_frames_clear[s])
    head_angles_time_corr.to_csv(os.path.join(mscope.path, 'processed files', 'head_angles_time_adjusted.csv'), sep=',', index=False)

print('Size calcium data ' + str(len(df_extract_rawtrace_detrended)))
print('Size head data ' + str(len(head_angles_time_corr)))
print('Size reg_bad_frames ' + str(len(reg_bad_frames)))