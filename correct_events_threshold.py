# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S2.xlsx')
protocol_type = 'split'

for session_data_idx in range(np.shape(session_data)[0]):
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    # Session data and inputs
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                             frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    trials_ses_name.insert(len(trials_ses_name), 'late washout')
    trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]
    [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
        mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)
    roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)

    # Find calcium events - label as synchronous or asynchronous
    df_events_extract = mscope.get_events(df_extract, 0, 'df_events_extract')  # 0 for no detrending
    df_events_extract_rawtrace = mscope.get_events(df_extract_rawtrace, 1,
                                                   'df_events_extract_rawtrace')  # 1 for detrending"

    df_events_extract = df_events_extract.drop(reg_bad_frames)
    df_events_extract_rawtrace = df_events_extract_rawtrace.drop(reg_bad_frames)
    df_events_extract.to_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract.csv'), sep=',', index=False)
    df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'),
                                      sep=',', index=False)

# roi_plot = np.int64(np.random.choice(roi_list)[3:])
# print('ROI ' + str(roi_plot))
# trial_plot = np.random.choice(trials)
# print('Trial ' + str(trial_plot))
# int_find = ''.join(x for x in df_events_extract_rawtrace.columns[2] if x.isdigit())
# int_find_idx = df_events_extract_rawtrace.columns[2].find(int_find)
# if df_events_extract_rawtrace.columns[2][:int_find_idx] == 'ROI':
#     df_type = 'ROI'
# else:
#     df_type = 'cluster'
# idx_nr = df_type + str(roi_plot)
# df_dff_trial = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, idx_nr])  # get dFF for the desired trial
# df_dff_time = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, 'time'])  # get dFF for the desired trial
# fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
# idx_trial = np.where(trials == trial_plot)[0][0]
# ax.plot(df_dff_time, df_dff_trial, color='black')
# events_plot = np.where(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial_plot, idx_nr])[0]
# for e in events_plot:
#     ax.scatter(df_dff_time[e], df_dff_trial[e], s=60,
#                color='orange')
# ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
# ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
# plt.xticks(fontsize=mscope.fsize - 4)
# plt.yticks(fontsize=mscope.fsize - 4)
# plt.setp(ax.get_yticklabels(), visible=False)
# ax.tick_params(axis='y', which='y', length=0)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# plt.tick_params(axis='y', labelsize=0, length=0)

# #compute firing rates and ISI...
# isi_df = mscope.compute_isi(df_events_extract_rawtrace, 'raw', [])
# roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
# fr_roi = np.zeros(len(roi_list))
# for count_r, r in enumerate(roi_list):
#     fr_roi[count_r] = 1 / isi_df.loc[isi_df['roi'] == r, 'isi'].mean()
#
# plt.figure()
# plt.scatter(np.arange(0, len(fr_roi)), fr_roi)