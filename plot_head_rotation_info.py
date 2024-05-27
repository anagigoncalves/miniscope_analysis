# # -*- coding: utf-8 -*-
# # %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import pandas as pd
import warnings
# warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Projects\\Dev\\miniscope_analysis\\')
import miniscope_session_class
# import locomotion_class
# path_session_data = 'J:\\Miniscope processed files'
# session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
# save_path = 'J:\\Miniscope processed files\\Analysis on population data\\\Head rotation\\split ipsi fast S1\\'

# for s in range(len(session_data)):
#     ses_info = session_data.iloc[s, :]
#     print(ses_info)
#     date = ses_info[3]
#     # path inputs
#     path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
#     session_type = path.split('\\')[-4].split(' ')[0]
#     session_id = session_type + '_' + ses_info[2]
path = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast\\MC9194\\2021_07_02\\'
mscope = miniscope_session_class.miniscope_session(path)

# Session data and inputs
animal = mscope.get_animal_id()
# session = loco.get_session_id()
#[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
# clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
# [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
# [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
head_angles = mscope.compute_head_angles(trials)
# [pca_headangles, trial_id] = mscope.pca_head_angles(head_angles, trials, plot_data=1)


