# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# path inputs
path = 'J:\\Miniscope processed files\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split ipsi fast S1 050421\\'
session_type = path.split('\\')[-4].split(' ')[0]

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()

frames_dFF = np.load(os.path.join(path, 'processed files', 'black_frames.npy'))
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials, clusters_rois, colors_cluster, idx_roi_cluster_ordered,
 ref_image, frames_dFF] = mscope.load_processed_files()
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()


isi_clusters = mscope.compute_isi(df_events_trace_clusters, 'raw', 'isi_clusters')
[cv_clusters, cv2_clusters] = mscope.compute_isi_cv(isi_clusters, trials)
#make time column continuous
# time_continuous = []
# for count_trial, trial in enumerate(trials):
#     time_continuous.extend(cv2_clusters.loc[cv2_clusters['trial']==trial, 'time'] + (loco.trial_time * count_trial))
# cv2_clusters_cont = cv2_clusters
# cv2_clusters_cont['time'] = time_continuous
cv2_clusters_notnan = cv2_clusters.dropna()
cv2_clusters_notnan_trial = cv2_clusters_notnan.loc[cv2_clusters_notnan['trial']==1].iloc[:, [0, 2, 3]]
cv2_clusters_notnan_trial_pivot = cv2_clusters_notnan_trial.pivot(columns='roi', values='cv2')
sns.heatmap(cv2_clusters_notnan_trial_pivot,cmap='viridis')