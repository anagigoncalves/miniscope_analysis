# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'E:\\Miniscope processed files\\'
session_data = pd.read_excel('E:\\Miniscope processed files\\session_data_split_S1.xlsx')
s=0
ses_info = session_data.iloc[s, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
traces_type = 'raw'
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
[df_trace_clusters_ave, df_trace_clusters_std] = mscope.clusters_dataframe(df_extract_rawtrace_detrended, clusters_rois,
                                                                           0,
                                                                           1)  # no detrending because it comes from detrended traces
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.6
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                  distance_neurons, np.array([1, 2, 3]), th_cluster,
                                                                  colormap_cluster, plot_data, print_plots)
[clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                 idx_roi_cluster, centroid_ext)
[df_trace_clusters_ave, df_trace_clusters_std] = mscope.clusters_dataframe(df_extract_rawtrace_detrended, clusters_rois,
                                                                           0,
                                                                           1)  # no detrending because it comes from detrended traces
dFF_trial = df_trace_clusters_ave.loc[df_trace_clusters_ave['trial'] == 3]  # get dFF for the desired trial
dFF_trial_std = df_trace_clusters_std.loc[df_trace_clusters_std['trial'] == 3]  # get dFF for the desired trial
fig, ax = plt.subplots(figsize=(15, 10), tight_layout=True)
count_r = 0
for idx_r, r in enumerate(dFF_trial.columns[2:]):
 plt.plot(dFF_trial['time'], dFF_trial[r] + (count_r / 2), color=colors_cluster[idx_r])
 plt.fill_between(dFF_trial['time'], dFF_trial[r] + (count_r / 2) - dFF_trial_std[r], dFF_trial[r] + (count_r / 2) + dFF_trial_std[r], color=colors_cluster[idx_r], alpha=0.3)
 count_r += 1
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 2)
ax.set_ylabel('Calcium trace for trial 3', fontsize=mscope.fsize - 2)
ax.set_xlim([15, 45])
plt.xticks(fontsize=mscope.fsize - 2)
plt.yticks(fontsize=mscope.fsize - 2)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='y', which='y', length=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
plt.savefig('E:\\Miniscope figures\\cluster_average_example')
