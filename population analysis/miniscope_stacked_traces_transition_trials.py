# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files\\'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data_all.xlsx')
for s in range(len(session_data)):
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
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    centroids_mediolateral = []
    for c in range(len(centroid_ext)):
        centroids_mediolateral.append(centroid_ext[c][0])
    distance_neurons_ordered = np.argsort(centroids_mediolateral)
    rois_ordered_distance_str = []
    for i in distance_neurons_ordered:
        rois_ordered_distance_str.append(roi_list[i])

    columns_ordered_ratio = ['time', 'trial'] + rois_ordered_distance_str[::3]
    df_extract_rawtrace_detrended_ordered = df_extract_rawtrace_detrended[columns_ordered_ratio]

    if session_type == 'split':
        trials_plot = np.array([trials_ses[0, 1], trials_ses[1, 0], trials_ses[1, 1], trials_ses[2, 0]])
    if session_type == 'tied' and animal != 'MC8855':
        trials_plot = np.array([trials_ses[0, 1], trials_ses[1, 1], trials_ses[2, 1]])
    if session_type == 'tied' and animal == 'MC8855':
        trials_plot = np.array([trials_ses[0, 1], trials_ses[1, 1]])
    mscope.plot_stacked_traces(df_extract_rawtrace_detrended_ordered, traces_type, trials, trials_plot, plot_data, 1)
    plt.close('all')




