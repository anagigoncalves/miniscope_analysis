# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mp
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

path_session_data = 'C:\\Users\\Ana\\Desktop\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\Ana\\Desktop\\Miniscope processed files\\session_data_MC13420.xlsx')
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
    colors_session = mscope.colors_session(session_type, trials, 1)

    # For sessions where trial structure is different
    if ses_info['protocol'] == 'tied baseline':
        greys = mp.cm.get_cmap('Greys', 14)
        oranges = mp.cm.get_cmap('Oranges', 23)
        purples = mp.cm.get_cmap('Purples', 23)
        colors_switched = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: purples(23),
                        8: purples(19), 9: purples(16), 10: purples(13), 11: purples(10), 12: purples(6), 13: oranges(23),
                        14: oranges(19), 15: oranges(16), 16: oranges(13), 17: oranges(10), 18: oranges(6)}
        colors_slow_first = {1: purples(23), 2: purples(19), 3: purples(16), 4: purples(13), 5: purples(10), 6: purples(6),
                        7: greys(14), 8: greys(12), 9: greys(10), 10: greys(8), 11: greys(6), 12: greys(4),
                        13: oranges(23), 14: oranges(19), 15: oranges(16), 16: oranges(13), 17: oranges(10), 18: oranges(6)}
        if ses_info['animal'] == 'MC9226' and ses_info['session'] == 'S2':
            colors_session = {1: greys(4), 2: greys(4), 3: greys(4), 4: greys(4), 5: greys(4), 6: greys(4),
                              7: greys(14), 8: greys(12), 9: greys(10), 10: greys(8), 11: greys(6), 12: greys(4),
                              13: oranges(21), 14: oranges(19), 15: oranges(17), 16: oranges(15), 17: oranges(13), 18: oranges(11),
                              19: oranges(9), 20: oranges(7), 21: purples(19), 22: purples(17), 23: purples(15),
                              24: purples(13), 25: purples(11), 26: purples(9)}
        if ses_info['animal'] == 'MC9226' and ses_info['session'] == 'S3':
            colors_session = colors_slow_first
        if ses_info['animal'] == 'MC10221' and ses_info['session'] == 'S1':
            colors_session = colors_slow_first
        if ses_info['animal'] == 'MC10221' and ses_info['session'] == 'S2':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9513' and ses_info['session'] == 'S1':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9513' and ses_info['session'] == 'S2':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9194' and ses_info['session'] == 'S1':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9194' and ses_info['session'] == 'S2':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9308' and ses_info['session'] == 'S1':
            colors_session = colors_switched
        if ses_info['animal'] == 'MC9308' and ses_info['session'] == 'S2':
            colors_session = colors_switched
    np.save(os.path.join(mscope.path, 'processed files', 'colors_session.npy'), colors_session)

    # FOV coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
    if session_type == 'split':
        colors_phases = ['black', 'crimson', 'teal']
    if session_type == 'tied':
        colors_phases = ['black', 'orange', 'purple']
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

    plot_ratio = 4
    for cluster_plot in np.arange(1, len(clusters_rois) + 1):
        mscope.plot_stacked_traces_singleROI(df_trace_clusters_ave, traces_type, cluster_plot, trials, colors_session,
                                             plot_ratio, plot_data, print_plots)
    plt.close('all')