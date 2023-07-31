# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:59:55 2023

@author: User
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')

path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_tied.xlsx')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'

ind_vars = ['FR-HL x difference', 'FR-HR x difference', 'FR-FL x difference']
# ind_vars = ['FR-FL x difference']
sta = [[] for _ in range(len(ind_vars))]
sta_shuffled = [[] for _ in range(len(ind_vars))]
sta_zs = [[] for _ in range(len(ind_vars))]

for s in range(len(session_data)):
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

    # Upload STA data
    for v, var_name in enumerate(ind_vars):
        # Upload STA
        file_path = os.path.join(path_session_data, 'STA', f'{animal}_{session_id}_{var_name}_STA.npy')
        sta[v].append(np.load(file_path))
        # Upload STA shuffled
        file_path = os.path.join(path_session_data, 'STA', f'{animal}_{session_id}_{var_name}_STAchance.npy')
        sta_shuffled[v].append(np.load(file_path))
        # Upload STA z-scored
        file_path = os.path.join(path_session_data, 'STA', f'{animal}_{session_id}_{var_name}_STAzs.npy')
        sta_zs[v].append(np.load(file_path))




window = np.arange(-330, 330 + 1)  # Samples
interval = [-80, 0]
sta_blocks_all = [[] for _ in range(len(ind_vars))]
sta_clust_all = [[] for _ in range(len(ind_vars))]
peaks_pos_all = [[] for _ in range(len(ind_vars))]
latency_pos_all = [[] for _ in range(len(ind_vars))] 
peaks_neg_all = [[] for _ in range(len(ind_vars))]
latency_neg_all = [[] for _ in range(len(ind_vars))]
ratio_all = [[] for _ in range(len(ind_vars))]
save_plot = True
for v in range(len(ind_vars)):
    for s in range(len(session_data)):
        ses_info = session_data.iloc[s, :]
        # print(ses_info)
        date = ses_info[3]
        path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
        path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
        mscope = miniscope_session_class.miniscope_session(path)
        loco = locomotion_class.loco_class(path_loco)
        session_type = path.split('\\')[-4].split(' ')[0]
        session_id = session_type + '_' + ses_info[2]
        animal = mscope.get_animal_id()
        session = loco.get_session_id()
        sr_cam = 330
        [_, _, _, _, _, _, _, _, trials, clusters_rois, colors_cluster, colors_session, _, _, _] = mscope.load_processed_files()
        [trials_ses, _, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)
        
        if len(clusters_rois) > 1:
            cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))]) - 1
            cluster_transition_idx = np.concatenate(([0], cluster_transition_idx, [sta_zs[v][s].shape[0]]))
        else:
            cluster_transition_idx = np.array([0]) 
            
        if session_type == 'split':
            split_blocks = nxb.split_expblocks(trials_ses)
        else:
            split_blocks = trials_ses

        sta_blocks = nxb.sta_expblocks(sta_zs[v][s], trials, split_blocks)
        sta_blocks_all[v].append(sta_blocks)

        sta = np.swapaxes(sta_zs[v][s], 0, 1)
        sta_clust = nxb.sta_clusters(sta, cluster_transition_idx, window) #sta_block
        sta_clust_all[v].append(sta_clust)
        
        nxb.plot_zoom_sta(sta_clust, window, [-165, 0], colors_session, colors_cluster, ind_vars[v], session_id, animal, save_plot)

        peaks_pos, latency_pos, peaks_neg, latency_neg, ratio = nxb.peaks_latency(sta_clust, interval, sr_cam) # sta_clust or sta
        # peaks_pos, latency_pos, peaks_neg, latency_neg, ratio = nxb.peaks_latency_doublepeak(sta_clust, interval, sr_cam)
        peaks_pos_all[v].append(peaks_pos)
        latency_pos_all[v].append(latency_pos)
        peaks_neg_all[v].append(peaks_neg)
        latency_neg_all[v].append(latency_neg)
        ratio_all[v].append(ratio)

        # Study learning for each animal/cluster
        nxb.tuning_change(latency_pos, latency_neg, peaks_pos, peaks_neg, ind_vars[v], session_id, animal, trials, interval, colors_cluster, save_plot)
                
        
    # Pie chart of positive and negative peaks for all the clusters
    data_sum = np.sum(np.vstack(ratio_all[v]), axis = 0)
    nxb.modulation_ratio(data_sum, ind_vars[v], session_id, save_plot)

    # Distribution of latency for clusters positive and negative
    nxb.lat_peaks_distr(latency_pos_all[v], latency_neg_all[v], interval, session_id, ind_vars[v], save_plot) 

    # Heatmap all animals
    nxb.plot_sta_all(sta_blocks_all[v], window, ind_vars[v], session_id, save_plot, condition = 'blocks') #sta_blocks_all[v] #sta_zs[v] #sta_clust_all[v]
        
        