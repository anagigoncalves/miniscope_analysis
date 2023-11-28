import os
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.stats import linregress
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
import sync_analysis_class
sync = sync_analysis_class.population_synchrony('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')

path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'

save_plot = False
plot_data = False
window = np.arange(-330, 330 + 1) # Samples
sync_class = ['async', 'low sync', 'mid sync', 'high sync']
sync_class_colors = ['gray', 'lightgreen', 'green', 'DarkGreen']
sync_behaviorChange = {sync_class[0]: [], sync_class[1]: [], sync_class[2]: [], sync_class[3]: []}

# Loop thorugh animals
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = ses_info[0]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [_, _, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, _, _, _, trials,
     clusters_rois, colors_cluster, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
    [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, _, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)
    
    # Sort ROIs in the dataframe of neural activity by cluster
    df_spikes, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois)
    if len(cluster_transition_idx) > 1:
        cluster_transition_idx = np.concatenate(([0], cluster_transition_idx, [df_spikes.shape[1]]))
    else:
        cluster_transition_idx = np.concatenate((cluster_transition_idx, [df_spikes.shape[1]]))
    # flatten the list of ROIs 'clusters_rois'
    rois_sorted = []
    for i in range(len(clusters_rois)):
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
        
    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, _, _, _, _, _] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        final_tracks_trials.append(final_tracks)

    # Get body kinematic variables (body position, speed, acceleration) on x axis 
    bodycenter, bodyspeed, bodyacc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 91, polyorder = 3)
    variable = bodyacc
    var_name = 'body acceleration'
    params = {'colors trials': colors_session, 'y tick labels': [block[1] for block in trials_ses], 'y ticks': [block[1] - 1 for block in trials_ses], 'variable': var_name, 'mouse ID': animal, 'session': session_id, 'save': save_plot}
    
    # Loop through clusters
    for c in range(1, len(cluster_transition_idx)):
        if cluster_transition_idx[c] - cluster_transition_idx[c-1] > 3:
            # Compute population synchrony
            df_spikes_clust = df_spikes.iloc[:, cluster_transition_idx[c-1]:cluster_transition_idx[c]]
            if c > 1:
                df_spikes_clust.insert(0, 'time', df_spikes['time'])
                df_spikes_clust.insert(1, 'trial', df_spikes['trial'])
            coactive_rois = sync.get_coactive_rois(df_spikes_clust)
            async_value = 1/df_spikes_clust.iloc[:,2:].shape[1]
            sync_distr = coactive_rois[coactive_rois['coactive_rois'] > async_value].iloc[:, 2].values
            quartiles = np.percentile(sync_distr, [33, 66])

            # Loop through synchrony classes
            for idx, sync_threshold in enumerate([[0, async_value], [async_value, quartiles[0]], [quartiles[0], quartiles[1]], [quartiles[1], 1]]):
                # Find high synchrony bouts
                sync_spikes_popul = coactive_rois[(coactive_rois['coactive_rois'] > sync_threshold[0]) & (coactive_rois['coactive_rois'] <= sync_threshold[1])]

                if len(sync_spikes_popul) > 3:
                    # Map population high synchrony event timestamps into behavioral timestamps
                    sync_spikes_idx_behav_popul = []
                    for tr_idx, tr in enumerate(trials):
                        sync_spikes_idx_behav_popul.append([np.abs(bcam_time[tr_idx] - ts).argmin() for ts in sync_spikes_popul[sync_spikes_popul['trial'] == tr]['time']])
                
                    # Compute syncSTA population
                    sync_sta_popul, peri_sync_event_traces, excluded_data = nxb.compute_sta(sync_spikes_idx_behav_popul, variable, window, zscore = True)
                
                    # Plot syncSTA population
                    if plot_data:
                        nxb.plot_sta_rois(sync_sta_popul, window, f'cluster {c+1} {sync_class[idx]}', params)
                    
                    # Compute absolute value of the peaks of each behavioral trace in a window of 250 ms before the high synchrony bouts
                    # peaks = np.max(np.abs(np.diff(np.concatenate(peri_sync_event_traces)[:, 250:330], axis=1)), axis=1)
                    peaks = np.max(np.abs(np.concatenate(peri_sync_event_traces)[:, 250:330]), axis=1)
                    sync_spikes_popul_aligned = []
                    for tr_idx, tr in enumerate(trials):   
                        sync_spikes_popul_tr = sync_spikes_popul[sync_spikes_popul['trial'] == tr].reset_index(drop = True)
                        excluded_data_tr = excluded_data[excluded_data[:, 0] == tr_idx]
                        remove_start = np.sum(excluded_data_tr[:,1] < abs(window[0]))
                        remove_end = len(excluded_data_tr) - remove_start
                        if remove_start != 0:
                            sync_spikes_popul_tr = sync_spikes_popul_tr.drop(np.arange(0, remove_start))
                        if remove_end != 0:
                            sync_spikes_popul_tr = sync_spikes_popul_tr.drop(np.arange(sync_spikes_popul_tr.shape[0] - remove_end, sync_spikes_popul_tr.shape[0]))                  
                        sync_spikes_popul_aligned.append(sync_spikes_popul_tr.iloc[:, 2].values)
                    sync_behaviorChange[sync_class[idx]].append(np.vstack((np.concatenate(sync_spikes_popul_aligned), peaks)))

# behav_label ='max |jerk| ($m/s^3$)'
# behav_label ='|acceleration peaks| ($m/s^2$)'
behav_label ='|z-scored acceleration peaks|'

mean_behav_class = []
std_behav_class = []
sync = []
behav = []
for syn_c, clusters in sync_behaviorChange.items():
    mean_clust = [np.nanmean(cluster[1]) for cluster in clusters]
    std_clust = [np.nanstd(cluster[1]) for cluster in clusters] 
    behav_clust_class = [cluster[1] for cluster in clusters]
    sync_clust_class = [cluster[0] for cluster in clusters]
    mean_behav_class.append(np.array(mean_clust))
    std_behav_class.append(np.array(std_clust))
    sync.append(np.array(np.concatenate(sync_clust_class)))
    behav.append(np.array(np.concatenate(behav_clust_class)))
mean_behav_class = np.array(mean_behav_class)
std_behav_class = np.array(std_behav_class)
sync = np.concatenate(sync)
behav = np.concatenate(behav)

# 
plt.figure()
plt.scatter(sync, behav, c = 'gray', s = 1)
slope, intercept, r_value, p_value, std_err = linregress(sync, behav)
plt.plot(sync, slope * sync + intercept, color='crimson')
plt.xlabel('Fraction of co-active ROIs', fontsize = 15)
plt.ylabel(behav_label, fontsize = 15)
   
# Clusters' mean x sync                                   
fig, ax = plt.subplots()
boxplot = ax.boxplot(mean_behav_class.T, patch_artist=True)
ax.set_xticklabels(sync_class, fontsize = 15)
ax.set_ylabel('mean ' + behav_label, fontsize = 15)
for box, color in zip(boxplot['boxes'], sync_class_colors):
    box.set_facecolor(color)
 
# Clusters' std x sync
fig, ax = plt.subplots()
boxplot = ax.boxplot(std_behav_class.T, patch_artist=True)
ax.set_xticklabels(sync_class, fontsize = 15)
ax.set_ylabel('std ' + behav_label, fontsize = 15)
for box, color in zip(boxplot['boxes'], sync_class_colors):
    box.set_facecolor(color)
 
# Number of events in each synchrony class
n_events_sync_classes = []
for syn_c, clusters in sync_behaviorChange.items():
    n_events_sync_classes.append(sum([cluster.shape[1] for cluster in clusters]))
n_events_sync_classes = np.array(n_events_sync_classes)
plt.figure()
for i, c in enumerate(sync_class_colors):
    plt.bar(i, n_events_sync_classes[i], color = c)
plt.ylabel('Count of events', fontsize = 15)