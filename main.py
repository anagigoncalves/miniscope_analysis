# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:33:42 2023

@author: User
"""
###################
#    Load data    #
###################


# Import modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
from scipy.stats import zscore
from scipy.signal import correlate
from scipy.signal import savgol_filter
warnings.filterwarnings('ignore')


# Path inputs
path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
path_loco = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\TM TRACKING FILES\\split ipsi fast S1 050421\\'
path_analysis = 'C:\\Users\\User\\Desktop\\Climbing fibers and instructive error signals\\Code'
session_type = path.split('\\')[-4].split(' ')[0]
version_mscope = 'v4'
plot_data = 0
load_data = 0
print_plots = 0
save_data = 0
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24


# Import classes
os.chdir('C:\\Users\\User\\Desktop\\Guilherme\\Mice (openlab)\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
import df_behav_class
nxb = df_behav_class.df_behav_analysis(path_analysis)


# Create plots folders
path_plots = os.path.join(path, 'Plots')
if not os.path.exists(path_plots):
    os.mkdir(path_plots)
    

# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()
trials = mscope.get_trial_id()
strobe_nr_txt = loco.bcam_strobe_number() 
trial_start_blip_nr = loco.trial_start_blips()
ops_s2p = mscope.get_s2p_parameters()
# print(ops_s2p)
session_type = path.split(mscope.delim)[-4].split(' ')[0]  # tied or split
colors_session = mscope.colors_session(session_type, trials, 1)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
if session_type == 'split':
    colors_phases = ['black', 'crimson', 'teal']
if session_type == 'tied':
    colors_phases = ['black', 'orange', 'purple']
traces_type = 'raw'


# Load miniscope data
frames_dFF = np.load(os.path.join(path, 'processed files', 'black_frames.npy'))
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
[df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()
time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
th_cluster = 0.65
colormap_cluster = 'hsv'
[colors_cluster, idx_roi_cluster] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                      distance_neurons, trials_baseline, th_cluster,
                                                                      colormap_cluster, plot_data, print_plots)
[clusters_rois, idx_roi_cluster_ordered] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                     idx_roi_cluster, centroid_ext)
frame_time = mscope.get_miniscope_frame_time(trials, frames_dFF, version_mscope)  # get frame time for each trial


# Load behavioral data
filelist = loco.get_track_files(animal, session)
param_name = 'step_length'
p = 'FR'
p2 = 'FL'
st_strides_trials = []
sw_strides_trials = []
final_tracks_trials = []
param_trials = []
param_trials_fr_mean = np.zeros(len(trials))
stride_duration_trials = []
final_tracks_forwadloco_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = loco.get_paws_rel(final_tracks, 'X')
    final_tracks_forwadloco = loco.final_tracks_forwardlocomotion(final_tracks, st_strides_mat)
    final_tracks_forwadloco_trials.append(final_tracks_forwadloco)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
    param_trials_fr_mean[count_trial] = np.nanmean(param_trials[-1][0])-np.nanmean(param_trials[-1][2])
final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-st')
[sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, p, p2, sym = 1, remove_nan = 1)  # SL symmetry for each stride


##################
#    Analysis    #
##################


# Clusters
mscope.plot_roi_clustering_spatial(ref_image, colors_cluster, idx_roi_cluster_ordered, coord_ext, plot_data = True, print_plots = False)


# SL sym curve across trials
mscope.plot_sl_sym_session(param_trials_fr_mean, trials_ses, trials, session_type, colors_session, plot_data, print_plots)


# df/f traces single ROI
traces_type = 'raw'
roi_plot = 3
line_ratio = 0.5
mscope.plot_stacked_traces_singleROI(df_extract_rawtrace_detrended, traces_type, roi_plot, trials, colors_session, line_ratio, plot_data = True, print_plots = False)


# Event count single ROIs for trial
roi_plot = 3
traces_type = 'raw'   
mscope.get_event_count_wholetrial(df_events_extract_rawtrace, traces_type, colors_session, trials, roi_plot, plot_data = True, print_plots = False)
           

# Firing rate distribution for trial and mean firing rate
colors_clusters = ['red', 'yellow', 'green', 'purple', 'blue']
nxb.fr_distr_trial(df_events_extract_rawtrace, trials, clusters_rois, colors_clusters, colors_session, save_plot = False)


# Isi, cv and cv2
traces_type = 'raw'
csv_name = 'MC8855_isi'
isi_df = mscope.compute_isi(df_events_extract_rawtrace, traces_type, csv_name)
# [isi_cv_df, isi_cv2_df] = mscope.compute_isi_cv(isi_df, trials)
mean_isi = isi_df.iloc[:, 0].mean()
print(mean_isi)
median_isi = isi_df.iloc[:, 0].median()
print(median_isi)
std_isi = isi_df.iloc[:, 0].std()
print(std_isi)
p15 = np.quantile(isi_df.iloc[:, 0].dropna(), 0.15)
print(p15)
min_isi = isi_df.iloc[:, 0].min()
print(min_isi)
max_isi = isi_df.iloc[:, 0].max()
print(max_isi)
bin_size = 0.05
fig, ax = plt.subplots()
ax.hist(isi_df.iloc[:, 0], bins=int((max_isi-min_isi)/bin_size), color = 'gray')
plt.axvline(median_isi, color='red', linestyle = '--')
ax.set_xlabel('ISI (s)', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_xlim(0, 2)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()


# Population cross-correlation
mscope.compute_clustered_traces_events_correlations(df_events_extract_rawtrace, df_extract_rawtrace_detrended, clusters_rois, colors_cluster, trials, plot_data = True, print_plots = False)


##############################################################################

# Align dF/F and behavior (body position, speed, acceleration, SL symmetry) for each trial and desired epoch
# df = df_events_extract_rawtrace
# df = df_trace_clusters_ave
df = mscope.norm_traces(df_extract_rawtrace_detrended, norm_name = 'zscore', axis = 'session') # Normalize dF/F traces
plot_type = 'popul_heatmap'  # 'popul_heatmap', 'cluster_traces' or 'popul_raster'
window = [0, 60]
nxb.df_behav_align(df, clusters_rois, frame_time, final_tracks_trials, sl_time_all_array, sl_sym_all_array, trials, plot_type, window, save_plot = False)


# Find timestamps of behavioral recording matching neural activity timestamps
behav_ts_idx = nxb.find_behav_ts(df_extract_rawtrace_detrended, bcam_time)

# Get kinematic variables
win_len = 81
polyorder = 3
bodycenter_aligned, bodyspeed_aligned, bodyacc_aligned = nxb.kinematic(final_tracks_trials, trials, behav_ts_idx, win_len, polyorder)
bodycenter = []
bodyspeed = []
bodyacc = []
for tr in range(trials[-1]):
    bodycenter_trial = np.nanmean(final_tracks_trials[tr][0,:4, :],axis=0)*nxb.pixel_to_mm # Get bodycenter position (x-axis) for the desired trial and interval
    bodycenter.append(bodycenter_trial)        
    bodyspeed_trial = savgol_filter(nxb.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=1) # Get body speed
    bodyspeed.append(bodyspeed_trial)
    bodyacc_trial = savgol_filter(nxb.inpaint_nans(bodycenter_trial),win_len,polyorder,deriv=2) # Get body acceleration
    bodyacc.append(bodyacc_trial)
    

# Spike-triggered average
df_events, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois) # Sort ROIs by cluster
window = np.arange(-330, 330 + 1) # define the time window in samples
variable = bodyspeed
var_name = 'Speed (z-score)'
zs_signal = True
save_plot = True
plot_data = True
sta_tr_all, sta_roi, _ = nxb.sta(df_events, variable, var_name, bcam_time, window, zs_signal, plot_data, save_plot)

# Normalize STA on STA you would observe by chance (randomly shuffled CSs timestamps)
iter_n = 20
shuffled_spikes_ts = nxb.shuffle_spikes_ts(df_events_extract, iter_n)
# Compute STA for each iteration of CSs timestamps shuffling
sta_shuffled_ts_all = []
for i in range(iter_n):
    sta_shuffled_ts = nxb.sta_shuffled(shuffled_spikes_ts[i], variable, var_name, bcam_time, window)
    sta_shuffled_ts_all.append(sta_shuffled_ts)
    
# Average the STA traces obtained from the different iteration of random shuffling of CS timestamps
sta_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
stsd_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
for n in range(df_events.iloc[:, 2:].shape[1]):
    roi_sta_tr = [sublist[n] for sublist in sta_shuffled_ts_all]
    for tr in range(trials[-1]):
        sta_chance[n, tr] = np.mean([array[tr] for array in roi_sta_tr], axis=0)
        stsd_chance[n, tr] = np.std([array[tr] for array in roi_sta_tr], axis=0)

# Plot observed STA, STA that you would expect by chance and normalized STA
sta_zs = nxb.plot_sta_shuffled(sta_roi, sta_chance, stsd_chance, df_events, window, var_name)

# Detect min and max z-score for every peak
max_abs_zs = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1]))
max_abs_zs[:] = np.nan
latency = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1]))
latency[:] = np.nan
for n in range(df_events.iloc[:, 2:].shape[1]):
    for tr in range(trials[-1]):
        max_zs = max(sta_zs[n, tr, 248:413]) #-0.25 to +0.25
        min_zs = min(sta_zs[n, tr, 248:413])
        if max_zs > abs(min_zs): #and max_zs > 2
            max_abs_zs[n, tr] = max_zs
            latency[n, tr] = (np.argmax(sta_zs[n, tr, 248:413])/sr_cam)-0.25
        elif abs(min_zs) > max_zs: #and min_zs < -2
            max_abs_zs[n, tr] = min_zs
            latency[n, tr] = (np.argmin(sta_zs[n, tr, 248:413])/sr_cam)-0.25

# Plot distribution of max z-score for the whole session
median_max_zs = np.nanmedian(max_abs_zs, axis=1)
min_val = min(median_max_zs)
max_val = max(median_max_zs)
bin_size = 0.5
bin_edges = np.arange(int(min_val), int(max_val) + bin_size, bin_size)
hist, _ = np.histogram(median_max_zs, bins=bin_edges)
plt.bar(bin_edges[:-1], hist, bin_size, color='white')
for i in range(len(bin_edges)-1):
    if bin_edges[i+1] < -1: # -2
        plt.bar(bin_edges[i], hist[i], bin_size, color='blue', edgecolor='black', linewidth=1.2)
    elif bin_edges[i] > 1.5: # 2
        plt.bar(bin_edges[i], hist[i], bin_size, color='yellow', edgecolor='black', linewidth=1.2)
    else:
        plt.bar(bin_edges[i], hist[i], bin_size, color='black', edgecolor='black', linewidth=1.2)
plt.xlabel('Max z-score', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.title('Max z-score', fontsize = 15)

# Boxplot latency of max z-score for the whole session
idx_neg = np.where(max_abs_zs < -2)
idx_pos = np.where(max_abs_zs > 2)
latency_neg = latency[idx_neg]
latency_pos = latency[idx_pos]
fig, ax = plt.subplots()
boxplot = ax.boxplot([latency_neg, latency_pos], showfliers=False, patch_artist=True)
ax.set_xticklabels([]) 
ax.tick_params(bottom=False)
colors = ['blue', 'yellow']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(2)
for median in boxplot['medians']:
    median.set(color='black', linewidth=1.7)
for whisker in boxplot['whiskers']:
    whisker.set(color='black', linewidth=1.7)
for cap in boxplot['caps']:
    cap.set(color='black', linewidth=1.7)
ax.set_ylabel('Latency (s)', fontsize = 15)
plt.show()
plt.title('Latency of max z-score')


##############################################################################

# Phase maps
p1 = 0
colors_clusters = ['red', 'yellow', 'green', 'purple', 'blue']
nxb.phasemap(final_tracks_trials_phase, df_events_trace_clusters, bcam_time, p1, colors_clusters, st_align = False, save_plot = False)


##############################################################################

# PCA on df/f
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = df_trace_clusters_ave.iloc[:, 2:]
# df = df_extract_rawtrace_detrended.iloc[:, 2:]


# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df) 

# Center the data
df_centered = df_scaled - df_scaled.mean(axis=0)

# Perform PCA 3D
pca = PCA(n_components=3)
PC = pca.fit_transform(df_centered)

# Colormap
# cmap = plt.cm.get_cmap('Reds')
# colors = cmap(values)

# Plot
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PC[:, 0], PC[:, 1], PC[:,2], marker='.', c='navy')
# for i in range(len(df)):
#     ax.scatter(PC[i, 0], PC[i, 1], PC[i,2], marker='.', c=colors_session[trial_idx.iloc[i]]) # To plot PCs colorcoded by trial
ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
ax.set_zlabel('PC3', fontsize=20)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Plot the explained variance
plt.figure()
plt.bar(range(1, pca.n_components_+1), pca.explained_variance_ratio_ * 100)
plt.xticks(range(1, pca.n_components_+1), ['PC{}'.format(i) for i in range(1, pca.n_components_+1)], fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Explained Variance (%)', fontsize=30)
plt.ylim([0, np.max(pca.explained_variance_ratio_*100)+10])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Plot the loadings
plt.figure()
plt.imshow(pca.components_, cmap='viridis', aspect='auto')
colorbar = plt.colorbar()
plt.xticks(range(len(df.columns)), df.columns, rotation=30, fontsize=25)
colorbar.ax.tick_params(labelsize=25)  # Set the font size of the tick labels
# plt.gca().axes.get_yaxis().set_visible(False)
plt.yticks(range(0, pca.n_components_), ['PC{}'.format(i) for i in range(1, pca.n_components_+1)], fontsize=25)

# # Plot on PC in time
# fig, ax = plt.subplots()
# ax.plot(np.linspace(0, 23, len(PC)), PC[:,0])

# # Plot one PC in time per trial
# for i in range(0, 21):
#     plt.figure() 
#     plt.plot(np.linspace(0, 60, len(PC[trial_changes[i]:trial_changes[i+1]])), PC[trial_changes[i]:trial_changes[i+1],0])
#     plt.xlabel('PC1') 
#     plt.ylabel('PC2')
#     plt.title('Figure {}'.format(i+2))
# plt.show()

# # Plot neural trajectories 
# plt.plot(PC[trial_changes[i]:trial_changes[i+1],0], PC[trial_changes[i]:trial_changes[i+1],1])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()


# # t-SNE on df/f
# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=3, random_state=42)
# tsne_results = tsne.fit_transform(df_scaled)

# fig, ax = plt.subplots()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], marker='.', c = 'navy')
# ax.set_xlabel('t-SNE 1')
# ax.set_ylabel('t-SNE 2')
# ax.set_zlabel('t-SNE 3')


# # Isomap on df/f
# from sklearn.manifold import Isomap

# isomap = Isomap(n_components=3, n_neighbors=30)
# isomap_results = isomap.fit_transform(df_scaled)

# fig, ax = plt.subplots()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(isomap_results[:, 0], isomap_results[:, 1], isomap_results[:, 2], marker='.')
# ax.set_xlabel('Isomap 1')
# ax.set_ylabel('Isomap 2')
# ax.set_zlabel('Isomap 3')


##############################################################################

# TCA on df/f
import tensortools as tt

df = df_trace_clusters_ave
# Create an empty tensor with dimensions frames, neurons, and trials
num_frames = 1772
num_neurons = df.shape[1] - 2  # Subtract 2 for the time and trial columns
num_trials = len(trials)
tensor = np.zeros((num_frames, num_neurons, num_trials))

# Iterate over the trials and fill in the tensor with the corresponding frames
for i, trial_id in enumerate(trials):
    trial_frames = df[df['trial'] == trial_id].iloc[:, 2:].values # Extract activity of all the neurons for one trial 
    if len(trial_frames) > num_frames:
        trial_frames = trial_frames[:num_frames, :] # Resize all the trials to be the same length of the shortest trial
    tensor[:, :, i] = trial_frames

# Fit an ensemble of models, 4 random replicates / optimization runs per model rank
ensemble = tt.Ensemble(fit_method="ncp_hals") # nonnegative TCA
ensemble.fit(tensor, ranks=range(1, 9), replicates=6)

fig, axes = plt.subplots(1, 2)
tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
fig.tight_layout()

# Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.
num_components = 2
replicate = 0
tt.plot_factors(ensemble.factors(num_components)[replicate])  # plot the low-d factors

plt.show()