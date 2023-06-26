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
import matplotlib.pyplot as plt


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
strobe_nr_txt = loco.bcam_strobe_number() 
trial_start_blip_nr = loco.trial_start_blips()
session_type = path.split(mscope.delim)[-4].split(' ')[0]  # tied or split
if session_type == 'split':
    colors_phases = ['black', 'crimson', 'teal']
if session_type == 'tied':
    colors_phases = ['black', 'orange', 'purple']
traces_type = 'raw'


# Load miniscope data
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
[df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()
time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
centroid_ext = mscope.get_roi_centroids(coord_ext)
distance_neurons = mscope.distance_neurons(centroid_ext, 0)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal)
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)


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
roi_plot = 3
mscope.plot_stacked_traces_singleROI(df_extract_rawtrace_detrended, 'raw' , roi_plot, trials, colors_session, 0.5, plot_data = True, print_plots = False)


# Event count single ROIs for trial
roi_plot = 3
mscope.get_event_count_wholetrial(df_events_extract_rawtrace, 'raw' , colors_session, trials, roi_plot, plot_data = True, print_plots = False)
           

# Isi, cv and cv2
isi_df = mscope.compute_isi(df_events_extract_rawtrace, 'raw', 'MC8855_isi')
mean_isi = isi_df.iloc[:, 0].mean()
std_isi = isi_df.iloc[:, 0].std()
min_isi = isi_df.iloc[:, 0].min()
max_isi = isi_df.iloc[:, 0].max()
bin_size = 0.05
plt.hist(isi_df.iloc[:, 0], bins=int((max_isi-min_isi)/bin_size), color = 'gray')
plt.xlabel('ISI (s)', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xlim(0, 2)
plt.show()


# Population cross-correlation
mscope.compute_clustered_traces_events_correlations(df_events_extract_rawtrace, df_extract_rawtrace_detrended, clusters_rois, colors_cluster, trials, plot_data = True, print_plots = False)


##############################################################################


# Align dF/F and behavior (body position, speed, acceleration) for each trial and desired epoch
# df = df_events_extract_rawtrace # Spike events for 'popul_raster'
# df = df_trace_clusters_ave # Cluster traces for 'cluster_traces'
df = mscope.norm_traces(df_extract_rawtrace_detrended, norm_name = 'zscore', axis = 'session') # Normalized dF/F traces for 'popul_heatmap'
plot_type = 'popul_heatmap'  # 'popul_heatmap', 'cluster_traces' or 'popul_raster'
window = [0, 60] # Desired time window in seconds
nxb.df_behav_align(df, clusters_rois, frame_time, final_tracks_trials, sl_time_all_array, sl_sym_all_array, trials, plot_type, window, save_plot = False)


# Get kinematic variables (body position, speed, acceleration)
win_len = 81 # In samples
polyorder = 3
bodycenter, bodyspeed, bodyacc = nxb.kinematic(final_tracks_trials, trials, win_len, polyorder)
# Find timestamps of behavioral recording matching the ones of neural activity and compute kinematic variables downsampled and aligned to df/f
behav_ts_idx = nxb.find_behav_ts(df_extract_rawtrace_detrended, bcam_time)
bodycenter_aligned, bodyspeed_aligned, bodyacc_aligned = nxb.kinematic_aligned(final_tracks_trials, trials, behav_ts_idx, win_len, polyorder) 


# Compute spike-triggered average (STA) of kinematic variables
window = np.arange(-330, 330 + 1) # In samples
variable = bodyspeed
df_events, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois) # Sort ROIs by cluster
sta_allrois, signal_chunks_allrois = nxb.sta(df_events, variable, bcam_time, window, trials)
# Plot STA
save_plot = False
plot_data = True
var_name = 'Speed'
rois_sorted=[]
for i in range(len(clusters_rois)): # flatten 'clusters_rois'
    rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
nxb.plot_sta(sta_allrois,signal_chunks_allrois, window, trials, blocks, block_colors, split_blocks, rois_sorted, var_name, save_plot)


# Shuffle CS timestamps
iter_n = 10
shuffled_spikes_ts = nxb.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)

# Compute STA for each iteration of CSs timestamps shuffling
sta_shuffled_ts_alliter = []
for i in range(iter_n):
    sta_shuffled_ts = nxb.sta_shuffled(shuffled_spikes_ts[i], variable, bcam_time, window, trials)
    sta_shuffled_ts_alliter.append(sta_shuffled_ts)   
    
# Average the STA traces obtained from the different iterations of random shuffling of CS timestamps
sta_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
stsd_chance = np.zeros((df_events.iloc[:, 2:].shape[1], trials[-1], len(window)))
for n in range(df_events.iloc[:, 2:].shape[1]):
    roi_sta_tr = [sublist[n] for sublist in sta_shuffled_ts_alliter]
    for tr in range(trials[-1]):
        sta_chance[n, tr] = np.mean([array[tr] for array in roi_sta_tr], axis=0)
        stsd_chance[n, tr] = np.std([array[tr] for array in roi_sta_tr], axis=0)
        
# Standardize observed STA on STA computed with shuffled data
sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
for n in range(len(sta_allrois)):
    sta_zs[n] = (sta_allrois[n] - sta_chance[n]) / stsd_chance[n]
    
# Plot observed STA, STA you would expect by chance and standardized STA
nxb.plot_sta_shuffled(sta_zs, sta_allrois, sta_chance, window, var_name, trials_ses, rois_sorted, animal, save_plot)


###### TRANSFORM THIS INTO A METHOD ######
# Detect min and max z-score for every peak
sr_cam = 330
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
signif_thresh = 2
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
latency_negzs = latency[np.where(max_abs_zs < -2)]
latency_poszs = latency[np.where(max_abs_zs > 2)]
fig, ax = plt.subplots()
boxplot = ax.boxplot([latency_negzs, latency_poszs], showfliers=False, patch_artist=True)
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


# Detect drops in body position
tr_trans_idx = nxb.trial_transition_idx(df_events) # Find indexes of trial transitions
bodycenter_aligned_tr = []
for tr in range(1, len(tr_trans_idx)): # Re-organize bodycenter array by trial
    bodycenter_aligned_tr.append(bodycenter_aligned[tr_trans_idx[tr-1]:tr_trans_idx[tr]])
TimePntThres = 100
ampl=20
nxb.peak_detection(bodycenter_aligned_tr, ampl, TimePntThres, trials)

# Compute heatmap 



##############################################################################

    
# Phase maps
p1 = 0 # Reference paw (FR = 0)
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

# Plot
fig, ax = plt.subplots()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(PC[:, 0], PC[:, 1], PC[:,2], marker='.', c='navy')
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