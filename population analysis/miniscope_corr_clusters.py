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

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
corr_data_all = []
corr_data_all_bs = []
animal_in = []
trials_in = []
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
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    if session_type == 'split':
        colors_phases = ['black', 'crimson', 'teal']
    if session_type == 'tied':
        colors_phases = ['black', 'orange', 'purple']
    time_cumulative = mscope.cumulative_time(df_extract_rawtrace_detrended, trials)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)
    [df_trace_clusters_ave, df_trace_clusters_std, df_events_trace_clusters] = mscope.load_processed_files_clusters()

    trials_baseline_idx = []
    for t in trials_baseline:
        idx_trial = np.where(trials_baseline==t)[0][0]
        trials_baseline_idx.append(idx_trial)

    corr_data_trials = np.zeros((len(clusters_rois), len(trials)))
    for c in range(len(clusters_rois)):
        for count_t, t in enumerate(trials):
            corr_data_trials[c, count_t] = np.nanmean(np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == t, clusters_rois[c]].corr()).flatten())
    corr_data_trials_bs = corr_data_trials-np.nanmean(corr_data_trials[:, trials_baseline-1])
    corr_data_all.append(corr_data_trials)
    corr_data_all_bs.append(corr_data_trials_bs)
    animal_in.append(animal)
    trials_in.append(trials)

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
def get_colors_plot(animal_name, color_animals):
    if animal_name=='MC8855':
        color_plot = color_animals[0]
    if animal_name=='MC9194':
        color_plot = color_animals[1]
    if animal_name=='MC10221':
        color_plot = color_animals[2]
    if animal_name=='MC9513':
        color_plot = color_animals[3]
    if animal_name=='MC9226':
        color_plot = color_animals[4]
    return color_plot

fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
rectangle = plt.Rectangle((6.5, 0.5), 10, 1.6, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for a in range(len(corr_data_all)):
    if a == 3:
        ax.plot(np.arange(4, 27), np.nanmean(corr_data_all[a], axis=0)+(a/3), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
        ax.fill_between(np.arange(4, 27), np.nanmean(corr_data_all[a], axis=0)+(a/3) - np.nanstd(corr_data_all[a], axis=0),
                        np.nanmean(corr_data_all[a], axis=0) + np.nanstd(corr_data_all[a], axis=0)+(a/3),
                        color=get_colors_plot(animal_in[a], color_animals), alpha=0.3)
    else:
        ax.plot(trials_in[a], np.nanmean(corr_data_all[a], axis=0)+(a/3), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
        ax.fill_between(trials_in[a], np.nanmean(corr_data_all[a], axis=0)+(a/3) - np.nanstd(corr_data_all[a], axis=0),
                        np.nanmean(corr_data_all[a], axis=0) + np.nanstd(corr_data_all[a], axis=0)+(a/3),
                        color=get_colors_plot(animal_in[a], color_animals), alpha=0.3)
ax.set_title('Mean cluster correlation', fontsize=mscope.fsize - 2)
ax.legend(animal_in, frameon=False, fontsize=mscope.fsize-4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Miniscope processed files\\corr_summary_cluster_mean_raw', dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
rectangle = plt.Rectangle((6.5, -0.1), 10, 1, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for a in range(len(corr_data_all)):
    if a == 3:
        ax.plot(np.arange(4, 27), np.nanmean(corr_data_all_bs[a], axis=0)+(a/5), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
        ax.fill_between(np.arange(4, 27), np.nanmean(corr_data_all_bs[a], axis=0)+(a/5) - np.nanstd(corr_data_all_bs[a], axis=0),
                        np.nanmean(corr_data_all_bs[a], axis=0) + np.nanstd(corr_data_all_bs[a], axis=0)+(a/5),
                        color=get_colors_plot(animal_in[a], color_animals), alpha=0.3)
    else:
        ax.plot(trials_in[a], np.nanmean(corr_data_all_bs[a], axis=0)+(a/5), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
        ax.fill_between(trials_in[a], np.nanmean(corr_data_all_bs[a], axis=0)+(a/5) - np.nanstd(corr_data_all_bs[a], axis=0),
                        np.nanmean(corr_data_all_bs[a], axis=0) + np.nanstd(corr_data_all_bs[a], axis=0)+(a/5),
                        color=get_colors_plot(animal_in[a], color_animals), alpha=0.3)
ax.set_title('Mean cluster correlation baseline subtracted', fontsize=mscope.fsize - 2)
ax.legend(animal_in, frameon=False, fontsize=mscope.fsize-4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Miniscope processed files\\corr_summary_cluster_mean_raw_bs', dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
rectangle = plt.Rectangle((6.5, -0.1), 10, 1, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for a in range(len(corr_data_all)):
    if a == 3:
        ax.plot(np.arange(4, 27), np.transpose(corr_data_all_bs[a])+(a/5), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
    else:
        ax.plot(trials_in[a], np.transpose(corr_data_all_bs[a])+(a/5), marker='o',
                color=get_colors_plot(animal_in[a], color_animals), markersize=5, linewidth=3)
ax.set_title('Mean cluster correlation baseline subtracted', fontsize=mscope.fsize - 2)
ax.legend(animal_in, frameon=False, fontsize=mscope.fsize-4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Miniscope processed files\\corr_summary_cluster_mean_raw_allclusters_bs', dpi=mscope.my_dpi)
