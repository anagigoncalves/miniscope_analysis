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
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
save_path = 'J:\\Thesis\\for figures\\'
event_count_loco_slow_all = []
event_count_loco_fast_all = []
event_count_loco_baseline_all = []
event_count_loco_animals = []
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

    # FOV coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    colors_session = mscope.colors_session(animal, session_type, trials, 1)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    st_strides_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        st_strides_trials.append(st_strides_mat)

    # Get trial structure
    trials_fast = []
    trials_slow = []
    trials_baseline = []
    for i in range(len(trials_ses)):
        if trials_ses_name[i] == 'fast speed':
            trials_fast = np.arange(trials_ses[i][0], trials_ses[i][-1] + 1)
        if trials_ses_name[i] == 'slow speed':
            trials_slow = np.arange(trials_ses[i][0], trials_ses[i][-1] + 1)
        if trials_ses_name[i] == 'baseline speed':
            trials_baseline = np.arange(trials_ses[i][0], trials_ses[i][-1] + 1)
    trials_fast_idx = np.where(np.isin(trials, trials_fast))[0]
    trials_slow_idx = np.where(np.isin(trials, trials_slow))[0]
    trials_baseline_idx = np.where(np.isin(trials, trials_baseline))[0]

    for roi_plot in df_events_extract_rawtrace.columns[2:]:
        event_count_loco = mscope.get_event_count_locomotion(df_events_extract_rawtrace, traces_type, colors_session, trials,
                                                             bcam_time, st_strides_trials, np.int64(roi_plot[3:]), 0, 0)
        event_count_loco_slow_all.append(np.nanmean(event_count_loco[trials_slow_idx]))
        event_count_loco_fast_all.append(np.nanmean(event_count_loco[trials_fast_idx]))
        event_count_loco_baseline_all.append(np.nanmean(event_count_loco[trials_baseline_idx]))
        event_count_loco_animals.append(animal)

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

#CONSIDER DOING ONLY S1 SESSIONS, OR S2
animals = ['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226']
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True, sharey=True)
for i in range(len(event_count_loco_slow_all)):
    plt.scatter(1+np.random.rand(), event_count_loco_slow_all[i], s=5, color=get_colors_plot(event_count_loco_animals[i], color_animals))
    plt.scatter(5+np.random.rand(), event_count_loco_fast_all[i], s=5, color=get_colors_plot(event_count_loco_animals[i], color_animals))
    plt.scatter(3+np.random.rand(), event_count_loco_baseline_all[i], s=5, color=get_colors_plot(event_count_loco_animals[i], color_animals))
for count_a, a in enumerate(animals):
    animal_idx = np.where(np.isin(event_count_loco_animals, a))[0]
    slow_values = []
    fast_values = []
    bs_values = []
    for i in animal_idx:
        slow_values.append(np.array(event_count_loco_slow_all)[i])
        fast_values.append(np.array(event_count_loco_fast_all[i]))
        bs_values.append(np.array(event_count_loco_baseline_all[i]))
    plt.plot(np.array([1.5, 3.5, 5.5]), np.array([np.mean(slow_values), np.mean(bs_values), np.mean(fast_values)]), linewidth=7, color=color_animals[count_a])
ax.set_xticks([1.5, 3.5, 5.5])
ax.set_xticklabels(['slow', 'baseline', 'fast'])
ax.set_xlabel('Speed', fontsize=mscope.fsize - 4)
ax.set_ylabel('Firing rate during\nforward locomotion', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
plt.savefig(os.path.join(save_path, 'rois_fr_forwardloco_speed'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True, sharey=True)
for count_a, a in enumerate(animals):
    animal_idx = np.where(np.isin(event_count_loco_animals, a))[0]
    slow_values = []
    fast_values = []
    bs_values = []
    for i in animal_idx:
        slow_values.append(np.array(event_count_loco_slow_all)[i])
        fast_values.append(np.array(event_count_loco_fast_all[i]))
        bs_values.append(np.array(event_count_loco_baseline_all[i]))
    animal_values = [slow_values, bs_values, fast_values]
    violin_parts = plt.violinplot(animal_values, positions=[0 + count_a, 6 + count_a, 12 + count_a])
    for pc in violin_parts['bodies']:
        pc.set_color(color_animals[count_a])
    violin_parts['cbars'].set_color(color_animals[count_a])
    violin_parts['cmins'].set_color(color_animals[count_a])
    violin_parts['cmaxes'].set_color(color_animals[count_a])
ax.set_xticks([3, 9, 15])
ax.set_xticklabels(['slow', 'baseline', 'fast'])
ax.set_xlabel('Speed', fontsize=mscope.fsize - 4)
ax.set_ylabel('Firing rate during\nforward locomotion', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 4)
plt.savefig(os.path.join(save_path, 'rois_fr_forwardloco_speed_violin'), dpi=mscope.my_dpi)

slow_all = []
fast_all = []
bs_all = []
for count_a, a in enumerate(animals):
    animal_idx = np.where(np.isin(event_count_loco_animals, a))[0]
    for i in animal_idx:
        slow_all.append(event_count_loco_slow_all[i])
        fast_all.append(event_count_loco_fast_all[i])
        bs_all.append(event_count_loco_baseline_all[i])
slow_all_arr = np.array(slow_all)
fast_all_arr = np.array(fast_all)
bs_all_arr = np.array(bs_all)
slow_notnan = slow_all_arr[~np.isnan(slow_all_arr)]
fast_notnan = fast_all_arr[~np.isnan(fast_all_arr)]
bs_notnan = bs_all_arr[~np.isnan(bs_all_arr)]
plot_values = [slow_notnan, bs_notnan, fast_notnan]
color_plots = ['purple', 'black', 'orange']
fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True, sharey=True)
violin_parts = plt.violinplot(plot_values, positions=[1, 2 ,3])
for count_pc, pc in enumerate(violin_parts['bodies']):
    pc.set_color(color_plots[count_pc])
violin_parts['cbars'].set_color(color_plots)
violin_parts['cmins'].set_color(color_plots)
violin_parts['cmaxes'].set_color(color_plots)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['slow', 'baseline', 'fast'])
ax.set_xlabel('Speed', fontsize=mscope.fsize - 4)
ax.set_ylabel('Firing rate during\nforward locomotion', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 4)
plt.savefig(os.path.join(save_path, 'rois_fr_forwardloco_speed_violin_pooled'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(7, 3), tight_layout=True, sharey=True)
violin_parts = plt.violinplot(plot_values, positions=[1, 2 ,3])
for count_pc, pc in enumerate(violin_parts['bodies']):
    pc.set_color('black')
violin_parts['cbars'].set_color('dimgray')
violin_parts['cmins'].set_color('dimgray')
violin_parts['cmaxes'].set_color('dimgray')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['0.175', '0.225', '0.300'])
ax.set_xlabel('Speed (m/s)', fontsize=mscope.fsize - 4)
ax.set_ylabel('Event rate during\nforward locomotion\n(Hz)', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 4)
plt.savefig(os.path.join(save_path, 'rois_fr_forwardloco_speed_violin_pooled_thesis'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'rois_fr_forwardloco_speed_violin_pooled_thesis.svg'), dpi=mscope.my_dpi)