# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

protocol = 'split contra fast'
animal = 'MC13419'

save_path = 'J:\\Thesis\\for figures\\405 control\\'

# path inputs
path_405 = 'J:\\Miniscope processed files\\TM RAW FILES\split contra fast 405\\MC13419\\2022_05_31\\'
path_loco_405 = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split contra fast 405nm S2 310522\\'
mscope = miniscope_session_class.miniscope_session(path_405)
loco = locomotion_class.loco_class(path_loco_405)
coord_ext_405 = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
ref_image_405 = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)
traces_405 = pd.read_csv(os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
events_405 = pd.read_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'))
frames_dFF_405 = np.load(os.path.join(mscope.path, 'processed files', 'black_frames.npy'), allow_pickle=True)
[trigger_nr, strobe_nr, frames_loco_405, trial_start, bcam_time_405] = loco.get_tdms_frame_start(animal, 2, frames_dFF_405)
filelist_405 = loco.get_track_files(animal, 2)
st_strides_trials_405 = []
for count_trial, f in enumerate(filelist_405):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco_405[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    st_strides_trials_405.append(st_strides_mat)
        
path_480 = 'J:\\Miniscope processed files\\TM RAW FILES\split contra fast\\MC13419\\2022_05_31\\'
path_loco_480 = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split contra fast S1 310522\\'
mscope = miniscope_session_class.miniscope_session(path_480)
loco = locomotion_class.loco_class(path_loco_480)
coord_ext_480 = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
ref_image_480 = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)
traces_480 = pd.read_csv(os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
events_480 = pd.read_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'))
frames_dFF_480 = np.load(os.path.join(mscope.path, 'processed files', 'black_frames.npy'), allow_pickle=True)
[trigger_nr, strobe_nr, frames_loco_480, trial_start, bcam_time_480] = loco.get_tdms_frame_start(animal, 1, frames_dFF_480)
filelist_480 = loco.get_track_files(animal, 1)
st_strides_trials_480 = []
for count_trial, f in enumerate(filelist_480):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco_480[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    st_strides_trials_480.append(st_strides_mat)

# Compare ROIs distribution over the reference image
plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext_405)):
    plt.scatter(coord_ext_405[r][:, 0], coord_ext_405[r][:, 1], s=1, alpha=0.6)
plt.imshow(ref_image_405, cmap='gray',
           extent=[0, np.shape(ref_image_405)[1] / mscope.pixel_to_um, np.shape(ref_image_405)[0] / mscope.pixel_to_um, 0])
plt.title('405', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.savefig(os.path.join(save_path, 'ref_image_405_' + animal + '_' + protocol.replace(' ', '_')), dpi=mscope.my_dpi)

plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext_480)):
    plt.scatter(coord_ext_480[r][:, 0], coord_ext_480[r][:, 1], s=1, alpha=0.6)
plt.imshow(ref_image_480, cmap='gray',
           extent=[0, np.shape(ref_image_480)[1] / mscope.pixel_to_um, np.shape(ref_image_480)[0] / mscope.pixel_to_um, 0])
plt.title('480', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.savefig(os.path.join(save_path, 'ref_image_480_' + animal + '_' + protocol.replace(' ', '_')), dpi=mscope.my_dpi)

# Compare traces for the same ROIs
trial_plot = 3
traces_405_trial = traces_405.loc[traces_405['trial'] == trial_plot]
fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
for count_r, r in enumerate(traces_405_trial.columns[2:]):
    plt.plot(traces_405_trial['time'], traces_405_trial[r] + (count_r / 2), color='black')
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax.set_ylabel('Calcium trace 405 for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='y', which='y', length=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
plt.savefig(os.path.join(save_path, 'traces_405_' + animal + '_' + protocol.replace(' ', '_')), dpi=mscope.my_dpi)

traces_480_trial = traces_480.loc[traces_480['trial'] == trial_plot]
fig, ax = plt.subplots(figsize=(10, 20), tight_layout=True)
for count_r, r in enumerate(traces_480_trial.columns[2:]):
    plt.plot(traces_480_trial['time'], traces_480_trial[r] + (count_r / 2), color='black')
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax.set_ylabel('Calcium trace 480 for trial ' + str(trial_plot), fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='y', which='y', length=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
plt.savefig(os.path.join(save_path, 'traces_480_' + animal + '_' + protocol.replace(' ', '_')), dpi=mscope.my_dpi)

#Firing rate plots
trials = np.arange(1, 27)
trials_baseline_idx = np.arange(0, 6)
trials_split_idx = np.arange(6, 16)
trials_washout_idx = np.arange(16, 26)

event_count_loco_split_all = []
event_count_loco_washout_all = []
event_count_loco_baseline_all = []
for roi_plot in events_405.columns[2:]:
    event_count_loco = np.zeros((len(trials)))
    for count_t, t in enumerate(trials):
        bcam_trial = bcam_time_405[count_t]
        events = np.array(
            events_405.loc[(events_405['trial'] == t) & (events_405[roi_plot] == 1), 'time'])
        st_on = st_strides_trials_405[count_t][0][:, 0, -1]  # FR paw
        st_off = st_strides_trials_405[count_t][0][:, 1, -1]  # FR paw
        time_forwardloco = []
        event_clean_list = []
        for s in range(len(st_on)):
            time_forwardloco.append(bcam_trial[int(st_off[s])] - bcam_trial[int(st_on[s])])
            if len(np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]) > 0:
                event_clean_list.append(len(
                    np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]))
        time_forwardloco_trial = np.sum(time_forwardloco)
        event_count_loco[count_t] = np.sum(event_clean_list) / time_forwardloco_trial
    event_count_loco_split_all.append(np.nanmean(event_count_loco[trials_split_idx]))
    event_count_loco_washout_all.append(np.nanmean(event_count_loco[trials_washout_idx]))
    event_count_loco_baseline_all.append(np.nanmean(event_count_loco[trials_baseline_idx]))

colors_violins = ['black', 'crimson', 'blue']
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True, sharey=True)
animal_values = [event_count_loco_baseline_all, event_count_loco_split_all, event_count_loco_washout_all]
violin_parts = plt.violinplot(animal_values, positions=[0, 1,2])
for c, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_violins[c])
violin_parts['cbars'].set_color(colors_violins)
violin_parts['cmins'].set_color(colors_violins)
violin_parts['cmaxes'].set_color(colors_violins)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['baseline', 'split', 'washout'])
ax.set_xlabel('Trial type', fontsize=mscope.fsize - 4)
ax.set_ylabel('Firing rate during\nforward locomotion', fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim([1, 2])
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\405 control\\', 'roi_fr_forwardloco_split_contra_fast_405_violin_' + animal), dpi=mscope.my_dpi)

event_count_loco_split_all = []
event_count_loco_washout_all = []
event_count_loco_baseline_all = []
for roi_plot in events_480.columns[2:]:
    event_count_loco = np.zeros((len(trials)))
    for count_t, t in enumerate(trials):
        bcam_trial = bcam_time_480[count_t]
        events = np.array(
            events_480.loc[(events_480['trial'] == t) & (events_480[roi_plot] == 1), 'time'])
        st_on = st_strides_trials_480[count_t][0][:, 0, -1]  # FR paw
        st_off = st_strides_trials_480[count_t][0][:, 1, -1]  # FR paw
        time_forwardloco = []
        event_clean_list = []
        for s in range(len(st_on)):
            time_forwardloco.append(bcam_trial[int(st_off[s])] - bcam_trial[int(st_on[s])])
            if len(np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]) > 0:
                event_clean_list.append(len(
                    np.where((events >= bcam_trial[int(st_on[s])]) & (events <= bcam_trial[int(st_off[s])]))[0]))
        time_forwardloco_trial = np.sum(time_forwardloco)
        event_count_loco[count_t] = np.sum(event_clean_list) / time_forwardloco_trial
    event_count_loco_split_all.append(np.nanmean(event_count_loco[trials_split_idx]))
    event_count_loco_washout_all.append(np.nanmean(event_count_loco[trials_washout_idx]))
    event_count_loco_baseline_all.append(np.nanmean(event_count_loco[trials_baseline_idx]))

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True, sharey=True)
animal_values = [event_count_loco_baseline_all, event_count_loco_split_all, event_count_loco_washout_all]
violin_parts = plt.violinplot(animal_values, positions=[0, 1,2])
for c, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors_violins[c])
violin_parts['cbars'].set_color(colors_violins)
violin_parts['cmins'].set_color(colors_violins)
violin_parts['cmaxes'].set_color(colors_violins)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['baseline', 'split', 'washout'])
ax.set_xlabel('Trial type', fontsize=mscope.fsize - 4)
ax.set_ylabel('Firing rate during\nforward locomotion', fontsize=mscope.fsize - 4)
ax.set_ylim([1, 2])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 6)
plt.savefig(os.path.join('J:\\Thesis\\for figures\\405 control\\', 'roi_fr_forwardloco_split_contra_fast_480_violin_' + animal), dpi=mscope.my_dpi)



