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

path_session_data = 'E:\\Miniscope processed files\\'
session_data = pd.read_excel('E:\\Miniscope processed files\\session_data_split_S1.xlsx')
if not os.path.exists(path_session_data + 'STA difference between paws'):
    os.mkdir(path_session_data + 'STA difference between paws')
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
    centroid_ext = mscope.get_roi_centroids(coord_ext)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    paws_rel_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        paws_rel_trials.append(paws_rel)

    cmap = plt.get_cmap('magma')
    color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
    def get_colors_plot(animal_name, color_animals):
        if animal_name == 'MC8855':
            color_plot = color_animals[0]
        if animal_name == 'MC9194':
            color_plot = color_animals[1]
        if animal_name == 'MC10221':
            color_plot = color_animals[2]
        if animal_name == 'MC9513':
            color_plot = color_animals[3]
        if animal_name == 'MC9226':
            color_plot = color_animals[4]
        return color_plot

    window = 0.2
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    for roi in df_events_extract_rawtrace.columns[2:]:
        paw_diff_trial_means = []
        event_trial_prob = []
        for trial in trials:
            trial_idx = np.where(trials == trial)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (
                        df_events_extract_rawtrace['trial'] == trial), 'time'])
            paw_diff_trial = paws_rel_trials[trial_idx][0] - paws_rel_trials[trial_idx][2]
            bins = np.arange(0, bcam_trial[-1], window)
            bcam_trial_idx_bins = np.digitize(bcam_trial, bins) #returns the indices of the bins to which each bcam timestamp belongs
            events_trial_idx_bins = np.digitize(events_trial, bins)  # returns the indices of the bins to which each calcium event time belongs
            paw_diff_trial_means.extend([np.nanmean(paw_diff_trial[bcam_trial_idx_bins[:len(paw_diff_trial)] == i]) for i in range(len(bins))])
            event_trial_prob.extend([len(events_trial[events_trial_idx_bins[:len(events_trial)] == i])/len(events_trial) for i in range(len(bins))])

        paw_diff_trial_means_arr = np.array(paw_diff_trial_means)
        event_trial_prob_arr = np.array(event_trial_prob)
        # all event probabilities for all ROIs
        # plt.scatter(paw_diff_trial_means_arr[event_trial_prob_arr > 0], event_trial_prob_arr[event_trial_prob_arr > 0], s=1, color=get_colors_plot(animal, color_animals))
        # bin paw difference values to get the mean probability
        bins_pawdiff = np.arange(np.nanmin(paw_diff_trial_means_arr), np.nanmax(paw_diff_trial_means_arr), 5)
        pawdiff_idx_bins = np.digitize(paw_diff_trial_means_arr, bins_pawdiff)  # returns the indices of the bins to which each bcam timestamp belongs
        event_prob_bin = [np.nanmean(event_trial_prob_arr[pawdiff_idx_bins[:len(event_trial_prob_arr)] == i]) for i in range(len(bins_pawdiff))]
        ax.scatter(bins_pawdiff, event_prob_bin, s=10, color=get_colors_plot(animal, color_animals))
    ax.set_xlabel('FR-FL paw difference (mm)', fontsize=mscope.fsize - 4)
    ax.set_ylabel('Calcium event probability\n(count in bin/total count)', fontsize=mscope.fsize - 4)
    plt.xticks(fontsize=mscope.fsize - 4)
    plt.yticks(fontsize=mscope.fsize - 4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_eventprobability'), dpi=mscope.my_dpi)
    plt.close('all')

plt.figure()
plt.scatter(paw_diff_trial_means_arr, event_trial_prob_arr, s=50)