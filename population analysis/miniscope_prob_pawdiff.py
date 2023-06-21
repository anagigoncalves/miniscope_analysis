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

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
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

    iter_n = 50
    shuffled_spikes = []
    trials_n = len(trials)
    cum_tr_len = np.arange(0, (trials_n * mscope.trial_time) + mscope.trial_time, mscope.trial_time, dtype=int)
    for n in df_events_extract_rawtrace.columns[2:]:
        all_spikes_ts = np.array([])
        for count_t, trial in enumerate(trials):
            df_events_tr = df_events_extract_rawtrace[df_events_extract_rawtrace.trial == trial]
            events_idx = np.array(df_events_tr.index[df_events_tr[n] == 1])
            spikes_ts = np.array(df_events_tr.time[events_idx]) + mscope.trial_time * count_t
            all_spikes_ts = np.concatenate((all_spikes_ts, spikes_ts))
        isi = np.diff(all_spikes_ts)
        for i in range(iter_n):
            np.random.shuffle(isi)
        shuffled_spikes_ts = np.insert(np.cumsum(isi), 0, 0)
        shuffled_spikes_ts_all = []
        for count_t, trial in enumerate(trials):
            shuffled_data_trial_sorted = shuffled_spikes_ts[(cum_tr_len[count_t] < shuffled_spikes_ts) &
                (shuffled_spikes_ts <= cum_tr_len[count_t+1])] - (mscope.trial_time * count_t)
            shuffled_spikes_ts_all.append(shuffled_data_trial_sorted)
        shuffled_spikes.append(shuffled_spikes_ts_all)

    window = 0.2
    max_pawdiff = []
    min_pawdiff = []
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    for count_roi, roi in enumerate(df_events_extract_rawtrace.columns[2:]):
        paw_diff_trial_means = []
        event_trial_prob = []
        event_trial_prob_shuffled = []
        for count_trial, trial in enumerate(trials):
            trial_idx = np.where(trials == trial)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (
                        df_events_extract_rawtrace['trial'] == trial), 'time'])
            events_trial_shuffled = shuffled_spikes[count_roi][count_trial]
            paw_diff_trial = paws_rel_trials[trial_idx][0] - paws_rel_trials[trial_idx][2]
            bins = np.arange(0, bcam_trial[-1], window)
            bcam_trial_idx_bins = np.digitize(bcam_trial, bins) #returns the indices of the bins to which each bcam timestamp belongs
            events_trial_idx_bins = np.digitize(events_trial, bins)  # returns the indices of the bins to which each calcium event time belongs
            events_trial_idx_bins_shuffled = np.digitize(events_trial_shuffled, bins)
            paw_diff_trial_means.extend([np.nanmean(paw_diff_trial[bcam_trial_idx_bins[:len(paw_diff_trial)] == i]) for i in range(len(bins))])
            event_trial_prob.extend([len(events_trial[events_trial_idx_bins[:len(events_trial)] == i])/len(events_trial) for i in range(len(bins))])
            event_trial_prob_shuffled.extend([len(events_trial_shuffled[events_trial_idx_bins_shuffled[:len(events_trial_shuffled)] == i]) / len(events_trial_shuffled) for i in range(len(bins))])
        paw_diff_trial_means_arr = np.array(paw_diff_trial_means)
        max_pawdiff.append(np.nanmax(paw_diff_trial_means_arr))
        min_pawdiff.append(np.nanmin(paw_diff_trial_means_arr))
        event_trial_prob_arr = np.array(event_trial_prob)
        event_trial_prob_arr_shuffled = np.array(event_trial_prob_shuffled)
        # all event probabilities for all ROIs
        # plt.scatter(paw_diff_trial_means_arr[event_trial_prob_arr > 0], event_trial_prob_arr[event_trial_prob_arr > 0], s=1, color=get_colors_plot(animal, color_animals))
        # bin paw difference values to get the mean probability
        bins_pawdiff = np.arange(-50, 50, 10)
        # bins_pawdiff = np.arange(np.nanmin(paw_diff_trial_means_arr), np.nanmax(paw_diff_trial_means_arr), 5)
        pawdiff_idx_bins = np.digitize(paw_diff_trial_means_arr, bins_pawdiff)  # returns the indices of the bins to which each bcam timestamp belongs
        event_prob_bin = [np.nanmean(event_trial_prob_arr[pawdiff_idx_bins[:len(event_trial_prob_arr)] == i]) for i in range(len(bins_pawdiff))]
        event_prob_bin_shuffled = [np.nanmean(event_trial_prob_arr_shuffled[pawdiff_idx_bins[:len(event_trial_prob_arr_shuffled)] == i]) for i in range(len(bins_pawdiff))]
        ax.scatter(bins_pawdiff, event_prob_bin, s=10, color=get_colors_plot(animal, color_animals))
        ax.scatter(bins_pawdiff+2, event_prob_bin_shuffled, s=10, color='lightgray')
    ax.set_xlabel('FR-FL paw difference (mm)', fontsize=mscope.fsize - 4)
    ax.set_ylabel('Calcium event probability\n(count in bin/total count)', fontsize=mscope.fsize - 4)
    ax.set_ylim([-0.0025, 0.02])
    ax.set_xlim([-50, 50])
    plt.xticks(fontsize=mscope.fsize - 4)
    plt.yticks(fontsize=mscope.fsize - 4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_eventprobability'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(10, 5))
    ax = ax.ravel()
    ax[0].hist(np.array(max_pawdiff), bins=10)
    ax[0].set_title('Max of pawdiff values')
    ax[1].hist(np.array(min_pawdiff), bins=10)
    ax[1].set_title('Min of pawdiff values')
    plt.savefig(os.path.join(path_session_data, 'STA difference between paws',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_histmaxmin'), dpi=mscope.my_dpi)
    plt.close('all')
