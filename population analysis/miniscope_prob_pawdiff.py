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
if not os.path.exists(path_session_data + 'CS probability at difference between paws'):
    os.mkdir(path_session_data + 'CS probability at difference between paws')
if not os.path.exists(path_session_data + 'Paw difference at CS time'):
    os.mkdir(path_session_data + 'Paw difference at CS time')
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
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

    # import glob
    # import nptdms as tdms
    # tdmslist = glob.glob(loco.path + '*.tdms')
    # h5list = glob.glob(loco.path + '*.h5')
    # tdms_sr = 10000
    # h5filename = []
    # for f in h5list:
    #     if loco.delim == '/':
    #         h5filename.append(f.split('/')[-1])
    #     if loco.delim == '\\':
    #         h5filename.append(f.split('\\')[-1])
    # h5_trialorder = []
    # h5list_animal = []
    # for f in h5filename:
    #     if f.split('_')[0] == animal and int(f.split('_')[7]) == session:
    #         h5_trialorder.append(int(f.split('_')[8][:-3]))
    #         h5list_animal.append(loco.path + f)
    # # order tdms list for the animal and session that are going to be computed
    # tdmslist_animal = []
    # for f in tdmslist:
    #     if loco.delim == '/':
    #         filename = f.split('/')[-1]
    #     else:
    #         filename = f.split('\\')[-1]
    #     filename_split = filename.split('_')
    #     if filename_split[0] == animal and np.int64(filename_split[7]) == session:
    #         tdmslist_animal.append(f)
    # trialorder_tdms = []
    # for f in tdmslist_animal:
    #     if loco.delim == '/':
    #         filename = f.split('/')[-1]
    #     else:
    #         filename = f.split('\\')[-1]
    #     filename_split = filename.split('_')
    #     trialorder_tdms.append(np.int64(filename_split[8][:-5]))
    # trial_order_idx = np.argsort(trialorder_tdms)
    # tdmslist_animal_ordered = list(np.array(tdmslist_animal)[trial_order_idx])
    # trial = []
    # triggers = []
    # strobes = []
    # frame_rec_start = []
    # mscope_align_time = []  # if miniscope start is before behavior cam (should never happen)
    # frame_time_bcam = []
    # frame_nr = []
    # for f in tdmslist_animal_ordered:
    #     tdms_file = tdms.TdmsFile.read(f)
    #     trial.append(int(f.split('_')[-1][:-5]))
    #     h5_trial = np.where(np.array(h5_trialorder) == int(f.split('_')[-1][:-5]))[0][0]
    #     track_df = pd.read_hdf(h5list_animal[h5_trial], 'df_with_missing')
    #     frame_nr.append(track_df.iloc[:, 0].index[-1]+1)
    #     tdms_groups = tdms_file.groups()
    #     tdms_channels = tdms_groups[0].channels()
    #     ch1 = tdms_channels[0].data  # triggers
    #     ch2 = tdms_channels[1].data  # miniscope pulse
    #     ch3 = tdms_channels[2].data  # strobes
    #     triggers.append(len(np.where(np.diff(ch1) > 1)[0] + 1))
    #     idx_peaks = np.where(np.diff(ch3) > 1)[0] + 1
    #     cam_start = idx_peaks[0]
    #     strobes.append(len(idx_peaks))
    #     idx_miniscope_start = np.where(ch2)[0][0]  # tdms time point from bcam at which mscope frame started
    #     frame_behavior_miniscope = len(np.where(idx_peaks < idx_miniscope_start)[
    #                                        0]) + 1  # frame from bcam at which mscope frame started (tdms starts before trigger)
    #     if frame_behavior_miniscope == 1:  # if miniscope start is before behavior cam (should never happen)
    #         mscope_idx_0 = np.where(ch2 == 1)[0][0]
    #         mscope_start = (cam_start - mscope_idx_0) / tdms_sr
    #         mscope_align_time.append(mscope_start)
    #         frame_rec_start.append(0)
    #         frame_time_bcam.append(((
    #                                             idx_peaks - cam_start) / tdms_sr) + mscope_start)  # need to add to bcam time the difference of time since mscope started
    #     else:
    #         frame_rec_start.append(frame_behavior_miniscope)
    #         mscope_start = 0
    #         mscope_align_time.append(mscope_start)
    #         cam_frames_time = (idx_peaks - idx_peaks[0]) / tdms_sr
    #         frame_time_bcam.append(cam_frames_time)  # bcam time starts from 0
    # if len(frames_dFF) == 0:
    #     frame_rec_start_full = frame_rec_start
    # else:
    #     frame_rec_start_full = np.zeros(len(triggers))
    #     for i in range(len(frame_rec_start)):
    #         frame_rec_start_full[i] = frame_rec_start[i] + (frames_dFF[i] / 30) * loco.sr
    # # change frame_time_bcam to account for black frames that were excluded
    # bcam_time = []
    # for t in range(len(frame_time_bcam)):
    #     if frame_rec_start_full[t] / loco.sr > frame_time_bcam[t][
    #         0]:  # if black frames is larger than the gap between mscope start and bcam start
    #         bcam_time_lostframes = frame_time_bcam[t][:frame_nr[t]]  # if it lost frames it was at the end
    #         bcam_time_clean = bcam_time_lostframes[int(frame_rec_start_full[t]):]
    #         bcam_time.append(
    #             bcam_time_clean - bcam_time_clean[0])  # because extra frames will be deleted from tracking
    #     else:
    #         bcam_time.append(frame_time_bcam[t][:frame_nr[t]])
    #         frame_rec_start_full[t] = 0
    #
    # [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(filelist[3], 0.9, int(
    #     frames_loco[3]))
    # print(frame_nr[3])
    # print(len(final_tracks[0, 0, :]))
    # print(len(bcam_time[3]))

     # Order ROIs mediolateral
    roi_list = mscope.get_roi_list(df_events_extract_rawtrace)
    centroids_mediolateral = []
    for c in range(len(centroid_ext)):
        centroids_mediolateral.append(centroid_ext[c][0])
    distance_neurons_ordered = np.argsort(centroids_mediolateral)
    rois_ordered_distance_str = []
    for i in distance_neurons_ordered:
        rois_ordered_distance_str.append(roi_list[i])

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

    #Event count for paw difference values against a generated Poisson distribution for each trial
    # fig, ax = plt.subplots(5, 5, tight_layout=True, sharey=True, sharex=True)
    # ax = ax.ravel()
    mean_pawdiff_events = np.zeros((len(df_events_extract_rawtrace.columns[2:]), len(trials)))
    mean_pawdiff_events_generated = np.zeros((len(df_events_extract_rawtrace.columns[2:]), len(trials)))
    for count_roi, roi in enumerate(rois_ordered_distance_str):
        for count_trial, trial in enumerate(trials):
            trial_idx = np.where(trials == trial)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (df_events_extract_rawtrace['trial'] == trial), 'time'])
            paw_diff_trial = paws_rel_trials[trial_idx][0] - paws_rel_trials[trial_idx][2]
            events_trial_idx = [np.argmin(np.abs(e - bcam_trial)) for e in events_trial]
            paw_diff_trial_events = paw_diff_trial[events_trial_idx]
            mean_pawdiff_events[count_roi, count_trial] = np.nanmean(paw_diff_trial_events)
            isi = np.diff(events_trial)
            #generate the ISI from a random exponential distribution
            #exponential distribution is the probability distribution of time between events in a Poisson process
            e_dist = np.random.exponential(np.nanmean(isi), len(isi))
            spikes_generated = np.insert(np.cumsum(e_dist), 0, 0)
            events_trial_idx_generated = [np.argmin(np.abs(e - bcam_trial)) for e in spikes_generated]
            paw_diff_trial_events_generated = paw_diff_trial[events_trial_idx_generated]
            mean_pawdiff_events_generated[count_roi, count_trial] = np.nanmean(paw_diff_trial_events_generated)
            # ax[count_trial].hist(paw_diff_trial_events, bins=20)
            # ax[count_trial].hist(paw_diff_trial_events_generated, bins=20, alpha=0.5)
            # ax[count_trial].set_title('trial ' + str(trial))
    fig, ax = plt.subplots(1, 2, tight_layout=True)
    ax = ax.ravel()
    sns.heatmap(mean_pawdiff_events, ax=ax[0], cmap='viridis')
    sns.heatmap(mean_pawdiff_events_generated, cmap='viridis')
    ax[0].vlines(trials_ses[0, 1], *ax[0].get_ylim(), color='white', linestyle='dashed')
    ax[0].vlines(trials_ses[1, 1], *ax[0].get_ylim(), color='white', linestyle='dashed')
    ax[0].set_yticks(np.arange(0, len(rois_ordered_distance_str), 8))
    ax[0].set_yticklabels(rois_ordered_distance_str[::8])
    ax[0].set_xticks(trials)
    ax[0].set_xticklabels(list(map(str, trials)), rotation=45, fontsize=mscope.fsize - 10)
    ax[0].set_xlabel('Trials', fontsize=mscope.fsize - 8)
    ax[0].set_ylabel('ROI', fontsize=mscope.fsize - 8)
    ax[0].set_title('Mean of paw difference\ndistribution for real events', fontsize=mscope.fsize - 8)
    ax[1].vlines(trials_ses[0, 1], *ax[0].get_ylim(), color='white', linestyle='dashed')
    ax[1].vlines(trials_ses[1, 1], *ax[0].get_ylim(), color='white', linestyle='dashed')
    ax[1].set_yticks(np.arange(0, len(rois_ordered_distance_str), 8))
    ax[1].set_yticklabels(rois_ordered_distance_str[::8])
    ax[1].set_xticks(trials)
    ax[1].set_xticklabels(list(map(str, trials)), rotation=45, fontsize=mscope.fsize - 10)
    ax[1].set_xlabel('Trials', fontsize=mscope.fsize - 8)
    ax[1].set_ylabel('ROI', fontsize=mscope.fsize - 8)
    ax[1].set_title('Generated calcium events from Poisson\ndistribution for each trial', fontsize=mscope.fsize - 8)
    plt.savefig(os.path.join(path_session_data, 'Paw difference at CS time',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_pawdiff_eventtime_generatedevents'), dpi=mscope.my_dpi)

    # SHUFFLING SPIKES
    iter_n = 1000
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
        event_trial_nr = []
        event_trial_total = []
        event_trial_prob_shuffled = []
        event_trial_nr_shuffled = []
        event_trial_total_shuffled = []
        for count_trial, trial in enumerate(trials):
            trial_idx = np.where(trials == trial)[0][0]
            bcam_trial = bcam_time[trial_idx]
            events_trial = np.array(df_events_extract_rawtrace.loc[(df_events_extract_rawtrace[roi] == 1) & (df_events_extract_rawtrace['trial'] == trial), 'time'])
            events_trial_shuffled = shuffled_spikes[count_roi][count_trial]
            paw_diff_trial = paws_rel_trials[trial_idx][0] - paws_rel_trials[trial_idx][2]
            bins = np.arange(0, bcam_trial[-1], window)
            bcam_trial_idx_bins = np.digitize(bcam_trial, bins) #returns the indices of the bins to which each bcam timestamp belongs
            events_trial_idx_bins = np.digitize(events_trial, bins)  # returns the indices of the bins to which each calcium event time belongs
            events_trial_idx_bins_shuffled = np.digitize(events_trial_shuffled, bins)
            paw_diff_trial_means.extend([np.nanmean(paw_diff_trial[bcam_trial_idx_bins[:len(paw_diff_trial)] == i]) for i in range(len(bins))])
            event_trial_prob.extend([np.divide(len(events_trial[events_trial_idx_bins[:len(events_trial)] == i]), len(events_trial)) for i in range(len(bins))])
            event_trial_prob_shuffled.extend([np.divide(len(events_trial_shuffled[events_trial_idx_bins_shuffled[:len(events_trial_shuffled)] == i]), len(events_trial_shuffled)) for i in range(len(bins))])
        paw_diff_trial_means_arr = np.array(paw_diff_trial_means)
        max_pawdiff.append(np.nanmax(paw_diff_trial_means_arr))
        min_pawdiff.append(np.nanmin(paw_diff_trial_means_arr))
        event_trial_prob_arr = np.array(event_trial_prob)
        event_trial_prob_arr_shuffled = np.array(event_trial_prob_shuffled)
        # bin paw difference values to get the mean probability
        bins_pawdiff = np.arange(-50, 50, 5)
        pawdiff_idx_bins = np.digitize(paw_diff_trial_means_arr, bins_pawdiff)  # returns the indices of the bins to which each bcam timestamp belongs
        event_prob_bin = [np.nanmean(event_trial_prob_arr[pawdiff_idx_bins[:len(event_trial_prob_arr)] == i]) for i in range(len(bins_pawdiff))]
        event_prob_bin_shuffled = [np.nanmean(event_trial_prob_arr_shuffled[pawdiff_idx_bins[:len(event_trial_prob_arr_shuffled)] == i]) for i in range(len(bins_pawdiff))]
        ax.scatter(bins_pawdiff, event_prob_bin, s=10, color=get_colors_plot(animal, color_animals))
        ax.scatter(bins_pawdiff+1, event_prob_bin_shuffled, s=10, color='lightgray')
    ax.set_xlabel('FR-FL paw difference (mm)', fontsize=mscope.fsize - 4)
    ax.set_ylabel('Calcium event probability\n(count in bin/total count)', fontsize=mscope.fsize - 4)
    # ax.set_ylim([-0.0025, 0.02])
    ax.set_xlim([-50, 50])
    plt.xticks(fontsize=mscope.fsize - 4)
    plt.yticks(fontsize=mscope.fsize - 4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_session_data, 'CS probability at difference between paws',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_eventprobability'), dpi=mscope.my_dpi)

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(10, 5))
    ax = ax.ravel()
    ax[0].hist(np.array(max_pawdiff), bins=10)
    ax[0].set_title('Max of pawdiff values')
    ax[1].hist(np.array(min_pawdiff), bins=10)
    ax[1].set_title('Min of pawdiff values')
    plt.savefig(os.path.join(path_session_data, 'CS probability at difference between paws',
                             animal + '_' + ses_info['protocol'].replace(' ', '_') + '_S' + str(session) + '_FR-FL_histmaxmin'), dpi=mscope.my_dpi)
    plt.close('all')
