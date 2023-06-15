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

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    sl_sym_mean = np.zeros(len(trials))
    coo_sym_mean = np.zeros(len(trials))
    ds_sym_mean = np.zeros(len(trials))
    fr_fl_diff_mean = np.zeros(len(trials))
    fr_hr_diff_mean = np.zeros(len(trials))
    hr_hl_diff_mean = np.zeros(len(trials))
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        fr_fl_diff_mean[count_trial] = np.nanmean(paws_rel[0]-paws_rel[2])
        fr_hr_diff_mean[count_trial] = np.nanmean(paws_rel[0] - paws_rel[1])
        hr_hl_diff_mean[count_trial] = np.nanmean(paws_rel[1] - paws_rel[3])
        sl_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
        sl_sym_mean[count_trial] = np.nanmean(sl_trials[0])-np.nanmean(sl_trials[2])
        ds_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'double_support')
        ds_sym_mean[count_trial] = np.nanmean(ds_trials[0])-np.nanmean(ds_trials[2])
        coo_trials = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'coo')
        coo_sym_mean[count_trial] = np.nanmean(coo_trials[0])-np.nanmean(coo_trials[2])

    # FR-FL difference curve across trials
    fr_fl_diff_baseline = np.nanmean(fr_fl_diff_mean[:trials_ses[0, 1]])
    fr_fl_diff_bs = fr_fl_diff_mean - fr_fl_diff_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(fr_fl_diff_bs)), 10,
                                  max(fr_fl_diff_bs) - min(fr_fl_diff_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(fr_fl_diff_bs), colors='grey', linestyles='--')
    ax.plot(trials, fr_fl_diff_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, fr_fl_diff_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('FR-FL difference', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'FR-FL_difference_curve'), dpi=mscope.my_dpi)

    # FR-HR difference curve across trials
    fr_hr_diff_baseline = np.nanmean(fr_hr_diff_mean[:trials_ses[0, 1]])
    fr_hr_diff_bs = fr_hr_diff_mean - fr_hr_diff_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(fr_hr_diff_bs)), 10,
                                  max(fr_hr_diff_bs) - min(fr_hr_diff_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(fr_hr_diff_bs), colors='grey', linestyles='--')
    ax.plot(trials, fr_hr_diff_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, fr_hr_diff_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('FR-FL difference', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'FR-HR_difference_curve'), dpi=mscope.my_dpi)

    # HR-HL difference curve across trials
    hr_hl_diff_baseline = np.nanmean(hr_hl_diff_mean[:trials_ses[0, 1]])
    hr_hl_diff_bs = hr_hl_diff_mean - hr_hl_diff_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(hr_hl_diff_bs)), 10,
                                  max(hr_hl_diff_bs) - min(hr_hl_diff_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(hr_hl_diff_bs), colors='grey', linestyles='--')
    ax.plot(trials, hr_hl_diff_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, hr_hl_diff_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('FR-FL difference', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'HR-HL_difference_curve'), dpi=mscope.my_dpi)

    # SL sym curve across trials
    sl_sym_baseline = np.nanmean(sl_sym_mean[:trials_ses[0, 1]])
    sl_sym_bs = sl_sym_mean - sl_sym_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(sl_sym_bs)), 10,
                                  max(sl_sym_bs) - min(sl_sym_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(sl_sym_bs), colors='grey', linestyles='--')
    ax.plot(trials, sl_sym_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, sl_sym_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('Step length symmetry', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'sl_sym_curve'), dpi=mscope.my_dpi)

    # DS sym curve across trials
    ds_sym_baseline = np.nanmean(ds_sym_mean[:trials_ses[0, 1]])
    ds_sym_bs = ds_sym_mean - ds_sym_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(ds_sym_bs)), 10,
                                  max(ds_sym_bs) - min(ds_sym_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(ds_sym_bs), colors='grey', linestyles='--')
    ax.plot(trials, ds_sym_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, ds_sym_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('Double support symmetry', fontsize=20)
    plt.gca().invert_yaxis()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'ds_sym_curve'), dpi=mscope.my_dpi)

    # COO sym curve across trials
    coo_sym_baseline = np.nanmean(coo_sym_mean[:trials_ses[0, 1]])
    coo_sym_bs = coo_sym_mean - coo_sym_baseline
    fig, ax = plt.subplots(figsize=(5, 6), tight_layout=True)
    if session_type == 'split':
        rectangle = plt.Rectangle((trials_ses[0, 1] + 0.5, min(sl_sym_bs)), 10,
                                  max(sl_sym_bs) - min(sl_sym_bs), fc='grey', alpha=0.3)
        ax.add_patch(rectangle)
    ax.hlines(0, 1, len(coo_sym_bs), colors='grey', linestyles='--')
    ax.plot(trials, coo_sym_bs, color='black')
    for count_t, t in enumerate(trials):
        idx_trial = np.where(trials == t)[0][0]
        ax.scatter(t, coo_sym_bs[idx_trial], s=80, color=colors_session[t])
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('COO symmetry', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(mscope.path, 'images', 'coo_sym_curve'), dpi=mscope.my_dpi)
    plt.close('all')
