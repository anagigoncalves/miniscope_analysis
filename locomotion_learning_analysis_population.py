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

path_session_data = 'E:\\Miniscope processed files'
session_data = pd.read_excel('E:\\Miniscope processed files\\session_data_split_ipsi.xlsx') # animals that did split right fast in S1
if not os.path.exists(path_session_data+'\\split-belt locomotion analysis'):
    os.mkdir(path_session_data+'\\split-belt locomotion analysis')
swing_x_rel_all = []
swing_y_rel_all = []
phase_bin_paw_all = []
swing_inst_vel_all = []
swing_z_all = []
df_values = []
df_speed = []
df_param = []
df_animal = []
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
     clusters_rois, colors_cluster, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    path_save = path_session_data+'\\split-belt locomotion analysis\\' + animal + ' session ' + str(session) + '\\'
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stance_speed']
    param_sym = np.zeros((len(param_sym_name), 23))
    stance_speed = np.zeros((4, 23))
    filelist = loco.get_track_files(animal, session)
    if animal != 'MC8855':
        filelist = filelist[3:] #because all other sessions have 26 trials, ignore the first 3 baseline trials to be comparable
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF[count_trial])
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                                param)
            if param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_trial] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])

    # # # Plot
    # baseline subtracion of parameters
    param_sym_bs = np.zeros(np.shape(param_sym))
    for p in range(np.shape(param_sym)[0] - 1): # all except stance speed
        bs_mean = np.nanmean(param_sym[p, :3])
        param_sym_bs[p, :] = param_sym[p, :] - bs_mean

    # plot symmetry baseline subtracted
    for p in range(np.shape(param_sym)[0] - 1):
        fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
        rectangle = plt.Rectangle((3 + 0.5, min(param_sym_bs[p, :].flatten())), 10,
                                  max(param_sym_bs[p, :].flatten()) - min(param_sym_bs[p, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs[p, :]), colors='grey', linestyles='--')
        plt.plot(np.linspace(1, len(param_sym_bs[p, :]), len(param_sym_bs[p, :])), param_sym_bs[p, :], linewidth=2, color='black')
        ax.set_xlabel('Trial', fontsize=20)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        np.save(path_save + param_sym_name[p], param_sym_bs[p, :])
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym_bs', dpi=128)

    # plot stance speed
    fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
    rectangle = plt.Rectangle((3 + 0.5, -0.3), 10, 0.2, fc='dimgrey', alpha=0.3)
    plt.gca().add_patch(rectangle)
    for p in range(4):
        ax.plot(np.linspace(1, len(stance_speed[p, :]), len(stance_speed[p, :])), stance_speed[p, :], color=paw_colors[p],
                   linewidth=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Trial', fontsize=20)
        ax.set_ylabel('Stance speed', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    np.save(path_save + 'stance_speed', stance_speed)
    if print_plots:
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + 'stance_speed', dpi=96)
    plt.close('all')

# Get control data
control_path = 'E:\\Miniscope processed files\\Non-miniscope locomotion data\\Treadmill adaptation\\grouped output'
main_miniscope_path = 'E:\\Miniscope processed files\\split-belt locomotion analysis\\'
folders_animals = os.listdir(main_miniscope_path)
# Group the data
param_split = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length']
param_split_miniscope_values = np.zeros((len(param_split), len(folders_animals), 23))
stance_speed_miniscope_values = np.zeros((len(folders_animals), 4, 23))
for count_p, g in enumerate(param_split):
    for count_a, a in enumerate(folders_animals):
        miniscope_path = main_miniscope_path + a
        param_tied_miniscope_file = np.load(miniscope_path + '\\' + param_split[count_p] + '.npy')
        param_split_miniscope_values[count_p, count_a, :] = param_tied_miniscope_file
for count_a, a in enumerate(folders_animals):
    miniscope_path = main_miniscope_path + a
    stance_speed_miniscope_file = np.load(miniscope_path + '\\stance_speed.npy')
    stance_speed_miniscope_values[count_a, :, ] = stance_speed_miniscope_file
param_split_control_values = np.zeros((len(param_split), 10, 23))
for count_p, g in enumerate(param_split):
    param_tied_control_file = np.load(control_path + '\\' + param_split[count_p] + '.npy')
    param_split_control_values[count_p, :, :] = param_tied_control_file
stance_speed_control_values = np.load(control_path + '\\' + 'stance_speed.npy')

max_rect = np.array([3, 6, 8, 10, 18])
min_rect = np.array([-8, -15, -5, -2, -1])
# Plot learning curves
for p in range(np.shape(param_split)[0]):
    param_split_miniscope_values_mean = np.nanmean(param_split_miniscope_values[p, :, :], axis=0)
    param_split_miniscope_values_std = np.nanstd(param_split_miniscope_values[p, :, :], axis=0)/np.sqrt(np.shape(param_split_miniscope_values)[1])
    fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
    rectangle = plt.Rectangle((3 + 0.5, min_rect[p]), 10,
                              max_rect[p] - min_rect[p],
                              fc='dimgrey', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, len(param_split_miniscope_values_mean), colors='grey', linestyles='--')
    param_split_control_values
    plt.plot(np.linspace(1, len(param_split_control_values[p, 0, :]), len(param_split_control_values[p, 0, :])), np.nanmean(param_split_control_values[p, :, :], axis=0), linewidth=2, color='black')
    plt.fill_between(np.linspace(1, len(param_split_miniscope_values_mean), len(param_split_miniscope_values_mean)),
             np.nanmean(param_split_control_values[p, :, :], axis=0)-(np.nanstd(param_split_control_values[p, :, :], axis=0)/np.sqrt(np.shape(param_split_control_values)[1])),
                     np.nanmean(param_split_control_values[p, :, :], axis=0)+(np.nanstd(param_split_control_values[p, :, :], axis=0)/np.sqrt(np.shape(param_split_control_values)[1])), alpha=0.3, color='black')
    plt.plot(np.linspace(1, len(param_split_miniscope_values_mean), len(param_split_miniscope_values_mean)), param_split_miniscope_values_mean, linewidth=2, color='teal')
    plt.fill_between(np.linspace(1, len(param_split_miniscope_values_mean), len(param_split_miniscope_values_mean)),
             param_split_miniscope_values_mean-param_split_miniscope_values_std,
                     param_split_miniscope_values_mean+param_split_miniscope_values_std, alpha=0.3, color='teal')
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_split[p].replace('_', ' '), fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if print_plots:
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_session_data+'\\split-belt locomotion analysis\\' + param_split[p] + '_sym_bs', dpi=128)

# plot stance speed
fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
rectangle = plt.Rectangle((3 + 0.5, -0.3), 10, 0.2, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for p in range(4):
    ax.plot(np.linspace(1, len(stance_speed_miniscope_values[0, p, :]), len(stance_speed_miniscope_values[0, p, :])),
            np.nanmean(stance_speed_miniscope_values[:, p, :], axis=0), color=paw_colors[p],
               linewidth=2)
    ax.fill_between(np.linspace(1, len(stance_speed_miniscope_values[0, p, :]), len(stance_speed_miniscope_values[0, p, :])),
            np.nanmean(stance_speed_miniscope_values[:, p, :], axis=0)-(np.nanstd(stance_speed_miniscope_values[:, p, :], axis=0)/np.sqrt(np.shape(stance_speed_miniscope_values)[0])),
    np.nanmean(stance_speed_miniscope_values[:, p, :], axis=0)+(np.nanstd(stance_speed_miniscope_values[:, p, :], axis=0)/np.sqrt(np.shape(stance_speed_miniscope_values)[0])), color=paw_colors[p],
               alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Stance speed', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
if print_plots:
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    plt.savefig(path_session_data+'\\split-belt locomotion analysis\\' + 'stance_speed_miniscopes', dpi=96)

fig, ax = plt.subplots(figsize=(5, 10), tight_layout=True)
rectangle = plt.Rectangle((3 + 0.5, -0.4), 10, 0.3, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for p in range(4):
    ax.plot(np.linspace(1, len(stance_speed_control_values[p, 0, :]), len(stance_speed_control_values[p, 0, :])),
            np.nanmean(stance_speed_control_values[p, :, :], axis=0), color=paw_colors[p],
               linewidth=2)
    ax.fill_between(np.linspace(1, len(stance_speed_control_values[p, 0, :]), len(stance_speed_control_values[p, 0, :])),
            np.nanmean(stance_speed_control_values[p, :, :], axis=0)-(np.nanstd(stance_speed_control_values[p, :, :], axis=0)/np.sqrt(np.shape(stance_speed_control_values)[1])),
    np.nanmean(stance_speed_control_values[p, :, :], axis=0)+(np.nanstd(stance_speed_control_values[p, :, :], axis=0)/np.sqrt(np.shape(stance_speed_control_values)[1])), color=paw_colors[p],
               alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel('Stance speed', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
if print_plots:
    if not os.path.exists(path_save):
        os.mkdir(path_save)
    plt.savefig(path_session_data+'\\split-belt locomotion analysis\\' + 'stance_speed_control', dpi=96)
