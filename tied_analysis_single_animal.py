#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:26:04 2020

@author: anagoncalves
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# path inputs
path_loco = 'J:\\Miniscope processed files\\Non-miniscope locomotion data\\Treadmill tied\\'
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
speed_range = np.arange(0.1, 0.4, 0.05)
speed_bins = np.array([1, 2, 3])  # 0.15 0.2 0.25
speed_bin_side = 2
frames_dFF = 0  # black frames removed before ROI segmentation

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import locomotion_class

loco = locomotion_class.loco_class(path_loco)
if not os.path.exists(path_loco + 'gait parameters'):
    os.mkdir(path_loco + 'gait parameters')

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

count_animal = 0
for animal in animal_list:
    session = int(session_list[count_animal])
    path_save = path_loco + 'gait parameters\\' + animal + ' session ' + str(session) + '\\'
    # get h5 list of files
    filelist = loco.get_track_files(animal, session)
    tied_trials = loco.trials_ordered(filelist)
    exclude_bad_strides = 1
    axis = 'X'
    final_tracks_trials = []
    bodycenter_trials = []
    tracks_tail_trials = []
    joints_elbow_trials = []
    joints_wrist_trials = []
    st_strides_trials = []
    sw_pts_trials = []
    paws_rel_X_trials = []
    paws_rel_Y_trials = []
    paws_rel_Z_trials = []
    print('Getting trial information for ' + animal + ' session ' + session_list[count_animal])
    for f in filelist:
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF)
        final_tracks_trials.append(final_tracks)
        bodycenter_trials.append(bodycenter)
        tracks_tail_trials.append(tracks_tail)
        joints_elbow_trials.append(joints_elbow)
        joints_wrist_trials.append(joints_wrist)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, exclude_bad_strides)
        st_strides_trials.append(st_strides_mat)
        sw_pts_trials.append(sw_pts_mat)
        paws_rel_X_trials.append(loco.get_paws_rel(final_tracks, 'X'))
        paws_rel_Y_trials.append(loco.get_paws_rel(final_tracks, 'Y'))
        paws_rel_Z_trials.append(loco.get_paws_rel(final_tracks, 'Z'))

    # speed indices for all paws
    paws = ['FR', 'HR', 'FL', 'HL']
    stride_idx_bins_paws = []
    speed_stride_bins_paws = []
    for p in paws:
        stride_idx_bins_trials = []
        speed_stride_bins_trials = []
        t = 0
        for f in filelist:
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF)
            speed_L = float(f.split('_')[5].replace(',', '.'))
            speed_stride = loco.get_stride_speed(speed_L, final_tracks, st_strides_trials[t])
            strides = loco.get_stride_trajectories(final_tracks, st_strides_trials[t], p, 'X', 0, 75)
            [stride_idx_bins, param_mat_bins, stride_trajectory_bins] = loco.bin_strides(speed_stride, strides, p,
                                                                                         speed_range)
            stride_idx_bins_trials.append(stride_idx_bins)
            speed_stride_bins_trials.append(param_mat_bins)
            t += 1
        stride_idx_bins_paws.append(stride_idx_bins_trials)
        speed_stride_bins_paws.append(speed_stride_bins_trials)

    if len(stride_idx_bins_paws[0]) == 0 and len(stride_idx_bins_paws[1]) == 0 and len(stride_idx_bins_paws[2]) == 0:
        print('Bad session, no strides')
        count_animal += 1

    else:
        # intralimb parameter distribution
        print('Intralimb parameters for ' + animal + ' session ' + session_list[count_animal])
        param_tied = ['stance_duration', 'swing_duration', 'cadence', 'swing_length']
        fig, ax = plt.subplots(3, 2, figsize=(20, 20), tight_layout=True)
        ax = ax.ravel()
        count_p = 0
        for g in param_tied:
            param_paw = []
            speed_paw = []
            for t in range(len(tied_trials)):
                param_mat = loco.compute_gait_param(bodycenter_trials[t], final_tracks_trials[t], paws_rel_X_trials[t],
                                                    st_strides_trials[t], sw_pts_trials[t], g)
                for b in range(len(speed_range) - 1):
                    if len(stride_idx_bins_paws[0][t][b]) > 0:
                        param_paw.extend(param_mat[0][stride_idx_bins_paws[0][t][b]])  # do for FR paw
                        speed_paw.extend(np.repeat(speed_range[b], len(param_mat[0][stride_idx_bins_paws[0][t][b]])))
            param_events = {'values': param_paw, 'speed': speed_paw}
            df = pd.DataFrame(param_events)
            lplot = sns.lineplot(x=df['speed'], y=df['values'], ax=ax[count_p], color='black')
            ax[count_p].set_xlabel('Speed')
            ax[count_p].set_title(g.replace('_', ' ') + ' FR paw')
            ax[count_p].set_ylabel(g.replace('_', ' '))
            ax[count_p].spines['right'].set_visible(False)
            ax[count_p].spines['top'].set_visible(False)
            count_p += 1
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            np.save(path_save + g, df)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'param_intralimb' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)

        # plot phase in reference to FR
        print('Stance phasing for ' + animal + ' session ' + session_list[count_animal])
        from scipy.stats import circmean

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        phase_bin_paw = np.zeros((4, len(speed_range) - 1))
        for p in range(4):
            param_paw = []
            speed_paw = []
            for t in range(len(tied_trials)):
                param_mat = loco.compute_gait_param(bodycenter_trials[t], final_tracks_trials[t], paws_rel_X_trials[t],
                                                    st_strides_trials[t], sw_pts_trials[t], 'phase_st')
                for b in range(len(speed_range) - 1):
                    if len(stride_idx_bins_paws[0][t][b]) > 0:
                        param_paw.extend(param_mat[0][p][stride_idx_bins_paws[0][t][b]])  # do for FR paw
                        speed_paw.extend(np.repeat(speed_range[b], len(param_mat[0][p][stride_idx_bins_paws[0][t][b]])))
            phase_bin = np.zeros(len(speed_range) - 1)
            for b in range(len(speed_range) - 1):
                param_idx = np.where(speed_paw == speed_range[b])[0]
                param_array = np.array(param_paw)
                phase_bin[b] = circmean(param_array[param_idx.astype(int)], nan_policy='omit')
            phase_bin_paw[p, :] = phase_bin
            ax.scatter(phase_bin, speed_range[:-1], c=paw_colors[p], s=20)
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        np.save(path_save + 'phase_st', phase_bin_paw)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'phase_stance' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)

        # plot trajectories aligned to FR paw
        print('Paw trajectories for ' + animal + ' session ' + session_list[count_animal])
        stride_pts = 100
        param_traj = ['swing_inst_vel', 'stride_nose_yrel', 'stride_nose_zrel', 'tail_y_relbase', 'tail_z_relbase',
                      'swing_z', 'swing_z_elbow', 'swing_z_wrist']
        fig, ax = plt.subplots(3, 3, figsize=(20, 20), tight_layout=True)
        ax = ax.ravel()
        count_p = 0
        for p_traj in param_traj:
            param_bins_mean = np.zeros((len(speed_bins), stride_pts))
            b_count = 0
            for b in speed_bins:
                param_paw = []
                for t in range(len(tied_trials)):
                    param_mat = loco.compute_trajectories(paws_rel_X_trials[t], bodycenter_trials[t],
                                                          final_tracks_trials[t], joints_elbow_trials[t],
                                                          joints_wrist_trials[t], tracks_tail_trials[t],
                                                          st_strides_trials[t], sw_pts_trials[t], p_traj)
                    if len(stride_idx_bins_paws[0][t][b]) > 0:
                        if p_traj == 'tail_y_relbase' or p_traj == 'tail_z_relbase':
                            param_paw.extend(param_mat[0][stride_idx_bins_paws[0][t][b][:-1] - 1, 7, :])
                        else:
                            param_paw.extend(param_mat[0][stride_idx_bins_paws[0][t][b][:-1] - 1])  # do for FR paw
                param_paw_vstack = np.vstack(param_paw)
                param_bins_mean[b_count, :] = np.nanmean(param_paw_vstack, axis=0)
                ax[count_p].plot(np.linspace(0, 100, stride_pts), param_bins_mean[b_count, :],
                                 linewidth=speed_bins[b_count] / 2, color='black')
                if count_p > 1:
                    ax[count_p].set_xlabel('% stride')
                else:
                    ax[count_p].set_xlabel('% swing')
                ax[count_p].set_title(p_traj.replace('_', ' ') + ' FR paw')
                ax[count_p].set_ylabel(p_traj.replace('_', ' '))
                ax[count_p].spines['right'].set_visible(False)
                ax[count_p].spines['top'].set_visible(False)
                b_count += 1
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            np.save(path_save + p_traj, param_bins_mean)
            count_p += 1
        # traj limb side
        print('Side view trajectories for ' + animal + ' session ' + session_list[count_animal])
        param_limb_side = ['swing_z', 'swing_z_wrist', 'swing_z_elbow', 'swing_z_pos', 'swing_z_wrist_pos',
                           'swing_z_elbow_pos']
        count_p = 0
        params_mean = np.zeros((len(param_limb_side), stride_pts))
        for p_traj in param_limb_side:
            param_paw = []
            for t in range(len(tied_trials)):
                param_mat = loco.compute_trajectories(paws_rel_X_trials[t], bodycenter_trials[t],
                                                      final_tracks_trials[t], joints_elbow_trials[t],
                                                      joints_wrist_trials[t], tracks_tail_trials[t],
                                                      st_strides_trials[t], sw_pts_trials[t], p_traj)
                if len(stride_idx_bins_paws[0][t][speed_bin_side]) > 0:
                    param_paw.extend(param_mat[0][stride_idx_bins_paws[0][t][speed_bin_side][:-1] - 1])  # do for FR paw
            param_paw_vstack = np.vstack(param_paw)
            params_mean[count_p, :] = np.nanmean(param_paw_vstack, axis=0)
            count_p += 1
        traj_pos = np.dstack((params_mean[3, :], params_mean[4, :], params_mean[5, :]))
        traj_z = np.dstack((params_mean[0, :], params_mean[1, :], params_mean[2, :]))
        ax[8].plot(np.transpose(traj_pos[0, :, :]), np.transpose(traj_z[0, :, :]), marker='o', color='black', alpha=0.5)
        ax[8].set_xlabel('x excursion relative to body')
        ax[8].set_ylabel('z amplitude')
        ax[8].set_title('speed ' + str(np.round(speed_range[speed_bin_side], decimals=2)))
        ax[8].spines['right'].set_visible(False)
        ax[8].spines['top'].set_visible(False)
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        np.save(path_save + 'xside_excursion', traj_pos)
        np.save(path_save + 'z_excursion', traj_z)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'trajectories_FR_paw' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)

        # base of support
        print('Base of support for ' + animal + ' session ' + session_list[count_animal])
        fig, ax = plt.subplots(1, 2, figsize=(10, 10), tight_layout=True)
        ax = ax.ravel()
        swing_x_rel = np.zeros((4, len(speed_bins), stride_pts))
        swing_y_rel = np.zeros((4, len(speed_bins), stride_pts))
        for p in range(4):
            param_bins_x_mean = np.zeros((len(speed_bins), stride_pts))
            param_bins_y_mean = np.zeros((len(speed_bins), stride_pts))
            b_count = 0
            for b in speed_bins:
                param_x_paw = []
                param_y_paw = []
                for t in range(len(tied_trials)):
                    param_x_mat = loco.compute_trajectories(paws_rel_X_trials[t], bodycenter_trials[t],
                                                            final_tracks_trials[t], joints_elbow_trials[t],
                                                            joints_wrist_trials[t], tracks_tail_trials[t],
                                                            st_strides_trials[t], sw_pts_trials[t], 'swing_x_rel')
                    param_y_mat = loco.compute_trajectories(paws_rel_Y_trials[t], bodycenter_trials[t],
                                                            final_tracks_trials[t], joints_elbow_trials[t],
                                                            joints_wrist_trials[t], tracks_tail_trials[t],
                                                            st_strides_trials[t], sw_pts_trials[t], 'swing_y_rel')
                    if len(stride_idx_bins_trials[t][b]) > 0:
                        param_x_paw.extend(param_x_mat[p][stride_idx_bins_paws[p][t][b][:-1] - 1])
                        param_y_paw.extend(param_y_mat[p][stride_idx_bins_paws[p][t][b][:-1] - 1])
                param_paw_x_vstack = np.vstack(param_x_paw)
                param_bins_x_mean[b_count, :] = np.nanmean(param_paw_x_vstack, axis=0)
                param_paw_y_vstack = np.vstack(param_y_paw)
                param_bins_y_mean[b_count, :] = np.nanmean(param_paw_y_vstack, axis=0)
                ax[0].plot(param_bins_y_mean[b_count, :], param_bins_x_mean[b_count, :],
                           linewidth=speed_bins[b_count] / 2, color='black')
                ax[0].set_title('base of support')
                ax[0].set_ylabel('swing x rel')
                ax[0].set_xlabel('swing y rel')
                ax[0].spines['right'].set_visible(False)
                ax[0].spines['top'].set_visible(False)
                b_count += 1
            swing_x_rel[p, :, :] = param_bins_x_mean
            swing_y_rel[p, :, :] = param_bins_y_mean
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        np.save(path_save + 'swing_x_rel', swing_x_rel)
        np.save(path_save + 'swing_y_rel', swing_y_rel)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'bos_supports' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)
        count_animal += 1
        plt.close('all')