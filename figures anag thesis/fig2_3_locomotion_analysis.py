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
print_plots = 0
paw_colors = ['#e52c27', '#3854a4', '#ad4397', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

speed_range = np.arange(0.1, 0.4, 0.05)
speed_bins = np.array([1, 2, 3]) #0.15 0.2 0.25
speed_bin_side = 2

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
if not os.path.exists(path_session_data+'\\tied belt locomotion analysis'):
    os.mkdir(path_session_data+'\\tied belt locomotion analysis')
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
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    path_save = path_session_data+'\\tied belt locomotion analysis\\' + animal + ' session ' + str(session) + '\\'

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
    print('Getting trial information for ' + animal + ' session ' + str(session))
    for t, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF[t])
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, exclude_bad_strides)
        st_strides_trials.append(st_strides_mat)
        sw_pts_trials.append(sw_pts_mat)
        final_tracks_trials.append(final_tracks)
        bodycenter_trials.append(loco.compute_bodycenter(final_tracks, 'X'))
        tracks_tail_trials.append(tracks_tail)
        joints_elbow_trials.append(joints_elbow)
        joints_wrist_trials.append(joints_wrist)
        paws_rel_X_trials.append(loco.get_paws_rel(final_tracks, 'X'))
        paws_rel_Y_trials.append(loco.get_paws_rel(final_tracks, 'Y'))
        paws_rel_Z_trials.append(loco.get_paws_rel(final_tracks, 'Z'))

    # speed indices for all paws
    stride_idx_bins_paws = []
    speed_stride_bins_paws = []
    for p in paws:
        stride_idx_bins_trials = []
        speed_stride_bins_trials = []
        for t, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF[t])
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, exclude_bad_strides)
            speed_L = float(f.split('_')[5].replace(',', '.'))
            speed_stride = loco.get_stride_speed(speed_L, final_tracks, st_strides_mat)
            strides = loco.get_stride_trajectories(final_tracks, st_strides_mat, p, 'X', 0, 75)
            [stride_idx_bins, param_mat_bins, stride_trajectory_bins] = loco.bin_strides(speed_stride, strides, p,
                                                                                         speed_range)
            stride_idx_bins_trials.append(stride_idx_bins)
            speed_stride_bins_trials.append(param_mat_bins)
        stride_idx_bins_paws.append(stride_idx_bins_trials)
        speed_stride_bins_paws.append(speed_stride_bins_trials)

    if len(stride_idx_bins_paws[0]) == 0 and len(stride_idx_bins_paws[1]) == 0 and len(stride_idx_bins_paws[2]) == 0:
        print('Bad session, no strides')

    else:
        # intralimb parameter distribution
        print('Intralimb parameters for ' + animal + ' session '  + str(session))
        param_tied = ['stance_duration', 'swing_duration', 'cadence', 'swing_length', 'coo', 'double_support']
        fig, ax = plt.subplots(3, 2, figsize=(20, 20), tight_layout=True)
        ax = ax.ravel()
        for count_p, g in enumerate(param_tied):
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
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            np.save(path_save + g, df)
            df_values.extend(param_paw)
            df_speed.extend(speed_paw)
            df_param.extend(np.repeat(g, len(param_paw)))
            df_animal.extend(np.repeat(animal, len(param_paw)))
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'param_intralimb' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)

        # plot phase in reference to FR
        print('Stance phasing for ' + animal + ' session ' + str(session))
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
        phase_bin_paw_all.append(phase_bin_paw)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'phase_stance' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)

        # plot trajectories aligned to FR paw
        print('Paw trajectories for ' + animal + ' session '  + str(session))
        stride_pts = 100
        param_traj = ['swing_inst_vel', 'stride_nose_yrel', 'stride_nose_zrel', 'tail_y_relbase', 'tail_z_relbase',
                      'swing_z', 'swing_z_elbow', 'swing_z_wrist']
        fig, ax = plt.subplots(3, 3, figsize=(20, 20), tight_layout=True)
        ax = ax.ravel()
        for count_p, p_traj in enumerate(param_traj):
            param_bins_mean = np.zeros((len(speed_bins), stride_pts))
            for b_count, b in enumerate(speed_bins):
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
            if count_p == 0:
                swing_inst_vel_all.append(param_bins_mean)
            if count_p == 5:
                swing_z_all.append(param_bins_mean)
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            np.save(path_save + p_traj, param_bins_mean)
        # traj limb side
        print('Side view trajectories for ' + animal + ' session '  + str(session))
        param_limb_side = ['swing_z', 'swing_z_wrist', 'swing_z_elbow', 'swing_z_pos', 'swing_z_wrist_pos',
                           'swing_z_elbow_pos']
        params_mean = np.zeros((len(param_limb_side), stride_pts))
        for count_p, p_traj in enumerate(param_limb_side):
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
        print('Base of support for ' + animal + ' session '  + str(session))
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        swing_x_rel = np.zeros((4, len(speed_bins), stride_pts))
        swing_y_rel = np.zeros((4, len(speed_bins), stride_pts))
        for p in range(4):
            param_bins_x_mean = np.zeros((len(speed_bins), stride_pts))
            param_bins_y_mean = np.zeros((len(speed_bins), stride_pts))
            for b_count, b in enumerate(speed_bins):
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
                ax.plot(param_bins_y_mean[b_count, :], param_bins_x_mean[b_count, :],
                           linewidth=speed_bins[b_count] / 2, color='black')
                ax.set_title('base of support')
                ax.set_ylabel('Swing x rel')
                ax.set_xlabel('Swing y rel')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
            swing_x_rel[p, :, :] = param_bins_x_mean
            swing_y_rel[p, :, :] = param_bins_y_mean
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        np.save(path_save + 'swing_x_rel', swing_x_rel)
        np.save(path_save + 'swing_y_rel', swing_y_rel)
        swing_x_rel_all.append(swing_x_rel)
        swing_y_rel_all.append(swing_y_rel)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + 'bos_supports' + '_' + animal + '_' + str(session), dpi=loco.my_dpi)
    plt.close('all')

main_control_path = 'J:\\Miniscope processed files\\Non-miniscope locomotion data\\\Treadmill tied\\gait parameters\\'
folders_animals = os.listdir(main_control_path)
# Group the data and plot intralimb parameters
df_intralimb = pd.DataFrame(columns=['values', 'speed', 'parameter', 'animal'])
df_intralimb['values'] = df_values
df_intralimb['speed'] = df_speed
df_intralimb['parameter'] = df_param
df_intralimb['animal'] = df_animal
param_tied = ['stance_duration', 'swing_duration', 'cadence', 'swing_length']
param_tied_control_values = np.zeros((len(param_tied), len(folders_animals), len(speed_range)-1))
param_tied_control_speed = np.zeros((len(param_tied), len(folders_animals), len(speed_range)-1))
for count_p, g in enumerate(param_tied):
    for count_a, a in enumerate(folders_animals):
        control_path = main_control_path + a
        param_tied_control_file = np.load(control_path + '\\' + param_tied[count_p] + '.npy')
        param_tied_control_df = pd.DataFrame({'values': param_tied_control_file[:, 0], 'speed': param_tied_control_file[:, 1]})
        param_tied_control_mean = param_tied_control_df.groupby(['speed'])['values'].mean()
        param_tied_control_speed[count_p, count_a, :] = param_tied_control_mean.index
        param_tied_control_values[count_p, count_a, :] = param_tied_control_mean.values
ylabel = ['Stance duration (ms)', 'Swing duration (ms)', 'Cadence ($\mathregular{ms^{-1}}$)', 'Swing length (mm)']
for count_p, g in enumerate(param_tied):
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    df_intralimb_plot = df_intralimb.loc[df_intralimb['parameter'] == g]
    df_intralimb_plot_animal_mean_speed = np.zeros((len(df_intralimb['animal'].unique()), len(speed_range)-1))
    df_intralimb_plot_animal_mean_values = np.zeros((len(df_intralimb['animal'].unique()), len(speed_range) - 1))
    for count_a, a in enumerate(df_intralimb['animal'].unique()):
        df_intralimb_plot_animal = df_intralimb_plot.loc[df_intralimb_plot['animal'] == a]
        df_intralimb_plot_animal_mean = df_intralimb_plot_animal.groupby(['speed'])['values'].mean()
        df_intralimb_plot_animal_mean_speed[count_a, :] = df_intralimb_plot_animal_mean.index
        df_intralimb_plot_animal_mean_values[count_a, :] = df_intralimb_plot_animal_mean.values
    ax.plot(df_intralimb_plot_animal_mean_speed[0, :], np.nanmean(param_tied_control_values[count_p, :, :], axis=0), color='black', linewidth=2)
    ax.fill_between(df_intralimb_plot_animal_mean_speed[0, :],
                     np.nanmean(param_tied_control_values[count_p, :, :], axis=0)-np.nanstd(param_tied_control_values[count_p, :, :], axis=0),
    np.nanmean(param_tied_control_values[count_p, :, :], axis=0)+np.nanstd(param_tied_control_values[count_p, :, :], axis=0), color='black', alpha=0.3)
    ax.plot(df_intralimb_plot_animal_mean_speed[0, :], np.nanmean(df_intralimb_plot_animal_mean_values, axis=0), color='darkviolet', linewidth=2)
    ax.fill_between(df_intralimb_plot_animal_mean_speed[0, :],
                     np.nanmean(df_intralimb_plot_animal_mean_values, axis=0)-np.nanstd(df_intralimb_plot_animal_mean_values, axis=0),
    np.nanmean(df_intralimb_plot_animal_mean_values, axis=0)+np.nanstd(df_intralimb_plot_animal_mean_values, axis=0), color='darkviolet', alpha=0.3)
    ax.set_xlabel('Speed (m/s)', fontsize=20)
    # ax.set_title(g.replace('_', ' ') + ' FR paw', fontsize=20)
    ax.set_ylabel(ylabel[count_p], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_param_interlimb_FR_' + g, dpi=256)
    plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_param_interlimb_FR_' + g + '.svg', dpi=256)
#plt.savefig('J:\\Miniscope processed files\\tied belt locomotion analysis\\param_intralimb.png')

# Plot phases and trajectories
phase_bin_paw_control_mat = np.zeros((len(folders_animals), 4, len(speed_range)-1))
swing_z_control_mat = np.zeros((len(folders_animals), 100))
swing_inst_vel_control_mat = np.zeros((len(folders_animals), 100))
swing_x_rel_control_mat = np.zeros((len(folders_animals), 4, 100))
swing_y_rel_control_mat = np.zeros((len(folders_animals), 4, 100))
for count_a, a in enumerate(folders_animals):
    control_path = main_control_path + a
    phase_st_animal = np.load(control_path + '\\phase_st.npy')
    phase_bin_paw_control_mat[count_a, :, :] = phase_st_animal
    swing_z_animal = np.load(control_path + '\\swing_z.npy')
    swing_z_control_mat[count_a, :] = swing_z_animal[speed_bin_side, :]
    swing_inst_vel_animal = np.load(control_path + '\\swing_inst_vel.npy')
    swing_inst_vel_control_mat[count_a, :] = swing_inst_vel_animal[speed_bin_side, :]
    swing_x_rel_animal = np.load(control_path + '\\swing_x_rel.npy')
    swing_x_rel_control_mat[count_a, :, :] = swing_x_rel_animal[:, speed_bin_side, :]
    swing_y_rel_animal = np.load(control_path + '\\swing_y_rel.npy')
    swing_y_rel_control_mat[count_a, :, :] = swing_y_rel_animal[:, speed_bin_side, :]
phase_bin_paw_all_mat = np.zeros((len(phase_bin_paw_all), 4, len(speed_range)-1))
swing_z_all_mat = np.zeros((len(phase_bin_paw_all), 100))
swing_inst_vel_all_mat = np.zeros((len(phase_bin_paw_all), 100))
swing_x_rel_all_mat = np.zeros((len(phase_bin_paw_all), 4, 100))
swing_y_rel_all_mat = np.zeros((len(phase_bin_paw_all), 4, 100))
for a in range(len(phase_bin_paw_all)):
    phase_bin_paw_all_mat[a, :, :] = phase_bin_paw_all[a]
    swing_z_all_mat[a, :] = swing_z_all[a][speed_bin_side]
    swing_inst_vel_all_mat[a, :] = swing_inst_vel_all[a][speed_bin_side]
    swing_x_rel_all_mat[a, :, ] = swing_x_rel_all[a][:, speed_bin_side, :]
    swing_y_rel_all_mat[a, :, ] = swing_y_rel_all[a][:, speed_bin_side, :]

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
for p in range(4):
    ax.scatter(np.nanmean(phase_bin_paw_all_mat[:, p, :], axis=0), speed_range[:-1], c=paw_colors[p], marker='*', s=120)
    ax.scatter(np.nanmean(phase_bin_paw_control_mat[:, p, :], axis=0), speed_range[:-1], c=paw_colors[p], s=120, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticks(speed_range[:-1:2])
# plt.savefig('J:\\Miniscope processed files\\tied belt locomotion analysis\\phase_st.png')
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_phase_st', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_phase_st.svg', dpi=256)

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.plot(np.linspace(0, 100, 100), np.nanmean(swing_inst_vel_control_mat, axis=0), color='black', linewidth=2)
ax.fill_between(np.linspace(0, 100, 100),
                 np.nanmean(swing_inst_vel_control_mat, axis=0)-np.nanstd(swing_inst_vel_control_mat, axis=0),
                np.nanmean(swing_inst_vel_control_mat, axis=0)+np.nanstd(swing_inst_vel_control_mat, axis=0), color='black', alpha=0.3)
ax.plot(np.linspace(0, 100, 100), np.nanmean(swing_inst_vel_control_mat, axis=0), color='darkviolet', linewidth=2)
ax.fill_between(np.linspace(0, 100, 100),
                 np.nanmean(swing_inst_vel_all_mat, axis=0)-np.nanstd(swing_inst_vel_all_mat, axis=0),
np.nanmean(swing_inst_vel_all_mat, axis=0)+np.nanstd(swing_inst_vel_all_mat, axis=0), color='darkviolet', alpha=0.3)
ax.set_xlabel('% swing (norm)', fontsize=20)
ax.set_ylabel('Swing instantaneous\nvelocity (m/s)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_inst_vel', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_inst_vel.svg', dpi=256)
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.plot(np.linspace(0, 100, 100), np.nanmean(swing_z_control_mat, axis=0), color='black', linewidth=2)
ax.fill_between(np.linspace(0, 100, 100),
                 np.nanmean(swing_z_control_mat, axis=0)-np.nanstd(swing_z_control_mat, axis=0),
np.nanmean(swing_z_control_mat, axis=0)+np.nanstd(swing_z_control_mat, axis=0), color='black', alpha=0.3)
ax.plot(np.linspace(0, 100, 100), np.nanmean(swing_z_all_mat, axis=0), color='darkviolet', linewidth=2)
ax.fill_between(np.linspace(0, 100, 100),
                 np.nanmean(swing_z_all_mat, axis=0)-np.nanstd(swing_z_all_mat, axis=0),
np.nanmean(swing_z_all_mat, axis=0)+np.nanstd(swing_z_all_mat, axis=0), color='darkviolet', alpha=0.3)
ax.set_xlabel('% swing (norm)', fontsize=20)
ax.set_ylabel('swing amplitude (mm)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z.svg', dpi=256)
# quantification of maximum z amplitude
swing_z_max_control = np.nanmax(swing_z_control_mat, axis=1)
swing_z_max_miniscope = np.nanmax(swing_z_all_mat, axis=1)
swing_z_max_df = pd.DataFrame({'max': np.concatenate((swing_z_max_control, swing_z_max_miniscope)),
'group': np.concatenate((np.repeat(0, len(swing_z_max_control)), np.repeat(1, len(swing_z_max_miniscope))))})
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
sns.boxplot(x='group', y='max', data=swing_z_max_df,
            medianprops=dict(color='black'), palette={0: 'darkgrey', 1: 'darkviolet'}, showfliers=False)
ax.set_xticklabels(['Without\nminiscopes', 'With\nminiscopes'])
ax.set_xlabel('')
ax.set_ylabel('Peak swing\namplitude', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylim([3, 4.5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z_max', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z_max.svg', dpi=256)
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for p in range(4):
    ax.plot(np.nanmean(swing_y_rel_control_mat[:, p, :], axis=0), np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0), color='black', linewidth=2)
    ax.fill_between(np.nanmean(swing_y_rel_control_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0) - np.nanstd(swing_x_rel_control_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0) + np.nanstd(swing_x_rel_control_mat[:, p, :], axis=0), color='black',
                       alpha=0.3)
    ax.plot(np.nanmean(swing_y_rel_all_mat[:, p, :], axis=0), np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0), color='darkviolet', linewidth=2)
    ax.fill_between(np.nanmean(swing_y_rel_all_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0) - np.nanstd(swing_x_rel_all_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0) + np.nanstd(swing_x_rel_all_mat[:, p, :], axis=0), color='darkviolet',
                       alpha=0.3)
    ax.set_xlabel('Y relative to bodycenter (mm)', fontsize=20)
    ax.set_ylabel('X relative to bodycenter (mm)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_bos', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_bos.svg', dpi=256)
# plt.savefig('J:\\Miniscope processed files\\tied belt locomotion analysis\\trajectories.png')




