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

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_ipsi.xlsx') # animals that did split right fast in S1
if not os.path.exists(path_session_data+'\\split-belt locomotion analysis'):
    os.mkdir(path_session_data+'\\split-belt locomotion analysis')

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

    param_sym_all = np.zeros((len(param_sym_name), 26))
    param_sym_all[:] = np.nan
    filelist = loco.get_track_files(animal, session)
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF[count_trial])
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat,
                                                param)
            if animal=='MC8855':
                param_sym_all[count_p, count_trial+3] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
            else:
                param_sym_all[count_p, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
    param_sym_bs_all = param_sym_all[1] - np.nanmean(param_sym_all[1, :6])
    np.save(path_save + param_sym_name[1]+'_all', param_sym_bs_all)

# Get control data
main_miniscope_path = 'J:\\Miniscope processed files\\split-belt locomotion analysis\\'
folders_animals = [filename for filename in os.listdir(main_miniscope_path) if filename.startswith("M")]
# Group the data
param_split = ['step_length']
param_split_miniscope_values = np.zeros((len(param_split), len(folders_animals), 23))
for count_p, g in enumerate(param_split):
    for count_a, a in enumerate(folders_animals):
        miniscope_path = main_miniscope_path + a
        param_tied_miniscope_file = np.load(miniscope_path + '\\' + param_split[count_p] + '.npy')
        param_split_miniscope_values[count_p, count_a, :] = param_tied_miniscope_file

max_rect = [2]
min_rect = [-9]
p=0
# Plot learning curve
param_split_miniscope_values_mean = np.nanmean(param_split_miniscope_values[p, :, :], axis=0)
param_split_miniscope_values_std = np.nanstd(param_split_miniscope_values[p, :, :], axis=0)/np.sqrt(np.shape(param_split_miniscope_values)[1])
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
rectangle = plt.Rectangle((3 + 0.5, min_rect[p]), 10,
                          max_rect[p] - min_rect[p],
                          fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
plt.hlines(0, 1, len(param_split_miniscope_values_mean), colors='grey', linestyles='--')
for count_a in range(np.shape(param_split_miniscope_values)[1]):
    plt.plot(np.linspace(1, len(param_split_miniscope_values_mean), len(param_split_miniscope_values_mean)),
             param_split_miniscope_values[p, count_a, :],
             linewidth=1, color='darkgray')
plt.plot(np.linspace(1, len(param_split_miniscope_values_mean), len(param_split_miniscope_values_mean)), param_split_miniscope_values_mean,
    linewidth=4, color='black')
ax.set_xlabel('Trial', fontsize=24)
ax.set_ylabel('Step length (mm)', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\miniscope_step_length', dpi=256)
#plt.savefig('J:\\Thesis\\figuresChapter2\\miniscope_step_length.svg', format='svg', dpi=256)

fig, ax = plt.subplots(figsize=(7, 3), tight_layout=True)
rectangle = plt.Rectangle((6.5, 0.1), 10, 0.25, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
ax.scatter([1, 2, 3, 4, 5, 6], [0.225, 0.225, 0.225, 0.225, 0.225, 0.225], color='black', s=60)
ax.scatter([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], color='black', s=60)
ax.scatter([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15], color='black', s=60)
ax.scatter([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225, 0.225], color='black', s=60)
ax.set_xlabel('Trial', fontsize=18)
ax.set_ylabel('Belt speed (m/s)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\split_session_protocol', dpi=128)
#plt.savefig('J:\\Thesis\\figuresChapter2\\split_session_protocol.svg', dpi=128)