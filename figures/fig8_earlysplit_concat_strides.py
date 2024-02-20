import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig pca\\'
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S1.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split ipsi fast'
align_event = 'st'
align_dimension = 'phase'
bins = np.arange(0, 105, 10)  # 5 deg
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

paws = ['FR', 'HR', 'FL', 'HL']
# for the order ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.64, 1],
                     [6.48, 1.5],
                     [6.48, 1.7]]) #AP, ML

os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

#for s in range(len(session_data)):
animal = 'MC9513'
s = 4
paw = 0

ses_info = session_data.iloc[s, :]
print(ses_info)
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
session_id = session_type + '_' + ses_info[2]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

animal = mscope.get_animal_id()
session = loco.get_session_id()
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
 frames_dFF] = mscope.load_processed_files()
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)

# Load behavioral data
filelist = loco.get_track_files(animal, session)
st_strides_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
        frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    st_strides_trials.append(st_strides_mat)

trial_id_cumulative = []
stride_idx_cumulative = []
for count_t, trial in enumerate(trials):
    t = np.where(trials == trial)[0][0]
    for s in range(np.shape(st_strides_trials[t][paw])[0]):
        if count_t == 0 and s == 0:
            stride_idx_cumulative.append(1)
            trial_id_cumulative.append(trial)
        else:
            stride_idx_cumulative.append(stride_idx_cumulative[-1] + 1)
            trial_id_cumulative.append(trial)
stride_idx_cumulative_arr = np.array(stride_idx_cumulative)
trial_id_cumulative_arr = np.array(trial_id_cumulative)
stride_idx_cumulative_window = stride_idx_cumulative_arr[np.where(trial_id_cumulative_arr>4)[0][0]:np.where(trial_id_cumulative_arr<7)[0][-1]]
trial_id_cumulative_window = trial_id_cumulative_arr[np.where(trial_id_cumulative_arr>4)[0][0]:np.where(trial_id_cumulative_arr<7)[0][-1]]

events_phase_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_events_stride_trial_rois.npy'), allow_pickle=True)
stride_id_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_cumulative_idx_rois.npy'), allow_pickle=True)
trial_id_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_trial_id_rois.npy'), allow_pickle=True)
nr_rois = len(events_phase_animal)
events_mat = np.zeros((nr_rois, 600, len(bins)))
for roi in range(nr_rois):
    data_trial = trial_id_animal[roi][paw][np.where(trial_id_animal[roi][paw]>4)[0][0]:np.where(trial_id_animal[roi][paw]<7)[0][-1]]
    data_stride = stride_id_animal[roi][paw][np.where(trial_id_animal[roi][paw]>4)[0][0]:np.where(trial_id_animal[roi][paw]<7)[0][-1]]
    data_stride_reset = data_stride-data_stride[0]
    data_stride_reset_nonan = data_stride_reset[~np.isnan(data_stride_reset)]
    data_cs = events_phase_animal[roi][paw][np.where(trial_id_animal[roi][paw]>4)[0][0]:np.where(trial_id_animal[roi][paw]<7)[0][-1]]
    data_cs_bin = np.digitize(data_cs, bins)-1
    events_mat[roi, data_stride_reset_nonan, data_cs_bin] = 1
events_mat_reshape = np.reshape(events_mat, ((nr_rois, events_mat.shape[1]*events_mat.shape[2])))

fig, ax = plt.subplots(tight_layout=True)
sns.heatmap(events_mat_reshape)
ax.axvline(np.where(trial_id_cumulative_window==6)[0][0]*len(bins-1))

