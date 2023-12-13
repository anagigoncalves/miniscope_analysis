# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data +'\\session_data_split_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
protocol_type = 'split'
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.25, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
#for quantification of peaks and throughs
xaxis_new_0 = np.where(xaxis == 0)[0][0]
xaxis_new = xaxis[xaxis_start:xaxis_new_0]
xaxis_new_minus_250ms = np.argmin(np.abs(np.abs(xaxis_new) - 0.25))
animal_order = ['MC10221', 'MC9513', 'MC9226', 'MC8855', 'MC9194']
fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89],
                     [6.80, 1.75],
                     [6.98, 1.47],
                     [6.39, 1.62]]) #AP, ML
var_names = 'Body acceleration'
cond_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']

sta_zoom_all = []
sta_animal_id = []
sta_animal_length = []
for count_f, f in enumerate(animal_order):
    session_data_idx = np.where(session_data['animal'] == f)[0][0]
    ses_info = session_data.iloc[session_data_idx, :]
    date = ses_info[3]
    # path inputs
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    session_type = path.split('\\')[-4].split(' ')[0]
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    # Session data and inputs
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    trials_ses_name.insert(len(trials_ses_name), 'late washout')
    trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]
    # [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
    #     mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

    sta = np.load(
        os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_names.replace(' ', '_') + '.npy'))

    # Get cluster global coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = fov_coords[count_f]
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)
    trials_ses_split = trials_ses.flatten()[1:]
    for count_c in range(len(clusters_rois)):  # 0 are ROIs that don't overlap with reference session
        sta_zoom = np.zeros((len(clusters_rois[count_c]), len(cond_name), xaxis_end - xaxis_start))
        sta_zoom[:] = np.nan
        for count_t, t in enumerate(trials_ses_name):  # if odd is -1, if even is the next
            if count_t % 2 == 0:
                trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t] - 1)[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
            if count_t % 2 != 0:
                trial_start_idx = trials_idx[np.where(trials == trials_ses_split[count_t])[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses_split[count_t] + 1)[0][0]]
            sta_zoom[:, count_t, :] = np.nanmean(
                sta[idx_roi_cluster_ordered == count_c + 1, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end],
                axis=1)
        sta_zoom_all.append(sta_zoom)
    sta_animal_length.append(np.shape(sta)[0])
sta_zoom_all_concat = np.concatenate(sta_zoom_all)

#ANIMALS SUMMARY HEATMAP
fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    hm = sns.heatmap(sta_zoom_all_concat[:, t, :], vmax=np.nanpercentile(sta_zoom_all_concat, 99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat, 0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat[t])[1]]))
    ax[t].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    for line in np.cumsum(sta_animal_length):
        ax[t].axhline(y=line, color='white', linewidth=2)
    ax[t].set_yticks(np.cumsum(sta_animal_length))
    ax[t].set_yticklabels(animal_order)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    ax[t].set_ylabel('Animals', fontsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[t].set_title(cond_name[t], fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary'), dpi=mscope.my_dpi)
