# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mp
from scipy import signal as sig
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_tied_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA paws\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.25, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89],
                     [6.80, 1.75],
                     [6.98, 1.47],
                     [6.39, 1.62]]) #AP, ML
sort_type = 'ML'
var_names = ['FR', 'HR', 'FL', 'HL']

sta_zoom_all_concat_vars = []
sta_zoom_all_cluster_size = []
for var in var_names:
    sta_zoom_all = []
    animal_list = []
    sta_animal_id = []
    sta_cluster_size = []
    sta_ap = []
    sta_ml = []
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
        [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                                 frames_dFF)
        [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
        trials_ses_name.insert(len(trials_ses_name), 'late washout')
        trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]
        [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
            mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

        sta_zs = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '.npy'))

        # Get cluster global coordinates - use only overlapping ROIs
        centroid_ext_overlap = mscope.get_roi_centroids(coord_ext[np.where(coord_ext_overlap > 0)[0]])
        fov_coord = fov_coords[count_f]
        cluster_coord = mscope.get_coordinates_cluster(centroid_ext_overlap, fov_coord,
                                                       coord_ext_overlap[np.where(coord_ext_overlap > 0)[0]])
        clusters_in_session_all = np.unique(coord_ext_overlap)
        if len(np.where(clusters_in_session_all == 0)[0]) > 0:
            clusters_in_session = np.delete(clusters_in_session_all, np.where(clusters_in_session_all == 0)[0][0])
        else:
            clusters_in_session = np.delete(clusters_in_session_all, np.where(clusters_in_session_all == 0)[0])

        # SEPARATE BY TRIALS YOU WANT TO PLOT AND THE OVERLAPPING CLUSTERS
        for count_c, c in enumerate(clusters_in_session):  # 0 are ROIs that don't overlap with reference session
            sta_zs_zoom = np.nanmean(sta_zs[coord_ext_overlap == c, :, xaxis_start:xaxis_end], axis=1)
            # save also cluster, animal id, AP and ML global coordinates
            sta_zoom_all.append(sta_zs_zoom)
            sta_animal_id.append(animal)
            sta_cluster_size.append(np.shape(sta_zs_zoom)[0])
            sta_ap.append(cluster_coord[count_c, 0])
            sta_ml.append(cluster_coord[count_c, 1])
    sort_ml = np.argsort(sta_ml)
    sort_ap = np.argsort(sta_ap)
    if sort_type == 'ML':
        sta_zs_zoom_all_sort = []
        for i in sort_ml:
            sta_zs_zoom_all_sort.append(sta_zoom_all[i])
    if sort_type == 'AP':
        sta_zs_zoom_all_sort = []
        for i in sort_ap:
            sta_zs_zoom_all_sort.append(sta_zoom_all[i])
    if sort_type == 'none':
        sta_zs_zoom_all_sort = sta_zoom_all
    sta_zoom_all_concat = np.concatenate(sta_zs_zoom_all_sort)
    sta_zoom_all_concat_vars.append(sta_zoom_all_concat)
    sta_zoom_all_cluster_size.append(np.cumsum(np.array(sta_cluster_size)))

#ANIMALS SUMMARY
fig, ax = plt.subplots(1, 4, figsize=(25, 10), tight_layout='True', sharey=True)
for count_v, var in enumerate(var_names):
    hm = sns.heatmap(sta_zoom_all_concat_vars[count_v], vmax=np.nanpercentile(sta_zoom_all_concat_vars[count_v],99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_vars[count_v], 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis == 0)[0][0]-xaxis_start, color='white')
    ax[count_v].set_yticks(sta_zoom_all_cluster_size[count_v])
    ax[count_v].tick_params(left=False)
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # for a in np.cumsum(sta_zoom_all_cluster_size[count_v])[:-1]:
    #     ax[count_v].axhline(y=a, c='k', linestyle='--')
    if sort_type == 'ML':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[count_v].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=6)
    ax[count_v].set_title(var, fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_paws_animal_summary_sort_' + sort_type), dpi = mscope.my_dpi)