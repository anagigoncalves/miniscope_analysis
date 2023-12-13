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
session_data = pd.read_excel(path_session_data +'\\session_data_tied_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig sta\\'
protocol_type = 'tied'
sort_type = 'ML'
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.75, 0.5])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
#for quantification of peaks and throughs
xaxis_new_0 = np.where(xaxis == 0)[0][0]
xaxis_new = xaxis[xaxis_start:xaxis_new_0]
xaxis_new_minus_250ms = np.argmin(np.abs(np.abs(xaxis_new) - 0.25))
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
fov_coords = np.array([[6.27, 0.53],
                     [6.61, 0.89],
                     [6.80, 1.75],
                     [6.98, 1.47],
                     [6.39, 1.62]]) #AP, ML
var_names = ['Body position', 'Body speed', 'Body acceleration', 'Body jerk']

sta_zoom_all_concat_vars = []
sta_zoom_all_cluster_size = []
sta_zoom_all_concat_vars_notzscored = []
sta_zoom_all_concat_vars_shuffled = []
for var in var_names:
    sta_zoom_all = []
    sta_zoom_all_notzscore = []
    sta_zoom_all_shuffled = []
    sta_animal_id = []
    sta_cluster_size = []
    sta_ap = []
    sta_ml = []
    if var == 'Body acceleration':
        rois_pos_all = []
        rois_neg_all = []
        rois_neutral_all = []
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

        sta_zs = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '_zscored.npy'))

        sta = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '.npy'))

        sta_shuffled = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '_shuffled.npy'))

        # Get cluster global coordinates
        centroid_ext = mscope.get_roi_centroids(coord_ext)
        fov_coord = fov_coords[count_f]
        cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

        for count_c in range(len(clusters_rois)): #0 are ROIs that don't overlap with reference session
            # do trial average
            sta_zs_zoom = np.nanmean(sta_zs[idx_roi_cluster_ordered == count_c+1, :, xaxis_start:xaxis_end], axis=1)
            sta_zoom = np.nanmean(sta[idx_roi_cluster_ordered == count_c+1, :, xaxis_start:xaxis_end], axis=1)
            sta_zoom_shuffled = np.nanmean(sta_shuffled[idx_roi_cluster_ordered == count_c + 1, :, xaxis_start:xaxis_end], axis=1)
            #save also cluster, animal id, AP and ML global coordinates
            sta_zoom_all.append(sta_zs_zoom)
            sta_zoom_all_notzscore.append(sta_zoom)
            sta_zoom_all_shuffled.append(sta_zoom_shuffled)
            sta_animal_id.append(animal)
            sta_cluster_size.append(np.shape(sta_zs_zoom)[0])
            sta_ap.append(cluster_coord[count_c, 0])
            sta_ml.append(cluster_coord[count_c, 1])
            if var == 'Body acceleration':
                # Get average values in 250ms before CS to then quantify peaks and throughs
                sta_zs_zoom_250mswindow = sta_zs_zoom[:, xaxis_new_minus_250ms:xaxis_new_0-xaxis_start]
                sta_zs_zoom_250mswindow_max = np.max(sta_zs_zoom_250mswindow, axis=1)
                sta_zs_zoom_250mswindow_min = np.min(sta_zs_zoom_250mswindow, axis=1)
                rois_pos = np.where(sta_zs_zoom_250mswindow_max >= 2)[0]
                rois_neg = np.where(sta_zs_zoom_250mswindow_min <= -2)[0]
                #if there are significant peaks in both choose the largest - ONLY ONCE FOR TIED BASELINE S1
                #MC9226 CLUSTER 1
                if len(list(set(rois_pos).intersection(rois_neg)))>0:
                    for i in list(set(rois_pos).intersection(rois_neg)):
                        if sta_zs_zoom_250mswindow_max[i]>np.abs(sta_zs_zoom_250mswindow_min[i]):
                            rois_neg = np.delete(rois_neg, np.where(rois_neg==i)[0][0])
                        if sta_zs_zoom_250mswindow_max[i] < np.abs(sta_zs_zoom_250mswindow_min[i]):
                            rois_pos = np.delete(rois_pos, np.where(rois_pos==i)[0][0])
                rois_neutral = np.setdiff1d(np.arange(0, np.shape(sta_zs_zoom)[0]), np.concatenate((rois_pos, rois_neg)))
                rois_pos_all.extend(rois_pos)
                rois_neg_all.extend(rois_neg)
                rois_neutral_all.extend(rois_neutral)
    sort_ml = np.argsort(sta_ml)
    sort_ap = np.argsort(sta_ap)
    if sort_type == 'ML':
        sta_zs_zoom_all_sort = []
        sta_zoom_all_sort = []
        sta_shuffled_zoom_all_sort = []
        for i in sort_ml:
            sta_zs_zoom_all_sort.append(sta_zoom_all[i])
            sta_zoom_all_sort.append(sta_zoom_all_notzscore[i])
            sta_shuffled_zoom_all_sort.append(sta_zoom_all_shuffled[i])
    if sort_type == 'AP':
        sta_zs_zoom_all_sort = []
        sta_zoom_all_sort = []
        sta_shuffled_zoom_all_sort = []
        for i in sort_ap:
            sta_zs_zoom_all_sort.append(sta_zoom_all[i])
            sta_zoom_all_sort.append(sta_zoom_all_notzscore[i])
            sta_shuffled_zoom_all_sort.append(sta_zoom_all_shuffled[i])
    if sort_type == 'none':
        sta_zs_zoom_all_sort = sta_zoom_all
        sta_zoom_all_sort = sta_zoom_all_notzscore
        sta_shuffled_zoom_all_sort = sta_zoom_all_shuffled
    sta_zoom_all_concat = np.concatenate(sta_zs_zoom_all_sort)
    sta_zoom_all_concat_notzscored = np.concatenate(sta_zoom_all_sort)
    sta_zoom_all_concat_shuffled = np.concatenate(sta_shuffled_zoom_all_sort)
    sta_zoom_all_concat_vars.append(sta_zoom_all_concat)
    sta_zoom_all_concat_vars_notzscored.append(sta_zoom_all_concat_notzscored)
    sta_zoom_all_concat_vars_shuffled.append(sta_zoom_all_concat_shuffled)
    sta_zoom_all_cluster_size.append(np.cumsum(np.array(sta_cluster_size)))

#ANIMALS SUMMARY HEATMAP
for count_v, var in enumerate(var_names):
    fig, ax = plt.subplots(figsize=(5, 10), tight_layout='True')
    hm = sns.heatmap(sta_zoom_all_concat_vars_notzscored[count_v], vmax=np.nanpercentile(sta_zoom_all_concat_vars_notzscored[count_v], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_vars_notzscored[count_v], 0.5), cmap='coolwarm')
    ax.set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars[count_v])[1]]))
    ax.set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax.axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax.set_yticks(sta_zoom_all_cluster_size[count_v])
    ax.set_xlabel('Time around event (s)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax.set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax.set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax.set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
# ax.set_title(var, fontsize=16)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_notzscored_sort_'+sort_type+'_'+var), dpi=mscope.my_dpi)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_notzscored_sort_'+sort_type+'_'+var+'.svg), dpi=mscope.my_dpi)

#ROIS SUMMARY PEAKS AND THROUGHS PIE CHART + SPATIAL MAP OF ROIS AND SIG INCREASES
labels = ['Significant\nincreases\n>2 STD', '\nSignificant\ndecreases\n<2 STD', '']
sizes = [len(rois_pos_all), len(rois_neg_all), len(rois_neutral_all)]
colors = ['firebrick', 'dodgerblue', 'lightgray']
explode = (0.05, 0.05, 0.05)
fig2, ax2 = plt.subplots(figsize=(5, 5), tight_layout=True)
patches, texts, autotexts = ax2.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
for i in range(len(texts)):
    texts[i].set_fontsize(16)
    autotexts[i].set_fontsize(10)
    autotexts[i].set_color('white')
    autotexts[i].set_fontweight('bold')
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig2 = plt.gcf()
fig2.gca().add_artist(centre_circle)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_jerk_zscored_quantification'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, len(var_names), figsize=(20, 10), tight_layout='True')
for count_v, var in enumerate(var_names):
    hm = sns.heatmap(sta_zoom_all_concat_vars[count_v], vmax=np.nanpercentile(sta_zoom_all_concat_vars[count_v], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_vars[count_v], 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax[count_v].set_yticks(sta_zoom_all_cluster_size[count_v])
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[count_v].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[count_v].set_title(var, fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_zscored_sort_'+sort_type), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, len(var_names), figsize=(20, 10), tight_layout='True')
for count_v, var in enumerate(var_names):
    hm = sns.heatmap(sta_zoom_all_concat_vars_shuffled[count_v], vmax=np.nanpercentile(sta_zoom_all_concat_vars_shuffled[count_v], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_vars_shuffled[count_v], 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax[count_v].set_yticks(sta_zoom_all_cluster_size[count_v])
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[count_v].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[count_v].set_title(var, fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_shuffled_sort_'+sort_type), dpi=mscope.my_dpi)

fig, ax = plt.subplots(1, len(var_names), figsize=(20, 10), tight_layout='True')
for count_v, var in enumerate(var_names):
    idx_notsig = np.where((sta_zoom_all_concat_vars[count_v] < 2) & (sta_zoom_all_concat_vars[count_v] > -2))
    sta_zoom_notzscore_sig = sta_zoom_all_concat_vars_notzscored[count_v].copy()
    sta_zoom_notzscore_sig[idx_notsig] = np.nan
    hm = sns.heatmap(sta_zoom_notzscore_sig, vmax=np.nanpercentile(sta_zoom_notzscore_sig, 99.5),
                vmin=np.nanpercentile(sta_zoom_notzscore_sig, 0.5), cmap='coolwarm', ax=ax[count_v])
    ax[count_v].set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars[count_v])[1]]))
    ax[count_v].set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax[count_v].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    ax[count_v].set_yticks(sta_zoom_all_cluster_size[count_v])
    ax[count_v].set_xlabel('Time around event (s)', fontsize=20)
    ax[count_v].tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ml), 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax[count_v].set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        ax[count_v].set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=12)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    ax[count_v].set_title(var, fontsize=16)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_notzscored_sig_sort_'+sort_type), dpi=mscope.my_dpi)