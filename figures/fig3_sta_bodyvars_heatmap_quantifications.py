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
sort_type = 'none'
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-1, 0.5])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
#for quantification of peaks and throughs
xaxis_new_0 = np.where(xaxis == 0)[0][0]
xaxis_new = xaxis[xaxis_start:xaxis_new_0]
xaxis_new_minus_250ms = np.argmin(np.abs(np.abs(xaxis_new) - 0.25))
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.48, 1.5],
                     [6.64, 1],
                     [6.48, 1.5]]) #AP, ML
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
        roi_coordinates = []
        rois_val_all = []
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

        sta_zs = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '_zscored.npy'))

        sta = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '.npy'))

        sta_shuffled = np.load(
            os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var.replace(' ', '_') + '_shuffled.npy'))

        # Get rois global coordinates
        centroid_ext = mscope.get_roi_centroids(coord_ext)
        centroid_ext_swap = np.array(centroid_ext)[:, [1, 0]]
        fov_coord = fov_coords[count_f]
        fov_corner = np.array([fov_coord[0] - 0.5, fov_coord[1] - 0.5])
        centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner

        #do trial average
        sta_zs_zoom = np.nanmean(sta_zs[:, :, xaxis_start:xaxis_end], axis=1)
        sta_zoom = np.nanmean(sta[:, :, xaxis_start:xaxis_end], axis=1)
        sta_zoom_shuffled = np.nanmean(sta_shuffled[:, :, xaxis_start:xaxis_end],
                                       axis=1)
        # save also cluster, animal id, AP and ML global coordinates
        sta_zoom_all.append(sta_zs_zoom)
        sta_zoom_all_notzscore.append(sta_zoom)
        sta_zoom_all_shuffled.append(sta_zoom_shuffled)
        sta_animal_id.append(animal)
        sta_cluster_size.append(np.shape(sta_zs_zoom)[0])
        sta_ap.extend(centroid_dist_corner[:, 0])
        sta_ml.extend(centroid_dist_corner[:, 1])
        if var == 'Body acceleration':
            roi_coordinates.extend(centroid_dist_corner)
            # Get average values in 250ms before CS to then quantify peaks and throughs
            sta_zs_zoom_250mswindow = sta_zs_zoom[:, xaxis_new_minus_250ms:xaxis_new_0 - xaxis_start]
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
            rois_val = np.zeros(np.shape(sta_zs_zoom_250mswindow)[0])
            rois_val[rois_pos] = 1
            rois_val[rois_neg] = -1
            rois_pos_all.extend(rois_pos)
            rois_neg_all.extend(rois_neg)
            rois_neutral_all.extend(rois_neutral)
            rois_val_all.extend(rois_val)
    #sort ML or AP values
    sort_ml = np.argsort(sta_ml)
    sort_ap = np.argsort(sta_ap)
    #concatenate ROIs across animals
    sta_zoom_all_concat = np.concatenate(sta_zoom_all)
    sta_zoom_all_concat_notzscored = np.concatenate(sta_zoom_all_notzscore)
    sta_zoom_all_concat_shuffled = np.concatenate(sta_zoom_all_shuffled)
    if sort_type == 'ML':
        sta_zs_zoom_all_sort = []
        sta_zoom_all_sort = []
        sta_shuffled_zoom_all_sort = []
        for i in sort_ml:
            sta_zs_zoom_all_sort.append(sta_zoom_all_concat[i])
            sta_zoom_all_sort.append(sta_zoom_all_concat_notzscored[i])
            sta_shuffled_zoom_all_sort.append(sta_zoom_all_concat_shuffled[i])
    if sort_type == 'AP':
        sta_zs_zoom_all_sort = []
        sta_zoom_all_sort = []
        sta_shuffled_zoom_all_sort = []
        for i in sort_ap:
            sta_zs_zoom_all_sort.append(sta_zoom_all_concat[i])
            sta_zoom_all_sort.append(sta_zoom_all_concat_notzscored[i])
            sta_shuffled_zoom_all_sort.append(sta_zoom_all_concat_shuffled[i])
    if sort_type == 'none':
        sta_zs_zoom_all_sort = sta_zoom_all
        sta_zoom_all_sort = []
        for i in range(len(sta_zoom_all_concat_notzscored)):
            sta_zoom_all_sort.append(sta_zoom_all_concat_notzscored[i])
        sta_shuffled_zoom_all_sort = sta_zoom_all_shuffled
    sta_zoom_all_concat_vars.append(sta_zs_zoom_all_sort)
    sta_zoom_all_concat_vars_notzscored.append(sta_zoom_all_sort)
    sta_zoom_all_concat_vars_shuffled.append(sta_shuffled_zoom_all_sort)
    sta_zoom_all_cluster_size.append(np.cumsum(np.array(sta_cluster_size)))
roi_coordinates_arr = np.array(roi_coordinates)

#ANIMALS SUMMARY HEATMAP
for count_v, var in enumerate(var_names):
    fig, ax = plt.subplots(figsize=(5, 10), tight_layout='True')
    hm = sns.heatmap(sta_zoom_all_concat_vars_notzscored[count_v], vmax=np.nanpercentile(sta_zoom_all_concat_vars_notzscored[count_v], 99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat_vars_notzscored[count_v], 0.5), cmap='coolwarm')
    ax.set_xticks(np.array([0, np.where(xaxis == 0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat_vars_notzscored[count_v])[1]]))
    ax.set_xticklabels([str(np.round(xaxis[xaxis_start], 2)), '0', str(np.round(xaxis[xaxis_end], 2))], fontsize=20)
    ax.axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='white', linewidth=2)
    yticks_plot = np.arange(0, np.shape(sta_zoom_all_concat_vars[count_v])[0], 50)
    ax.set_yticks(yticks_plot)
    ax.set_xlabel('Time around event (s)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if sort_type == 'ML':
        sta_ml_sorted = np.round(np.sort(sta_ml), 2)
        sta_ml_plot = np.linspace(sta_ml_sorted[0], sta_ml_sorted[-1], num=len(yticks_plot))
        ax.set_yticklabels(list(map(str, np.round(sta_ml_plot, 2))), fontsize=12, rotation=45)
    if sort_type == 'AP':
        ax.set_yticklabels(list(map(str, np.round(np.sort(sta_ap), 2))), fontsize=12, rotation=45)
    if sort_type == 'none':
        # ax.set_ylabel('   '.join(sta_animal_id[::-1]), fontsize=12)
        ax.set_ylabel('ROI', fontsize=20)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # ax.set_title(var, fontsize=16)
    plt.savefig(os.path.join(save_path,
                             'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_animal_summary_notzscored_sort_'+sort_type+'_'+var), dpi=mscope.my_dpi)

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
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_acc_zscored_quantification'), dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path,
                         'sta_bodyvars_' + load_path.split('\\')[-2].replace(' ','_') + '_acc_zscored_quantification.svg'), dpi=mscope.my_dpi)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(roi_coordinates_arr[:, 1], roi_coordinates_arr[:, 0], s=5, c=rois_val_all, cmap='coolwarm')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.gca().invert_yaxis()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
cbar = plt.colorbar(sc)
cbar.ax.tick_params(labelsize=20)
plt.savefig(os.path.join(save_path, 'sta_bodyvars_bodyacc_max_250_0_roilocation'), dpi=256)
plt.savefig(os.path.join(save_path, 'sta_bodyvars_bodyacc_max_250_0_roilocation.svg'), dpi=256)