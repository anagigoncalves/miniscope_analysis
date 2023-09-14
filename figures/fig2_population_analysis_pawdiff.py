# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_tied_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA bodyvars\\tied baseline S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
protocol_type = 'tied'
if protocol_type == 'tied':
    cond_name = ['slow', 'baseline', 'fast']
    colors_cond = ['purple', 'black', 'orange']
if protocol_type == 'split':
    cond_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
    colors_cond = ['black', (0.403921568627451, 0.0, 0.05098039215686274, 1.0),
                   (0.9896613190730839, 0.7597147950089126, 0.6663101604278074, 1.0),
                   (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0),
                   (0.7935828877005348, 0.8702317290552584, 0.9429590017825312, 1.0)]
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.5, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]
idx_minus100 = np.where(xaxis == -0.1)[0][0] - xaxis_start
idx_0 = np.where(xaxis == 0)[0][0] - xaxis_start
animal_order = ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
var_name = 'Body acceleration'

sta_zoom_all = []
animal_list = []
sta_animal_transition = []
sta_animal_minus100 = []
sta_animal_0 = []
sta_animal_colors_cluster = []
for f in animal_order:
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
    [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
        mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

    sta_zs = np.load(
        os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'))

    #TODO ROIS SHOULD BE ORDERED BY CLUSTER
    if protocol_type == 'tied':
        sta_zs_zoom = np.zeros((np.shape(sta_zs)[0], len(cond_name), xaxis_end-xaxis_start))
        sta_zs_zoom[:] = np.nan
        for count_c, c in enumerate(trials_ses_name):
            if trials_ses_name[count_c] == 'baseline speed':
                bs_idx = trials_ses_name.index('baseline speed')
                trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                sta_zs_zoom[:, 1, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
            if trials_ses_name[count_c] == 'slow speed':
                bs_idx = trials_ses_name.index('slow speed')
                trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                sta_zs_zoom[:, 0, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
            if trials_ses_name[count_c] == 'fast speed':
                bs_idx = trials_ses_name.index('fast speed')
                trial_start_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 0])[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses[bs_idx, 1])[0][0]]
                sta_zs_zoom[:, 2, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
    if protocol_type == 'split':
        sta_zs_zoom = np.zeros((np.shape(sta_zs)[0], len(cond_name), xaxis_end-xaxis_start))
        sta_zs_zoom[:] = np.nan
        trials_ses = trials_ses.flatten()[1:]
        for count_c, c in enumerate(trials_ses_name): #if odd is -1, if even is the next
            if count_c % 2 == 0:
                trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c]-1)[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
            if count_c % 2 != 0:
                trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
                trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c]+1)[0][0]]
            sta_zs_zoom[:, count_c, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)
    sta_zoom_all.append(sta_zs_zoom)

sta_zoom_all_concat = np.concatenate(sta_zoom_all)
#ANIMALS SUMMARY
fig, ax = plt.subplots(1, np.shape(sta_zoom_all_concat)[1], figsize=(25, 10), tight_layout='True', sharey=True)
for t in range(np.shape(sta_zoom_all_concat)[1]):
    hm = sns.heatmap(sta_zoom_all_concat[:, t, :], vmax=np.nanpercentile(sta_zoom_all_concat,99.5),
                vmin=np.nanpercentile(sta_zoom_all_concat,0.5), cmap='coolwarm', ax=ax[t])
    ax[t].set_xticks(np.array([0, np.where(xaxis==0)[0][0]-xaxis_start, np.shape(sta_zoom_all_concat)[2]]))
    ax[t].set_xticklabels([str(xaxis[xaxis_start]), '0', str(np.round(xaxis[xaxis_end],2))], fontsize=20)
    ax[t].axvline(x=np.where(xaxis==0)[0][0]-xaxis_start, color='k')
    ax[t].set_ylabel('   '.join(animal_list[::-1]), fontsize=18)
    ax[t].set(yticklabels=[])
    ax[t].tick_params(left=False)
    ax[t].set_xlabel('Time around event (s)', fontsize=20)
    ax[t].tick_params(axis='both', which='major', labelsize=16)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    for a in np.cumsum(sta_animal_transition)[:-1]:
        ax[t].axhline(y=a, c='k', linestyle='--')
    ax[t].set_title(cond_name[t], fontsize=16)
# plt.savefig(os.path.join(save_path,
#                          'sta_bodyvars_' + var_name.replace(' ', '_') + '_animal_summary'), dpi=mscope.my_dpi)