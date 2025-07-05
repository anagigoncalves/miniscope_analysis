# # -*- coding: utf-8 -*-
# # %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\\Head rotation\\split ipsi fast S1\\'
protocol = 'split ipsi fast'

for s in range(len(session_data)):
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
    # Session data and inputs
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
    clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    head_angles = pd.read_csv(os.path.join(mscope.path, 'processed files', 'head_angles.csv'))
    head_angles_corr = mscope.correct_gimbal_lock(head_angles)
    head_angles_arr = np.array(head_angles_corr.iloc[:, :3])
    pca = PCA(n_components=3)
    principalComponents_3CP = pca.fit_transform(head_angles_arr)
    # Plot 2d
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    idx_bs = np.where(head_angles_corr['trial'] < trials_ses[0, 1])[0]
    idx_es = np.where(head_angles_corr['trial'] == trials_ses[1, 0])[0]
    idx_ls = np.where(head_angles_corr['trial'] == trials_ses[1, 1])[0]
    idx_ae = np.where(head_angles_corr['trial'] == trials_ses[2, 0])[0]
    plt.scatter(principalComponents_3CP[idx_bs, 0], principalComponents_3CP[idx_bs, 1], s=1, color='black')
    plt.scatter(principalComponents_3CP[idx_es, 0], principalComponents_3CP[idx_es, 1], s=1, color='crimson')
    plt.scatter(principalComponents_3CP[idx_ls, 0], principalComponents_3CP[idx_ls, 1], s=1, color='salmon')
    plt.scatter(principalComponents_3CP[idx_ae, 0], principalComponents_3CP[idx_ae, 1], s=1, color='dodgerblue')
    ax.set_title('First 2 PCs\nexplained variance of ' + str(
        np.round(np.cumsum(pca.explained_variance_ratio_)[1], decimals=3)), fontsize=20)
    ax.set_xlabel('PC component 1', fontsize=16)
    ax.set_ylabel('PC component 2', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, 'pca_headmovements_' + protocol.replace(' ', '_') + '_' + animal),
                dpi=mscope.my_dpi)
    plt.close('all')


