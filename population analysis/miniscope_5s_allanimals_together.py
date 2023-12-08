# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
fsize = 24

plot_protocol = 'split ipsi fast'
plot_session = 'S1'

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'E:\\Miniscope processed files'
session_data = pd.read_excel('E:\\Miniscope processed files\\session_data_split_S1.xlsx')
mean_data_animals = []
animal_in = []
for s in range(len(session_data)):
    ses_info = session_data.iloc[s, :]
    if ses_info['protocol'] == plot_protocol and ses_info['session'] == plot_session:
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
        [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
         clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
        [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)

        # Order ROIs by cluster
        if len(clusters_rois) == 1:
            clusters_rois_flat = clusters_rois[0]
        else:
            clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])-1
        df_events_extract_zscore_clustered = df_events_extract_rawtrace[clusters_rois_flat]
        df_extract_rawtrace_detrended_zscore = mscope.norm_traces(df_extract_rawtrace_detrended, 'zscore', 'session')

        # raw signal clustered - CLUSTERS
        time_beg_vec = np.arange(0, 60, 5)
        time_end_vec = np.arange(5, 60+5, 5)
        w = 0 # only from 0 to 5 seconds for each trial
        mean_data_trials_rois = []
        for count_c, c in enumerate(clusters_rois):
            data_trials = np.zeros(len(trials))
            for count_t, t in enumerate(trials):
                df_data = df_extract_rawtrace_detrended_zscore.loc[df_extract_rawtrace_detrended_zscore['trial']==t, c].mean(axis=1)
                data_trials[count_t] = df_data.iloc[time_beg_vec[w]*mscope.sr:time_end_vec[w] * mscope.sr].mean(axis=0)
            mean_data_trials_rois.append(data_trials-np.nanmean(data_trials[trials_baseline-1]))
        mean_data_animals.append(mean_data_trials_rois)
    animal_in.append(animal)

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
def get_colors_plot(animal_name, color_animals):
    if animal_name=='MC8855':
        color_plot = color_animals[0]
    if animal_name=='MC9194':
        color_plot = color_animals[1]
    if animal_name=='MC10221':
        color_plot = color_animals[2]
    if animal_name=='MC9513':
        color_plot = color_animals[3]
    if animal_name=='MC9226':
        color_plot = color_animals[4]
    return color_plot
fig, ax = plt.subplots(figsize=(7, 15), tight_layout=True)
rectangle = plt.Rectangle((6.5, -1), 10, 9, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for a in range(len(mean_data_animals)):
    for c in range(len(mean_data_animals[a])):
        if a == 3:
            ax.plot(np.arange(4, 27), mean_data_animals[a][c] + a*1.7, marker='o', linewidth=2, color=get_colors_plot(animal_in[a], color_animals))
        elif a == 2:
            ax.plot(np.arange(1, 24), mean_data_animals[a][c] + a * 1.7, marker='o', linewidth=2,
                    color=get_colors_plot(animal_in[a], color_animals))
        else:
            ax.plot(np.arange(1, 27), mean_data_animals[a][c] + a * 1.7, marker='o', linewidth=2,
                    color=get_colors_plot(animal_in[a], color_animals))
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 4)
ax.set_ylabel('Calcium response 1st 5s', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('E:\\Miniscope processed files\\avg_activity_clusters_1st_5s_split_ipsi_fast_S1', dpi=mscope.my_dpi)
