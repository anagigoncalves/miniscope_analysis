# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
corr_data_all = []
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

    # FOV coordinates
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    fov_coord = np.array([-6.64, np.nanmean(np.array([1, 2.5]))])
    cluster_coord = mscope.get_coordinates_cluster(centroid_ext, fov_coord, idx_roi_cluster_ordered)

    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)

    # ROIs correlation with mediolateral distance
    corr_rois_mat = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trials_baseline[-1]].iloc[:, 2:].corr()
    distance_neurons_save = distance_neurons[1:, 0]
    corr_data_save = corr_rois_mat.iloc[1:, 0]
    corr_data = np.zeros((len(distance_neurons[1:, 0]), 2))
    for i in range(len(distance_neurons_save)):
        corr_data[i, 0] = distance_neurons_save[i]
        corr_data[i, 1] = corr_data_save[i]
    corr_data_all.append(corr_data)

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
#order
#MC8855, MC9194, MC10221, MC9513, MC9226
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True, sharey=True)
for a in range(len(color_animals)-1):
    plt.scatter(corr_data_all[a][:, 0], corr_data_all[a][:, 1], s=15, color=color_animals[a])
    z = np.polyfit(corr_data_all[a][:, 0], corr_data_all[a][:, 1], 1)
    p = np.poly1d(z)
    plt.plot(corr_data_all[a][:, 0], p(corr_data_all[a][:, 0]), linewidth=3, color=color_animals[a])
ax.set_xlabel('Mediolateral distance (\u03BCm)', fontsize=mscope.fsize - 2)
ax.set_ylabel('Correlation between ROIs', fontsize=mscope.fsize - 2)
# ax.legend(['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226'], fontsize=mscope.fsize - 4, frameon=False)
# ax.legend(['Animal 1', 'Animal 2', 'Animal 3', 'Animal 4', 'Animal 5'], fontsize=mscope.fsize - 2, frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=mscope.fsize - 2)
plt.savefig('J:\\Thesis\\figuresChapter2\\rois_correlation_distance', dpi=mscope.my_dpi)
# plt.savefig('J:\\Thesis\\figuresChapter2\\rois_correlation_distance.svg', dpi=mscope.my_dpi)
# plt.savefig(os.path.join(path_session_data, 'corr_rois_mediolateral_distance_population_legend'), dpi=mscope.my_dpi)

