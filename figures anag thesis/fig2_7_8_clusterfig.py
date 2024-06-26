# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import pandas as pd
import seaborn as sns
import warnings
import scipy.stats
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24
plot_example = 0

# for the order ['MC8855', 'MC9194', 'MC10221', 'MC9226', 'MC9513']
fov_coords = np.array([[6.12, 0.5],
                     [6.24, 1],
                     [6.48, 1.7],
                     [6.64, 1],
                     [6.48, 1.5]]) #AP, ML
th_cluster = [0.6, 0.7, 0.8, 1, 0.6]

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
corr_data_all = []
roi_coordinates = []
cluster_id_rois = []
cluster_id_cumsum = 0
cluster_id_rois_split = []
cluster_id_rois_split_animal = []
cluster_id_cumsum_split = 0
sim_trials = np.zeros((len(session_data), 26))
upper_ci_trials = np.zeros((len(session_data), 26))
lower_ci_trials = np.zeros((len(session_data), 26))
sim_trials[:] = np.nan
upper_ci_trials[:] = np.nan
lower_ci_trials[:] = np.nan
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
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    distance_neurons = mscope.distance_neurons(centroid_ext, 0)

    # Compute ROI coordinates
    coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
    centroid_ext = mscope.get_roi_centroids(coord_ext)
    centroid_ext_arr = np.array(centroid_ext)
    #Flip coords horizontally and vertically because image in miniscope is flipped
    centroid_ext_flip = np.zeros(np.shape(centroid_ext_arr))
    centroid_ext_flip[:, 1] = 1000-centroid_ext_arr[:, 0]
    centroid_ext_flip[:, 0] = 1000-centroid_ext_arr[:, 1]
    #Need to swap again, because now ML and AP are swapped
    #Adjust for the FOV coordinates to get global coordinates
    centroid_ext_swap = np.array(centroid_ext_flip)[:, [1, 0]] 
    fov_coord = fov_coords[s]
    fov_corner = np.array([fov_coord[1] - 0.5, fov_coord[0] - 0.5]) #ML is the centroid[:, 0] and AP the centroid[:, 1]
    centroid_dist_corner = (np.array(centroid_ext_swap) * 0.001) + fov_corner
    roi_coordinates.extend(centroid_dist_corner)
    if s == 0:
        max_prev_animal = 0
    else:
        max_prev_animal = np.max(cluster_id)
    cluster_id = idx_roi_cluster_ordered+max_prev_animal
    cluster_id_rois.extend(cluster_id)
    cluster_id_rois_split_animal.extend(np.repeat(animal, len(cluster_id)))

    #Clusters split
    colormap_cluster = 'hsv'
    [colors_cluster_split, idx_roi_cluster_split] = mscope.compute_roi_clustering(df_extract_rawtrace_detrended, centroid_ext,
                                                                      distance_neurons, trials_split, th_cluster[s],
                                                                      colormap_cluster, 0, 0)
    [clusters_rois_split, idx_roi_cluster_ordered_split] = mscope.get_rois_clusters_mediolateral(df_extract_rawtrace_detrended,
                                                                                     idx_roi_cluster_split, centroid_ext)

    if animal == 'MC8855': #for split ipsi S1 3-10-10
        trials_idx = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    if animal == 'MC9226': #for split ipsi S1 3-10-7
        trials_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
    else:
        trials_idx = np.arange(0, 26)

    # compute correlation matrix for trial 1
    if animal == 'MC9226':
        clusters_rois_flat = clusters_rois[0]
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
    else:
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
    df_traces_clustered = df_extract_rawtrace_detrended[clusters_rois_flat]
    df_corr_trial_t1 = df_traces_clustered.loc[df_traces_clustered['trial'] == trials[0]].iloc[:, 2:].corr()
    # get upper values
    mask = np.triu_indices(df_corr_trial_t1.shape[0], k=1)
    df_corr_trial_upper_t1 = df_corr_trial_t1.values[mask]
    for count_t, trial in enumerate(trials):
        #compute correlation matrix for each trial
        df_corr_trial = df_traces_clustered.loc[df_traces_clustered['trial'] == trial].iloc[:, 2:].corr()
        df_corr_trial_upper = df_corr_trial.values[mask]
        #do first fisher transformation https://en.wikipedia.org/wiki/Fisher_transformation
        [sim, p_value] = scipy.stats.spearmanr(np.arctanh(df_corr_trial_upper_t1), np.arctanh(df_corr_trial_upper))
        #CI calculation https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
        count = len(df_corr_trial_upper_t1)
        lower_ci = np.tanh(np.arctanh(sim) - (1.96 * (1.0/np.sqrt(len(df_corr_trial_upper_t1)-3))))
        upper_ci = np.tanh(np.arctanh(sim) + (1.96 * (1.0/np.sqrt(len(df_corr_trial_upper_t1)-3))))
        upper_ci_trials[s, trials_idx[count_t]] = upper_ci
        lower_ci_trials[s, trials_idx[count_t]] = lower_ci
        sim_trials[s, trials_idx[count_t]] = sim

    if animal == 'MC8855' and plot_example:
        # ROIs
        plt.figure(figsize=(7, 7), tight_layout=True)
        for r in range(len(coord_ext)):
            plt.scatter(coord_ext[r][:, 0], coord_ext[r][:, 1], color=colors_cluster[idx_roi_cluster_ordered[r] - 1], s=1, alpha=0.6)
        plt.imshow(ref_image, cmap='gray',
                   extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um,
                           0])
        plt.xlabel('FOV in micrometers', fontsize=24)
        plt.ylabel('FOV in micrometers', fontsize=24)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\MC8855_rois_cluster', dpi=256)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\MC8855_rois_cluster.svg', dpi=256)

        # Order ROIs by cluster
        trial_plot = 3
        cluster_transition_idx = np.cumsum([len(clusters_rois[c]) for c in range(len(clusters_rois))])
        clusters_rois_flat = np.transpose(sum(clusters_rois, []))
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'time')
        clusters_rois_flat = np.insert(clusters_rois_flat, 0, 'trial')
        df_traces_clustered = df_extract_rawtrace_detrended[clusters_rois_flat]
        fig, ax = plt.subplots(figsize=(15, 10), tight_layout=True)
        sc = sns.heatmap(df_traces_clustered.loc[
                        df_traces_clustered['trial'] == trial_plot].iloc[:, 2:].corr(), cmap='Greys')
        cbar = sc.collections[0].colorbar
        cbar.ax.tick_params(labelsize=20)
        ax.set_xticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
        ax.set_xticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        ax.set_yticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
        ax.set_yticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix', dpi=mscope.my_dpi)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix.svg', dpi=mscope.my_dpi)
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        sc = sns.heatmap(df_traces_clustered.loc[
                        df_traces_clustered['trial'] == trial_plot].iloc[:, 2:].corr(), cmap='Greys', cbar=None)
        ax.set_xticks(np.arange(0, len(df_traces_clustered.columns[2:]), 12))
        ax.set_xticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        ax.set_yticks(np.arange(0, len(df_traces_clustered.columns[2:]), 12))
        ax.set_yticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix_square', dpi=mscope.my_dpi)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix_square.svg', dpi=mscope.my_dpi)
        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        ax.set_xticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
        ax.set_xticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        ax.set_yticks(np.arange(0, len(df_traces_clustered.columns[2:]), 4))
        ax.set_yticklabels(df_traces_clustered.columns[2::4], rotation=45, fontsize=10)
        for c in range(len(clusters_rois)):
            ax.axvline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
            ax.axvline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
            ax.axhline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
            ax.axhline(cluster_transition_idx[c], color=colors_cluster[c], linewidth=2)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix_lines', dpi=mscope.my_dpi)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\correlation_matrix_lines.svg', dpi=mscope.my_dpi)

        dFF_trial = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot]  # get dFF for the desired trial
        fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
        for count_c, c in enumerate(clusters_rois):
            mean_data = np.array(dFF_trial[c].mean(axis=1))
            std_data = np.array(dFF_trial[c].std(axis=1))
            plt.plot(dFF_trial.loc[dFF_trial['trial'] == trial_plot, 'time'], mean_data+(count_c*1.5),
                     color=colors_cluster[count_c])
            plt.fill_between(dFF_trial.loc[dFF_trial['trial'] == trial_plot, 'time'], mean_data+(count_c*1.5)-std_data,
                     mean_data+(count_c*1.5)+std_data, color=colors_cluster[count_c], alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 2)
        ax.set_xlim([10, 45])
        plt.xticks(fontsize=mscope.fsize - 2)
        plt.yticks(fontsize=mscope.fsize - 2)
        #plt.setp(ax.get_yticklabels(), visible=False)
        #ax.tick_params(axis='y', which='y', length=0)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\traces_cluster', dpi=mscope.my_dpi)
        plt.savefig('J:\\Thesis\\figuresChapter2\\fig7 - cluster example\\traces_cluster.svg', dpi=mscope.my_dpi)

    # ROIs correlation with mediolateral distance
    corr_rois_mat = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trials_baseline[-1]].iloc[:, 2:].corr()
    distance_neurons_save = distance_neurons[1:, 0]
    corr_data_save = corr_rois_mat.iloc[1:, 0]
    corr_data = np.zeros((len(distance_neurons[1:, 0]), 2))
    for i in range(len(distance_neurons_save)):
        corr_data[i, 0] = distance_neurons_save[i]
        corr_data[i, 1] = corr_data_save[i]
    corr_data_all.append(corr_data)
roi_coordinates_arr = np.array(roi_coordinates)
cluster_id_arr = np.array(cluster_id_rois)

cmap = plt.get_cmap('magma')
color_animals = [cmap(i) for i in np.linspace(0, 1, 6)]
#order
#MC8855, MC9194, MC10221, MC9513, MC9226
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True, sharey=True)
for a in range(len(color_animals)-1):
    plt.scatter(corr_data_all[a][:, 0], corr_data_all[a][:, 1], s=15, color=color_animals[a])
    #TODO regression can be exponential
    #z = np.polyfit(corr_data_all[a][:, 0], corr_data_all[a][:, 1], 1)
    #p = np.poly1d(z)
    #plt.plot(corr_data_all[a][:, 0], p(corr_data_all[a][:, 0]), linewidth=3, color=color_animals[a])
ax.set_xlabel('Mediolateral distance (\u03BCm)', fontsize=20)
ax.set_ylabel('Correlation between ROIs', fontsize=20)
# ax.legend(['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226'], fontsize=mscope.fsize - 4, frameon=False)
# ax.legend(['Animal 1', 'Animal 2', 'Animal 3', 'Animal 4', 'Animal 5'], fontsize=mscope.fsize - 2, frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\rois_correlation_distance', dpi=mscope.my_dpi)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\rois_correlation_distance.svg', dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
rectangle = plt.Rectangle((6.5, 0), 10, 1, fc='dimgrey', alpha=0.3)
plt.gca().add_patch(rectangle)
for a in range(len(color_animals)-1):
    plt.plot(np.arange(1, 27)[1:], sim_trials[a, 1:], marker='o', color=color_animals[a ])
    plt.fill_between(np.arange(1, 27)[1:], lower_ci_trials[a, 1:], upper_ci_trials[a, 1:], color=color_animals[a], alpha=0.3)
ax.set_xlabel('Trials', fontsize=20)
ax.set_ylabel('Similarity in correlation matrix', fontsize=20)
ax.set_ylim([0, 1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\corr_matrix_similarity', dpi=mscope.my_dpi)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\corr_matrix_similarity.svg', dpi=mscope.my_dpi)

cmap = mp.cm.jet
bounds = np.unique(cluster_id_arr)
norm = mp.colors.BoundaryNorm(bounds, cmap.N)
fig, ax = plt.subplots(figsize=(13, 10))
sc = ax.scatter(roi_coordinates_arr[:, 0], roi_coordinates_arr[:, 1], c=cluster_id_arr, s=15, cmap=cmap, norm=norm)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.gca().invert_yaxis()
ax2 = fig.add_axes([0.9, 0.1, 0.05, 0.8])
cb = plt.colorbar(mp.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax2, orientation='vertical', ticks=bounds, boundaries=bounds, format='%1i')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\cluster_map_all_animals_split', dpi=mscope.my_dpi)
plt.savefig('J:\\Thesis\\figuresChapter2\\fig8 - cluster tied vs split\\cluster_map_all_animals_split.svg', dpi=mscope.my_dpi)