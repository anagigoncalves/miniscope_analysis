# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:26:03 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from scipy.interpolate import interp1d
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

os.chdir('C:\\Users\\User\\Documents\\Phaser\\gait_signatures-main')
import util
import phaser

reference_trial = 0
sr_cam = 330  
plot_data = True; save_data = True; save_plot = True
paws = ['FR', 'HR', 'FL', 'HL']
paw_color = ['red', 'magenta', 'blue', 'cyan']
data_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\Behavior'
load_data_path = os.path.join(data_path, 'Paw position all animals split ipsi fast.npy')
save_data_path = os.path.join(data_path, 'Global phase all animals split ipsi fast.npy')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\Behavioral manifold'

# Load data
with open(load_data_path, 'rb') as file:
    data = pickle.load(file)
animals = list(data.keys()) # List of animals ID
dataset = {}

# Utils
def find_phase(k):
    y = np.array(k)
    y = util.detrend(y.T).T
    phsr = phaser.Phaser(y=y)
    k[:] = phsr.phaserEval(y)[0,:]
    return k

def findpeaks(array, peak_threshold=None, trough_threshold=None):
    peaks = []
    troughs = []
    current_state = 'peak'  
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] > array[i + 1] and array[i] > peak_threshold:
            if current_state == 'trough':
                peaks.append(i)
                current_state = 'peak'
        elif array[i - 1] > array[i] < array[i + 1] and array[i] < trough_threshold:
            if current_state == 'peak':
                troughs.append(i)
                current_state = 'trough'
    return peaks, troughs

def interpolate_arrays(list_of_arrays, target_length=None):
    interpolated_arrays = []
    for array in list_of_arrays:
        N, _ = array.shape
        indices = np.linspace(0, N - 1, num=target_length)
        interpolated_columns = [interp1d(np.arange(N), array[:, i])(indices) for i in range(3)]
        interpolated_array = np.column_stack(interpolated_columns)
        interpolated_arrays.append(interpolated_array)
    return interpolated_arrays

def image_saver(save_path, folder_name, file_name):
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    plt.savefig(os.path.join(save_path, folder_name + '\\', file_name + '.png'))

def get_rotation_angle(a, b):
    cross_covariance_matrix = np.dot(a, b.T)
    u, _, vh = np.linalg.svd(cross_covariance_matrix)
    rotation_matrix = np.dot(vh.T, u.T)
    rotation_angle_rad = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    # axis = (rotation_matrix - rotation_matrix.T) / (2 * np.sin(rotation_angle_rad))
    return rotation_angle_deg

# Main loop
for animal in animals:
    # Retrieve data for one experimental animal
    paws_position = data[animal]['paws positions']
    trial_id = data[animal]['trial id']
    trials = np.unique(trial_id)
    color_trials = data[animal]['color trials']
    experimental_blocks = data[animal]['experimental blocks']
    body_speed = data[animal]['body speed']
    sw_idx = data[animal]['swing idx']
    st_idx = data[animal]['stance idx']
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_fit = pca.fit(paws_position)
    principal_components = [pca.transform(paws_position.iloc[np.where(trial_id == trial)[0]]) for trial_idx, trial in enumerate(trials)]
    
    # Compute global phase with Phaser
    global_phase = []; cycle_duration = []; interp_cycles = []; avg_trajectory = []; cycle_onset = []; cycle_offset = []
    for trial_idx, trial in enumerate(trials):
        global_phase_unwrapped = find_phase(paws_position.iloc[np.where(trial_id == trial)[0]].T)
        global_phase.append(np.mod(global_phase_unwrapped.iloc[0, :], 2 * np.pi).values) # wrap phase
        peaks, troughs = findpeaks(global_phase[trial_idx], peak_threshold=6.1, trough_threshold=0.3)
        cycle_onset.append(troughs); cycle_offset.append(peaks)
        cycle_duration.append(np.diff(troughs) * (1 / sr_cam))

    # Interpolate cycles in the behavioral manifold and average them    
        cycles = []
        for i in range(len(peaks)):
            onset = troughs[i]; offset = peaks[i]
            cycles.append(principal_components[trial_idx][onset:offset+1, :])
        interp_cycles.append(interpolate_arrays(cycles, target_length=200))
        avg_trajectory.append([np.mean([cycle[:, col] for cycle in interp_cycles[trial_idx]], axis = 0) for col in range(3)])
    
    # Plot data
    if plot_data:
        # Plot example of global phase
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(len(global_phase[reference_trial][200:2000] - 1)) / sr_cam, global_phase[0][200:2000], color = 'dimgray')
        plt.xlabel('Time (s)', fontsize=18); plt.ylabel('Global phase (rad)', fontsize=18)
        # Plot distribution of cycles duration
        plt.subplot(1, 2, 2)
        plt.hist(np.concatenate(cycle_duration), bins=40, color='dimgray')
        plt.axvline(np.mean(np.concatenate(cycle_duration)), color='black', linestyle = '--')
        plt.xlabel('Cycle duration (s)', fontsize=18); plt.ylabel('Count', fontsize=18)
        plt.tight_layout()
        image_saver(save_path, animal, 'global phase')
        plt.close()
    
        # Plot 3D manifold and average trajectory
        for trial_idx, trial in enumerate(trials):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  
            ax.plot(
                principal_components[trial_idx][:, 0],
                principal_components[trial_idx][:, 1],
                principal_components[trial_idx][:, 2],
                color = 'dimgray', alpha = 0.3, linewidth = 0.5)
            ax.set_xlabel('PC1', fontsize=15)
            ax.set_ylabel('PC2', fontsize=15)
            ax.set_zlabel('PC3', fontsize=15)
            ax.plot(avg_trajectory[trial_idx][0], avg_trajectory[trial_idx][1], avg_trajectory[trial_idx][2], color = color_trials[trial], linewidth = 2)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.grid(False)
            if save_plot:
                image_saver(save_path, animal, f'behavioral manifold trial {trial}')
                plt.close()
        
        # Map global phase onto 3D manifold for an example trial
        colormap = plt.cm.get_cmap('viridis')  
        normalize = plt.Normalize(0, 2*np.pi)
        colors = colormap(normalize(global_phase[reference_trial]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            principal_components[reference_trial][:, 0],
            principal_components[reference_trial][:, 1],
            principal_components[reference_trial][:, 2],
            color = colors, s = 1)
        cbar = plt.colorbar(scatter, orientation='vertical'); cbar.set_label('Global phase', fontsize=15)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        cbar.set_ticks([0, 0.5, 1]); cbar.set_ticklabels(['0', 'π', '2π'])
        ax.view_init(elev=45, azim=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        if save_plot:
            image_saver(save_path, animal, f'global phase behavioral manifold')
            plt.close()
        
        # Map trajectory speed onto 3D manifold for an example trial
        traj_speed = np.diff(principal_components[reference_trial], axis=0) / (1/sr_cam)
        traj_speed = np.vstack((np.zeros((1, 3)), traj_speed))
        traj_speed = np.linalg.norm((traj_speed), axis=1)
        colormap = plt.cm.get_cmap('RdBu')
        normalize = plt.Normalize(np.percentile(traj_speed, 10), np.percentile(traj_speed, 90))
        colors = colormap(normalize(traj_speed))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            principal_components[reference_trial][:, 0],
            principal_components[reference_trial][:, 1],
            principal_components[reference_trial][:, 2],
            color = colors, s = 1)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(elev=10, azim=30)
        if save_plot:
            image_saver(save_path, animal, f'trajectory speed behavioral manifold')
            plt.close()
    
        # Map body speed onto 3D manifold for an example trial
        colormap = plt.cm.get_cmap('RdBu')
        normalize = plt.Normalize(np.percentile(body_speed[1], 10), np.percentile(body_speed[1], 90))
        colors = colormap(normalize(body_speed[1]))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            principal_components[reference_trial][:, 0],
            principal_components[reference_trial][:, 1],
            principal_components[reference_trial][:, 2],
            color = colors, s = 1)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(elev=30, azim=0)
        if save_plot:
            image_saver(save_path, animal, f'body speed behavioral manifold')
            plt.close()
        
        # Map stance onto 3D manifold for an example trial
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p, paw in enumerate(paws):
            st_idx_paw = np.array(st_idx[paw][reference_trial]).astype(int)
            scatter = ax.scatter(
                principal_components[reference_trial][:, 0],
                principal_components[reference_trial][:, 1],
                principal_components[reference_trial][:, 2],
                color = 'lightgray', alpha = 0.1,  s = 0.5)
            scatter = ax.scatter(
                principal_components[reference_trial][st_idx_paw, 0],
                principal_components[reference_trial][st_idx_paw, 1],
                principal_components[reference_trial][st_idx_paw, 2],
                color = paw_color[p], s = 30, marker = '.', zorder=50)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(elev=90, azim=0)
        if save_plot:
            image_saver(save_path, animal, f'stance behavioral manifold')
            plt.close()
    
        # Map swing onto 3D manifold for an example trial
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for p, paw in enumerate(paws):
            sw_idx_paw = np.array(sw_idx[paw][1]).astype(int)
            scatter = ax.scatter(
                principal_components[reference_trial][:, 0],
                principal_components[reference_trial][:, 1],
                principal_components[reference_trial][:, 2],
                color = 'lightgray', alpha = 0.1,  s = 0.5)
            scatter = ax.scatter(
                principal_components[reference_trial][sw_idx_paw, 0],
                principal_components[reference_trial][sw_idx_paw, 1],
                principal_components[reference_trial][sw_idx_paw, 2],
                color = paw_color[p], s = 30, marker = '.', zorder = 50)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        ax.view_init(elev=90, azim=0)
        if save_plot:
            image_saver(save_path, animal, f'swing behavioral manifold')
            plt.close()
        
        # Plot distribution of swing and stance along the global phase
        sw_phase = {}
        st_phase = {}
        for p, paw in enumerate(paws):
            sw_phase[paw] = [global_phase[trial_idx][np.array(sw_idx[paw][trial_idx]).astype(int)] for trial_idx, trial in enumerate(trials)]
            st_phase[paw] = [global_phase[trial_idx][np.array(st_idx[paw][trial_idx]).astype(int)] for trial_idx, trial in enumerate(trials)] 
        for trial_idx, trial in enumerate(trials):
            fig, ax = plt.subplots(2, 1, figsize = (8, 8), subplot_kw=dict(polar=True))
            for p, paw in enumerate(paws):  
                ax[0].hist(st_phase[paw][trial_idx], bins=36, color=paw_color[p], density=True, alpha = 0.7)
                ax[1].hist(sw_phase[paw][trial_idx], bins=36, color=paw_color[p], density=True, alpha = 0.7)
            ax[0].set_title('stance', fontsize=15, loc='left')
            ax[1].set_title('swing', fontsize=15, loc='left')
            plt.suptitle(f'Trial {trial}', fontsize=15)
            plt.tight_layout
            if save_plot:   
                image_saver(save_path, animal, f'trial{trial} sw-st distr')
                plt.close()
        
        # Plot median of swing and stance position along the global phase for all trials togheter
        fig, ax = plt.subplots(2, 1, figsize = (8, 8), subplot_kw=dict(polar=True))
        for trial_idx, trial in enumerate(trials):
            for p, paw in enumerate(paws):  
                ax[0].scatter(np.median(st_phase[paw][trial_idx]), trial_idx, color=paw_color[p], s = 10)
                ax[1].scatter(np.median(sw_phase[paw][trial_idx]), trial_idx, color=paw_color[p], s = 10)
        ax[0].set_title('stance', fontsize=15, loc='left')
        ax[1].set_title('swing', fontsize=15, loc='left')
        plt.tight_layout
        for a in ax:
            a.yaxis.grid(False)
            a.set_yticks([])
        if save_plot:   
            image_saver(save_path, animal, f'sw-st distr all trials')
            plt.close()
    
        # Plot each principal component in phase
        fig, axes = plt.subplots(3, 1, figsize = (5, 15))
        for pc in range(3):
            for trial_idx, trial in enumerate(trials):
                axes[pc].plot(avg_trajectory[trial_idx][pc], color = color_trials[trial], linewidth = 2)
            axes[pc].set_xticks([0, 100, 199])
            axes[pc].set_xticklabels([0, 'π', '2π'], fontsize=15)
            axes[pc].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axes[pc].spines['right'].set_visible(False)
            axes[pc].spines['top'].set_visible(False)
            axes[2].set_xlabel('Global phase', fontsize=15)
            axes[pc].set_ylabel(f'PC{pc+1}', fontsize=15)
            axes[pc].set_xlim(0, 199)
        if save_plot:
            image_saver(save_path, animal, f'principal components phase')
            plt.close()
            
        # Plot cumulative explained variance ratio
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))  
        axes[0].plot(range(1, 4), np.cumsum(pca.explained_variance_ratio_), marker='.', markersize=12, linewidth=2, color='blue')
        axes[0].set_xlabel('Principal Components', fontsize = 15)
        axes[0].set_ylabel('Cumulative Variance Explained', fontsize = 15)
    
        # Plot loadings
        im = axes[1].imshow(pca.components_, cmap='viridis', aspect='auto')
        axes[1].set_ylabel('Principal Components', fontsize = 15)
        axes[1].set_xlabel('Features', fontsize = 15)
        fig.colorbar(im, ax=axes[1], orientation='vertical')
        axes[1].set_xticks(np.arange(len(paws_position.columns)))
        axes[1].set_xticklabels(paws_position.columns)
        axes[1].set_yticks(np.arange(0, 3, 1))
        axes[1].set_yticklabels(['PC1', 'PC2', 'PC3'])
        plt.tight_layout()
        if save_plot:
            image_saver(save_path, animal, 'pca performance')
            plt.close()
        
        # Plot average trajectories by trial
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  
        for trial_idx, trial in enumerate(trials):
            ax.plot(avg_trajectory[trial_idx][0], avg_trajectory[trial_idx][1], avg_trajectory[trial_idx][2], color = color_trials[trial], linewidth = 2)
        ax.set_xlabel('PC1', fontsize=15); ax.set_ylabel('PC2', fontsize=15); ax.set_zlabel('PC3', fontsize=15)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(False)
        if save_plot:
            image_saver(save_path, animal, 'average trajectory')
            plt.close()
        
        # Plot euclidean distance to manifold's center of mass
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        for trial_idx, trial in enumerate(trials):
            center = np.mean(principal_components[trial_idx], axis=0)
            center_distance = np.mean([[np.sqrt(np.sum(((sample - center)**2))) for sample in cycle] for cycle in interp_cycles[trial_idx]], axis=0)
            ax.plot(np.linspace(0, 2 * np.pi, 200), center_distance, color=color_trials[trial], linewidth=2)
        ax.spines['polar'].set_visible(False)
        ax.set_title('Euclidean distance from center of mass', fontsize=15)
        if save_plot:
            image_saver(save_path, animal, f'distance from center of mass')
            plt.close()    
        
        # Plot distance of each cycle to mean trajectory
        plt.figure()
        dist2avg_mean = []
        for trial in trials:
            dist2avg = [np.linalg.norm(cycle.T - np.array(avg_trajectory[trial_idx]), axis = 0) for cycle in interp_cycles[trial_idx]]
            dist2avg_mean.append(np.mean(dist2avg, axis = 0))
        hm = sns.heatmap(np.flipud(dist2avg_mean), cmap = 'viridis')
        hm.collections[0].colorbar.set_label('Distance to mean trajectory', fontsize=15)
        plt.ylabel('Trials', fontsize=15)
        plt.xlabel('Global phase (rad)', fontsize=15)
        plt.xticks([0, 100, 199], labels=[0, 'π', '2π'], fontsize=12)
        plt.yticks([trials[0], trials[-1]], labels=[trials[-1], trials[0]], fontsize=12) 
        plt.xlim(0, 199)
        if save_plot:
            image_saver(save_path, animal, f'distance to mean trajectory')
            plt.close()
            
        # Get rotation angle with respect to baseline
        plt.figure()
        rotation_angle = []
        for trial_idx, trial in enumerate(trials):
            rotation_angle.append(get_rotation_angle(np.array(avg_trajectory[trial_idx]), np.array(avg_trajectory[0])))
            plt.scatter(trial, -1*rotation_angle[trial_idx], color = color_trials[trial], s = 60)
        plt.plot(np.arange(1, len(rotation_angle)+1), -1*np.array(rotation_angle), c = 'black')
        plt.ylabel('Rotation to baseline (deg)', fontsize=18)
        plt.xlabel('Trial', fontsize=18)
        plt.xlim(1, len(rotation_angle))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if save_plot:
            image_saver(save_path, animal, f'principal angles')
            plt.close()
            
    dataset[animal] = {'behavioral manifold': principal_components, 'global phase': global_phase, 'interpolated cycles': interp_cycles, 'average behavioral manifold': avg_trajectory, 'cycle onset': cycle_onset, 'cycle offset': cycle_offset}
plt.show()

# Save data
with open(save_data_path, 'wb') as file:
    pickle.dump(dataset, file)