# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# import classes
path_code = 'C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\'
os.chdir(path_code)
import df_behav_class
nxb = df_behav_class.df_behav_analysis(path_code)

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\STA FR-FL\\split ipsi fast S1\\'
session_type = save_path.split('\\')[-2].split(' ')[0]
dotsize = 60
window = np.arange(-330, 330 + 1)  # Samples
xaxis = window/330
xaxis_start = np.where(xaxis >= -0.25)[0][0]
xaxis_end = np.where(xaxis >= 0)[0][0]
time_minus100 = np.argmin(np.abs(xaxis+0.1))
time_0 = np.argmin(np.abs(xaxis))
animals = ['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226']
animal_name_plots = ['Animal 1', 'Animal 2', 'Animal 3', 'Animal 4', 'Animal 5']

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

if session_type == 'split':
    protocol = 'split_ipsi_fast'
    fig, ax = plt.subplots(2, 3, figsize=(10, 10), tight_layout=True, sharey=True)
    ax = ax.ravel()
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))[[0, -1], :, :]
        sta_zs_rois_split = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_split.npy'))[[0, -1], :, :]
        sta_zs_rois_washout = np.load(
            os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_washout.npy'))[[0, -1], :, :]
        colors_cluster = np.load(os.path.join(save_path, animal + '_' + protocol, 'colors_cluster.npy'))
        sta_zs_rois_concat = np.concatenate((sta_zs_rois_bs, sta_zs_rois_split, sta_zs_rois_washout), axis=0)
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal != 'MC8855':
                ax[count_a].scatter(np.array([1, 6]), xaxis[np.argmax(sta_zs_rois_bs[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize, color=colors_cluster[count_c])
            if animal == 'MC8855':
                ax[count_a].scatter(np.array([4, 6]), xaxis[np.argmax(sta_zs_rois_bs[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize, color=colors_cluster[count_c])
            ax[count_a].scatter(np.array([7, 16]), xaxis[np.argmax(sta_zs_rois_split[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize, color=colors_cluster[count_c])
            if animal != 'MC9226':
                ax[count_a].scatter(np.array([17, 26]), xaxis[np.argmax(sta_zs_rois_washout[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize, color=colors_cluster[count_c])
            if animal == 'MC9226':
                ax[count_a].scatter(np.array([17, 23]), xaxis[np.argmax(sta_zs_rois_washout[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize,
                           color=colors_cluster[count_c])
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal == 'MC8855':
                ax[count_a].plot(np.array([4, 6, 7, 16, 17, 26]), xaxis[np.argmax(sta_zs_rois_concat[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], color=colors_cluster[count_c])
            elif animal == 'MC9226':
                ax[count_a].plot(np.array([1, 6, 7, 16, 17, 23]), xaxis[np.argmax(sta_zs_rois_concat[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], color=colors_cluster[count_c])
            else:
                ax[count_a].plot(np.array([1, 6, 7, 16, 17, 26]), xaxis[np.argmax(sta_zs_rois_concat[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], color=colors_cluster[count_c])
        ax[count_a].axvline(x=6.5, linewidth=2, linestyle='dashed', color='black')
        ax[count_a].axvline(x=16.5, linewidth=2, linestyle='dashed', color='black')
        ax[count_a].set_xlabel('Trials', fontsize=12)
        ax[count_a].set_ylabel('FR-FL latency(s)', fontsize=12)
        ax[count_a].set_title('FR-FL latency before CS ', fontsize=12)
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_latency_split_ipsi_S1'), dpi=128)

if session_type == 'tied':
    protocol = 'tied_baseline'
    fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True, sharey=True)
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))
        sta_zs_rois_fast = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_fast.npy'))
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            ax.scatter((np.ones(np.shape(sta_zs_rois_bs)[0])*3)+np.random.rand(np.shape(sta_zs_rois_bs)[0]), xaxis[np.argmax(sta_zs_rois_bs[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize, color=colors_cluster[count_c])
            ax.scatter((np.ones(np.shape(sta_zs_rois_fast)[0])*5)+np.random.rand(np.shape(sta_zs_rois_fast)[0]), xaxis[np.argmax(sta_zs_rois_fast[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize,
                   color=colors_cluster[count_c])
        if animal != 'MC8855':
            sta_zs_rois_slow = np.load(
                os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_slow.npy'))
            for count_c in range(np.shape(sta_zs_rois_slow)[2]):
                ax.scatter(np.ones(np.shape(sta_zs_rois_slow)[0]) + np.random.rand(np.shape(sta_zs_rois_slow)[0]), xaxis[np.argmax(sta_zs_rois_slow[:, xaxis_start:xaxis_end, count_c], axis=1)+xaxis_start], s=dotsize,
                       color=colors_cluster[count_c])
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(['slow', 'baseline', 'fast'], fontsize=20)
    ax.set_xlabel('Belt speed', fontsize=20)
    ax.set_ylabel('FR-FL latency (s)', fontsize=20)
    ax.set_title('FR-FL latency before CS ', fontsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_latency_tied_session_S1'), dpi=128)




