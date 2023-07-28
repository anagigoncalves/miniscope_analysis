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
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_tied_S1.xlsx')
save_path = 'J:\\Miniscope processed files\\STA FR-FL\\tied baseline S1\\'
session_type = save_path.split('\\')[-2].split(' ')[0]
dotsize = 60
window = np.arange(-330, 330 + 1)  # Samples
xaxis = window/330
xaxis_start = np.where(xaxis >= -0.5)[0][0]
xaxis_end = np.where(xaxis >= 0.25)[0][0]
time_minus100 = np.argmin(np.abs(xaxis+0.1))
time_0 = np.argmin(np.abs(xaxis))
animals = ['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226']
animal_name_plots = ['Animal 1', 'Animal 2', 'Animal 3', 'Animal 4', 'Animal 5']
if session_type == 'split':
    protocol = 'split_ipsi_fast'
    fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True, sharey=True)
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))
        sta_zs_rois_split = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_split.npy'))
        sta_zs_rois_washout = np.load(
            os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_washout.npy'))
        sta_zs_rois_concat = np.concatenate((sta_zs_rois_bs, sta_zs_rois_split, sta_zs_rois_washout), axis=0)
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal != 'MC8855':
                ax.scatter(np.arange(1, 7), sta_zs_rois_bs[:, time_minus100, count_c], s=dotsize, color='black')
            if animal == 'MC8855':
                ax.scatter(np.arange(4, 7), sta_zs_rois_bs[:, time_minus100, count_c], s=dotsize, color='black')
            ax.scatter(np.arange(7, 17), sta_zs_rois_split[:, time_minus100, count_c], s=dotsize,
                       color='red')
            if animal != 'MC9226':
                ax.scatter(np.arange(17, 27), sta_zs_rois_washout[:, time_minus100, count_c], s=dotsize,
                           color='blue')
            if animal == 'MC9226':
                ax.scatter(np.arange(17, 24), sta_zs_rois_washout[:, time_minus100, count_c], s=dotsize,
                           color='blue')
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal == 'MC8855':
                ax.plot(np.arange(4, 27), sta_zs_rois_concat[:, time_minus100, count_c], color='black')
            elif animal == 'MC9226':
                ax.plot(np.arange(1, 24), sta_zs_rois_concat[:, time_minus100, count_c], color='black')
            else:
                ax.plot(np.arange(1, 27), sta_zs_rois_concat[:, time_minus100, count_c], color='black')
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('FR-FL at -100ms', fontsize=20)
    ax.set_title('FR-FL at -100ms before CS ', fontsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_-100ms_split_ipsi_S1'), dpi=128)

    fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True, sharey=True)
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))
        sta_zs_rois_split = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_split.npy'))
        sta_zs_rois_washout = np.load(
            os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_washout.npy'))
        sta_zs_rois_concat = np.concatenate((sta_zs_rois_bs, sta_zs_rois_split, sta_zs_rois_washout), axis=0)
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal != 'MC8855':
                ax.scatter(np.arange(1, 7), sta_zs_rois_bs[:, time_0, count_c], s=dotsize, color='black')
            if animal == 'MC8855':
                ax.scatter(np.arange(4, 7), sta_zs_rois_bs[:, time_0, count_c], s=dotsize, color='black')
            ax.scatter(np.arange(7, 17), sta_zs_rois_split[:, time_0, count_c], s=dotsize,
                       color='red')
            if animal != 'MC9226':
                ax.scatter(np.arange(17, 27), sta_zs_rois_washout[:, time_0, count_c], s=dotsize,
                           color='blue')
            if animal == 'MC9226':
                ax.scatter(np.arange(17, 24), sta_zs_rois_washout[:, time_0, count_c], s=dotsize,
                           color='blue')
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            if animal == 'MC8855':
                ax.plot(np.arange(4, 27), sta_zs_rois_concat[:, time_0, count_c], color='black')
            elif animal == 'MC9226':
                ax.plot(np.arange(1, 24), sta_zs_rois_concat[:, time_0, count_c], color='black')
            else:
                ax.plot(np.arange(1, 27), sta_zs_rois_concat[:, time_0, count_c], color='black')
    ax.set_xlabel('Trials', fontsize=20)
    ax.set_ylabel('FR-FL at 0ms', fontsize=20)
    ax.set_title('FR-FL at CS ', fontsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_0ms_split_ipsi_S1'), dpi=128)

if session_type == 'tied':
    protocol = 'tied_baseline'
    fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True, sharey=True)
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))
        sta_zs_rois_fast = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_fast.npy'))
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            ax.scatter((np.ones(np.shape(sta_zs_rois_bs)[0])*3)+np.random.rand(np.shape(sta_zs_rois_bs)[0]), sta_zs_rois_bs[:, time_minus100, count_c], s=dotsize, color='black')
            ax.scatter((np.ones(np.shape(sta_zs_rois_fast)[0])*5)+np.random.rand(np.shape(sta_zs_rois_fast)[0]), sta_zs_rois_fast[:, time_minus100, count_c], s=dotsize,
                   color='orange')
        if animal != 'MC8855':
            sta_zs_rois_slow = np.load(
                os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_slow.npy'))
            for count_c in range(np.shape(sta_zs_rois_slow)[2]):
                ax.scatter(np.ones(np.shape(sta_zs_rois_slow)[0]) + np.random.rand(np.shape(sta_zs_rois_slow)[0]), sta_zs_rois_slow[:, time_minus100, count_c], s=dotsize,
                       color='purple')
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(['slow', 'baseline', 'fast'], fontsize=20)
    ax.set_xlabel('Belt speed', fontsize=20)
    ax.set_ylabel('FR-FL at -100ms', fontsize=20)
    ax.set_title('FR-FL at -100ms before CS ', fontsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_-100ms_tied_session_S1'), dpi=128)

    fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True, sharey=True)
    for count_a, animal in enumerate(animals):
        sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_bs.npy'))
        sta_zs_rois_fast = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_fast.npy'))
        for count_c in range(np.shape(sta_zs_rois_bs)[2]):
            ax.scatter((np.ones(np.shape(sta_zs_rois_bs)[0])*3)+np.random.rand(np.shape(sta_zs_rois_bs)[0]), sta_zs_rois_bs[:, time_minus100, count_c], s=dotsize, color='black')
            ax.scatter((np.ones(np.shape(sta_zs_rois_fast)[0])*5)+np.random.rand(np.shape(sta_zs_rois_fast)[0]), sta_zs_rois_fast[:, time_minus100, count_c], s=dotsize,
                   color='orange')
        if animal != 'MC8855':
            sta_zs_rois_slow = np.load(
                os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_FR-FL_displacement_difference_slow.npy'))
            for count_c in range(np.shape(sta_zs_rois_slow)[2]):
                ax.scatter(np.ones(np.shape(sta_zs_rois_slow)[0]) + np.random.rand(np.shape(sta_zs_rois_slow)[0]), sta_zs_rois_slow[:, time_minus100, count_c], s=dotsize,
                       color='purple')
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(['slow', 'baseline', 'fast'], fontsize=20)
    ax.set_xlabel('Belt speed', fontsize=20)
    ax.set_ylabel('FR-FL at 0ms', fontsize=20)
    ax.set_title('FR-FL at CS ', fontsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(os.path.join(save_path, 'sta_bodyvars_FR-FL_displacement_difference_0ms_tied_session_S1'), dpi=128)



