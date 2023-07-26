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
save_path = 'J:\\Miniscope processed files\\STA bodyvars\\tied baseline S1\\'
session_type = save_path.split('\\')[-2].split(' ')[0]

window = np.arange(-330, 330 + 1)  # Samples
xaxis = window/330
xaxis_start = np.where(xaxis >= -0.5)[0][0]
xaxis_end = np.where(xaxis >= 0.25)[0][0]
# Loop through independent variables to compute and plot STAs of each one
vars = ['Body_position', 'Body_speed', 'Body_acceleration']
animals = ['MC8855', 'MC9194', 'MC10221', 'MC9513', 'MC9226']
animal_name_plots = ['Animal 1', 'Animal 2', 'Animal 3', 'Animal 4', 'Animal 5']
if session_type == 'split':
    protocol = 'split_ipsi_fast'
if session_type == 'tied':
    protocol = 'tied_baseline'
for var in range(len(vars)):
    fig, ax = plt.subplots(1, 5, figsize=(25, 7), tight_layout=True, sharey=True)
    ax = ax.ravel()
    for count_a, animal in enumerate(animals):
        var_name = vars[var]
        if session_type == 'split':
            sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_bs.npy'))
            sta_zs_rois_split = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_split.npy'))
            sta_zs_rois_washout = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_washout.npy'))
            for c in range(np.shape(sta_zs_rois_bs)[2]):
                ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0), color='black')
                ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)-np.nanstd(sta_zs_rois_bs[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)+np.nanstd(sta_zs_rois_bs[:, :, c], axis=0), color='black', alpha=0.3)
                ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_split[:, :, c], axis=0), color='red')
                ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_split[:, :, c], axis=0)-np.nanstd(sta_zs_rois_split[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_split[:, :, c], axis=0)+np.nanstd(sta_zs_rois_split[:, :, c], axis=0), color='red', alpha=0.3)
                ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_washout[:, :, c], axis=0), color='blue')
                ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_washout[:, :, c], axis=0)-np.nanstd(sta_zs_rois_washout[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_washout[:, :, c], axis=0)+np.nanstd(sta_zs_rois_washout[:, :, c], axis=0), color='blue', alpha=0.3)
        if session_type == 'tied':
            sta_zs_rois_bs = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_bs.npy'))
            if animal != 'MC8855':
                sta_zs_rois_slow = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_slow.npy'))
            sta_zs_rois_fast = np.load(os.path.join(save_path, animal + '_' + protocol, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_fast.npy'))
            for c in range(np.shape(sta_zs_rois_bs)[2]):
                ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0), color='black')
                ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)-np.nanstd(sta_zs_rois_bs[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_bs[:, :, c], axis=0)+np.nanstd(sta_zs_rois_bs[:, :, c], axis=0), color='black', alpha=0.3)
                if animal != 'MC8855':
                    ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_slow[:, :, c], axis=0),
                               color='purple')
                    ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_slow[:, :, c], axis=0)-np.nanstd(sta_zs_rois_slow[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_slow[:, :, c], axis=0)+np.nanstd(sta_zs_rois_slow[:, :, c], axis=0), color='purple', alpha=0.3)
                ax[count_a].plot(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_fast[:, :, c], axis=0), color='orange')
                ax[count_a].fill_between(xaxis[xaxis_start:xaxis_end], np.nanmean(sta_zs_rois_fast[:, :, c], axis=0)-np.nanstd(sta_zs_rois_fast[:, :, c], axis=0),
                                    np.nanmean(sta_zs_rois_fast[:, :, c], axis=0)+np.nanstd(sta_zs_rois_fast[:, :, c], axis=0), color='orange', alpha=0.3)
            ax[count_a].axvline(x=0, linewidth=2, linestyle='dashed', color='black')
            ax[count_a].set_xlabel('Time (s)', fontsize=20)
            ax[count_a].set_ylabel(var_name.replace('_',' '), fontsize=20)
            ax[count_a].set_title(animal_name_plots[count_a], fontsize=24)
            ax[count_a].spines['right'].set_visible(False)
            ax[count_a].spines['top'].set_visible(False)
            ax[count_a].tick_params(axis='both', which='major', labelsize=18)
        plt.savefig(os.path.join(save_path, 'sta_bodyvars_' + var_name.replace(' ', '_') + '_' + protocol+'_grc'), dpi=128)


