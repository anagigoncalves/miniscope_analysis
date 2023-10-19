# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
#CLTM FULL LEFT AND RIGHT ARE SWITCHED
path_save = 'J:\\Thesis\\for methods\\'
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

s = 1
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

# Plot example of event detection
roi_plot = 256
trial_plot = 5
df_dff_trial = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, 'ROI'+str(roi_plot)]  # get dFF for the desired trial
fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
idx_trial = np.where(trials == trial_plot)[0][0]
ax.plot(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, 'time'], df_dff_trial, color='black', linewidth=2)
events_plot = np.where(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial_plot, 'ROI'+str(roi_plot)])[0]
for e in events_plot:
    ax.scatter(np.array(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial_plot, 'time'])[e], df_dff_trial.iloc[e], s=240,
               color='orange')
ax.set_xlabel('Time (s)', fontsize=26)
ax.set_ylabel(u'ΔF/F', fontsize=26)
plt.xlim([5.5, 24])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(path_save + 'example_events', dpi=mscope.my_dpi)
# plt.savefig(path_save + 'example_events.svg', format='svg', dpi=mscope.my_dpi)

# Plot about Miniscope LED power
software_val = np.array([0, 10, 20, 30, 40, 50, 60, 70])
res10kohm_val = np.array([0, 0.165, 0.242, 0.36, 0.473, 0.603, 0.727, 0.982])
res24kohm_val = np.array([0, 0.082, 0.124, 0.16, 0.202, 0.279, 0.346, 0.433])
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(software_val, res10kohm_val, marker='o', color='black', label='10kOhm')
ax.plot(software_val, res24kohm_val, marker='o', color='black', linestyle='dashed', label='24kOhm')
ax.legend(frameon=False, fontsize=16, loc='upper left')
ax.set_xlabel('LED software value', fontsize=20)
ax.set_ylabel('Power (mW)', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(path_save + 'miniscope_power', dpi=256)
# plt.savefig(path_save + 'miniscope_power.svg', format='svg', dpi=256)

# Plot DLC examples
dotsize = 200
project_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\DeepLocoTM_MdV_Miniscopes_ClosedLoop\\tmLOCO-dd-2019-08-29\\labeled-data\\'
animal_folder = 'MC16946_60_25_0.15_0.15_tied_1_2'
img_name = 'img13229.png'
csv_name = 'CollectedData_dd.csv'
#load image
im = plt.imread(os.path.join(project_path, animal_folder, img_name))
#load csv
labeled_data = pd.read_csv(os.path.join(project_path, animal_folder, csv_name), header=1)
img_path = os.path.join('labeled-data', animal_folder, img_name)
coord_row = np.where(labeled_data.iloc[:, 0] == img_path)[0][0]
coords = labeled_data.iloc[coord_row, 1:].astype(float)
colors_features = ['orange', 'orange', 'lightgray', 'lightgray', 'sienna', 'sienna', 'red', 'blue', 'magenta',
    'cyan', 'red', 'blue', 'magenta', 'cyan', 'red', 'blue', 'magenta', 'cyan', 'red', 'blue', 'magenta', 'cyan']
colors_features_full = colors_features + ['green'] * 30
cmap_cbar = mpl.colors.ListedColormap(['orange', 'lightgray', 'sienna', 'red', 'blue', 'magenta', 'cyan', 'green'])
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
boundslabel = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig(path_save + 'cl_tm_full.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig(path_save + 'cl_tm_full_cbar.svg', format='svg', dpi=256)

project_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\DeepLocoTM_MdV_Miniscopes\\tmLOCO-dd-2019-08-29\\labeled-data\\'
animal_folder = 'MC7354_25_80_1_tied_0,350_0,350_1_10'
img_name = 'img01010.png'
csv_name = 'CollectedData_dd.csv'
#load image
im = plt.imread(os.path.join(project_path, animal_folder, img_name))
#load csv
labeled_data = pd.read_csv(os.path.join(project_path, animal_folder, csv_name), header=1)
img_path = os.path.join('labeled-data', animal_folder, img_name)
coord_row = np.where(labeled_data.iloc[:, 0] == img_path)[0][0]
coords = labeled_data.iloc[coord_row, 1:].astype(float)
colors_features = ['orange', 'orange', 'lightgray', 'lightgray', 'sienna', 'sienna', 'red', 'blue', 'magenta',
    'cyan', 'red', 'blue', 'magenta', 'cyan', 'red', 'blue', 'magenta', 'cyan', 'red', 'blue', 'magenta', 'cyan']
colors_features_full = colors_features + ['green'] * 30
cmap_cbar = mpl.colors.ListedColormap(['orange', 'lightgray', 'sienna', 'red', 'blue', 'magenta', 'cyan', 'green'])
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
boundslabel = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig(path_save + 'miniscope_tm_full.svg', format='svg', dpi=256)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig(path_save + 'miniscope_tm_full.png', format='png', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig(path_save + 'miniscope_tm_cbar.svg', format='svg', dpi=256)

project_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\DeepLocoTM_MdV_Miniscope_ClosedLoop-AnaG-2023-04-02-MOBILE\\labeled-data\\'
animal_folder = 'MC16848_149_22_0.275_0.275_tied_1_8_crop'
img_name = 'img02399.png'
csv_name = 'CollectedData_AnaG.csv'
im = plt.imread(os.path.join(project_path, animal_folder, img_name))
#load csv
labeled_data = pd.read_csv(os.path.join(project_path, animal_folder, csv_name), header=1)
img_path = os.path.join('labeled-data', animal_folder, img_name)
coord_row = np.where(labeled_data.iloc[:, 0] == img_path)[0][0]
coords = labeled_data.iloc[coord_row, 1:].astype(float)
colors_features = ['red', 'magenta', 'blue', 'cyan']
cmap_cbar = mpl.colors.ListedColormap(['red', 'magenta', 'blue', 'cyan'])
bounds = [1, 2, 3, 4, 5]
boundslabel = ['FR', 'HR', 'FL', 'HL']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features[count_i])
plt.savefig(path_save + 'cltm_crop.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig(path_save + '\cltm_crop_cbar.svg', format='svg', dpi=256)

project_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\DeepLocoTM_ClosedLoop-Tailbase-Alice-2023-05-23-MOBILE\\labeled-data\\'
animal_folder = 'MC16850_148_24_0.275_0.275_tied_1_17_crop'
img_name = 'img09327.png'
csv_name = 'CollectedData_Alice.csv'
im = plt.imread(os.path.join(project_path, animal_folder, img_name))
#load csv
labeled_data = pd.read_csv(os.path.join(project_path, animal_folder, csv_name), header=1)
img_path = os.path.join('labeled-data', animal_folder, img_name)
coord_row = np.where(labeled_data.iloc[:, 2] == img_name)[0][0]
coords = labeled_data.iloc[coord_row, 3:].astype(float)
colors_features = ['red', 'magenta', 'blue', 'cyan', 'green']
cmap_cbar = mpl.colors.ListedColormap(['red', 'magenta', 'blue', 'cyan', 'green'])
bounds = [1, 2, 3, 4, 5, 6]
boundslabel = ['FR', 'HR', 'FL', 'HL', 'Tailbase']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features[count_i])
plt.savefig(path_save + 'cltm_crop_tailbase.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig(path_save + 'cltm_crop_tailbase_cbar.svg', format='svg', dpi=256)
