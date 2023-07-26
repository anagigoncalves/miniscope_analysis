import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

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
colors_features = ['orange', 'orange', 'lightgray', 'lightgray', 'sienna', 'sienna', 'blue', 'red', 'cyan',
    'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta']
colors_features_full = colors_features + ['green'] * 30
cmap_cbar = mpl.colors.ListedColormap(['orange', 'lightgray', 'sienna', 'blue', 'red', 'cyan', 'magenta', 'green'])
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
boundslabel = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig('J:\\Miniscope figures\\for methods\\cl_tm_full.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig('J:\\Miniscope figures\\for methods\\cl_tm_full_cbar.svg', format='svg', dpi=256)

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
colors_features = ['orange', 'orange', 'lightgray', 'lightgray', 'sienna', 'sienna', 'blue', 'red', 'cyan',
    'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta']
colors_features_full = colors_features + ['green'] * 30
cmap_cbar = mpl.colors.ListedColormap(['orange', 'lightgray', 'sienna', 'blue', 'red', 'cyan', 'magenta', 'green'])
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
boundslabel = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig('J:\\Miniscope figures\\for methods\\miniscope_tm_full.svg', format='svg', dpi=256)
fig1, ax1 = plt.subplots(figsize=(25, 15), tight_layout=True)
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=dotsize, color=colors_features_full[count_i])
plt.savefig('J:\\Miniscope figures\\for methods\\miniscope_tm_full.png', format='png', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig('J:\\Miniscope figures\\for methods\\miniscope_tm_cbar.svg', format='svg', dpi=256)

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
plt.savefig('J:\\Miniscope figures\\for methods\\cltm_crop.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig('J:\\Miniscope figures\\for methods\\cltm_crop_cbar.svg', format='svg', dpi=256)

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
plt.savefig('J:\\Miniscope figures\\for methods\\cltm_crop_tailbase.svg', format='svg', dpi=256)
fig2, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,
            ticks=bounds, spacing='proportional', orientation='horizontal')
plt.savefig('J:\\Miniscope figures\\for methods\\cltm_crop_tailbase_cbar.svg', format='svg', dpi=256)
