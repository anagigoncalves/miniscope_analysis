import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

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

colors_features = ['orange', 'orange', 'brown', 'brown', 'lightgray', 'lightgray', 'blue', 'red', 'cyan',
    'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta', 'blue', 'red', 'cyan', 'magenta']
colors_features_full = colors_features + ['green'] * 30

cmap_cbar = mpl.colors.ListedColormap(['orange', 'brown', 'lightgray', 'blue', 'red', 'cyan', 'magenta', 'green'])
#bounds = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
bounds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
boundslabel = ['nose', 'ear', 'bodycenter', 'FR', 'FL', 'HR', 'HL', 'tail']
norm = mpl.colors.BoundaryNorm(bounds, cmap_cbar.N)

fig, ax1 = plt.subplots()
im_plot = plt.imshow(im)
for count_i, i in enumerate(np.arange(0, len(coords), 2)):
    plt.scatter(coords[i], coords[i+1], s=20, color=colors_features_full[count_i])
    
fig, ax2 = plt.subplots(figsize=(5,1))
cb2 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap_cbar, norm=norm,  
            ticks=bounds, spacing='proportional', orientation='horizontal')
cb2.ax2.set_yticklabels(boundslabel)
