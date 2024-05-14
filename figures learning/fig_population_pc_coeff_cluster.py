import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import pandas as pd
import seaborn as sns

# Input data
load_path = 'J:\\LocoCF\\miniscopes learning\\'
protocol = 'split ipsi fast'

# Load data
pc_coeff = pd.read_csv(os.path.join(load_path, 'pc_coeff_df_' + '_'.join(protocol.split(' ')) + '.csv'))

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(pc_coeff['PC1'].values, pc_coeff['PC2'].values, c=pc_coeff['cluster_id'].values, cmap='jet', s=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('PC2 weights', fontsize=20)
ax.set_xlabel('PC1 weights', fontsize=20)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(pc_coeff['PC2'].values, pc_coeff['PC3'].values, c=pc_coeff['cluster_id'].values, cmap='jet', s=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('PC3 weights', fontsize=20)
ax.set_xlabel('PC2 weights', fontsize=20)

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(pc_coeff['PC1'].values, pc_coeff['PC3'].values, c=pc_coeff['cluster_id'].values, cmap='jet', s=10)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('PC3 weights', fontsize=20)
ax.set_xlabel('PC1 weights', fontsize=20)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter(pc_coeff['PC1'].values, pc_coeff['PC2'].values, pc_coeff['PC3'].values, c=pc_coeff['cluster_id'].values,
        s=10, cmap='jet')
