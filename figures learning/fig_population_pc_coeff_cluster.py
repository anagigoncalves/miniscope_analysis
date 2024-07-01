import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import KMeans as kmean
from sklearn.metrics import silhouette_score

# Input data
protocol = 'tied baseline'
load_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\' + protocol + ' S1\\'
save_path = 'J:\\LocoCF\\Miniscopes cluster cells in PC space (tied baseline session)\\cluster pc space\\' + protocol + ' S1\\'

# Load PC coefficients data
pc_coeff = pd.read_csv(os.path.join(load_path, 'pc_coeff_df_' + '_'.join(protocol.split(' ')) + '.csv'))

#### Get optimal number of clusters based on silhouette score
range_n_clusters = np.arange(2, 21)
silhouette_avg = np.zeros(len(range_n_clusters))
sse = np.zeros(len(range_n_clusters))
for idx, n_clusters in enumerate(range_n_clusters):
    cluster_obj = kmean(n_clusters=n_clusters, init='k-means++', n_init=1000, max_iter=300,
                        random_state=10).fit(pc_coeff[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])
    silhouette_avg[idx] = silhouette_score(pc_coeff[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']], cluster_obj.labels_)
    sse[idx] = cluster_obj.inertia_

#Plot sum squared error (SSE)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
plt.plot(range_n_clusters, sse, marker='o', color='black', linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('SSE', fontsize=20)
ax.set_xlabel('Number of clusters', fontsize=20)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_sse'), dpi=256)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_sse.svg'), dpi=256)

#Plot silhouette score
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
plt.plot(range_n_clusters, silhouette_avg, marker='o', color='black', linewidth=2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('Silhouette score', fontsize=20)
ax.set_xlabel('Number of clusters', fontsize=20)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_silhouette_score_avg'), dpi=256)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_silhouette_score_avg.svg'), dpi=256)
    
### Optimal number of clusters is maximum of silhouette score
clusters_PCA = kmean(n_clusters=np.argmax(silhouette_avg)+2, init='k-means++', n_init=1000, max_iter=300,
        random_state=10).fit(pc_coeff[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])

### Cluster output on PC space (first 2 components)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(pc_coeff['PC1'].values, pc_coeff['PC2'].values, c=clusters_PCA.labels_, cmap='jet', s=10)
for c in range(len(clusters_PCA.cluster_centers_)):
    ax.scatter(clusters_PCA.cluster_centers_[c, 0], clusters_PCA.cluster_centers_[c, 1], marker='*', color='black', s=50)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('PC2 weights', fontsize=20)
ax.set_xlabel('PC1 weights', fontsize=20)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_PC_space'), dpi=256)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_PC_space.svg'), dpi=256)

### Cluster output on anatomical space
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
sc = ax.scatter(pc_coeff['coord_x'], pc_coeff['coord_y'], s=15, c=clusters_PCA.labels_, cmap='jet')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylabel('AP coordinate (mm)', fontsize=20)
ax.set_xlabel('ML coordinate (mm)', fontsize=20)
ax.set_title('ROIs cluster output', fontsize=20)
plt.gca().invert_yaxis()
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_ROI_space'), dpi=256)
plt.savefig(os.path.join(save_path, protocol.replace(' ', '_') + '_kmeans_ROI_space.svg'), dpi=256)

# Save dataframe as csv
pc_coeff['cluster_pca'] = clusters_PCA.labels_
pc_coeff.to_csv(
    os.path.join(save_path, 'pc_coeff_df_clusters_' + protocol.replace(' ','_') + '.csv'), sep=',', index=False)


