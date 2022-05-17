from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import os
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# root_path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05'
root_path = '/media/careylab/Samsung_T5/TM RAW FILES/tied baseline/MC8855/2021_04_04'
path_video = os.path.join(root_path,'Registered video') 
path_results = os.path.join(root_path,'Neuropil_analysis')
# n_trials = 23
n_trials = 6

'''************************************ SINGLE TRIAL NEUROPIL SIGNAL ************************************
for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)    

    trial_name = 'T'+str(trial)+'_reg.tif'
    fname_video = os.path.join(path_video,trial_name)    
    neurpil_signal, n_grid_ij = neuropil.grid_division(fname_video)

    print('Saving neuropil signal for trial ', trial)
    neurpil_signal.to_csv(neuropil.fname_neuropil)
    np.savetxt(os.path.join(path_results,'n_grid_ij.csv'),n_grid_ij,delimiter=',')
#  '''

'''***************************************** CONCATENATE TRIALS *****************************************
neuropil_session_signal = []

for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)
    neuropil_trial_signal = pd.read_csv(neuropil.fname_neuropil)
    neuropil_session_signal.append(neuropil_trial_signal)    

df_session = pd.concat(neuropil_session_signal)
print(df_session.columns)
df_session.drop("Unnamed: 0", axis=1, inplace=True)
df_session.to_csv(os.path.join(path_results,'neuropil_session_signal.csv'))
#  '''

'''****************************************** SESSION CLUSTERS ******************************************
neuropil = Neuropil(path_results)
neuropil_session_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), index_col=0)
#TODO normalizzare segnale
cluster_image, cluster_idx, z = neuropil.neuropil_h_clustering(neuropil_session_signal,th_cluster=0.55)

print('Saving clustering output for entire session')
np.savetxt(os.path.join(path_results,'clusters_map.csv'),cluster_image,delimiter=',')
cluster_idx.to_csv(os.path.join(path_results,'clusters_idx.csv'))

# print('Loading clustering output for entire session')
# cluster_image = pd.read_csv(os.path.join(path_results,'clusters_map.csv'), index_col=0).to_numpy()
# cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'),index_col=0)

print('Create color map')
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy()) 
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

print('Plotting clustering output for entire session')
fig_cluster = plt.figure(figsize=(20,20), tight_layout=True)
plt.imshow(cluster_image, aspect='auto', cmap=cmap)            
for i,c in enumerate(cluster_idx_list):
    fig_cluster.text(0.9,0.9-i*0.025,'cluster index: '+str(c), size = 16, bbox=dict(boxstyle ='round', fc=cmap.colors[i+1]))
fig_cluster.savefig(os.path.join(path_results,'clusters.png'))

# cluster_map = Image.fromarray(cluster_image)
# cluster_map.save(os.path.join(path_results,'clusters_bw.png'), format="png")

print('Plotting dendogram')
fig_dendogram = plt.figure(figsize=(20,15), tight_layout=True)
dn = dendrogram(z,above_threshold_color='y',no_labels=True,orientation='top') 
fig_dendogram.savefig(os.path.join(path_results,'dendrogram.png'))

#  '''

'''**************************************** SESSION CORRELATION *****************************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col=0)
pixel_idx = cluster_idx['pixel_idx'].tolist()
neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), usecols=pixel_idx)

print('Compute correlation map for entire session')
neuropil_signal=neuropil_signal[pixel_idx]
corr_mat=neuropil_signal.corr('pearson')
print('Saving correlation matrix')
corr_mat.to_csv(os.path.join(path_results,'corr_mat.csv'))

# print('Loading correlation output for entire session')
# cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col=0)
# pixel_idx = cluster_idx['pixel_idx'].tolist()
# corr_mat = pd.read_csv(os.path.join(path_results,'corr_mat.csv'), index_col=0)

print('Create color map')
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy()) 
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]
index = cluster_idx.index

print('Plotting correlation map')
fig, ax = plt.subplots(1,figsize=(20,20))  
ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
# for i,c in enumerate(cluster_idx_list):
#     p = index[cluster_idx['cluster_idx']==c].to_list()
#     rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i+1], facecolor="none")
#     ax.add_patch(rect)
fig.savefig(os.path.join(path_results,'correlation_map.png'))
#  '''

'''******************************* FIXED CLUSTERS SINGLE TRIAL CORRELATION ******************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'))
pixel_idx = cluster_idx['pixel_idx'].tolist()
c = cluster_idx['cluster_idx'].to_numpy()
cluster_idx_list = np.unique(c)
index = cluster_idx.index
nr_clusters = len(cluster_idx_list)

print('Create color map')
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)
    
    neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), usecols=pixel_idx)

    print('Compute correlation map for trial ', trial)
    neuropil_signal=neuropil_signal[pixel_idx]
    corr_mat=neuropil_signal.corr('pearson')

    print('Saving correlation matrix')
    corr_mat.to_csv(neuropil.fname_corr_mat)

    # print('Loading correlation output for entire session')
    # corr_mat = pd.read_csv(neuropil.fname_corr_mat, index_col=0)

    print('Plotting correlation map')
    fig, ax = plt.subplots(1,figsize=(20,20))  
    ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
    for i,c in enumerate(cluster_idx_list):
        p = index[cluster_idx['cluster_idx']==c].to_list()
        rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i+1], facecolor="none")
        ax.add_patch(rect)
    fig.savefig(neuropil.fname_corr_map)

# '''

'''*************************************** SINGLE TRIAL CLUSTERS ****************************************
for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)

    neuropil_signal = pd.read_csv(neuropil.fname_neuropil, index_col=0)
    cluster_image, cluster_idx, z = neuropil.neuropil_h_clustering(neuropil_signal,th_cluster=0.5)

    print('Saving clustering output for trial ', trial)
    np.savetxt(neuropil.fname_clusters,cluster_image,delimiter=',')
    cluster_idx.to_csv(neuropil.fname_cluster_idx)

    # print('Loading clustering output for trial ', trial)
    # cluster_image = pd.read_csv(neuropil.fname_clusters, index_col=0).to_numpy()
    # cluster_idx = pd.read_csv(neuropil.fname_cluster_idx, index_col=0)

    print('Create color map')
    cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy()) 
    nr_clusters = len(cluster_idx_list)
    cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
    cmap.colors[0]=[1,1,1,1]

    print('Plotting clustering output for trial ', trial)
    fig_cluster = plt.figure(figsize=(20,20), tight_layout=True)
    plt.imshow(cluster_image, aspect='auto', cmap=cmap)            
    for i,c in enumerate(cluster_idx_list):
        fig_cluster.text(0.9,0.9-i*0.025,'cluster index: '+str(c), size = 16, bbox=dict(boxstyle ='round', fc=cmap.colors[i+1]))
    fig_cluster.savefig(neuropil.fname_clusters[:-4]+'.png')

    print('Plotting dendogram')
    fig_dendogram = plt.figure(figsize=(15,20), tight_layout=True)
    dn = dendrogram(z,above_threshold_color='y',no_labels=True,orientation='top') 
    fig_dendogram.savefig(neuropil.fname_dendrogram )
#  '''

'''************************************* SINGLE TRIAL CORRELATION ***************************************
for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)

    print('Loading clustering output for trial ', trial)
    cluster_idx = pd.read_csv(neuropil.fname_cluster_idx, index_col=0)
    pixel_idx = cluster_idx['pixel_idx'].tolist()
    neuropil_signal = pd.read_csv(neuropil.fname_neuropil, usecols=pixel_idx)

    print('Compute correlation map for trial ', trial)
    neuropil_signal=neuropil_signal[pixel_idx]
    corr_mat=neuropil_signal.corr('pearson')
    print('Saving correlation matrix')
    corr_mat.to_csv(neuropil.fname_corr_mat)

    print('Loading correlation output for trial ', trial)
    cluster_idx = pd.read_csv(neuropil.fname_cluster_idx, index_col='pixel_idx')
    pixel_idx = cluster_idx['pixel_idx'].tolist()
    corr_mat = pd.read_csv(neuropil.fname_corr_mat, index_col=0)

    print('Create color map for trial ', trial)
    cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy()) 
    nr_clusters = len(cluster_idx_list)
    cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
    cmap.colors[0]=[1,1,1,1]
    index = cluster_idx.index

    print('Plotting correlation map for trial ', trial)
    fig, ax = plt.subplots(1,figsize=(20,20))  
    ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
    for i,c in enumerate(cluster_idx_list):
        p = index[cluster_idx['cluster_idx']==c].to_list()
        rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i+1], facecolor="none")
        ax.add_patch(rect)
    fig.savefig(neuropil.fname_corr_map)
#  '''

'''************************************************* PCA ************************************************
cmap = cm.get_cmap('inferno', n_trials+1)
# tied
cmap.colors[0]=[0.5,0.5,0.5,0]
cmap.colors[1]=[0.5,0.5,0.5,0]
cmap.colors[2]=[0.5,0.5,0.5, 1]
# split
cmap.colors[3]=[1,0,0,1]
cmap.colors[4]=[1,0,0,0]
cmap.colors[5]=[1,0,0,0]
cmap.colors[6]=[1,0,0,0]
cmap.colors[7]=[1,0,0,0]
cmap.colors[8]=[1,0,0,0]
cmap.colors[9]=[1,0,0,0]
cmap.colors[10]=[1,0,0,0]
cmap.colors[11]=[1,0,0,0]
cmap.colors[12]=[0.5,0,0.5,1]
# washout
cmap.colors[13]=[0,0,1,1]
cmap.colors[14]=[0,0,1,0]
cmap.colors[15]=[0,0,1,0]
cmap.colors[16]=[0,0,1,0]
cmap.colors[17]=[0,0,1,0]
cmap.colors[18]=[0,0,1,0]
cmap.colors[19]=[0,0,1,0]
cmap.colors[20]=[0,0,1,0]
cmap.colors[21]=[0,0,1,0]
cmap.colors[22]=[0,0,1,0]
cmap.colors[23]=[0,0,0,1]

neuropil = Neuropil(path_results)
c = pd.read_csv(os.path.join(path_results,'trials_index.csv'), index_col=0)
c = c['trial_index'].to_numpy()
neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), index_col=0)

# neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), index_col=0)
# neuropil_signal = neuropil_signal.T
# c = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col=0)
# c = c['cluster_idx'].to_numpy()
# cluster_idx_list = np.unique(c)
# nr_clusters = len(cluster_idx_list)
# cmap = cm.get_cmap('inferno', int(nr_clusters)+1)

neuropil.pca_neuropil_signal(neuropil_signal,c=c, cmap=cmap)
#  '''

'''*************************************** ROI EXTRACT CLUSTERS *****************************************
rois = pd.read_csv(os.path.join(path_results,'df_extract_raw_tied.csv'))
rois = rois[rois.columns[2:]].replace(np.nan,0.0)

th_cluster = 0.45

print('Computing pairwise distance')
distance = pdist(rois.T,'correlation')

print('Performing hierachical clustering')
z = linkage(y=distance, method='complete', metric='euclidean')
idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output

d = {'roi_idx': rois.columns, 'cluster_idx': idx}
cluster_idx = pd.DataFrame(data=d)
cluster_idx = cluster_idx.sort_values('cluster_idx')
cluster_idx.to_csv(os.path.join(path_results,'roi_EXTRACT_clusters_idx.csv'))

cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
nr_clusters = len(cluster_idx_list)

fig = plt.figure(figsize=(20,15))
dn = dendrogram(z,above_threshold_color='y',no_labels=True,orientation='top') 
plt.show()
cmap = cm.get_cmap('viridis', int(nr_clusters)+1)

pixel_to_um = 0.608
coord_cell = np.load(os.path.join(path_results,'coord_tied.npy'),allow_pickle=True)
coord_cell = coord_cell*pixel_to_um

fig = plt.figure(figsize=(10, 10), tight_layout=True)
path_mask = path_results #'/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/first mask/'
mask = Image.open(os.path.join(path_mask,'Mask.png'))
cmap_mask = cm.get_cmap('inferno',2)
cmap_mask.colors[0]=[0,0,0,1]
cmap_mask.colors[1]=[1,1,1,0]
plt.imshow(mask, cmap=cmap_mask)
for i,r in enumerate(rois.columns):
    c = int(cluster_idx['cluster_idx'][cluster_idx['roi_idx']==r].to_numpy())
    plt.scatter(coord_cell[i][:, 0], coord_cell[i][:, 1], color=cmap.colors[c], s=1, alpha=0.6)
fig.savefig(os.path.join(path_results,'roi_EXTRACT_cluster_map.png'), transparent=True)
#  '''

'''****************************************  ROI FROM RAW VIDEO *****************************************
# extract raw signal from each roi 
from skimage import io
pixel_to_um = 0.608
coord_cell = np.load(os.path.join(path_results,'coord_tied.npy'),allow_pickle=True)
coord_cell = coord_cell*pixel_to_um
rois = pd.read_csv(os.path.join(path_results,'df_extract_raw_tied.csv'), index_col='time')

for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)
    trial_name = 'T'+str(trial)+'_reg.tif'
    fname_video = os.path.join(path_video,trial_name)
    image_stack = io.ImageCollection(fname_video, conserve_memory=True)
    image_stack = image_stack.concatenate()
    for i,r in enumerate(rois.columns[1:]):
        roi_signal = []
        for f in range(image_stack.shape[0]):
            roi_signal.append(np.mean(image_stack[f][coord_cell[i].astype(int)]))
        rois[r][rois['trial']==trial]=roi_signal

rois.to_csv(os.path.join(path_results,'df_rois_raw_split.csv'))
#  '''

# '''************************************* ROI CLUSTERS FROM RAW VIDEO ************************************
# compute clusters with raw signals
rois = pd.read_csv(os.path.join(path_results,'df_rois_raw_split.csv'))
rois = rois[rois.columns[2:]]

th_cluster = 0.5

distance = pdist(rois.T,'correlation')
print('Performing hierachical clustering')
z = linkage(y=distance, method='complete', metric='euclidean')
idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
d = {'roi_idx': rois.columns, 'cluster_idx': idx}
cluster_idx = pd.DataFrame(data=d)
cluster_idx = cluster_idx.sort_values('cluster_idx')
cluster_idx.to_csv(os.path.join(path_results,'roi_RAW_clusters_idx.csv'))
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('viridis', int(nr_clusters)+1)

pixel_to_um = 0.608
coord_cell = np.load(os.path.join(path_results,'coord_tied.npy'),allow_pickle=True)
coord_cell = coord_cell*pixel_to_um

fig = plt.figure(figsize=(10, 10), tight_layout=True)
mask = Image.open(os.path.join(path_results,'Mask.png'))
cmap_mask = cm.get_cmap('inferno',2)
cmap_mask.colors[0]=[0,0,0,1]
cmap_mask.colors[1]=[1,1,1,0]
plt.imshow(mask, cmap=cmap_mask)
for i,r in enumerate(rois.columns):
    c = int(cluster_idx['cluster_idx'][cluster_idx['roi_idx']==r].to_numpy())
    plt.scatter(coord_cell[i][:, 0], coord_cell[i][:, 1], color=cmap.colors[c], s=1, alpha=0.6)

fig.savefig(os.path.join(path_results,'roi_RAW_cluster_map.png'), transparent=True)
#  '''

'''********************************* ROI CORRELATION WITH CLUSTER SIGNAL ********************************

# correlation matrix
# event detection

#  '''