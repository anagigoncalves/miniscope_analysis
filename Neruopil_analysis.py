from this import d
from matplotlib.cbook import boxplot_stats
from pyparsing import col
from analytic_wavelet.transform import rotate
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from numpy import asarray, loadtxt, reshape
import os
import pandas as pd
from time import time
from scipy.stats import pearsonr
from pandas.plotting import table 
import seaborn as sns


path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'

'''************************************ SINGLE TRIAL NEUROPIL SIGNAL ************************************
for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)    
    neuropil.grid_division()

#  '''

'''***************************************** CONCATENATE TRIALS *****************************************
neuropil_session_signal = []

for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    neuropil_trial_signal = pd.read_csv(neuropil.fname_neuropil)
    neuropil_session_signal.append(neuropil_trial_signal)    

df_session = pd.concat(neuropil_session_signal
df_session.to_csv(os.path.join(path,'neuropil_session_signal.csv'))
#  '''

'''****************************************** SESSION CLUSTERS ******************************************
neuropil = Neuropil(path)
neuropil.neuropil_clustering(path, 'neuropil_session_signal.csv')
#  '''

'''**************************************** SESSION CORRELATION *****************************************
neuropil = Neuropil(path)
cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx.csv'), index_col='pixel_idx')
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy()) 
pixel_idx = cluster_idx.index.values.tolist()
neuropil = pd.read_csv(os.path.join(path,'neuropil_session_signal.csv'), usecols=pixel_idx)
neuropil=neuropil[pixel_idx]
corr_mat=neuropil.corr('pearson')
corr_mat.to_csv(os.path.join(path,'corr_mat.csv'))

index = cluster_idx.index
print('Plotting correlation map')
fig, ax = plt.subplots(1,figsize=(20,20))  
ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
for i,c in enumerate(cluster_idx_list):
    p = index[cluster_idx['cluster_idx']==c].to_list()
    rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i+1], facecolor="none")
    ax.add_patch(rect)
fig.savefig(os.path.join(path,'correlation_matrix.csv'))
#  '''

'''************************************ FIXED CLUSTERS SINGLE TRIAL *************************************
cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx.csv'))
c = cluster_idx['cluster_idx'].to_numpy()
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
index = cluster_idx.index
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    
    corr_mat = neuropil.clustered_pixels_correlation(cluster_idx, cmap)

    corr_mat = pd.read_csv(os.path.join(path,neuropil.fname_corr_mat), index_col=0)
    cluster_corr_mat = np.zeros((nr_clusters,nr_clusters))
    for i,x in enumerate(cluster_idx_list):
        px = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==x].to_list()
        for j,y in enumerate(cluster_idx_list):
            py = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==y].to_list()
            sub_set = corr_mat[px].loc[py].to_numpy()
            if x==y:
                sub_set[np.tril_indices(len(py), -1)] = np.nan
                cluster_corr_mat[i,j] = np.nanmean(sub_set)
            else:
                cluster_corr_mat[i,j] = np.mean(sub_set)
    fig, ax = plt.subplots(1,figsize=(20,20))  
    ax.matshow(cluster_corr_mat,cmap='coolwarm',vmin=0, vmax=1)
    ax.set_xticks(np.arange(nr_clusters))
    ax.set_yticks(np.arange(nr_clusters))
    labels = ['cluster ' +str(i) for i in cluster_idx_list]
    ax.set_xticklabels(labels,fontsize=20, rotation=45)
    ax.set_yticklabels(labels,fontsize=20)
    fig.savefig(os.path.join(path,'T'+str(trial)+'_cluster_corr_map.png'))
    cluster_corr_mat = pd.DataFrame(data=cluster_corr_mat,index=cluster_idx_list,columns=cluster_idx_list)
    cluster_corr_mat.to_csv(os.path.join(path,'T'+str(trial)+'_cluster_corr_mat.csv'))


    neuropil_signal = pd.read_csv(os.path.join(path,neuropil.fname_neuropil), index_col=0)
    neuropil_signal = neuropil_signal.T
    neuropil.pca_neuropil_signal(neuropil_signal,c=c, cmap=cmap)

# '''

'''*************************************** SINGLE TRIAL ANALYSIS ****************************************
for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    
    # cluster_image = neuropil.neuropil_clustering_by_trial(save = True)

    # cluster_idx = pd.read_csv(neuropil.fname_cluster_idx)
    # c = cluster_idx['cluster_idx'].to_numpy()
    # cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
    # index = cluster_idx.index
    # nr_clusters = len(cluster_idx_list)
    # cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
    # cmap.colors[0]=[1,1,1,1]

    # corr_mat = neuropil.clustered_pixels_correlation(cluster_idx, cmap)

    # # neuropil_signal = pd.read_csv(neuropil.fname_neuropil, index_col=0)
    # neuropil_signal = neuropil_signal.T
    # neuropil.pca_neuropil_signal(neuropil_signal,c=c, cmap=cmap)

    # corr_mat = pd.read_csv(neuropil.fname_corr_mat, index_col=0)
    # cluster_corr_mat = np.zeros((nr_clusters,nr_clusters))
    # for i,x in enumerate(cluster_idx_list):
    #     px = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==x].to_list()
    #     for j,y in enumerate(cluster_idx_list):
    #         py = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==y].to_list()
    #         sub_set = corr_mat[px].loc[py].to_numpy()
    #         if x==y:
    #             sub_set[np.tril_indices(len(py), -1)] = np.nan
    #             cluster_corr_mat[i,j] = np.nanmean(sub_set)
    #         else:
    #             cluster_corr_mat[i,j] = np.mean(sub_set)
    # fig, ax = plt.subplots(1,figsize=(20,20))  
    # ax.matshow(cluster_corr_mat,cmap='coolwarm',vmin=0, vmax=1)
    # ax.set_xticks(np.arange(nr_clusters))
    # ax.set_yticks(np.arange(nr_clusters))
    # labels = ['cluster ' +str(i) for i in cluster_idx_list]
    # ax.set_xticklabels(labels,fontsize=20, rotation=45)
    # ax.set_yticklabels(labels,fontsize=20)
    # fig.savefig(neuropil.fname_corr_map[-4:]+'clusters.png')

    # neuropil.clustered_pixel_trace(cmap, cluster_idx, k=50)

    neuropil.Ca_events_neuropil()
#  '''

'''************************************************* PCA ************************************************
neuropil = Neuropil(path)

# neuropil_signal = pd.read_csv(os.path.join(path,'neuropil_session_signal.csv'), index_col=0)
# c = pd.read_csv(os.path.join(path,'trials_index.csv'), index_col=0)
# c = c['trial_index'].to_numpy()
# cmap = cm.get_cmap('hsv', 24)

neuropil_signal = pd.read_csv(os.path.join(path,'neuropil_session_signal.csv'), index_col=0)
neuropil_signal = neuropil_signal.T
c = pd.read_csv(os.path.join(path,'clusters_idx.csv'), index_col=0)
c = c['cluster_idx'].to_numpy()
cluster_idx_list = np.unique(c)
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]


neuropil.pca_neuropil_signal(neuropil_signal,c=c, cmap=cmap)
#  '''
