from matplotlib.cbook import boxplot_stats
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import asarray, loadtxt
import os
import pandas as pd
from time import time
from scipy.stats import pearsonr
from pandas.plotting import table 
import seaborn as sns
import matplotlib.patches as patches

path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'
# neuropil = Neuropil(path, 1)

# # KEEP fix clusters trhought the all session
# neuropil_session_signal = []
for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    
    # neuropil_trial_signal = pd.read_csv(neuropil.fname_neuropil)
    
    # neuropil_session_signal.append(neuropil_trial_signal)

    # neuropil.grid_division()

    cluster_image = neuropil.neuropil_clustering_by_trial(save = True)

    # cluster_image = pd.read_csv(neuropil.fname_clusters)
    nr_clusters = cluster_image.max().max()
    cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
    cmap.colors[0]=[1,1,1,1]
    fig = plt.figure(figsize=(20,20), tight_layout=True)    
    plt.imshow(cluster_image, aspect='auto', cmap=cmap)
    for i,c in enumerate(cmap.colors[1:]):
        fig.text(0.9,0.9-i*0.025,'cluster index: '+str(i+1), size = 16, bbox=dict(boxstyle ='round', fc=c))
    # plt.show()
    fig.savefig(neuropil.fname_clusters[:-4]+'.png')

    corr_mat = neuropil.clustered_pixels_correlation()

    # corr_mat = pd.read_csv(neuropil.fname_corr_mat, index_col=0)
    cluster_idx = pd.read_csv(neuropil.fname_cluster_idx)
    index = cluster_idx.index
    nr_clusters = cluster_idx['cluster_idx'].to_list()[-1]
    cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
    cmap.colors[0]=[1,1,1,1]
    print('Plotting correlation map')
    fig, ax = plt.subplots(1,figsize=(20,20))  
    ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
    for i in range(1,nr_clusters+1):
        p = index[cluster_idx['cluster_idx']==i].to_list()
        rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i], facecolor="none")
        ax.add_patch(rect)
    fig.savefig(neuropil.fname_corr_map)

    # neuropil.Ca_events_neuropil()

# # DONE: concatenate all trial
# df_session = pd.concat(neuropil_session_signal)
# df_session.to_csv(os.path.join(path,'neuropil_session_signal.csv'))


# # DONE: compute clusters for the entire session
# tic = time() 
# neuropil.neuropil_clustering(path, 'neuropil_session_signal.csv',th_cluster=0.5)
# toc =time()
# print(toc-tic)
# cluster_image = np.loadtxt(os.path.join(path,'clusters_map.csv'))
# fig = plt.figure(figsize=(20,20), tight_layout=True)
# plt.imshow(cluster_image, aspect='auto', cmap='inferno')
# fig.savefig(os.path.join(path,'clusters_th_0.5.png'))

# # DONE riordinare matrice segnale neuropil secondo cluster index
# cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx.csv'), index_col='pixel_idx')
# print(cluster_idx.columns)
# cluster_idx = cluster_idx.sort_values('cluster_idx')
# cluster_idx.to_csv(os.path.join(path,'clusters_idx_sorted.csv'))

# # DONE pairwise correlation between pixel sorted by index cluster
# cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx_sorted.csv'), index_col='pixel_idx')
# pixel_idx = cluster_idx.index.values.tolist()
# neuropil = pd.read_csv(os.path.join(path,'neuropil_session_signal.csv'), usecols=pixel_idx)
# neuropil=neuropil[pixel_idx]
# corr_mat=neuropil.corr('pearson')
# corr_mat.to_csv(os.path.join(path,'corr_mat.csv'))

# # DONE plot correlation map 
# cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx_sorted.csv'))
# index = cluster_idx.index
# nr_clusters = cluster_idx['cluster_idx'].to_list()[-1]
# corr_mat=pd.read_csv(os.path.join(path,'corr_mat.csv'), index_col=0)
# plt.matshow(corr_mat,cmap='coolwarm')
# plt.colorbar()
# plt.show()
# fig, ax = plt.subplots(1)  
# ax.matshow(corr_mat,cmap='coolwarm')
# for i in range(1,nr_clusters+1):
#     p = index[cluster_idx['cluster_idx']==i].to_list()
#     rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=1, edgecolor='black', facecolor="none")
#     ax.add_patch(rect)
# fig.savefig(os.path.join(path,'corr_map.png'))
# TODO correlation distribution within and between clusters --> da matrice triangolare

# TODO how correlation change with the thre different conditions (tied,split,after effects)

# TODO how correlation change trial-by-trial
