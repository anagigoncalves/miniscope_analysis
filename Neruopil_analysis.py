from matplotlib.cbook import boxplot_stats
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from numpy import asarray, loadtxt
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

df_session = pd.concat(neuropil_session_signal)
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


# '''************************************** SINGLE TRIAL ANALYSIS ***************************************
cluster_idx = pd.read_csv(os.path.join(path,'clusters_idx.csv'))
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
index = cluster_idx.index
nr_clusters = len(cluster_idx_list)
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

for trial in range(12,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    
    # cluster_image = neuropil.neuropil_clustering_by_trial(save = True)

    corr_mat = neuropil.clustered_pixels_correlation(cluster_idx, cmap)

    # neuropil.clustered_pixel_trace(cmap, cluster_idx, k=50)

    # neuropil.Ca_events_neuropil()
#  '''

'''*************************************** PCA ***************************************
# neuropil.pca_neuropil_signal(path, 'neuropil_session_signal.csv',trials=np.arange(1,24))
#  '''
