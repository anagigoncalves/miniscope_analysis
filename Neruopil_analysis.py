from matplotlib import image
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
import os
import pandas as pd
from skimage import io
from PIL import Image
from time import time

path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'
# KEEP fix clusters trhought the all session
# neuropil_session_signal = []
# for trial in range(1,24):
#     print('trial number: ', trial)
#     neuropil = Neuropil(path, trial)
    
    # neuropil_trial_signal = pd.read_csv(neuropil.fname_neuropil)
    # neuropil_session_signal.append(neuropil_trial_signal)
    # neuropil.grid_division()

    # cluster_image = neuropil.neuropil_clustering_by_trial(save = True)
    # fig = plt.figure(figsize=(20,20), tight_layout=True)
    # plt.imshow(cluster_image, aspect='auto', cmap='inferno')
    # fig.savefig(neuropil.fname_clusters[:-4]+'_th_0.4.png')

    # neuropil.Ca_events_neuropil()

# df_session = pd.concat(neuropil_session_signal, ignore_index=True)
# df_session.to_csv(os.path.join(path,'neuropil_session_signal.csv'))
neuropil = Neuropil(path, 1)
# tic = time()
# neuropil.neuropil_clustering(path, 'neuropil_session_signal.csv',th_cluster=0.5, save = True)
# toc =time()
# print(toc-tic)
cluster_image = pd.read_csv(os.path.join(path,'clusters_idx.csv'))
# fig = plt.figure(figsize=(20,20), tight_layout=True)
# plt.imshow(cluster_image, aspect='auto', cmap='inferno')
# fig.savefig(os.path.join(path,'clusters_th_0.5.png'))
# TODO riordinare matrice segnale neuropil secondo cluster index
# TODO pairwise correlation between pixel sorted by index cluster
# TODO correlation distribution within and between clusters --> da matrice triangolare
# TODO how correlation change with the thre different conditions (tied,split,after effects)
# TODO how correlation change trial-by-trial
