from matplotlib import image
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
import os
import pandas as pd
from skimage import io
from PIL import Image

path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'


for trial in range(1,24):
    print('trial number: ', trial)
    neuropil = Neuropil(path, trial)
    
    # neuropil.grid_division()

    cluster_image = neuropil.neuropil_clustering(save = True)
    fig = plt.figure(figsize=(20,20), tight_layout=True)
    plt.imshow(cluster_image, aspect='auto', cmap='inferno')
    fig.savefig(neuropil.fname_clusters[:-4]+'_th_0.4.png')

    # TODO EVALUATE CLUSTER QUALITY:
    # Internal cluster validation : The clustering result is evaluated based on the data clustered itself (internal information) 
    #   without reference to external information.
    # External cluster validation : Clustering results are evaluated based on some externally known result, 
    #   such as externally provided class labels. --> CLUSTER OF ROIS
    # INTRA CLUSTER CORRELATION COEFFICIENT: compare variance within a cluster with variance between clusters --> v_b/v_w+v_b
    # Dunn index to identify sets of clusters that are compact, with a small variance between members of the cluster, 
    # and well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster variance.
    # Higher the Dunn index value, better is the clustering.

    # neuropil.Ca_events_neuropil()