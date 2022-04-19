import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from skimage import io
from wavelet_transform_fun import *
from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA
import os
fsize=18

class Neuropil:
    def __init__(self,path,trial):
        trial_name = 'T'+str(trial)+'_reg.tif'
        self.path = path
        self.trial = trial
        self.fname_video = os.path.join(path,trial_name)        
        self.fname_neuropil = os.path.join(self.path,'T'+str(self.trial)+'_neuropil_signal.csv')
        self.fname_clusters = os.path.join(self.path,'T'+str(self.trial)+'_clusters.csv')
        self.fname_events_neuropil = os.path.join(self.path,'T'+str(self.trial)+'_neuropil_events.csv')


    def grid_division(self, downscale_factor=5, save=True):
        f = self.fname_video
        image_stack = io.ImageCollection(f, conserve_memory=True)
        image_stack = image_stack.concatenate()
        ds_image_stack = downscale_local_mean(image_stack,(1,downscale_factor,downscale_factor))
        self.neurpil = pd.DataFrame()
        for i in range(ds_image_stack.shape[1]): 
            for j in range(ds_image_stack.shape[2]):
                signal_ij = ds_image_stack[:,i,j]  
                idx = 'i'+str(i)+'_j'+str(j)
                self.neurpil[idx]=signal_ij
        if save:
            self.neurpil.to_csv(self.fname_neuropil)
        return 

    def neuropil_clustering(self,th_cluster=0.4, save=True):
        if os.path.exists(self.fname_neuropil):
            neuropil = pd.read_csv(self.fname_neuropil)
        else:
            self.grid_division()
            neuropil = self.neurpil

        distance = pdist(neuropil.T,'correlation')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
        nr_clusters = np.unique(idx)

        n_grid_i = int(neuropil.columns[-1][1:neuropil.columns[-1].find('_')])
        n_grid_j = int(neuropil.columns[-1][neuropil.columns[-1].find('j')+1:]) 
        self.cluster_image = np.zeros((n_grid_i+1,n_grid_j+1))
        for i,pos in enumerate(neuropil.columns[1:]):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            self.cluster_image[ind_i,ind_j]=idx[i]
        if save:
            np.save(self.fname_clusters)
        return self.cluster_image

    def Ca_events_neuropil(self, show_fig=False, save=True):
        if os.path.exists(self.fname_neuropil):
            neuropil = pd.read_csv(self.fname_neuropil)
        else:
            self.grid_division()
            neuropil = self.neurpil

        self.events_neuropil=pd.DataFrame(columns=neuropil.columns)
        for i_j in neuropil.columns: 
            signal_ij=self.neurpil[i_j]
            cwt2d, ridges, events_cwt = wavelet_transform_morse(signal_ij,gamma=3,beta=2,min_scale=20,min_peak_position=15,min_freq=1, max_freq=20)
            self.events_neuropil[i_j]=events_cwt
            if show_fig:
                fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  
                plot_cwt2d_trace(cwt_ax, cwt2d, signal_ij)
                plot_ridges_trace_events(ridges_ax, ridges, signal_ij, events_cwt)
                plt.show()
            if save:
                self.events_neuropil.to_csv(self.fname_events_neuropil)
        return
    #TODO pca on neurpil signal        
    # def pca_neuropil_signal(self, show_fig=False, save=True):


    