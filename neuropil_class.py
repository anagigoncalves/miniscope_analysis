import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import os
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from skimage import io
from skimage.transform import downscale_local_mean
from wavelet_transform_fun import *
from sklearn.metrics import davies_bouldin_score as db_score
'''The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.'''
#from sklearn.decomposition import PCA

fsize=18

class Neuropil:
    def __init__(self,path,trial):
        trial_name = 'T'+str(trial)+'_reg.tif'
        self.path = path
        self.trial = trial
        self.fname_video = os.path.join(path,trial_name)
        if os.path.exists((os.path.join(path,'Mask.png'))):
            mask = Image.open(os.path.join(path,'Mask.png'))
            self.mask = asarray(mask)<255
            m = 'masked'
        else: 
            self.mask = None
            m = None
        if not os.path.exists(os.path.join(self.path,'T'+str(trial)+'_neuropil')):
            os.mkdir(os.path.join(self.path,'T'+str(trial)+'_neuropil'))
        self.fname_neuropil = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_neuropil_signal_'+m+'.csv')
        self.fname_clusters = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_clusters_'+m+'.csv')
        self.fname_dendrogram = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_dendrogram_'+m+'.png')
        self.fname_events_neuropil = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_neuropil_events_'+m+'.csv')
        self.fname_corr_mat = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_correlation_matrix_'+m+'.csv')


    def grid_division(self, downscale_factor=5, save=True):
        image_stack = io.ImageCollection(self.fname_video, conserve_memory=True)
        print('Spatial downsampling: grid size: ', downscale_factor)
        image_stack = image_stack.concatenate()
        if self.mask is not None:
            for f in range(image_stack.shape[0]):
                image_stack[f][self.mask]=0
        ds_image_stack = downscale_local_mean(image_stack,(1,downscale_factor,downscale_factor))
        # if self.mask is not None:
        #     ds_image_stack[]

        self.neurpil = pd.DataFrame()
        for i in range(ds_image_stack.shape[1]): 
            for j in range(ds_image_stack.shape[2]):
                signal_ij = ds_image_stack[:,i,j]  
                if not np.all((signal_ij==0)):
                    idx = 'i'+str(i)+'_j'+str(j)
                    self.neurpil[idx]=signal_ij
        
        if save:
            print('Saving neuropil signal for ',self.fname_video)
            self.neurpil.to_csv(self.fname_neuropil)
        return 

    def neuropil_clustering_by_trial(self,th_cluster=0.6, save=True):

        if os.path.exists(self.fname_neuropil):
            neuropil = pd.read_csv(self.fname_neuropil)
        else:
            self.grid_division()
            neuropil = self.neurpil
        print('Computing pairwise distance')
        distance = pdist(neuropil.T,'correlation')
        # self.m = squareform(distance)
        print('Performing hierachical clustering')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
        nr_clusters = np.unique(idx)
        # TODO exclude cluster < 3% of the all image size
        print(nr_clusters[-1])
        fig = plt.figure(figsize=(20,15))
        dn = dendrogram(z,above_threshold_color='y',no_labels=True,orientation='top') 

        n_grid_i = 0
        n_grid_j = 0
        for i,pos in enumerate(neuropil.columns[1:]):
            n_grid_i_old = int(pos[1:pos.find('_')])
            n_grid_j_old = int(pos[pos.find('j')+1:]) 
            if n_grid_i_old>n_grid_i:
                n_grid_i=n_grid_i_old
            if n_grid_j_old>n_grid_j:
                n_grid_j=n_grid_j_old

        self.cluster_image = np.zeros((n_grid_i+1,n_grid_j+1))
        for i,pos in enumerate(neuropil.columns[1:]):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            self.cluster_image[ind_i,ind_j]=idx[i]
        if save:
            print('Saving clustering output for '+self.fname_neuropil)
            np.savetxt(self.fname_clusters,self.cluster_image,delimiter=',')
            fig.savefig(self.fname_dendrogram)
        return self.cluster_image

    def pixel_trace():
        # TODO plot pixel trace: same color cluster, intensity.
        return

    def Ca_events_neuropil(self, show_fig=False, save=True):
        if os.path.exists(self.fname_neuropil):
            neuropil = pd.read_csv(self.fname_neuropil)
        else:
            self.grid_division()
            neuropil = self.neurpil

        self.events_neuropil=pd.DataFrame(columns=neuropil.columns)
        for i_j in neuropil.columns: 
            signal_ij=self.neurpil[i_j]
            print('Computing continuos wavelet transformation for neuropil signal')
            cwt2d, ridges, events_cwt = wavelet_transform_morse(signal_ij,gamma=3,beta=2,min_scale=20,min_peak_position=15,min_freq=1, max_freq=20)
            self.events_neuropil[i_j]=events_cwt
            if show_fig:
                fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  
                plot_cwt2d_trace(cwt_ax, cwt2d, signal_ij)
                plot_ridges_trace_events(ridges_ax, ridges, signal_ij, events_cwt)
                plt.show()
            if save:
                print('Saving detected events in '+self.fname_neuropil)
                self.events_neuropil.to_csv(self.fname_events_neuropil)
        return
    
    #TODO pca on neurpil signal        
    # def pca_neuropil_signal(self, show_fig=False, save=True):

    @staticmethod
    def neuropil_clustering(path,fname,th_cluster=0.6, save=True):
        neuropil = pd.read_csv(os.path.join(path,fname))
        print('Computing pairwise distance')
        distance = pdist(neuropil.T,'correlation')
        # self.m = squareform(distance)
        print('Performing hierachical clustering')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
        nr_clusters = np.unique(idx)
        # TODO exclude cluster < 3% of the all image size
        print(nr_clusters[-1])

        n_grid_i = 0
        n_grid_j = 0
        for i,pos in enumerate(neuropil.columns[2:]):
            
            n_grid_i_old = int(pos[1:pos.find('_')])
            n_grid_j_old = int(pos[pos.find('j')+1:]) 
            if n_grid_i_old>n_grid_i:
                n_grid_i=n_grid_i_old
            if n_grid_j_old>n_grid_j:
                n_grid_j=n_grid_j_old

        cluster_image = np.zeros((n_grid_i+1,n_grid_j+1))
        for i,pos in enumerate(neuropil.columns[2:]):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            cluster_image[ind_i,ind_j]=idx[i]
        if save:
            print('Saving clustering output')
            np.savetxt(os.path.join(path,'clusters_idx.csv'),cluster_image,delimiter=',')
        return 



    