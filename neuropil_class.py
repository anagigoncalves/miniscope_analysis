import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
from numpy import asarray
import os
import pandas as pd
from PIL import Image
from skimage import io
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from skimage.transform import downscale_local_mean
from wavelet_transform_fun import *
from sklearn.metrics import davies_bouldin_score as db_score



'''The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.'''


fsize=18

class Neuropil:
    def __init__(self,path,trial=1):
        self.path = path
        self.trial = trial
        if os.path.exists((os.path.join(path,'Mask.png'))):
            mask = Image.open(os.path.join(path,'Mask.png'))
            self.mask = asarray(mask)<255
        else: 
            self.mask = None
        if not os.path.exists(os.path.join(self.path,'T'+str(trial)+'_neuropil')):
            os.mkdir(os.path.join(self.path,'T'+str(trial)+'_neuropil'))
        self.fname_neuropil = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_neuropil_signal.csv')
        self.fname_clusters = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_clusters_trial.csv')
        self.fname_cluster_idx = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_clusters_idx.csv')
        self.fname_dendrogram = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_dendrogram.png')
        self.fname_events_neuropil = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_neuropil_events.csv')
        self.fname_corr_mat = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_correlation_matrix.csv')
        self.fname_corr_map = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_correlation_map.png')
        self.fname_pixel_traces = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_pixel_mean_traces.png')


    def grid_division(self, fname_video, downscale_factor=5):
        """Function to extract the pixel signal from the raw video (registered).
        Inputs:
            fname_video
            downscale_factor: spatial downsampling factor
        Outputs:
            neurpil_signal: DataFrame where each column is the fluorence trace of the pixel in_jm
            n_grid_i_j: downsampled image dimensions"""

        image_stack = io.ImageCollection(fname_video, conserve_memory=True)
        print('Spatial downsampling: grid size: ', downscale_factor)
        image_stack = image_stack.concatenate()
        if self.mask is not None:
            for f in range(image_stack.shape[0]):
                image_stack[f][self.mask]=0
        ds_image_stack = downscale_local_mean(image_stack,(1,downscale_factor,downscale_factor))

        n_grid_i_j = [ds_image_stack.shape[1],ds_image_stack.shape[2]]
        neurpil_signal = pd.DataFrame()
        for i in range(ds_image_stack.shape[1]): 
            for j in range(ds_image_stack.shape[2]):
                signal_ij = ds_image_stack[:,i,j]  
                if not np.all((signal_ij==0)):
                    idx = 'i'+str(i)+'_j'+str(j)
                    neurpil_signal[idx]=signal_ij
        
        # if save:

        #     neurpil_signal.to_csv(self.fname_neuropil)
        #     np.savetxt(os.path.join(self.path,'n_grid_ij.csv'),n_grid_i_j,delimiter=',')
        return neurpil_signal, n_grid_i_j

    def neuropil_h_clustering(self, neuropil_signal, th_cluster=0.6, x_pixels=122, y_pixels=122):
        """Function to performe the hierarchical clustering on the video pixels.
        Inputs:
            neurpil_signal: DataFrame where each column is the fluorence trace of the pixel in_jm
            th_cluster: float, threshold for clustering
            x_pixels: downsampled image x dimension
            y_pixels: downsampled image y dimension
        Outputs:
            cluster_image: Matrix where each element refers to the cluster index value (0 elsewhere)
            cluster_idx: DataFrame with two columns one referes to pixel index (in_jm), the other 
            z: clustering of linkage output"""

        print('Computing pairwise distance')
        distance = pdist(neuropil_signal.T,'correlation')
        # self.m = squareform(distance)

        print('Performing hierachical clustering')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output

        d = {'pixel_idx': neuropil_signal.columns, 'cluster_idx': idx}
        cluster_idx = pd.DataFrame(data=d)
        cluster_idx = cluster_idx.sort_values('cluster_idx')
        cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())      

        cluster_image = np.zeros((x_pixels+1,y_pixels+1))
        for i,pos in enumerate(neuropil_signal.columns):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            cluster_image[ind_i,ind_j]=idx[i]

        return cluster_image, cluster_idx, z

    def Ca_events_neuropil(self, neuropil, show_fig=True):

        events_neuropil=pd.DataFrame(columns=neuropil.columns)
        for i_j in neuropil.columns: 
            signal_ij=neuropil[i_j]
            print('Computing continuos wavelet transformation for neuropil signal')
            cwt2d, ridges, events_cwt = wavelet_transform_morse(signal_ij,gamma=3,beta=2,min_scale=20,min_peak_position=15,min_freq=1, max_freq=20)
            self.events_neuropil[i_j]=events_cwt
            if show_fig:
                fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  
                plot_cwt2d_trace(cwt_ax, cwt2d, signal_ij)
                plot_ridges_trace_events(ridges_ax, ridges, signal_ij, events_cwt)
                plt.show()

        return events_neuropil
        
    def norm_traces(self, df, cols):
        """Function to compute the norm traces.
            Inputs:
                df: dataframe containing for each column ROI/pixel raw trace
                cols: the pixels/ROI list contained in df to be normalized
            Outputs:
                df_zscored: dataframe with normalized traces"""
        df_zscored = pd.DataFrame(columns=df.columns, index=df.index.to_list())
        print('computing mean and std')

        print('computing zscored traces')
        for col in cols:
            print(col)
            mean_value = df[col].mean(axis=0)
            print(mean_value)
            std_value = df[col].std(axis=0)
            df_zscored[col] = (df[col] - mean_value)/std_value
        return df_zscored

