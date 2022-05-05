from cProfile import label
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
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
from sklearn.decomposition import PCA


'''The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score.'''


fsize=18

class Neuropil:
    def __init__(self,path,trial=1):
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
        self.fname_cluster_idx = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_clusters_idx_sorted_'+m+'.csv')
        self.fname_dendrogram = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_dendrogram_'+m+'.png')
        self.fname_events_neuropil = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_neuropil_events_'+m+'.csv')
        self.fname_corr_mat = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_correlation_matrix_'+m+'.csv')
        self.fname_corr_map = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_correlation_map_'+m+'.png')
        self.fname_pixel_traces = os.path.join(self.path,'T'+str(trial)+'_neuropil','T'+str(self.trial)+'_pixel_mean_traces.png')


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
        self.n_grid_i_j = [ds_image_stack.shape[1],ds_image_stack.shape[2]]
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
            np.savetxt(os.path.join(self.path,'n_grid_ij.csv'),self.n_grid_i_j,delimiter=',')
        return 

    def neuropil_clustering_by_trial(self, th_cluster=0.6, x_pixels=122, y_pixels=122, save=True, save_image=True):

        if os.path.exists(self.fname_neuropil):
            neuropil = pd.read_csv(self.fname_neuropil, index_col=0)
        else:
            self.grid_division()
            neuropil = self.neurpil

        print('Computing pairwise distance')
        distance = pdist(neuropil.T,'correlation')
        # self.m = squareform(distance)
        print('Performing hierachical clustering')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
        d = {'pixel_idx': neuropil.columns, 'cluster_idx': idx}
        cluster_idx = pd.DataFrame(data=d)
        cluster_idx = cluster_idx.sort_values('cluster_idx')
        cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())      

        # fig = plt.figure(figsize=(20,15))
        # dn = dendrogram(z,above_threshold_color='y',no_labels=True,orientation='top') 

        self.cluster_image = np.zeros((x_pixels+1,y_pixels+1))
        for i,pos in enumerate(neuropil.columns):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            self.cluster_image[ind_i,ind_j]=idx[i]
        if save:
            print('Saving clustering output for '+self.fname_neuropil)
            np.savetxt(self.fname_clusters,self.cluster_image,delimiter=',')
            cluster_idx.to_csv(self.fname_cluster_idx)
            # fig.savefig(self.fname_dendrogram)
        if save_image:
            nr_clusters = len(cluster_idx_list)
            cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
            cmap.colors[0]=[1,1,1,1]
            fig = plt.figure(figsize=(20,20), tight_layout=True)    
            plt.imshow(self.cluster_image, aspect='auto', cmap=cmap)
            for i,c in enumerate(cluster_idx_list):
                fig.text(0.9,0.9-i*0.025,'cluster index: '+str(c), size = 16, bbox=dict(boxstyle ='round', fc=cmap.colors[i+1]))
            # plt.show()
            fig.savefig(self.fname_clusters[:-4]+'.png')

        return self.cluster_image

    def clustered_pixels_correlation(self, cluster_idx, cmap, save=True, save_image=True, all_session=True):
        if all_session:
            fname_corr_map = os.path.join(self.path,'T'+str(self.trial)+'_corr_map_session_clusters.png')
            fname_corr_mat = os.path.join(self.path,'T'+str(self.trial)+'_corr_mat_session_clusters.csv')
        else:
            fname_corr_map = self.fname_corr_map
            fname_corr_mat = self.fname_corr_mat

        pixel_idx = cluster_idx['pixel_idx'].tolist()
        neuropil = pd.read_csv(self.fname_neuropil, usecols=pixel_idx)

        print('Computing pairwise correlation between pixels')
        neuropil=neuropil[pixel_idx]
        corr_mat=neuropil.corr('pearson')
        
        if save:
            print('Saving correlation matrix')
            corr_mat.to_csv(fname_corr_mat)

        if save_image:
            index = cluster_idx.index
            cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
            print('Plotting correlation map')
            fig, ax = plt.subplots(1,figsize=(20,20))  
            ax.matshow(corr_mat,cmap='coolwarm',vmin=0, vmax=1)
            for i,c in enumerate(cluster_idx_list):
                p = index[cluster_idx['cluster_idx']==c].to_list()
                rect = patches.Rectangle((p[0], p[0]), p[-1]-p[0]+1, p[-1]-p[0]+1, linewidth=5, edgecolor=cmap.colors[i+1], facecolor="none")
                ax.add_patch(rect)
            fig.savefig(fname_corr_map)


        return corr_mat

    def clustered_pixel_trace(self, cmap, cluster_idx, k=1, selected_clusters=None, save_image=True, all_session=True):

        if selected_clusters is None:
            cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
        else:
            cluster_idx_list = selected_clusters

        if all_session:
            fname_pixel_traces = os.path.join(self.path,'T'+str(self.trial)+'_pixel_mean_traces.png')
        else:
            fname_pixel_traces = self.fname_pixel_traces

        pixel_idx = cluster_idx.index.values.tolist()
        neuropil = pd.read_csv(self.fname_neuropil, usecols=pixel_idx)
        fig_mean,ax_mean= plt.subplots(figsize=(20,20), tight_layout=True)
        t = neuropil.index.values.tolist()
        for i,c in enumerate(cluster_idx_list):
            pixel_list = cluster_idx[cluster_idx['cluster_idx']==c].index.values.tolist()
            cluster_activity =[]
            for j,p in enumerate(pixel_list):
                cluster_activity.append(neuropil[p].to_numpy())
             
            mean_cluster_activity = np.mean(cluster_activity,axis=0)
            sd_cluster_activity = np.std(cluster_activity,axis=0)
            ax_mean.plot(t,mean_cluster_activity+i*k,color=cmap.colors[c], label='cluster'+str(c))
            ax_mean.fill_between(t,(mean_cluster_activity+i*k)+sd_cluster_activity,(mean_cluster_activity+i*k)-sd_cluster_activity,color=cmap.colors[c], alpha=0.8)
            ax_mean.legend()
        if save_image:
            fig_mean.savefig(fname_pixel_traces)       

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
    
    @staticmethod       
    def pca_neuropil_signal(path,fname,trials,show_fig=False, save=True):

        neuropil = pd.read_csv(os.path.join(path,fname), index_col=0)

        pca = PCA(n_components=3)
        principalComponents_3CP = pca.fit_transform(neuropil)

        fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
        plt.plot(np.arange(1, 4), np.cumsum(pca.explained_variance_ratio_), color='black')
        plt.scatter(3, np.cumsum(pca.explained_variance_ratio_)[2], color='red')
        ax.set_xlabel('PCA components', fontsize=14)
        ax.set_ylabel('Explained variance', fontsize=14)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(xs=principalComponents_3CP[:, 0], ys=principalComponents_3CP[:, 1], zs=principalComponents_3CP[:, 2], s=1)
        ax.set_title('First 3 PCs - explained variance of ' + str(np.round(np.cumsum(pca.explained_variance_ratio_)[2], decimals=3)), fontsize=24)
        ax.set_xlabel('PC component 1', fontsize=20)
        ax.set_ylabel('PC component 2', fontsize=20)
        # ax.set_zlabel('PC component 3', fontsize=20)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()
        return 

    @staticmethod
    def neuropil_clustering(path, fname, th_cluster=0.6, x_pixels=122, y_pixels=122,  save=True, save_image=True):

        neuropil = pd.read_csv(os.path.join(path,fname), index_col=0)
        print('Computing pairwise distance')
        distance = pdist(neuropil.T,'correlation')
        # self.m = squareform(distance)
        print('Performing hierachical clustering')
        z = linkage(y=distance, method='complete', metric='euclidean')
        idx = fcluster(z, th_cluster * distance.max(), 'distance')  # clustering of linkage output
        d = {'pixel_idx': neuropil.columns, 'cluster_idx': idx}
        cluster_idx = pd.DataFrame(data=d)
        cluster_idx = cluster_idx.sort_values('cluster_idx')
        cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
        nr_clusters = len(cluster_idx_list)
        cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
        cmap.colors[0]=[1,1,1,1]

        cluster_image = np.zeros((x_pixels+1,y_pixels+1))
        for i,pos in enumerate(neuropil.columns[1:]):
            ind_i = int(pos[1:pos.find('_')])
            ind_j = int(pos[pos.find('j')+1:])
            cluster_image[ind_i,ind_j]=idx[i]
        
        if save:
            print('Saving clustering output')
            np.savetxt(os.path.join(path,'clusters_map.csv'),cluster_image,delimiter=',')
            cluster_idx.to_csv(os.path.join(path,'clusters_idx.csv'))

        if save_image:
            fig = plt.figure(figsize=(20,20), tight_layout=True)
            plt.imshow(cluster_image, aspect='auto', cmap=cmap)            
            for i,c in enumerate(cluster_idx_list):
                fig.text(0.9,0.9-i*0.025,'cluster index: '+str(c), size = 16, bbox=dict(boxstyle ='round', fc=cmap.colors[i+1]))
            fig.savefig(os.path.join(path,'clusters.png'))

        return 



    