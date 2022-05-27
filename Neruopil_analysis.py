from matplotlib.axes import Axes
from neuropil_class import Neuropil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm, markers
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import os
import pandas as pd
from PIL import Image
from scipy.signal import spectrogram
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

root_path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05'
# root_path = '/media/careylab/Samsung_T5/TM RAW FILES/tied baseline/MC8855/2021_04_04'
path_video = os.path.join(root_path,'Registered video') 
path_results = os.path.join(root_path,'Neuropil_analysis')
# path_results = os.path.join(root_path,'ref mask')
n_trials = 23
# n_trials = 6
neuropil = Neuropil(path_results)
# TODO function/class for color map



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
print('computing zscore traces')
# df_session = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), index_col=0)
zscored_neuropil_signal = neuropil.norm_traces(df_session,df_session.columns)
zscored_neuropil_signal.to_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'))
#  '''
# zscored_neuropil_signal= pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), index_col=0)
# neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal.csv'), index_col=0)
# time_info = ['time', 'trial', 'trial_sample']
# new_columns = time_info+neuropil_signal.columns.to_list()
# # print(new_columns)
# zscored_neuropil_signal.columns=new_columns
# print(zscored_neuropil_signal.columns)
# zscored_neuropil_signal.to_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'))
# trials_df = pd.read_csv(os.path.join(path_results,'df_extract_raw_split.csv'), usecols=['time','trial'])
# zscored_neuropil_signal_new= pd.concat([trials_df, zscored_neuropil_signal], axis=1, ignore_index=True)
# zscored_neuropil_signal_new.to_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'))



'''****************************************** SESSION CLUSTERS ******************************************
neuropil = Neuropil(path_results)
neuropil_session_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), index_col=0)
cluster_image, cluster_idx, z = neuropil.neuropil_h_clustering(neuropil_session_signal,th_cluster=0.45)

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
neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=pixel_idx)

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

'''******************************************* CLUSTER TRACE ********************************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col='pixel_idx')
pixel_idx = cluster_idx.index.values.tolist()
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
# selected_clusters = cluster_idx_list
trials_df = pd.read_csv(os.path.join(path_results,'df_extract_raw_split.csv'), usecols=['trial'])
trials = trials_df['trial'].to_numpy()

selected_clusters = [1,2,3,4,5,6,7,8,9,11,13,14,15,18,20] # cluster 10, 12, 16, 17, 19 excluded
nr_clusters = len(cluster_idx_list)

print('Loading zscored cluster signal')
neuropil_signal_zscored = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=pixel_idx)

print('Create color map')
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]
fig_mean,ax_mean= plt.subplots(figsize=(25,20), tight_layout=True)
k=5
t = neuropil_signal_zscored.index.values.tolist()
mean_neuropil_signal = []
for i,c in enumerate(selected_clusters):
    pixel_list = cluster_idx[cluster_idx['cluster_idx']==c].index.values.tolist()

    cluster_activity =[]
    for j,p in enumerate(pixel_list):
        cluster_activity.append(neuropil_signal_zscored[p].to_numpy())        
    mean_cluster_activity = np.mean(cluster_activity,axis=0)
    sd_cluster_activity = np.std(cluster_activity,axis=0)

    print('Plotting  cluster ', c, ' activity')
    ax_mean.plot(t,mean_cluster_activity+i*k,color=cmap.colors[c], label='cluster'+str(c))
    ax_mean.hlines(i*k,t[0],t[-1], color=cmap.colors[c])
    # ax_mean.fill_between(t,(mean_cluster_activity+i*k)+sd_cluster_activity,(mean_cluster_activity+i*k)-sd_cluster_activity,color=cmap.colors[c], alpha=0.8)
    ax_mean.legend()
    mean_neuropil_signal.append(mean_cluster_activity)

for j in np.unique(trials):
    ax_mean.vlines(trials_df.index[trials_df['trial']==j][0], 0, i*k, linewidth=3)
fig_mean.savefig(os.path.join(path_results,'cluisters_trace.png'))
plt.close(fig_mean)

plt.imshow(mean_neuropil_signal, aspect='auto', vmax=5)  
for j in np.unique(trials):
    plt.vlines(trials_df.index[trials_df['trial']==j][0], 0, 14, linewidth=2, color='red')  
plt.show()
#  '''

'''**************************************** CLUSTER ACTIVITY MAT ****************************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col=0)
pixel_idx = cluster_idx['pixel_idx'].tolist()
neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=pixel_idx)


neuropil_signal=neuropil_signal[pixel_idx]
neuropil_signal=neuropil_signal.T
plt.imshow(neuropil_signal, aspect='auto', cmap='inferno', vmin=0)
plt.show()
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
    
    neuropil_signal = pd.read_csv(neuropil.fname_neuropil, usecols=pixel_idx)

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

'''*************************************** INTER CLUSTER CORRELATION ****************************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'))
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
cluster_idx_list = [1,2,3,4,5,6,7,8,9,11,13,14,15,18,20]
nr_clusters = len(cluster_idx_list)

print('Create color map')
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

for fc,fix_cluster in enumerate(cluster_idx_list):
    pixel_fix_cluster = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==fix_cluster].to_list()
    mat = np.zeros((nr_clusters,n_trials))
    fig,ax=plt.subplots(figsize=(20,20))
    for t,trial in enumerate(range(1,n_trials+1)):
        print('trial number: ', trial)
        neuropil = Neuropil(path_results, trial)
        corr_mat = pd.read_csv(neuropil.fname_corr_mat, index_col=0)
        for c,cluster in enumerate(cluster_idx_list):
            pixel_cluster = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==cluster].to_list()
            inter_cluster_corr = corr_mat[pixel_cluster].loc[pixel_fix_cluster].to_numpy()        
            mat[c,t] = np.mean(inter_cluster_corr)

    # ax.imshow(mat, aspect='auto',cmap='coolwarm', vmin=0, vmax=1)
    ax.plot(np.transpose(mat))
    ax.set_xticks(range(n_trials))
    ax.set_xticklabels(range(1,n_trials+1))
    ax.set_xlabel('# Trials')
    # ax.set_yticks(range(nr_clusters))
    # ax.set_yticklabels(cluster_idx_list)
    # ax.set_ylabel('Cluster idx')
    ax.set_title('Inter corre for Cluster '+str(fix_cluster))
    plt.show()
    plt.legend()
    # fig.savefig(os.path.join(path_results,'inter_corr_fixcluster_'+str(fix_cluster)+'.png'))
    plt.close(fig)


# '''

'''*************************************** SINGLE TRIAL CLUSTERS ****************************************
for trial in range(1,n_trials+1):
    print('trial number: ', trial)
    neuropil = Neuropil(path_results, trial)
    neuropil_signal = pd.read_csv(neuropil.fname_neuropil, index_col=0)
    # neuropil_signal = neuropil.norm_traces(neuropil_signal,neuropil_signal.columns)
    
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

'''************************************ SINGLE TRIAL CLUSTER TRACE **************************************
print('Loading clustering output for entire session')
cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'), index_col='pixel_idx')
pixel_idx = cluster_idx.index.values.tolist()
cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
# selected_clusters = cluster_idx_list
trials_df = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=['time', 'trial'])
trials = trials_df['trial'].to_numpy()

selected_clusters = [1,2,3,4,5,6,7,8,9,11,13,14,15,18,20] # cluster 10, 12, 16, 17, 19 excluded
nr_clusters = len(cluster_idx_list)

print('Loading zscored cluster signal')
neuropil_signal_zscored = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=pixel_idx)

print('Create color map')
cmap = cm.get_cmap('inferno', int(nr_clusters)+1)
cmap.colors[0]=[1,1,1,1]

k=5


mean_neuropil_signal = []
for t in np.unique(trials):
    trial_index = trials_df.index[trials_df['trial']==t]
    fig_mean,ax_mean= plt.subplots(figsize=(25,20), tight_layout=True)
    print('Trial ', t)
    for i,c in enumerate(selected_clusters):
        pixel_list = cluster_idx[cluster_idx['cluster_idx']==c].index.values.tolist()
        cluster_activity =[]
        for j,p in enumerate(pixel_list):
            cluster_activity.append(neuropil_signal_zscored[p][trial_index].to_numpy())        
        mean_cluster_activity = np.mean(cluster_activity,axis=0)
        sd_cluster_activity = np.std(cluster_activity,axis=0)

        print('Plotting  cluster ', c, ' activity')
        ax_mean.plot(range(len(mean_cluster_activity)),mean_cluster_activity+i*k,color=cmap.colors[c], label='cluster'+str(c))
        ax_mean.hlines(i*k,range(len(mean_cluster_activity))[0],range(len(mean_cluster_activity))[-1], color=cmap.colors[c])
        # ax_mean.fill_between(t,(mean_cluster_activity+i*k)+sd_cluster_activity,(mean_cluster_activity+i*k)-sd_cluster_activity,color=cmap.colors[c], alpha=0.8)
        ax_mean.legend()
        mean_neuropil_signal.append(mean_cluster_activity)
    fig_mean.savefig(os.path.join(path_results,'T'+str(t)+'_neuropil','cluisters_trace.png'), transparent=True)
    plt.close(fig_mean)
#  '''


'''************************************************* PCA ************************************************
newcolors = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                    'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                    'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
cmap = ListedColormap(newcolors, name='split_session')

neuropil = Neuropil(path_results)
trials = pd.read_csv(os.path.join(path_results,'df_extract_raw_split.csv'), usecols=['trial'])
trials = trials['trial'].to_numpy()

print('Loading data')
neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), index_col=0)

print('Computing PCA')
pca = PCA(n_components=3)
principalComponents_3CP = pca.fit_transform(neuropil_signal)
baricenter = np.zeros((len(np.unique(trials)),3))
for i,t in enumerate(np.unique(trials)):
    b_coord = np.mean(principalComponents_3CP[trials==t, :], axis=0)
    std_coord = np.std(principalComponents_3CP[trials==t, :], axis=0)
    baricenter[i,:]=b_coord

plot3d=False
plot2d=True
centroids=True

if plot3d:
    print('Plotting PCA results 3D')
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(xs=principalComponents_3CP[:, 0], ys=principalComponents_3CP[:, 1], zs=principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
    ax.set_title('First 3 PCs - explained variance of ' + str(np.round(np.cumsum(pca.explained_variance_ratio_)[2], decimals=3)), fontsize=24)
    ax.set_xlabel('PC component 1', fontsize=20)
    ax.set_ylabel('PC component 2', fontsize=20)
    ax.set_zlabel('PC component 3', fontsize=20)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

if plot2d:
    print('Plotting PCA results 2D')
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 10), tight_layout=True)
    # ax1.scatter(principalComponents_3CP[:, 0], principalComponents_3CP[:, 1], s=1, c=trials, cmap=cmap)
    ax1.set_xlabel('PC component 1', fontsize=20)
    ax1.set_ylabel('PC component 2', fontsize=20)        
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    # ax2.scatter(principalComponents_3CP[:, 0], principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
    ax2.set_xlabel('PC component 1', fontsize=20)
    ax2.set_ylabel('PC component 3', fontsize=20)        
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    # ax3.scatter(principalComponents_3CP[:, 1], principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
    ax3.set_xlabel('PC component 2', fontsize=20)
    ax3.set_ylabel('PC component 3', fontsize=20)        
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax2.set_title('First 3 PCs - explained variance of ' + str(np.round(np.cumsum(pca.explained_variance_ratio_)[2], decimals=3)), fontsize=24)
    if centroids:
        ax1.scatter(baricenter[:,0], baricenter[:,1], s=30, c=np.unique(trials), cmap=cmap)
        ax2.scatter(baricenter[:,0], baricenter[:,2], s=30, c=np.unique(trials), cmap=cmap)
        ax3.scatter(baricenter[:,1], baricenter[:,2], s=30, c=np.unique(trials), cmap=cmap)
    plt.show()
#  '''

'''******************************************** PCA CLUSTERS ********************************************
cmap = cm.get_cmap('rainbow', n_trials+1)
if n_trials == 23:
    cmap.colors[:] = ['darkgrey', 'darkgrey', 'darkgrey', 'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                    'crimson', 'crimson', 'crimson', 'crimson', 'crimson',
                    'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue']
if n_trials == 6:
    colors_bars = ['darkgrey', 'darkgrey', 'darkgrey', 'orange', 'orange', 'orange']

neuropil = Neuropil(path_results)
trials = pd.read_csv(os.path.join(path_results,'df_extract_raw_split.csv'), usecols=['trial'])
trials = trials['trial'].to_numpy()

cluster_idx = pd.read_csv(os.path.join(path_results,'clusters_idx.csv'))
# cluster_idx_list = np.unique(cluster_idx['cluster_idx'].to_numpy())
cluster_idx_list = [1,2,3,4,5,6,7,8,9,11,13,14,15,18,20]

for i in cluster_idx_list:
    pixel_idx = cluster_idx['pixel_idx'][cluster_idx['cluster_idx']==i].tolist()
    neuropil_signal = pd.read_csv(os.path.join(path_results,'neuropil_session_signal_norm.csv'), usecols=pixel_idx)

    print('computing PCA')
    pca = PCA(n_components=3)
    principalComponents_3CP = pca.fit_transform(neuropil)

    baricenter = np.zeros((len(np.unique(trials)),3))
    for i,t in enumerate(np.unique(trials)):
        b_coord = np.mean(principalComponents_3CP[trials==t, :], axis=0)
        std_coord = np.std(principalComponents_3CP[trials==t, :], axis=0)
        baricenter[i,:]=b_coord

    
    plot3d=False
    plot2d=True
    centroids=True

    if plot3d:
        print('3D plot')
        ax = plt.figure(figsize=(16,10)).gca(projection='3d')
        ax.scatter(xs=principalComponents_3CP[:, 0], ys=principalComponents_3CP[:, 1], zs=principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
        ax.set_title('First 3 PCs - explained variance of ' + str(np.round(np.cumsum(pca.explained_variance_ratio_)[2], decimals=3)), fontsize=24)
        ax.set_xlabel('PC component 1', fontsize=20)
        ax.set_ylabel('PC component 2', fontsize=20)
        ax.set_zlabel('PC component 3', fontsize=20)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.show()

    if plot2d:
        print('2D plot')
        fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 10), tight_layout=True)
        ax1.plot(principalComponents_3CP[:, 0], principalComponents_3CP[:, 1], s=1, c=trials, cmap=cmap)
        ax1.set_xlabel('PC component 1', fontsize=20)
        ax1.set_ylabel('PC component 2', fontsize=20)        
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax2.scatter(principalComponents_3CP[:, 0], principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
        ax2.set_xlabel('PC component 1', fontsize=20)
        ax2.set_ylabel('PC component 3', fontsize=20)        
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax3.scatter(principalComponents_3CP[:, 1], principalComponents_3CP[:, 2], s=1, c=trials, cmap=cmap)
        ax3.set_xlabel('PC component 2', fontsize=20)
        ax3.set_ylabel('PC component 3', fontsize=20)        
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax2.set_title('First 3 PCs - explained variance of ' + str(np.round(np.cumsum(pca.explained_variance_ratio_)[2], decimals=3)), fontsize=24)
        if centroids:
            ax1.scatter(baricenter[:,0], baricenter[:,1], s=30, c=np.unique(trials), cmap=cmap)
            ax2.scatter(baricenter[:,0], baricenter[:,2], s=30, c=np.unique(trials), cmap=cmap)
            ax3.scatter(baricenter[:,1], baricenter[:,2], s=30, c=np.unique(trials), cmap=cmap)
        plt.show()
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
coord_cell = np.load(os.path.join(path_results,'coord_split.npy'),allow_pickle=True)
coord_cell = coord_cell*pixel_to_um
rois = pd.read_csv(os.path.join(path_results,'df_extract_raw_split.csv'), index_col='time')

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

rois = neuropil.norm_traces(rois, rois.columns[1:])
rois.to_csv(os.path.join(path_results,'df_rois_raw_split_norm.csv'))
#  '''

# roi_signal = pd.read_csv(os.path.join(path_results,'df_rois_raw_split_norm.csv'))

# print('Compute correlation map for entire session')
# roi_signal=roi_signal[roi_signal.columns[2:]]
# roi_signal=roi_signal.T
# plt.imshow(roi_signal, aspect='auto', cmap='inferno', vmin=-1, vmax=5)
# plt.colorbar()
# plt.show()

'''************************************* ROI CLUSTERS FROM RAW VIDEO ************************************
# compute clusters with raw signals
rois = pd.read_csv(os.path.join(path_results,'df_rois_raw_split_norm.csv'))
rois = rois[rois.columns[2:]]

th_cluster = 0.1

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
coord_cell = np.load(os.path.join(path_results,'coord_split.npy'),allow_pickle=True)
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

# '''****************************************** FR-FL TRACE *****************************************
import locomotion_class

path_loco = '/media/careylab/Samsung_T5/TM TRACKING FILES/split ipsi fast S1 050421/'
frames_dFF = 0 #black frames removed before ROI segmentation

loco = locomotion_class.loco_class(path_loco)
animal = 'MC8855'
session = 1

#Get session protocol
filelist = loco.get_track_files(animal,session)
trial_type = []
for f in filelist:
    trial_type.append(f.split('_')[4])
split_1_idx = trial_type.index('split')
tied_trials = np.arange(1,split_1_idx+1)
split_trials = np.arange(split_1_idx+1,split_1_idx+11)
washout_trials = np.arange(split_trials[-1]+1,split_trials[-1]+11)

for i,f in enumerate(filelist):    
    fig,ax1 = plt.subplots(figsize=(25,20), tight_layout=True)
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f,0.9,frames_dFF)
    bodycenter = loco.compute_bodycenter(final_tracks,'X')
    # bodyspeed = loco.compute_bodyspeed(bodycenter)
    bodyacc = loco.compute_bodyacc(bodycenter)    
    # plt.specgram(bodyacc, Fs=330)
    # ax1.pcolormesh(t, f, Sxx, shading='gouraud')
    # ax.plot(final_tracks[0,0,:])
    # plt.plot(final_tracks[0,1,:])
    # ax.plot(final_tracks[0,2,:])
    # ax1=ax.twinx()
    # plt.plot(range(len(bodyacc))+(np.zeros((len(bodyacc)))+i*len(bodyacc)),bodyacc)
    plt.plot(bodyacc, 'white')
    plt.ylabel('# Trials')
    plt.xlabel('Time')
    # plt.set_yticklabels(range(1,len(filelist)+1))
    plt.ylim(-0.1,0.1)
    # plt.plot(final_tracks[0,3,:])
    # plt.plot(final_tracks[0,0,:]-final_tracks[0,2,:])
    # plt.title('trial '+str(i+1))
    fig.savefig(os.path.join(path_results,'bodyacc_trial'+str(i+1)+'.png'), transparent=True)
    plt.close(fig)


#  '''

'''****************************************** BODY ACCELERATION *****************************************
print('Computing body acceleration')
path_loco = '/media/careylab/Samsung_T5/TM TRACKING FILES/split ipsi fast S1 050421/'
frames_dFF = 0 #black frames removed before ROI segmentation
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
path_save = path_loco+'grouped output/'
    
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

#Get session protocol
count_animal = 0
Ntrials_all = np.zeros(len(animal_list))
tied_trials_all = []
split_trials_all = []
washout_trials_all = []
for animal in animal_list:
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal,session)
    Ntrials_all[count_animal] = len(filelist)
    trial_type = []

    for f in filelist:
        trial_type.append(f.split('_')[4])
    split_1_idx = trial_type.index('split')
    tied_trials = np.arange(1,split_1_idx+1)
    tied_trials_all.append(tied_trials)
    split_trials = np.arange(split_1_idx+1,split_1_idx+11)
    split_trials_all.append(split_trials)
    washout_trials = np.arange(split_trials[-1]+1,split_trials[-1]+11)
    washout_trials_all.append(washout_trials)
    count_animal += 1    
Ntrials = np.int64(np.max(Ntrials_all))

#symmery gait parameters
count_animal = 0
param_sym_name = ['coo','step_length','double_support','coo_stance','swing_length','phase_st','stance_speed']
param_sym = np.zeros((len(param_sym_name),len(animal_list),Ntrials))
param_phase = np.zeros((4,len(animal_list),Ntrials))
stance_speed = np.zeros((4,len(animal_list),Ntrials))
for animal in animal_list:
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal,session)
    triggers_ordered, strobes_ordered,frame_rec_start_full,mscope_align_time_ordered,bcam_time = loco.get_tdms_frame_start(animal,session,frames_dFF)
    
    for i,f in enumerate(filelist):
        
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f,0.9,frames_dFF)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks,1)
        bcam_trial = bcam_time[i - 1]
        st_on = st_strides_mat[0][:, 0, -1]  # FR paw
        st_off = st_strides_mat[0][:, 1, -1]  # FR paw
        # for s in range(len(st_on)):
            # TODO FORWARD ONLY
            # delete period of th edataframe not comprises in a stride
            # np.where((time >= bcam_trial[int(st_on[s])]) & (time <= bcam_trial[int(st_off[s])]))
        bodycenter = loco.compute_bodycenter(final_tracks,'X')
        bodyacc = loco.compute_bodyspeed(bodycenter)

        fig, ax1 = plt.subplots()
        ax1.plot(bodyacc)
        # ax2 = ax1.twinx()  
        # ax2.plot(bodycenter, 'red')
        plt.show()
#  '''


