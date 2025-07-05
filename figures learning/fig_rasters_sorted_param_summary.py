import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\LocoCF\\miniscopes learning\\Phase diff sorted rasters\\split ipsi fast\\'
load_pc_path = 'J:\\LocoCF\\miniscopes learning\\PCA validation and clusters (only baseline trials)\\'
protocol = load_path.split('\\')[-2]
save_path = 'J:\\LocoCF\\miniscopes learning\\Phase diff sorted rasters\\'
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
bins = 10
cluster = 2

df_prob_sum_fr = pd.read_csv(load_path + 'prob_sum_fr_df.csv')

# Load PC coefficients data
pc_coeff = pd.read_csv(os.path.join(load_pc_path, 'pc_coeff_df_clusters_' + '_'.join(protocol.split(' ')) + '.csv'))
cluster_idx = np.where(pc_coeff['cluster_pca'] == cluster)[0]

xaxis_fr = np.reshape(np.array(df_prob_sum_fr['param_val']), (np.int64(len(df_prob_sum_fr['param_val'])/bins), bins))
param_fr_st = np.reshape(np.array(df_prob_sum_fr['prob_st']), (np.int64(len(df_prob_sum_fr['prob_st'])/bins), bins))
param_fr_sw = np.reshape(np.array(df_prob_sum_fr['prob_sw']), (np.int64(len(df_prob_sum_fr['prob_sw'])/bins), bins))

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(np.nanmean(xaxis_fr[cluster_idx, :], axis=0), np.nanmean(param_fr_st[cluster_idx, :], axis=0), color='orange', linewidth=2)
ax.fill_between(np.nanmean(xaxis_fr[cluster_idx, :], axis=0),
        np.nanmean(param_fr_st[cluster_idx, :], axis=0)-np.nanstd(param_fr_st[cluster_idx, :], axis=0),
        np.nanmean(param_fr_st[cluster_idx, :], axis=0)+np.nanstd(param_fr_st[cluster_idx, :], axis=0), alpha=0.3, color='orange')
ax.plot(np.nanmean(xaxis_fr[cluster_idx, :], axis=0), np.nanmean(param_fr_sw[cluster_idx, :], axis=0), color='green', linewidth=2)
ax.fill_between(np.nanmean(xaxis_fr[cluster_idx, :], axis=0),
        np.nanmean(param_fr_sw[cluster_idx, :], axis=0)-np.nanstd(param_fr_sw[cluster_idx, :], axis=0),
        np.nanmean(param_fr_sw[cluster_idx, :], axis=0)+np.nanstd(param_fr_sw[cluster_idx, :], axis=0), alpha=0.3, color='green')
plt.savefig(os.path.join(save_path, 'coo_sorted_summary_fr_paw_cluster_' + str(cluster) + '_' + protocol.replace(' ', '_')), dpi=256)
plt.savefig(os.path.join(save_path, 'coo_sorted_summary_fr_paw_cluster_' + str(cluster) + '_' + protocol.replace(' ', '_') + '.svg'), dpi=256)

df_prob_sum_fl = pd.read_csv(load_path + 'prob_sum_fl_df.csv')

xaxis_fl = np.reshape(np.array(df_prob_sum_fl['param_val']), (np.int64(len(df_prob_sum_fl['param_val'])/bins), bins))
param_fl_st = np.reshape(np.array(df_prob_sum_fl['prob_st']), (np.int64(len(df_prob_sum_fl['prob_st'])/bins), bins))
param_fl_sw = np.reshape(np.array(df_prob_sum_fl['prob_sw']), (np.int64(len(df_prob_sum_fl['prob_sw'])/bins), bins))

fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(np.nanmean(xaxis_fl[cluster_idx, :], axis=0), np.nanmean(param_fl_st[cluster_idx, :], axis=0), color='orange', linewidth=2)
ax.fill_between(np.nanmean(xaxis_fl[cluster_idx, :], axis=0),
        np.nanmean(param_fl_st[cluster_idx, :], axis=0)-np.nanstd(param_fl_st[cluster_idx, :], axis=0),
        np.nanmean(param_fl_st[cluster_idx, :], axis=0)+np.nanstd(param_fl_st[cluster_idx, :], axis=0), alpha=0.3, color='orange')
ax.plot(np.nanmean(xaxis_fl[cluster_idx, :], axis=0), np.nanmean(param_fl_sw[cluster_idx, :], axis=0), color='green', linewidth=2)
ax.fill_between(np.nanmean(xaxis_fl[cluster_idx, :], axis=0),
        np.nanmean(param_fl_sw[cluster_idx, :], axis=0)-np.nanstd(param_fl_sw[cluster_idx, :], axis=0),
        np.nanmean(param_fl_sw[cluster_idx, :], axis=0)+np.nanstd(param_fl_sw[cluster_idx, :], axis=0), alpha=0.3, color='green')
plt.savefig(os.path.join(save_path, 'coo_sorted_summary_fl_paw_cluster_' + str(cluster) + '_' + protocol.replace(' ', '_')), dpi=256)
plt.savefig(os.path.join(save_path, 'coo_sorted_summary_fl_paw_cluster_' + str(cluster) + '_' + protocol.replace(' ', '_') + '.svg'), dpi=256)
