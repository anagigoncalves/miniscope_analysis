import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\LocoCF\\miniscopes learning\\DS sorted rasters\\split ipsi fast\\'
save_path = 'J:\\LocoCF\\miniscopes learning\\DS sorted rasters\\'
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
df_prob_sum_fr = pd.read_csv(load_path + 'prob_sum_fr_df.csv')

df_prob_sum_fr_animal = df_prob_sum_fr.loc[df_prob_sum_fr['animal'] == 'MC9194']
xaxis_fr = np.reshape(np.array(df_prob_sum_fr_animal['param_val']), (np.int64(len(df_prob_sum_fr_animal['param_val'])/20), 20))
param_fr_st = np.reshape(np.array(df_prob_sum_fr_animal['prob_st']), (np.int64(len(df_prob_sum_fr_animal['prob_st'])/20), 20))
param_fr_sw = np.reshape(np.array(df_prob_sum_fr_animal['prob_sw']), (np.int64(len(df_prob_sum_fr_animal['prob_sw'])/20), 20))

fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(np.nanmean(xaxis_fr, axis=0), np.nanmean(param_fr_st, axis=0), color='orange', linewidth=2)
ax.plot(np.nanmean(xaxis_fr, axis=0), np.nanmean(param_fr_sw, axis=0), color='green', linewidth=2)

df_prob_sum_fl = pd.read_csv(load_path + 'prob_sum_fl_df.csv')

xaxis_fl = np.reshape(np.array(df_prob_sum_fl['param_val']), (np.int64(len(df_prob_sum_fl['param_val'])/20), 20))
param_fl_st = np.reshape(np.array(df_prob_sum_fl['prob_st']), (np.int64(len(df_prob_sum_fl['prob_st'])/20), 20))
param_fl_sw = np.reshape(np.array(df_prob_sum_fl['prob_sw']), (np.int64(len(df_prob_sum_fl['prob_sw'])/20), 20))

fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
ax.plot(np.nanmean(xaxis_fl, axis=0), np.nanmean(param_fl_st, axis=0), color='orange', linewidth=2)
ax.plot(np.nanmean(xaxis_fl, axis=0), np.nanmean(param_fl_sw, axis=0), color='green', linewidth=2)