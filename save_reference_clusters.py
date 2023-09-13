# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import pandas as pd
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_MC9226.xlsx')
s = 0
ses_info = session_data.iloc[s, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
session_type = path.split('\\')[-4].split(' ')[0]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
traces_type = 'raw'
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext_reference_ses, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered_reference_ses, ref_image, frames_dFF] = mscope.load_processed_files()

np.save(os.path.join('J:\\Miniscope processed files\\TM RAW FILES\\reference clusters', 'coord_ext_reference_ses_' + animal + '.npy'), coord_ext_reference_ses, allow_pickle=True)
np.save(os.path.join('J:\\Miniscope processed files\\TM RAW FILES\\reference clusters', 'clusters_rois_idx_order_reference_ses_' + animal + '.npy'), idx_roi_cluster_ordered_reference_ses, allow_pickle=True)

# s = 1
# ses_info = session_data.iloc[s, :]
# date = ses_info[3]
# # path inputs
# path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
# path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
# session_type = path.split('\\')[-4].split(' ')[0]
# mscope = miniscope_session_class.miniscope_session(path)
# loco = locomotion_class.loco_class(path_loco)
#
# # Session data and inputs
# animal = mscope.get_animal_id()
# session = loco.get_session_id()
# traces_type = 'raw'
# [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
#  clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
#
# [coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = mscope.get_rois_aligned_reference_cluster(df_extract_rawtrace_detrended, coord_ext, animal)
#
# c=1
# idx_cluster = np.where(idx_roi_cluster_ordered_reference_ses == c)[0]
# rois_coordinates_cluster = np.array(list(chain.from_iterable(coord_ext_reference_ses[idx_cluster])))
# plt.figure()
# plt.scatter(rois_coordinates_cluster[:, 0], rois_coordinates_cluster[:, 1], color='blue')
# clusters_idx = np.where(coord_ext_overlap == c)[0]
# for i in clusters_idx:
#  plt.scatter(coord_ext[i][:, 0], coord_ext[i][:, 1], color='black')
