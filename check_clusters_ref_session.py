# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mp
from scipy import signal as sig
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S2.xlsx')

s = 4
ses_info = session_data.iloc[s, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
session_type = path.split('\\')[-4].split(' ')[0]
animal = mscope.get_animal_id()
session = loco.get_session_id()
# Session data and inputs
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session,
                                                                                         frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
trials_ses_name.insert(len(trials_ses_name), 'late washout')
trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1]+1), trials))[0]
[coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
    mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

rois_matched = np.where(coord_ext_overlap>0)[0]
plt.figure(figsize=(10, 10), tight_layout=True)
for r in rois_matched:
    plt.scatter(coord_ext[r][:, 0], coord_ext[r][:, 1], color=colors_cluster[np.int64(coord_ext_overlap[r]) - 1], s=1, alpha=0.6)
plt.imshow(ref_image, cmap='gray',
           extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um,
                   0])
plt.title('ROIs grouped by activity', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)


