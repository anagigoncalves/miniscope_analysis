# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as sp
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files'
session_data = pd.read_excel(path_session_data + '\\session_data_split_S1.xlsx')
load_path = path_session_data + '\\Analysis on population data\\STA paw spatial diff\\split ipsi fast S1\\'
save_path = 'J:\\Thesis\\for figures\\fig2\\'
var_name = 'FR-FL'
cond_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
colors_cond = ['black', (0.403921568627451, 0.0, 0.05098039215686274, 1.0),
               (0.9896613190730839, 0.7597147950089126, 0.6663101604278074, 1.0),
               (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0),
               (0.7935828877005348, 0.8702317290552584, 0.9429590017825312, 1.0)]
window = np.arange(-330, 330 + 1)  # Samples
zoom_in = np.array([-0.5, 0.25])
xaxis = window / 330
xaxis_start = np.where(xaxis >= zoom_in[0])[0][0]
xaxis_end = np.where(xaxis >= zoom_in[1])[0][0]

session_data_idx = 1 #example is MC9194
cluster_plot = 0
ses_info = session_data.iloc[session_data_idx, :]
date = ses_info[3]
# path inputs
path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                         ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                         date.split('_')[-3][2:] + '\\')
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)
session_type = path.split('\\')[-4].split(' ')[0]
animal = mscope.get_animal_id()
session = loco.get_session_id()
# Session data and inputs
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
 coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
 frames_dFF] = mscope.load_processed_files()
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
    trials, session_type, animal, session)
trials_ses_name.insert(len(trials_ses_name), 'late washout')
trials_idx = np.where(np.in1d(np.arange(trials[0], trials[-1] + 1), trials))[0]
[coord_ext_reference_ses, idx_roi_cluster_ordered_reference_ses, coord_ext_overlap, clusters_rois_overlap] = \
    mscope.get_rois_aligned_reference_cluster(df_events_extract_rawtrace, coord_ext, animal)

sta_zs = np.load(
    os.path.join(load_path, animal + ' ' + ses_info[0], 'sta_bodyvars_' + var_name.replace(' ', '_') + '_zscored.npy'))

sta_zs_zoom = np.zeros((np.shape(sta_zs)[0], len(cond_name), xaxis_end - xaxis_start))
sta_zs_zoom[:] = np.nan
trials_ses = trials_ses.flatten()[1:]
for count_c, c in enumerate(trials_ses_name):  # if odd is -1, if even is the next
    if count_c % 2 == 0:
        trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c] - 1)[0][0]]
        trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
    if count_c % 2 != 0:
        trial_start_idx = trials_idx[np.where(trials == trials_ses[count_c])[0][0]]
        trial_end_idx = trials_idx[np.where(trials == trials_ses[count_c] + 1)[0][0]]
    sta_zs_zoom[:, count_c, :] = np.nanmean(sta_zs[:, trial_start_idx:trial_end_idx, xaxis_start:xaxis_end], axis=1)

xaxis_short = xaxis[xaxis_start:xaxis_end]
idx_time0 = np.where(xaxis_short==0)[0][0]
def get_peakamp_latency(data, xaxis):
    idx_time0 = np.where(xaxis_short == 0)[0][0]
    data = np.transpose(sta_zs_zoom[roi, 0, :])
    data_filt = sp.medfilt(data - np.nanmean(data), 11)
    peaks_idx = sp.find_peaks(data_filt, width=10)[0]
    idx_closest_peak = np.argmax(peaks_idx - idx_time0)
    amp = data[peaks_idx[idx_closest_peak]]
    latency = xaxis_short[peaks_idx[idx_closest_peak]]
    return amp, latency
roi = 11
#find closest peak to 0
data = np.transpose(sta_zs_zoom[roi, 0, :])
data_filt = sp.medfilt(data-np.nanmean(data), 11)
peaks_idx = sp.find_peaks(data_filt, width=10)[0]
idx_closest_peak = np.argmax(peaks_idx-idx_time0)
amp = data[peaks_idx[idx_closest_peak]]
latency = xaxis_short[peaks_idx[idx_closest_peak]]
plt.plot(data, color='black')
plt.plot(data_filt, color='blue')
plt.scatter(peaks_idx, data_filt[peaks_idx], s=60, color='orange')