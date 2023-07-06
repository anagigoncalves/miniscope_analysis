# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class

path_session_data = 'J:\\Miniscope processed files\\'
session_data = pd.read_excel('J:\\Miniscope processed files\\session_data_split_S1.xlsx')

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
[df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace, coord_ext, reg_th, reg_bad_frames, trials,
 clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image, frames_dFF] = mscope.load_processed_files()

roi_plot = 161
trial_plot = 3
df_dff_trial = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, 'ROI'+str(roi_plot)]  # get dFF for the desired trial
fig, ax = plt.subplots(figsize=(20, 10), tight_layout=True)
idx_trial = np.where(trials == trial_plot)[0][0]
ax.plot(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial_plot, 'time'], df_dff_trial, color='black', linewidth=2)
events_plot = np.where(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial_plot, 'ROI'+str(roi_plot)])[0]
for e in events_plot:
    ax.scatter(np.array(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial_plot, 'time'])[e], df_dff_trial.iloc[e], s=120,
               color='orange')
ax.set_xlabel('Time (s)', fontsize=mscope.fsize - 2)
ax.set_ylabel('Calcium trace for trial ' + str(trial_plot), fontsize=mscope.fsize - 2)
plt.xlim([11, 36])
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='y', which='y', length=0)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
# plt.savefig('J:\\Miniscope figures\\for methods\\example_events', dpi=mscope.my_dpi)
plt.savefig('J:\\Miniscope figures\\for methods\\example_events.svg', format='svg', dpi=mscope.my_dpi)

