# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
path_save = 'J:\\Thesis\\figuresChapter2\\'
version_mscope = 'v4'
plot_data = 1
print_plots = 1
paw_colors = ['#e52c27', '#3854a4', '#ad4397', '#6fccdf']
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

#HEATMAP
clusters_rois_flat = np.transpose(sum(clusters_rois, []))
trial = 3
beg = 10
end = 45
fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
df_trial = df_extract_rawtrace_detrended.loc[(df_extract_rawtrace_detrended['trial'] == trial)&(df_extract_rawtrace_detrended['time']>beg)&(df_extract_rawtrace_detrended['time']<end)].iloc[:, 2:]  # Get df/f for the desired trial and interval
df_trial_ml = df_trial[clusters_rois_flat]
hm = sns.heatmap(df_trial_ml.T, cmap='viridis')
hm.figure.axes[-1].set_ylabel('\n\u0394F/F', size=20)
hm.figure.axes[-1].tick_params(labelsize=16)
ax.set_yticks(np.arange(0, np.shape(df_trial_ml)[1], 30))
ax.set_yticklabels(df_trial_ml.columns[np.arange(0, np.shape(df_trial_ml)[1], 30)])
ax.set_xticks(np.linspace(0, np.shape(df_trial_ml)[0], num=10))
ax.set_xticklabels(list(map(str, np.int64(np.linspace(beg, end, num=10)))), rotation=0)
ax.set_ylabel('ROIs', fontsize=20)
ax.set_xlabel('Time (s)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_heatmap', dpi=256)
plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_heatmap.svg', format='svg', dpi=256)

#ROIs
cmap_rois = plt.get_cmap('jet')
colors_rois = [cmap_rois(i) for i in np.linspace(0, 1, len(coord_ext))]
plt.figure(figsize=(7, 7), tight_layout=True)
for r in range(len(coord_ext)):
    plt.scatter(coord_ext[r][:, 0], coord_ext[r][:, 1], color=colors_rois[r], s=1, alpha=0.6)
plt.imshow(ref_image, cmap='gray',
           extent=[0, np.shape(ref_image)[1]/mscope.pixel_to_um, np.shape(ref_image)[0]/mscope.pixel_to_um,
                   0])
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)
plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_rois', dpi=256)
plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_rois.svg', format='svg', dpi=256)

trial = 2
rois = ['ROI97', 'ROI177']
data_time = np.array(df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial, 'time'])
fig, ax = plt.subplots(figsize=(15, 7), tight_layout=True)
for count_roi, roi in enumerate(rois):
    data_roi = df_extract_rawtrace_detrended.loc[df_extract_rawtrace_detrended['trial'] == trial, roi]
    data_events = np.where(df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == trial, roi])[0]
    ax.plot(data_time, data_roi + count_roi, color='black', linewidth=1.5)
    ax.scatter(data_time[data_events], np.repeat(np.nanmax(data_roi)+0.1, len(data_events)) + count_roi, s=600, marker='|', color='purple')
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_xlim([10, 30])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tick_params(axis='y', labelsize=0, length=0)
plt.setp(ax.get_yticklabels(), visible=False)
ax.tick_params(axis='both', which='major', labelsize=20)
# plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_events', dpi=256)
# plt.savefig(path_save + 'MC8855_trial3_splitipsifastS1_events.svg', format='svg', dpi=256)
