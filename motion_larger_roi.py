# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt

# path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
version_mscope = 'v4'

# import classes
os.chdir('C:\\Users\\Ana\\PycharmProjects\\MSCOPEproject\\')
import miniscope_session_class

mscope = miniscope_session_class.miniscope_session(path)

[dFF, dFF_trial, trials, keep_rois, reg_th, reg_bad_frames] = mscope.load_processed_dFF()
ref_image = mscope.get_ref_image()
# Registration bad moments - correct
[x_offset, y_offset, corrXY] = mscope.get_reg_data()
reg_good_frames = np.setdiff1d(np.arange(0, np.shape(dFF)[1]), reg_bad_frames)
x_offset_clean = x_offset[reg_good_frames]
y_offset_clean = y_offset[reg_good_frames]

# Load corresponding EXTRACT
thrs_spatial_weights = 0
[coord_cell, trace] = mscope.read_extract_output(thrs_spatial_weights)
fsize = 20
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for r in range(len(coord_cell)):
    ax.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], s=1)
ax.set_title('EXTRACT ROIs')
ax.imshow(ref_image,
          extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um, 0])
ax.set_xlabel('FOV in micrometers', fontsize=fsize - 4)
ax.set_ylabel('FOV in micrometers', fontsize=fsize - 4)
ax.tick_params(axis='x', labelsize=fsize - 4)
ax.tick_params(axis='y', labelsize=fsize - 4)

# Find radius of EXTRACT ROIs
[width_roi, height_roi, aspect_ratio] = mscope.get_roi_stats(coord_cell)
x_offset_minmax = np.abs(np.array([np.min(x_offset_clean), np.max(x_offset_clean)])) #corresponds to x in coord_cell
y_offset_minmax = np.abs(np.array([np.min(y_offset_clean), np.max(y_offset_clean)])) #corresponds to y in coord_cell

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.scatter(np.arange(1, len(coord_cell)+1), width_roi, s=10, color='blue')
ax.axhline(x_offset_minmax[0], color='black')
ax.axhline(x_offset_minmax[1], color='black')
ax.set_title('ROIs width and max and min offsets')

fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
ax.scatter(np.arange(1, len(coord_cell)+1), height_roi, s=10, color='blue')
ax.axhline(y_offset_minmax[0], color='black')
ax.axhline(y_offset_minmax[1], color='black')
ax.set_title('ROIs height and max and min offsets')

# for r in range(len(coord_cell)):
#     fig, ax = plt.subplots(figsize=(20, 20), tight_layout=True)
#     ax.scatter(coord_cell[r][:, 0], coord_cell[r][:, 1], s=1)
#     params, ell = mscope.fitEllipse(coord_cell[r], 1)
#     width = params[2]
#     ax.scatter(ell[:,0], ell[:,1], s=0.5, color='red')
#     ax.set_title('EXTRACT ROI width '+str(np.round(width,2)))
#     ax.imshow(ref_image,
#               extent=[0, np.shape(ref_image)[1] / mscope.pixel_to_um, np.shape(ref_image)[0] / mscope.pixel_to_um, 0])
#     ax.set_xlabel('FOV in micrometers', fontsize=fsize - 4)
#     ax.set_ylabel('FOV in micrometers', fontsize=fsize - 4)
#     ax.tick_params(axis='x', labelsize=fsize - 4)
#     ax.tick_params(axis='y', labelsize=fsize - 4)
#     bpress = plt.waitforbuttonpress()
#     plt.close('all')
