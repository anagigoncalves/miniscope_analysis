# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 21:39:54 2021

@author: anago
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff

#%% Create registered tiffs
#path inputs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\Miniscope pipeline\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
trials = mscope.get_trial_id()
frames_dFF = mscope.get_black_frames() #black frames removed before ROI segmentation
frame_time = mscope.get_miniscope_frame_time(trials,frames_dFF,version_mscope) #get frame time for each trial
reg_path = mscope.path +'\\Suite2p\\suite2p\\plane0\\reg_tif\\'
if not os.path.exists(mscope.path+'\\Registered video\\'):
    os.mkdir(mscope.path+'Registered video\\')
trial_end = np.cumsum([len(frame_time[k]) for k in range(len(frame_time))])
trial_beg = np.insert(trial_end[:-1],0,0)
tiflist = glob.glob(reg_path+'*.tif') 
tiflist_all = []
for tifff in tiflist:
    image_stack = tiff.imread(tifff)
    tiflist_all.append(image_stack)
tiflist_concat = np.concatenate(tiflist_all,axis=0)
for t in trials:
    trial_frames_full = tiflist_concat[trial_beg[t-1]:trial_end[t-1],:,:] 
    tiff.imsave(path+'Registered video\\T'+str(t)+'_reg.tif', trial_frames_full, bigtiff=True)
    
#%% Crop registered tiffs
path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\Miniscope pipeline\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
if not os.path.exists(mscope.path+'\\Crop video\\'):
    os.mkdir(mscope.path+'Crop video\\')
reg_path_tiff = mscope.path + 'Registered video\\'
tiflist_reg = glob.glob(reg_path_tiff + '*.tif') 
tiff_name = tiflist_reg[0].split('\\')[-1]
image_stack_ex = tiff.imread(tiflist_reg[0])
xstart = 75
ystart = 250
width = 100
crop_region = np.array([[xstart,ystart],[xstart+width,ystart+width]])
fig,ax = plt.subplots(tight_layout = True)
ax.imshow(image_stack_ex[0,:,:])
rect = patches.Rectangle((xstart,ystart),width,width,linewidth=2,edgecolor='red',facecolor='none')    
ax.add_patch(rect)
plt.savefig(path+'crop_xstart'+str(xstart)+'_ystart'+str(ystart)+'_width'+str(width))

#%% Create cropped tiffs
for tifff in tiflist_reg:
    tiff_name = tifff.split('\\')[-1]
    image_stack = tiff.imread(tifff)
    image_stack_crop = image_stack[:,crop_region[0,1]:crop_region[1,1],crop_region[0,0]:crop_region[1,0]]
    tiff.imsave(path+'Crop video\\T'+tiff_name.split('_')[0][1:]+'_reg_crop.tif', image_stack_crop, bigtiff=True)
