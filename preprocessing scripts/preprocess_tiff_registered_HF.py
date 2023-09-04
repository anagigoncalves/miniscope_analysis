# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:56:11 2023

@author: Ana
"""
import os
import glob
import numpy as np
import tifffile as tiff

#%% Create registered tiffs
#path inputs
path = r'D:\Titer analysis\TM RAW FILES\dilution 1 to 100\2020_01_27\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
reg_path = mscope.path +'\\Suite2p\\suite2p\\plane0\\reg_tif\\'
if not os.path.exists(mscope.path+'\\Registered video\\'):
    os.mkdir(mscope.path+'Registered video\\')
tiflist = glob.glob(reg_path+'*.tif') 
tiflist_all = []
for tifff in tiflist:
    print(tifff)
    image_stack = tiff.imread(tifff)
    tiflist_all.append(image_stack)
print('Concatenating reg tiffs from Suite2p')
if len(np.shape(tiflist_all[-1])) == 2:
    tiflist_concat2 = np.concatenate(tiflist_all[:-1],axis=0)
    tiflist_concat = np.concatenate((tiflist_concat2, np.swapaxes(np.dstack(tiflist_all[-1]), 2, 1)))
else:
    tiflist_concat = np.concatenate(tiflist_all,axis=0)
orig_tiflist = glob.glob(path+'*.tiff') 
trials = mscope.get_HF_trials(orig_tiflist)
orig_tiflist_ordered = []
for count_t, t in enumerate(trials):
    for l in orig_tiflist:
        if np.int64(l[l.rfind('T')+1:l.rfind('.')]) == t:
            orig_tiflist_ordered.append(l) 
frame_time = mscope.get_HF_frame_time(trials, orig_tiflist_ordered)
trial_end = np.cumsum([len(frame_time[k]) for k in range(len(frame_time))])
trial_beg = np.insert(trial_end[:-1],0,0)
for t in trials:
    idx_trial = np.where(trials == t)[0][0]
    print('Creating registered tiff for T'+str(t))
    trial_frames_full = tiflist_concat[trial_beg[idx_trial]:trial_end[idx_trial],:,:] 
    tiff.imsave(path+'Registered video\\T'+str(t)+'_reg.tif', trial_frames_full, bigtiff=True)
    
#%% Get registered tiffs and do downsampled version
#path inputs
reg_path_tiff = mscope.path + 'Registered video\\'
tiflist_reg = glob.glob(reg_path_tiff + '*.tif') 
tiflist_reg_ordered = []
for count_t, t in enumerate(trials):
    for l in tiflist_reg:
        if np.int64(l[l.rfind('T')+1:l.rfind('_')]) == t:
            tiflist_reg_ordered.append(l)   
path_ses = '\\Registered downsampled session\\'
if not os.path.exists(mscope.path+path_ses):
    os.mkdir(mscope.path+path_ses)
if not os.path.exists(mscope.path+path_ses+'\\EXTRACT\\'):
    os.mkdir(mscope.path+path_ses+'\\EXTRACT\\')
tiff_all = []
for tifff in tiflist_reg_ordered:
    print('Creating downsampled registered tiff from '+tifff)
    tiff_name = tifff.split('\\')[-1]
    image_stack = tiff.imread(tifff)
    tiff_all.append(image_stack[300:600,:,:]) #downsample 10s of each trial
tiff_concat = np.concatenate(tiff_all)    
tiff.imsave(mscope.path+path_ses+'reg_data_downsampled.tif', tiff_concat, bigtiff=True)