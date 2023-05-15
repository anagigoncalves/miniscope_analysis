# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 11:35:26 2022

@author: Ana
"""
import os
import glob
import numpy as np
import tifffile as tiff

#%% Create registered tiffs
#path inputs
path = 'J:\\TM RAW FILES\\split contra fast\\MC8855\\2021_04_06\\'
path_loco = 'J:\\TM TRACKING FILES\\split contra fast S1 060421\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
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
    print(tifff)
    image_stack = tiff.imread(tifff)
    tiflist_all.append(image_stack)
print('Concatenating reg tiffs from Suite2p')
tiflist_concat = np.concatenate(tiflist_all,axis=0)
for t in trials:
    idx_trial = np.where(trials == t)[0][0]
    print('Creating registered tiff for T'+str(t))
    trial_frames_full = tiflist_concat[trial_beg[idx_trial]:trial_end[idx_trial],:,:] 
    tiff.imsave(path+'Registered video\\T'+str(t)+'_reg.tif', trial_frames_full, bigtiff=True)

#%% Get registered tiffs and do downsampled version
#path inputs
path = 'J:\\TM RAW FILES\\split contra fast\\MC8855\\2021_04_06\\'
path_loco = 'J:\\TM TRACKING FILES\\split contra fast S1 060421\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
animal = mscope.get_animal_id()
session = loco.get_session_id()
reg_path_tiff = mscope.path + 'Registered video\\'
tiflist_reg = glob.glob(reg_path_tiff + '*.tif') 
path_ses = '\\Registered downsampled session\\'
if not os.path.exists(mscope.path+path_ses):
    os.mkdir(mscope.path+path_ses)
if not os.path.exists(mscope.path+path_ses+'\\EXTRACT\\'):
    os.mkdir(mscope.path+path_ses+'\\EXTRACT\\')
tiff_all = []
for tifff in tiflist_reg:
    print('Creating downsampled registered tiff from '+tifff)
    tiff_name = tifff.split('\\')[-1]
    image_stack = tiff.imread(tifff)
    tiff_all.append(image_stack[300:600,:,:]) #downsample 10s of each trial
tiff_concat = np.concatenate(tiff_all)    
tiff.imsave(mscope.path+path_ses+'reg_data_'+animal+'_'+path.split(mscope.delim)[-4].replace(' ','_')+'_S'+str(session)+'.tif', tiff_concat, bigtiff=True)