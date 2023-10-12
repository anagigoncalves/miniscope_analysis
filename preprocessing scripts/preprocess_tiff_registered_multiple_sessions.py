# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:04:09 2023

@author: Ana
"""
import os
import glob
import numpy as np
import tifffile as tiff

#%% Create registered tiffs
#path inputs
path = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast 405 with 480 videos for processing together\\MC13419\\2022_05_31\\'
# path_loco = 'J:\\TM TRACKING FILES\\split contra fast S1 060421\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
trials = np.arange(1, len(glob.glob(os.path.join(path, 'Suite2p') +'\\*.tif'))+1)
frames_dFF = mscope.get_black_frames() #black frames removed before ROI segmentation
frame_time = mscope.get_miniscope_frame_time(trials,frames_dFF,version_mscope) #get frame time for each trial
reg_path = mscope.path +'\\Suite2p\\suite2p\\plane0\\reg_tif\\'
if not os.path.exists(mscope.path+'\\Registered video\\'):
    os.mkdir(mscope.path+'Registered video\\')
trial_end = np.cumsum([len(frame_time[k]) for k in range(len(frame_time))])
trial_beg = np.insert(trial_end[:-1],0,0)
tiflist = glob.glob(reg_path+'*.tif') 
tiflist_frames = np.arange(0, len(tiflist)*50, 50)
tiflist_frames[-1] = trial_end[-1]
for t in trials:
    idx_trial = np.where(trials == t)[0][0]
    print('Creating registered tiff for T'+str(t))
    if t == 1: #see which ref tifs Suite2p correspond to each trial
        tif_idx_to_concat = np.where((tiflist_frames>=trial_beg[idx_trial])&(tiflist_frames<=trial_end[idx_trial]))[0]
        tif_idx_to_concat = np.append(tif_idx_to_concat,tif_idx_to_concat[-1]+1)
    if t > 1:
        tif_idx_to_concat = np.where((tiflist_frames>=trial_beg[idx_trial])&(tiflist_frames<=trial_end[idx_trial]))[0]
        tif_idx_to_concat = np.append(tif_idx_to_concat,tif_idx_to_concat[-1]+1) 
        tif_idx_to_concat = np.insert(tif_idx_to_concat, 0, tif_idx_to_concat[0]-1)
    if t == trials[-1]:
        tif_idx_to_concat = np.where((tiflist_frames>=trial_beg[idx_trial])&(tiflist_frames<=trial_end[idx_trial]))[0]
        tif_idx_to_concat = np.insert(tif_idx_to_concat, 0, tif_idx_to_concat[0]-1)
    tiflist_concat = [] #Concatenate only the corresponding ref tiffs
    for tifff in tif_idx_to_concat:
        print(tiflist[tifff])
        image_stack = tiff.imread(tiflist[tifff])
        tiflist_concat.append(image_stack)
    tiflist_concatenated = np.concatenate(tiflist_concat,axis=0)
    if t == 1: #Only create tiffs from the frames that correspond to that trial
        trial_frames_full = tiflist_concatenated[trial_beg[idx_trial]:trial_end[idx_trial],:,:] 
    if t > 1:
        tif_start_discard_len = len(np.arange(tiflist_frames[tif_idx_to_concat[0]],trial_beg[idx_trial]))
        trial_len = trial_end[idx_trial]-trial_beg[idx_trial]
        trial_frames_full = tiflist_concatenated[tif_start_discard_len:trial_len,:,:] 
    tiff.imsave(path+'Registered video\\T'+str(t)+'_reg.tif', trial_frames_full, bigtiff=True)

#%% Get registered tiffs and do downsampled version
#path inputs
path = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast 405 with 480 videos for processing together\\MC13419\\2022_05_31\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
animal = mscope.get_animal_id()
session = 2
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