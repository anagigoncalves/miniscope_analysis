# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:33:52 2021

@author: Ana
"""
import os
import glob
import numpy as np
import tifffile as tiff

#%% Create registered tiffs
#path inputs
path = 'G:\\O meu disco\\BACKUP\\Cajal\\Level2\\'
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

# #%% Get registered tiffs
# #path inputs
# path = 'E:\\TM RAW FILES\\split ipsi fast\\MC8855\\2021_04_05\\'
# perc_th = 5
# version_mscope = 'v4'
# #import classes
# os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\\Miniscope pipeline\\')
# import miniscope_session_class
# mscope = miniscope_session_class.miniscope_session(path)
# reg_path_tiff = mscope.path + 'Registered video\\'
# tiflist_reg = glob.glob(reg_path_tiff + '*.tif') 
        
# #%% Remove background registered tiffs
# path_bgsub = '\\Registered video without background\\'
# if not os.path.exists(mscope.path+path_bgsub):
#     os.mkdir(mscope.path+path_bgsub)
# for tifff in tiflist_reg:
#     tiff_name = tifff.split('\\')[-1]
#     image_stack = tiff.imread(tifff)
#     # Compute background of registered tiffs
#     perc_pixel = np.zeros((np.shape(image_stack)[1],np.shape(image_stack)[2]))
#     for i in range(np.shape(image_stack)[1]):
#          for j in range(np.shape(image_stack)[2]):
#              perc_pixel[i,j] = np.percentile(image_stack[:,i,j],perc_th)
#     perc_pixel_tile = np.tile(perc_pixel,(np.shape(image_stack)[0],1,1))
#     image_stack_bgsub = image_stack-perc_pixel_tile
#     tiff.imsave(mscope.path+path_bgsub+'T'+tiff_name.split('_')[0][1:]+'_reg_bgsub.tif', image_stack_bgsub, bigtiff=True)




