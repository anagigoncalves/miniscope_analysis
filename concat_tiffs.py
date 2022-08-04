# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:40:59 2022

@author: Ana
"""
import os
import glob
import numpy as np
import tifffile as tiff

#%% Get registered tiffs
#path inputs
path = 'I:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\'
path_loco = 'I:\\TM TRACKING FILES\\tied baseline\\tied baseline S1 040421\\'
version_mscope = 'v4'
#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\\Miniscope pipeline\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
animal = mscope.get_animal_id()
session = loco.get_session_id()
reg_path_tiff = mscope.path + 'Registered video\\'
tiflist_reg = glob.glob(reg_path_tiff + '*.tif') 

#%%
path_ses = '\\Registered downsampled session\\'
if not os.path.exists(mscope.path+path_ses):
    os.mkdir(mscope.path+path_ses)
if not os.path.exists(mscope.path+path_ses+'\\EXTRACT\\'):
    os.mkdir(mscope.path+path_ses+'\\EXTRACT\\')
tiff_all = []
for tifff in tiflist_reg:
    tiff_name = tifff.split('\\')[-1]
    image_stack = tiff.imread(tifff)
    tiff_all.append(image_stack[300:900,:,:]) #downsample 20s of each trial
tiff_concat = np.concatenate(tiff_all)    
tiff.imsave(mscope.path+path_ses+'reg_data_'+animal+'_'+path.split(mscope.delim)[-4].replace(' ','_')+'_S'+str(session)+'.tif', tiff_concat, bigtiff=True)
