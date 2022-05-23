# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:06:07 2021

@author: Ana
"""
import numpy as np
import tifffile as tiff
import glob
import os
import matplotlib.pyplot as plt

path = 'G:\\O meu disco\\BACKUP\\Miniscopes\\TM RAW FILES\\split_ipsi_fast_405\\MC12904\\2022_04_12\\'
delim = path[-1]
if delim == '/':
    path_output = path+'/Suite2p'
    path_images = path+'/images'
else:
    path_output = path+'\\Suite2p'
    path_images = path+'\\images'
if not os.path.exists(path_output):
    os.mkdir(path_output)
if not os.path.exists(path_images):
    os.mkdir(path_images)

heat_bool = input ("Correct for heating issue? 1 for yes, 0 for no: ") 

tiflist = glob.glob(path+'*.tif') #get list of tifs
tiff_boolean = 0
if not tiflist:
    tiflist = glob.glob(path+'*.tiff') 
    tiff_boolean = 1
trial_id = []
for t in range(len(tiflist)):
    tifname = tiflist[t].split(delim) 
    if tiff_boolean:
        tifname_split = tifname[-1].split('_')
        trial_id.append(int(tifname_split[-2]))
    else:
        trial_id.append(int(tifname[-1][1:-4])) #get trial order in that list    
trial_order = np.sort(trial_id) #reorder trials
files_ordered = [] #order tif filenames by file order
for f in range(len(tiflist)):
    tr_ind = np.where(trial_order[f] == trial_id)[0][0]
    files_ordered.append(tiflist[tr_ind])

fsize = 18
#read tiffs to do median of whole frame
for f in files_ordered:
    print('Processing '+ f.split(delim)[-1][:-4])
    image_stack = tiff.imread(f) #choose right tiff
    image_stack_gray = np.nanmean(image_stack,axis=3,dtype='float32')
    fig, ax = plt.subplots(figsize = (5,5), tight_layout=True) #plot histogram os video
    plt.hist(image_stack_gray.flatten(), color = 'gray', bins=100)
    ax.set_xlim([0,255])
    ax.set_xlabel('Pixel value', fontsize = fsize-4)
    ax.set_ylabel('Count', fontsize = fsize-4)
    ax.set_title(f.split(delim)[-1][:-4])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize = fsize-4)
    plt.yticks(fontsize = fsize-4)
    if delim == '/':
        if not os.path.exists(path+'images/histograms'):
            os.mkdir(path+'images/histograms')        
        plt.savefig(path+'images/histograms/'+ 'hist_'+f.split(delim)[-1][:-4], dpi=128)
    else:
        if not os.path.exists(path+'images\\histograms'):
            os.mkdir(path+'images\\histograms')        
        plt.savefig(path+'images\\histograms\\'+ 'hist_'+f.split(delim)[-1][:-4], dpi=128) 
    plt.close('all')                   
    mean_wholeframe = np.zeros(9)
    for i in range(9):
        mean_wholeframe[i] = np.nanmean(image_stack_gray[i,:,:].flatten(),dtype='float32')
    frame_start = np.argmax(np.diff(mean_wholeframe))+1
    image_stack_noblck = image_stack_gray[frame_start:,:,:]
    if int(heat_bool) == 1:
        roi_trace = np.zeros(np.shape(image_stack_noblck)[0])
        for frame in range(np.shape(image_stack_noblck)[0]):
            roi_trace[frame] = min(image_stack_noblck[frame,:,:].flatten()) #use min value for background
        frame_sub = np.zeros(np.shape(image_stack_noblck))
        for f1 in range(np.shape(image_stack)[1]):
            for f2 in range(np.shape(image_stack)[2]):
                frame_sub[:,f1,f2] = roi_trace
        image_stack_clean = np.subtract(image_stack_noblck,frame_sub,dtype='float32')
    if int(heat_bool) == 0:
        image_stack_clean = image_stack_noblck
    if delim == '/':
        tiff.imsave(path_output+'/'+f.split(delim)[-1][:-4]+'_blck'+str(frame_start+1)+'.tif',image_stack_clean,bigtiff = True)
    else:
        tiff.imsave(path_output+'\\'+f.split(delim)[-1][:-4]+'_blck'+str(frame_start+1)+'.tif',image_stack_clean,bigtiff = True)

