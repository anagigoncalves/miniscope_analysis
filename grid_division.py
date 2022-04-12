import numpy as np
import tifffile as tiff
import glob
import os
import matplotlib.pyplot as plt
fsize=18
f='D:\MC8855_splitipsifast_S1_2021_04_05\Raw video\T1.tif'
image_stack = tiff.imread(f) #choose right tiff
image_stack_gray = np.nanmean(image_stack,axis=3,dtype='float32')
fig, ax = plt.subplots(figsize = (5,5), tight_layout=True) #plot histogram os video
plt.hist(image_stack_gray.flatten(), color = 'gray', bins=100)
ax.set_xlim([0,255])
ax.set_xlabel('Pixel value', fontsize = fsize-4)
ax.set_ylabel('Count', fontsize = fsize-4)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize = fsize-4)
plt.yticks(fontsize = fsize-4)
plt.show()