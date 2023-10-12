# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class

# path inputs
path_405 = 'J:\\Miniscope processed files\\TM RAW FILES\\split contra fast 405\\MC13420\\2022_05_31\\'
mscope = miniscope_session_class.miniscope_session(path_405)
coord_ext_405 = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
ref_image_405 = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)

path_480 = 'J:\\Miniscope processed files\\TM RAW FILES\\split contra fast\\MC13420\\2022_05_31\\'
mscope = miniscope_session_class.miniscope_session(path_480)
coord_ext_480 = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
ref_image_480 = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)

plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext_405)):
    plt.scatter(coord_ext_405[r][:, 0], coord_ext_405[r][:, 1], s=1, alpha=0.6)
plt.imshow(ref_image_405, cmap='gray',
           extent=[0, np.shape(ref_image_405)[1] / mscope.pixel_to_um, np.shape(ref_image_405)[0] / mscope.pixel_to_um, 0])
plt.title('405', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)

plt.figure(figsize=(10, 10), tight_layout=True)
for r in range(len(coord_ext_480)):
    plt.scatter(coord_ext_480[r][:, 0], coord_ext_480[r][:, 1], s=1, alpha=0.6)
plt.imshow(ref_image_480, cmap='gray',
           extent=[0, np.shape(ref_image_480)[1] / mscope.pixel_to_um, np.shape(ref_image_480)[0] / mscope.pixel_to_um, 0])
plt.title('480', fontsize=mscope.fsize)
plt.xlabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.ylabel('FOV in micrometers', fontsize=mscope.fsize - 4)
plt.xticks(fontsize=mscope.fsize - 4)
plt.yticks(fontsize=mscope.fsize - 4)