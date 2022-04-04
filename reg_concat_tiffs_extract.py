import numpy as np
import tifffile as tiff
import glob
import h5py
import os

path = 'E:\\TM RAW FILES\\tied baseline\\MC8855\\2021_04_04\\Registered video\\'
delim = path[-1]
path_loco = 'E:\\TM TRACKING FILES\\tied baseline S1 040421\\'

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Code\Miniscope pipeline\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)

animal = mscope.get_animal_id()
session = loco.get_session_id()
protocol = mscope.get_protocol_id()
path_split = path.split(delim)
path_join = delim.join(path_split[:-2])
path_save = path_join + '\\EXTRACT\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)

tiflist = glob.glob(path+'*.tif')  # get list of tifs
trial_id = []
for t in range(len(tiflist)):
    tifname = tiflist[t].split(delim)
    tifname_split = tifname[-1].split('_')
    trial_id.append(int(tifname_split[0][1:]))
trial_order = np.sort(trial_id)  # reorder trials
files_ordered = []  # order tif filenames by file order
for f in range(len(tiflist)):
    tr_ind = np.where(trial_order[f] == trial_id)[0][0]
    files_ordered.append(tiflist[tr_ind])
# read tiffs to concatenate them
list_tiffs = []
for f in range(len(files_ordered)):
    image_stack = tiff.imread(files_ordered[f])  # choose right tiff
    list_tiffs.append(image_stack)

reg_tiffs = np.vstack(list_tiffs)

# create h5
hf = h5py.File(path_save+'reg_data_'+animal+'_'+protocol+'_S'+str(session)+'.h5', 'w')
hf.create_dataset('dataset', data=reg_tiffs)
hf.close()
