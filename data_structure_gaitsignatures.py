# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:37:45 2020

@author: Ana
"""
import os
import numpy as np

# path inputs
path_loco = 'G:\\My Drive\\Gait signatures\\Data for manifold\\Treadmill adaptation\\Rawdata\\'
print_plots = 1
frames_dFF = 0  # black frames removed before ROI segmentation

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class

loco = locomotion_class.loco_class(path_loco)
path_save = path_loco + 'grouped output\\'

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])
# summary gait parameters
for count_animal, animal in enumerate(animal_list):
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal, session)
    final_tracks_trials = []
    len_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, frames_dFF)
        final_tracks_trials.append(final_tracks)
        len_trials.append(np.shape(final_tracks)[-1])
    min_len_trials = np.min(len_trials)
    final_tracks_trials_x = np.zeros((len(filelist), min_len_trials, 5))
    for i in range(len(filelist)):
        final_tracks_trials_x[i, :, :] = loco.inpaint_nans(np.transpose(final_tracks_trials[i][0, :, :min_len_trials]))
    np.save('G:\\My Drive\\Gait signatures\\Data for manifold\\Treadmill adaptation\\' + animal + '_xexcursion.npy', final_tracks_trials_x, allow_pickle=True)
