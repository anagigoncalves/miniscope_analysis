# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 16:37:45 2020

@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean

#path inputs
path_loco = 'J:\\Miniscope processed files\\Non-miniscope locomotion data\\Treadmill adaptation\\'
print_plots  = 1
frames_dFF = 0 #black frames removed before ROI segmentation
paw_colors = ['red','magenta','blue','cyan']
plot_trials = np.array([3,4,12,13,23])
trials_name = ['last baseline', 'early split', 'late split', 'early washout', 'late washout']
color_trials = ['black','red','salmon','dodgerblue','turquoise']

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
path_save = path_loco+'grouped output\\'
    
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

#Get session protocol
count_animal = 0
Ntrials_all = np.zeros(len(animal_list))
tied_trials_all = []
split_trials_all = []
washout_trials_all = []
for animal in animal_list:
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal,session)
    Ntrials_all[count_animal] = len(filelist)
    trial_type = []
    for f in filelist:
        trial_type.append(f.split('_')[4])
    split_1_idx = trial_type.index('split')
    tied_trials = np.arange(1,split_1_idx+1)
    tied_trials_all.append(tied_trials)
    split_trials = np.arange(split_1_idx+1,split_1_idx+11)
    split_trials_all.append(split_trials)
    washout_trials = np.arange(split_trials[-1]+1,split_trials[-1]+11)
    washout_trials_all.append(washout_trials)
    count_animal += 1
Ntrials = np.int64(np.max(Ntrials_all))

#summary gait parameters
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stance_speed']
#param_sym_name = ['coo','step_length','double_support','coo_stance','swing_length', 'phase_st', 'stance_speed']
param_sym = np.zeros((len(param_sym_name),len(animal_list),Ntrials))
param_phase = np.zeros((4,len(animal_list),Ntrials))
stance_speed = np.zeros((4,len(animal_list),Ntrials))
for count_animal, animal in enumerate(animal_list):
    session = int(session_list[count_animal])
    filelist = loco.get_track_files(animal,session)
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f,0.9,frames_dFF)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks,1)
        paws_rel = loco.get_paws_rel(final_tracks,'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter,final_tracks,paws_rel,st_strides_mat,sw_pts_mat,param)
            # if param == 'phase_st':
            #     for p in range(4):
            #         param_phase[p,count_animal,count_trial] = np.degrees(circmean(param_mat[0][p],nan_policy = 'omit'))
            if param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
if not os.path.exists(path_save):
    os.mkdir(path_save)
np.save(path_save + 'stance_speed', stance_speed)

#%% Plot
#baseline subtracion of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
for p in range(np.shape(param_sym)[0]-1):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p,a,:tied_trials_all[a][-1]-1])
        param_sym_bs[p,a,:] = param_sym[p,a,:] - bs_mean #will give an error if animals did sessions of different sizes

#plot symmetry baseline subtracted
for p in range(np.shape(param_sym)[0]-1):
    fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
    rectangle = plt.Rectangle((tied_trials_all[0][-1]+0.5,min(param_sym_bs[p,:,:].flatten())), 10, max(param_sym_bs[p,:,:].flatten())-min(param_sym_bs[p,:,:].flatten()), fc='dimgrey',alpha=0.3) 
    plt.gca().add_patch(rectangle)
    plt.hlines(0,1,len(param_sym_bs[p,a,:]),colors='grey',linestyles='--')
    for a in range(np.shape(param_sym)[1]):
        plt.plot(np.linspace(1,len(param_sym_bs[p,a,:]),len(param_sym_bs[p,a,:])),param_sym_bs[p,a,:], label = animal_list[a], linewidth = 2)
    ax.set_xlabel('Trial', fontsize = 20)
    ax.legend(frameon = False)
    ax.set_ylabel(param_sym_name[p].replace('_',' '), fontsize = 20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    np.save(path_save + param_sym_name[p], param_sym_bs[p, :, :])
    if print_plots:
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save+param_sym_name[p]+'_sym_bs', dpi=128)

# fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
# p = 2
# a = 9
# rectangle = plt.Rectangle((tied_trials_all[0][-1]+0.5,min(param_sym_bs[p,:,:].flatten())), 10, max(param_sym_bs[p,:,:].flatten())-min(param_sym_bs[p,:,:].flatten()), fc='dimgrey',alpha=0.3)
# plt.gca().add_patch(rectangle)
# plt.hlines(0,1,len(param_sym_bs[p,a,:]),colors='grey',linestyles='--')
# plt.plot(np.linspace(1,len(param_sym_bs[p,a,:]),len(param_sym_bs[p,a,:])),param_sym_bs[p,a,:], color='black', linewidth = 2)
# ax.set_xlabel('Trial', fontsize = 20)
# ax.set_ylabel(param_sym_name[p].replace('_',' '), fontsize = 20)
# if p == 2:
#     plt.gca().invert_yaxis()
# plt.xticks(fontsize = 16)
# plt.yticks(fontsize = 16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# if print_plots:
#     plt.savefig('C:\\Users\\Ana\\Desktop\\'+animal_list[a]+'_doublesupport_sym_bs', dpi=128)

# #plot symmetry non-baseline subtracted
# for p in range(np.shape(param_sym)[0]-2):
#     fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
#     rectangle = plt.Rectangle((tied_trials_all[0][-1]+0.5,min(param_sym[p,:,:].flatten())), 10, max(param_sym[p,:,:].flatten())-min(param_sym[p,:,:].flatten()), fc='dimgrey',alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     plt.hlines(0,1,len(param_sym_bs[p,a,:]),colors='grey',linestyles='--')
#     for a in range(np.shape(param_sym)[1]):
#         plt.plot(np.linspace(1,len(param_sym[p,a,:]),len(param_sym[p,a,:])),param_sym[p,a,:], label = animal_list[a], linewidth = 2)
#     ax.set_xlabel('Trial', fontsize = 20)
#     ax.legend(frameon = False)
#     ax.set_ylabel(param_sym_name[p].replace('_',' '), fontsize = 20)
#     if p == 2:
#         plt.gca().invert_yaxis()
#     plt.xticks(fontsize = 16)
#     plt.yticks(fontsize = 16)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     if print_plots:
#         if not os.path.exists(path_save):
#             os.mkdir(path_save)
#         plt.savefig(path_save+param_sym_name[p]+'_sym', dpi=128)

# #plot phase and stance speed
# for a in range(np.shape(stance_speed)[1]):
#     fig, ax = plt.subplots(1,2,figsize=(10,10), tight_layout=True)
#     ax = ax.ravel()
#     rectangle = plt.Rectangle((tied_trials_all[0][-1]+0.5,10), 10, 210-(-10), fc='dimgrey',alpha=0.3)
#     for i in range(2):
#         if i == 0: #phase_st
#             data = param_phase[:,a,:]
#         if i == 1:
#             data = stance_speed[:,a,:]
#         for p in range(4):
#             ax[i].axvline(x = tied_trials_all[0][-1]+0.5, color='dimgray', linestyle='--')
#             ax[i].axvline(x = split_trials_all[0][-1]+0.5, color='dimgray', linestyle='--')
#             ax[i].plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
#             ax[i].spines['right'].set_visible(False)
#             ax[i].spines['top'].set_visible(False)
#             ax[i].set_xlabel('Trial', fontsize = 20)
#             ax[i].set_ylabel('Stance phasing', fontsize = 20)
#             ax[i].tick_params(axis='x',labelsize = 16)
#             ax[i].tick_params(axis='y',labelsize = 16)
#             ax[i].set_title(animal_list[a],fontsize=18)
#     if print_plots:
#         if not os.path.exists(path_save):
#             os.mkdir(path_save)
#         plt.savefig(path_save+'_'+animal_list[a]+'_phase_st_stancespeed', dpi=96)
#
# p = 1
# fig, ax = plt.subplots(figsize=(5,10), tight_layout=True)
# rectangle = plt.Rectangle((tied_trials_all[0][-1]+0.5,min(param_sym_bs[p,:,:].flatten())), 10, max(param_sym_bs[p,:,:].flatten())-min(param_sym_bs[p,:,:].flatten()), fc='dimgrey',alpha=0.1)
# plt.gca().add_patch(rectangle)
# plt.hlines(0,1,len(param_sym_bs[p,a,:]),colors='grey',linestyles='--')
# for a in range(np.shape(param_sym)[1]):
#     plt.plot(np.linspace(1,len(param_sym_bs[p,a,:]),len(param_sym_bs[p,a,:])),param_sym_bs[p,a,:], color = 'darkgray', linewidth = 2)
# plt.plot(np.linspace(1,len(param_sym_bs[p,0,:]),len(param_sym_bs[p,0,:])),np.nanmean(param_sym_bs[p,:,:],axis=0), color = 'black', linewidth = 2)
# ax.set_xlabel('Trial', fontsize = 20)
# ax.set_ylabel(param_sym_name[p].replace('_',' '), fontsize = 20)
# if p == 2:
#     plt.gca().invert_yaxis()
# plt.xticks(fontsize = 16)
# plt.yticks(fontsize = 16)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# if print_plots:
#     if not os.path.exists(path_save):
#         os.mkdir(path_save)
#     plt.savefig(path_save+param_sym_name[p]+'_sym_bs_all', dpi=128)
