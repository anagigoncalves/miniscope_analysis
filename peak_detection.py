# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:57:49 2023

@author: User
"""
import SlopeThreshold as ST
import os
import numpy as np
import matplotlib.pyplot as plt

path_data = 'E:\\Miniscope processed files\\TM TRACKING FILES\\split ipsi fast S1 050421\\'

# import class
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_data)
animal = 'MC8855'
session = 1
# Load behavioral data

filelist = loco.get_track_files(animal, session)
bodycenter_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.2, 0)
    bodycenter_trials.append(np.nanmean(final_tracks[0, :4, :], axis=0))

rawdata = -loco.inpaint_nans(bodycenter_trials[0])
Wn = 0.05 #0.1 for moving filter
acq_fq = 330
TimePntThres = 50
AmpPntThres = 30 # or TrueStd*2

# Filter the data
if isinstance(Wn, float):
    rawdata_filtered = ST.BesselFilter(rawdata, Wn, order=10, type='lowpass', filter='lfilter', Acq_Freq=0)
else:
    rawdata_filtered = rawdata

# Estimate Baseline
Baseline, Est_Std = ST.Estim_Baseline_PosEvents(rawdata_filtered, acq_fq, dtau=0.2, bmax_tslope=20, filtcut=1, graph=False)
# Calculate dF/F0:
F0 = Baseline + Est_Std * 2
dff = (rawdata_filtered - F0) / np.abs(F0)
dff = np.where(dff < 0, np.zeros_like(dff), dff)  # Remove negative dff values

# Find the most approximated noise amplitude estimation using the 'gaussian derivative' trick:
TrueStd, deriv_mean, deriv_std = ST.DerivGauss_NoiseEstim(dff, thres=2)

# Use that noise amplitude to define regions of slope change / stability
IncremSet, DecremSet, F_Values = ST.SlopeThreshold(rawdata_filtered, AmpPntThres, TimePntThres,
                                                CollapSeq=False, acausal=True, graph=None)
Ev_Onset = np.array(list(map(lambda x: x[0], IncremSet)))
Ev_Peaks = np.array(list(map(lambda x: x[1], IncremSet)))
Ev_Offset = np.array(list(map(lambda x: x[1], DecremSet)))

# find the offset corresponding to the onset
events_concat = np.concatenate((np.concatenate((Ev_Onset, Ev_Peaks)), Ev_Offset))
events_concat_type = np.concatenate((np.concatenate((np.repeat(1, len(Ev_Onset)), np.repeat(2, len(Ev_Peaks)))), np.repeat(3, len(Ev_Offset))))
events_concat_type_sort_idx = np.argsort(events_concat)
events_concat_sort = events_concat[events_concat_type_sort_idx]
events_concat_type_sort = events_concat_type[events_concat_type_sort_idx]
events_concat_onsets_idx = np.where(events_concat_type_sort==1)[0]
Ev_Onset_good = []
Ev_Peak_good = []
Ev_Offset_good = []
for i in events_concat_onsets_idx:
    if i < len(events_concat_sort)-2:
        if events_concat_type_sort[i+1] == 2:
            if events_concat_type_sort[i+2] == 3:
                Ev_Onset_good.append(events_concat_sort[i])
                Ev_Peak_good.append(events_concat_sort[i+1])
                Ev_Offset_good.append(events_concat_sort[i+2])

# compute peak as the maximum between onset and offset
Ev_Peak_max = []
for i in range(len(Ev_Onset_good)):
    Ev_Peak_max.append(np.argmax(rawdata[Ev_Onset_good[i]:Ev_Offset_good[i]])+Ev_Onset_good[i])

# if peaks smaller than range of baseline (baseline + est_std) discard
peaks_thr = np.nanmean(Baseline) + Est_Std
Ev_Peak_sel = []
Ev_Onset_sel = []
Ev_Offset_sel = []
for count_j, j in enumerate(Ev_Peak_max):
    if rawdata[j] > peaks_thr:
        Ev_Peak_sel.append(j)
        Ev_Onset_sel.append(Ev_Onset_good[count_j])
        Ev_Offset_sel.append(Ev_Offset_good[count_j])

plt.figure()
for i in range(len(Ev_Onset_sel)):
    rectangle = plt.Rectangle((Ev_Onset_sel[i], np.max(rawdata)), Ev_Offset_sel[i]-Ev_Onset_sel[i],
                          np.min(rawdata), fc='darkgrey', alpha=0.3, zorder=0)
    plt.gca().add_patch(rectangle)
plt.plot(rawdata, color='black')
plt.scatter(Ev_Peak_sel, rawdata[Ev_Peak_sel], s=150, marker = '.', c='k')
# plt.scatter(Ev_Onset_sel, rawdata[Ev_Onset_sel], s=300, marker = '.', c='g')
# plt.scatter(Ev_Offset_sel, rawdata[Ev_Offset_sel], s=150, marker = '.', c='r')