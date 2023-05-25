# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:57:49 2023

@author: User
"""
import SlopeThreshold as ST


rawdata = -bodycenter_aligned[trial_changes[-2]:trial_changes[-1]]
acq_fq = 30
TimePntThres = 200 
AmpPntThres = 10 # or TrueStd*2
    
# Derive noise amplitude
TrueStd, deriv_mean, deriv_std = ST.DerivGauss_NoiseEstim(rawdata, thres=2)

# Use that noise amplitude to define regions of slope change / stability
IncremSet, DecremSet, F_Values = ST.SlopeThreshold(rawdata, TrueStd*10, TimePntThres,
                                                   CollapSeq=False, acausal=True, graph=None)

peaks = []
onsets_pos = []
if len(IncremSet) > 0:
    if type(IncremSet[0]) is tuple:
        for i in range(len(IncremSet)):
            onsets_pos.append(IncremSet[i][0])
            if IncremSet[i][0] + TimePntThres >= len(rawdata):
                values_idx = np.arange(IncremSet[i][0], len(rawdata))
            else:
                values_idx = np.arange(IncremSet[i][0], IncremSet[i][0] + TimePntThres)
            peak_idx = np.argmax(rawdata[values_idx])
            peaks.append(IncremSet[i][0] + peak_idx)
            
# onsets_neg = []
# if len(DecremSet) > 0:
#     if type(DecremSet[0]) is tuple:
#         for i in range(len(DecremSet)):        
#             onsets_neg.append(DecremSet[i][1]) # DecremSet[i][1] because we want the offset of the peak

plt.figure()
plt.plot(rawdata)
plt.scatter(onsets_pos, rawdata[onsets_pos], s=60, marker = '.', c='g')
plt.scatter(peaks, rawdata[peaks], s=60, marker = '.', c='k')
# plt.scatter(onsets_neg, rawdata[onsets_neg], s=60, marker = '.', c='r')