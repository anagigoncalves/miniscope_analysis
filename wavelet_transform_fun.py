from analytic_wavelet import analytic_wavelet_transform, GeneralizedMorseWavelet
from libPeak import ridges_detection
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wavelet_transform_morse(trace,gamma=3,beta=2,min_scale=30,min_peak_position=15,min_freq=0.2, max_freq=15):
    
    morse = GeneralizedMorseWavelet(gamma=gamma, beta=beta)
    fs = morse.log_spaced_frequencies(high=np.pi/min_freq, low=np.pi/max_freq)
    psi, psi_f = morse.make_wavelet(trace.shape[-1], fs)
    cwt2d = analytic_wavelet_transform(np.real(trace), psi_f, np.isrealobj(psi))    
    cwt2d = abs(cwt2d)
    scales_ridges = ridges_detection(cwt2d)

    rifened_ridges=[]
    peaks_pos =[]
    for ridge in scales_ridges:
        cwt_ridge = []
        ind_x = ridge[0][:]
        ind_y = ridge[1][:]
        for c in range(len(ind_x)):
            cwt_ridge.append(cwt2d[ind_x[c],ind_y[c]])
        max_cwt_ridge = max(cwt_ridge)
        peak_pos = cwt_ridge.index(max_cwt_ridge)
        event_pos = np.where(cwt2d==max_cwt_ridge)[1]
        if len(ridge[0])>min_scale and peak_pos>min_peak_position:
            rifened_ridges.append(ridge)
            peaks_pos.append(event_pos)
    events_cwt = np.zeros((len(trace)))
    events_cwt[peaks_pos] = 1

    return cwt2d, rifened_ridges, events_cwt

def wavelet_transform_gaus2(trace,sampling_period=0.033,min_scale=10,min_peak_position=5, min_width=0.2, max_width=15, dw=0.5):
    widths = np.arange(min_width,max_width,dw)
    cwt2d, freq = pywt.cwt(trace,widths,wavelet='gaus2',sampling_period=sampling_period)
    scales_ridges = ridges_detection(cwt2d)

    rifened_ridges=[]
    peaks_pos =[]
    for ridge in scales_ridges:
        cwt_ridge = []
        ind_x = ridge[0][:]
        ind_y = ridge[1][:]
        for c in range(len(ind_x)):
            cwt_ridge.append(cwt2d[ind_x[c],ind_y[c]])
        max_cwt_ridge = max(cwt_ridge)
        peak_pos = cwt_ridge.index(max_cwt_ridge)
        event_pos = np.where(cwt2d==max_cwt_ridge)[1]
        if len(ridge[0])>min_scale and peak_pos>min_peak_position:
            rifened_ridges.append(ridge)
            peaks_pos.append(event_pos)
    events_cwt = np.zeros((len(trace)))
    events_cwt[peaks_pos] = 1

    return cwt2d, rifened_ridges, events_cwt


def plot_cwt2d_trace(cwt_ax, cwt2d, trace, events=None, cmap='seismic'):
    trace_ax = plt.twinx(cwt_ax)
    cwt_ax.imshow(cwt2d, cmap=cmap, aspect='auto',vmax=cwt2d.max(), vmin=-cwt2d.max())
    cwt_ax.invert_yaxis()
    if events is not None:
        trace_ax.plot(events, 'grey', alpha=0.7)
    trace_ax.plot(trace, 'black')
    return

def plot_ridges_trace_events(ridges_ax, ridges, trace, events, events_comp = None):
    trace_ax = plt.twinx(ridges_ax)
    if events_comp is not None:
        trace_ax.plot(events_comp, 'grey', alpha=0.7) 
        events = events*0.5
    trace_ax.plot(trace, 'black')
    for ridge in ridges:
        ridges_ax.scatter(ridge[1][:], ridge[0][:],s=4,color='red', marker ='.' )
    trace_ax.plot(events, 'purple', alpha=0.7)
    return

