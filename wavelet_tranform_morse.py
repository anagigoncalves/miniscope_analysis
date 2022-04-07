from lib2to3.pgen2.token import RIGHTSHIFTEQUAL
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import convolve
from analytic_wavelet import analytic_wavelet_transform, GeneralizedMorseWavelet, make_unpad_slices, masked_detrend, \
    wavelet_contourf, time_series_plot, load_test_data
from analytic_wavelet.ridge_analysis import _indicator_ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
# wavelet
data_raw = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv')
event = pd.read_csv('preprocessed_data/AnaG_events_all.csv')
ROIs = data_raw.columns.values[3:]
n_trials = 23
morse = GeneralizedMorseWavelet(gamma=3, beta=2)
fs = morse.log_spaced_frequencies(high=np.pi, low=np.pi / 41)
print(fs)

#frequencyevaluation --> psd
for r in ROIs[:1]:
    for t in range(1,2):#n_trials+1):
    
        data_raw[r][data_raw['trial']==t] = data_raw[r][data_raw['trial']==t].fillna(0)
        roi_trace = data_raw[r][data_raw['trial']==t].to_numpy()
        psi, psi_f = morse.make_wavelet(roi_trace.shape[-1], fs)
        wp = analytic_wavelet_transform(np.real(roi_trace), psi_f, np.isrealobj(psi))    
        wp = abs(wp)
        roi_events = event[r][event['trial']==t].to_numpy()

        all_ridges = []
        scale_ridges = []
    

        fig1,(ridges_ax, ax1) = plt.subplots(2,1, sharex=True)        
        ridges_ax.set_title(r + ' trail '+str(t))
        trace_ax = plt.twinx(ridges_ax)
        trace_ax.plot(roi_events, 'grey', alpha=0.7) 
        trace_ax.plot(roi_trace, 'black')
        trace_ax.set_ylim(0,1)         
        for i,f in enumerate(fs):
            w = int(np.ceil(1/f))
            peaks,_= signal.find_peaks(wp[i,:], width=w)
            y = (i)*np.ones(len(peaks))
            ridges_ax.scatter(peaks,y,s=4,color='red', marker ='.')
            all_ridges.append(peaks)
            scale_ridges.append(y)
    
        ax2 = plt.twinx(ax1)
        # wavelet_contourf(ax1, np.arange(wp.shape[-1]), 1/fs, np.sqrt(np.square(np.abs(wp)) + np.square(np.abs(wn))), levels=100) 

        ax1.imshow(wp, cmap='seismic', aspect='auto',vmax=wp.max(), vmin=wp.min())
        ax1.invert_yaxis()
        ax2.plot(roi_events, 'grey', alpha=0.7)
        ax2.plot(roi_trace, 'black')
        ax2.set_ylim(0,1)
        plt.show()

        


# for i in range(len(wp)):

#     plt.plot(wp[i,:])
#     plt.ylim(-1,1)
#     plt.show()


