from lib2to3.pgen2.token import RIGHTSHIFTEQUAL
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import convolve
from analytic_wavelet import analytic_wavelet_transform, GeneralizedMorseWavelet
from libPeak import ridges_detection, peaks_position
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
fs = morse.log_spaced_frequencies(high=np.pi/0.2, low=np.pi / 15)
# print(fs)
# data_raw['ROI1']= data_raw['ROI1'].fillna(0)
# roi_trace = data_raw['ROI1'].to_numpy()
# f, Pxx_den = signal.periodogram(roi_trace, 30)
# plt.semilogy(f, Pxx_den)
# plt.hlines(1e-3,0,15,'red')
# plt.ylim([1e-7, 1e2])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
for r in ROIs:
    for t in range(1,n_trials+1):
    
        data_raw[r][data_raw['trial']==t] = data_raw[r][data_raw['trial']==t].fillna(0)
        roi_trace = data_raw[r][data_raw['trial']==t].to_numpy()
        roi_events = event[r][event['trial']==t].to_numpy()

        psi, psi_f = morse.make_wavelet(roi_trace.shape[-1], fs)
        wp = analytic_wavelet_transform(np.real(roi_trace), psi_f, np.isrealobj(psi))    
        wp = abs(wp)
        ridges = ridges_detection(wp)
        #aggiungere due condizioni
        # numero minimo di punti che definiscono il ridge (25)
        min_scale = 35
        # se il CWT massimo dello specifico ridge è minore della 15th (circa 3Hz) scale lo escludo come rumore ad alta frequenza
        min_peak_position = 20
    
        fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  

        ridges_ax.set_title(r + ' trail '+str(t))
        trace_ax1 = plt.twinx(ridges_ax)
        trace_ax1.plot(roi_events, 'grey', alpha=0.7) 
        trace_ax1.plot(roi_trace, 'black')
        #trace_ax1.set_ylim(0,1) 
        rifened_ridges=[]
        for ridge in ridges:
            cwt_ridge = []
            ind_x = ridge[0][:]
            ind_y = ridge[1][:]
            for c in range(len(ind_x)):
                cwt_ridge.append(wp[ind_x[c],ind_y[c]])
            max_cwt_ridge = max(cwt_ridge)
            peak_pos = cwt_ridge.index(max_cwt_ridge)
            if len(ridge[0])>min_scale and peak_pos>min_peak_position:
                ridges_ax.scatter(ridge[1][:], ridge[0][:],s=4,color='red', marker ='.' )
                rifened_ridges.append(ridge) 

        trace_ax2 = plt.twinx(cwt_ax)
        cwt_ax.imshow(wp, cmap='seismic', aspect='auto',vmax=wp.max(), vmin=-wp.max())
        cwt_ax.invert_yaxis()
        # ax2.plot(roi_events, 'grey', alpha=0.7)
        trace_ax2.plot(roi_trace, 'black')
        #trace_ax2.set_ylim(0,1)
        plt.show()

        


# for i in range(len(wp)):

#     plt.plot(wp[i,:])
#     plt.ylim(-1,1)
#     plt.show()


