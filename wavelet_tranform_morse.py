from lib2to3.pgen2.token import RIGHTSHIFTEQUAL
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import convolve
from analytic_wavelet import analytic_wavelet_transform, GeneralizedMorseWavelet
from libPeak import ridges_detection
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
fs = morse.log_spaced_frequencies(high=np.pi, low=np.pi / 15)
print(fs)

#frequencyevaluation --> psd
for r in ROIs:
    for t in range(1,n_trials+1):
    
        data_raw[r][data_raw['trial']==t] = data_raw[r][data_raw['trial']==t].fillna(0)
        roi_trace = data_raw[r][data_raw['trial']==t].to_numpy()
        roi_events = event[r][event['trial']==t].to_numpy()

        psi, psi_f = morse.make_wavelet(roi_trace.shape[-1], fs)
        wp = analytic_wavelet_transform(np.real(roi_trace), psi_f, np.isrealobj(psi))    
        wp = abs(wp)
        ridges = ridges_detection(wp)
    
        fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  

        ridges_ax.set_title(r + ' trail '+str(t))
        trace_ax1 = plt.twinx(ridges_ax)
        trace_ax1.plot(roi_events, 'grey', alpha=0.7) 
        trace_ax1.plot(roi_trace, 'black')
        #trace_ax1.set_ylim(0,1) 
        for i in ridges:
            ridges_ax.scatter(i[1][:], i[0][:],s=4,color='red', marker ='.' )

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


