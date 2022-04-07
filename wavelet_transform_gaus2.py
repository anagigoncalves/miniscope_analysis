import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
# wavelet
data_raw = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv')
event = pd.read_csv('preprocessed_data/AnaG_events_all.csv')
ROIs = data_raw.columns.values[3:]
n_trials = 23

dt = 0.033  # 30 Hz sampling
widths = np.arange(1,30)
wavelet = 'gaus2'
#frequencyevaluation --> psd
for r in ROIs[:1]:
    for t in range(1,2):#,n_trials+1):
    
        data_raw[r][data_raw['trial']==t] = data_raw[r][data_raw['trial']==t].fillna(0)
        roi_trace = data_raw[r][data_raw['trial']==t].to_numpy()
        cwtmatr, freq = pywt.cwt(roi_trace,widths,wavelet=wavelet,sampling_period=dt)
        roi_events = event[r][event['trial']==t].to_numpy()
        all_ridges = []
        scale_ridges = []
        fig1,(ridges_ax, ax1) = plt.subplots(2,1, sharex=True)        
        ridges_ax.set_title(r + ' trail '+str(t))
        trace_ax = plt.twinx(ridges_ax)
        trace_ax.plot(roi_events, 'grey', alpha=0.7) 
        trace_ax.plot(roi_trace, 'black')
        trace_ax.set_ylim(0,1) 

        ridges_matrix = np.zeros((len(widths),len(roi_trace)))        
        for i,w in enumerate(widths):
            peaks,_= signal.find_peaks(cwtmatr[i,:], width=w)
            y = (i)*np.ones(len(peaks))
            #ridges_ax.scatter(peaks,y,s=4,color='red', marker ='.')
            all_ridges.append(peaks)
            scale_ridges.append(y)
            ridges_matrix[i][peaks]=True 

        ridges_ax.imshow(ridges_matrix, aspect='auto', cmap='Reds')
        ridges_ax.invert_yaxis()
        ax2 = plt.twinx(ax1)
        ax1.imshow(cwtmatr, cmap='seismic', aspect='auto',vmax=cwtmatr.max(), vmin=cwtmatr.min())
        ax1.invert_yaxis()
        # ax2.plot(roi_events, 'grey', alpha=0.7)
        ax2.plot(roi_trace, 'black')
        ax2.set_ylim(0,1)
        plt.show()

        


# for i in range(len(cwtmatr)):

#     plt.plot(cwtmatr[i,:])
#     plt.plot(roi_events, 'grey')
#     plt.text(0,1,str(widths[i]))
#     plt.ylim(-1,1)
#     plt.show()


