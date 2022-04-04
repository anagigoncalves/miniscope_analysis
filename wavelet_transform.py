import wave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
# wavelet
data_raw = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv')
data_bgsub = pd.read_csv('preprocessed_data/AnaG_traces_bgsub.csv')
event = pd.read_csv('preprocessed_data/AnaG_events_all.csv')
roi1_events = event['ROI2'][event['trial']==2].to_numpy()

roi1 = data_raw['ROI2'][data_raw['trial']==2].to_numpy()

dt = 0.33  # 30 Hz sampling
widths = np.arange(0.01, 0.33,0.01)
wavelet_list = pywt.wavelist()
wavelet = 'morl'
scales = pywt.scale2frequency(wavelet, widths) / dt
cwtmatr, freq = pywt.cwt(roi1,scales,wavelet=wavelet, method='fft')

fig, ax1 = plt.subplots()
ax1.imshow(cwtmatr, cmap='seismic', aspect='auto', vmax=abs(cwtmatr).max(), vmin=cwtmatr.min())
ax2 = plt.twinx(ax1)
#ax2.plot(roi1_events*0.5, 'grey')
ax2.plot(roi1, 'black', alpha = 0.5)

plt.show()


# for i in range(len(cwtmatr)):
#     plt.plot(cwtmatr[i,:])
#     plt.text(0,1,str(scales[i]))
#     plt.ylim(-1,1)
#     plt.show()


