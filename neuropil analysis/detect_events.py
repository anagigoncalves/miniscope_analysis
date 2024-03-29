import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wavelet_transform_fun import *
import os

# traces_df =  pd.read_excel('preprocessed_data/fiji.xlsx', sheet_name='trace', engine='openpyxl')   
# traces = pd.DataFrame(traces_df)
path = '/media/careylab/Samsung_T5/TM RAW FILES/split ipsi fast/MC8855/2021_04_05/Registered video/'
traces = pd.read_csv(os.path.join(path,'df_extract_raw_split.csv'))
# events_comp_df =  pd.read_csv('preprocessed_data/manual_events.csv')   
# events_comp = pd.DataFrame(events_comp_df)

ROIs = traces.columns.values[2:]
n_trials = 23

for r in ROIs:
    for t in range(1,n_trials+1):    
        traces[r][traces['trial']==t] = traces[r][traces['trial']==t].fillna(0)
        roi_trace = traces[r][traces['trial']==t].to_numpy()
        # roi_events_comp = events_comp[r][traces['trial']==t].to_numpy()
        cwt2d, ridges, events_cwt = wavelet_transform_morse(roi_trace, min_scale=20,min_peak_position=15,min_freq=0.5, max_freq=15)
        # cwt2d, ridges, events_cwt = wavelet_transform_gaus2(roi_trace)
        fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  
        cwt_ax.set_title(r + ' trial '+str(t))
        plot_cwt2d_trace(cwt_ax, cwt2d, roi_trace,cmap='seismic')
        plot_ridges_trace_events(ridges_ax, ridges, roi_trace, events_cwt)
        plt.show()


