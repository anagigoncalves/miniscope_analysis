import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fun_morse import *

traces = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv')
events_comp = pd.read_csv('preprocessed_data/AnaG_events_all.csv')

ROIs = traces.columns.values[3:]
n_trials = 23

for r in ROIs:
    for t in range(1,n_trials+1):    
        traces[r][traces['trial']==t] = traces[r][traces['trial']==t].fillna(0)
        roi_trace = traces[r][traces['trial']==t].to_numpy()
        roi_events_comp = events_comp[r][events_comp['trial']==t].to_numpy()

        cwt2d, ridges, events_cwt = wavelet_transform_morse(roi_trace)
        
        fig1,(cwt_ax,ridges_ax) = plt.subplots(2,1, sharex=True)  
        cwt_ax.set_title(r + ' trail '+str(t))
        plot_cwt2d_trace(cwt_ax, cwt2d, roi_trace)
        plot_ridges_trace_events(ridges_ax, ridges, roi_trace, events_cwt, events_comp)
        plt.show()


