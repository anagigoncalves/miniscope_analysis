import numpy as np
import pandas as pd
import os 

GT_event = pd.read_csv('preprocessed_data/GT_events.csv', sep=';')
traces = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv')
events = pd.read_csv('preprocessed_data/AnaG_events_all.csv')


writer1 = pd.ExcelWriter('preprocessed_data/fiji_all.xlsx',engine='xlsxwriter')
sheet1_name = 'trace'
traces.to_excel(writer1, sheet_name=sheet1_name)
sheet2_name = 'events'
events.to_excel(writer1, sheet_name=sheet2_name)
writer1.save()
