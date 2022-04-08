import numpy as np
import pandas as pd
import os 

GT_event = pd.read_csv('preprocessed_data/GT_events.csv', sep=';')
trial_time = pd.read_csv('preprocessed_data/AnaG_traces_raw.csv', usecols=['trial','time'])
ROIs = GT_event.columns

raw = np.load('preprocessed_data/trace_fiji.npy')
events = np.load('preprocessed_data/events_fiji.npy')
trace_fiji = pd.DataFrame(data=raw, columns=ROIs)
trace_fiji = pd.concat([trial_time, trace_fiji], axis=1)
events_fiji = pd.DataFrame(data=events, columns=ROIs)
events_fiji = pd.concat([trial_time, events_fiji], axis=1)
writer1 = pd.ExcelWriter('preprocessed_data/fiji.xlsx',engine='xlsxwriter')
sheet1_name = 'trace'
trace_fiji.to_excel(writer1, sheet_name=sheet1_name)
sheet2_name = 'events'
events_fiji.to_excel(writer1, sheet_name=sheet2_name)
writer1.save()

raw_bgsub = np.load('preprocessed_data/trace_fijibgsub.npy')
events_bgsub = np.load('preprocessed_data/events_fijibgsub.npy')
trace_fijibgsub = pd.DataFrame(data=raw_bgsub, columns=ROIs)
trace_fijibgsub = pd.concat([trial_time, trace_fijibgsub], axis=1)
events_fijibgsub = pd.DataFrame(data=events_bgsub, columns=ROIs)
events_fijibgsub = pd.concat([trial_time, events_fijibgsub], axis=1)
writer2 = pd.ExcelWriter('preprocessed_data/fijibgsub.xlsx',engine='xlsxwriter')
sheet1_name = 'trace'
trace_fijibgsub.to_excel(writer2, sheet_name=sheet1_name)
sheet2_name = 'events'
events_fijibgsub.to_excel(writer2, sheet_name=sheet2_name)
writer2.save()

raw_ext = np.load('preprocessed_data/trace_ext.npy')
events_ext = np.load('preprocessed_data/events_ext.npy')
trace_extract = pd.DataFrame(data=raw_ext, columns=ROIs)
trace_extract = pd.concat([trial_time, trace_extract], axis=1)
events_extract = pd.DataFrame(data=events_ext, columns=ROIs)
events_extract = pd.concat([trial_time, events_extract], axis=1)
writer3 = pd.ExcelWriter('preprocessed_data/extract.xlsx',engine='xlsxwriter')
sheet1_name = 'trace'
trace_extract.to_excel(writer3, sheet_name=sheet1_name)
sheet2_name = 'events'
events_extract.to_excel(writer3, sheet_name=sheet2_name)
writer3.save()