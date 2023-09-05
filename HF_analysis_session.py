# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import glob
import tifffile as tiff
import pandas as pd

# path inputs
save_path = 'D:\\Titer analysis\\session files\\'
version_mscope = 'v4'
plot_data = 1
load_data = 1
refine_events = 1
print_plots = 1
save_data = 1
fsize = 24

days = ['08', '10', '14', '17', '20', '23', '27']
dils = ['dilution_1_to_50']
for dil in dils:
    for day in days:
        path = 'D:\\Titer analysis\\TM RAW FILES\\' + dil.replace('_', ' ') + '\\2020_01_' + day + '\\'
        # path = 'C:\\Users\\Ana\\Desktop\\' + dil.replace('_', ' ') + '\\2020_01_' + day + '\\'
        session_type = path.split('\\')[-4].split(' ')[0]
        # import classes
        os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
        import miniscope_session_class
        mscope = miniscope_session_class.miniscope_session(path)

        # create plots folders
        path_images = os.path.join(path, 'images')
        path_events = os.path.join(path, 'images', 'events')
        if not os.path.exists(path_images):
            os.mkdir(path_images)
        if not os.path.exists(path_events):
            os.mkdir(path_events)

        # Trial structure, reference image and triggers
        orig_tiflist = glob.glob(path+'*.tiff')
        trials = mscope.get_HF_trials(orig_tiflist)
        ops_s2p = mscope.get_s2p_parameters()
        print(ops_s2p)
        traces_type = 'raw'

        if load_data == 0:
            ref_image = mscope.get_ref_image()
            orig_tiflist_ordered = []
            for count_t, t in enumerate(trials):
                for l in orig_tiflist:
                    if np.int64(l[l.rfind('T') + 1:l.rfind('.')]) == t:
                        orig_tiflist_ordered.append(l)
            frame_time = mscope.get_HF_frame_time(trials, orig_tiflist_ordered)
            trial_length_cumsum = mscope.cumulative_trial_length(frame_time)
            # Load ROIs and traces - EXTRACT
            thrs_spatial_weights = 0
            [coord_ext, df_extract_allframes] = mscope.read_extract_output(thrs_spatial_weights, frame_time, trials)

            # Good periods after motion correction
            th = 0.0067 # change with the notes from EXCEL
            [x_offset, y_offset, corrXY] = mscope.get_reg_data()  # registration bad moments
            [idx_to_nan, df_extract] = mscope.corr_FOV_movement(th, df_extract_allframes, corrXY)
            [width_roi_ext, height_roi_ext, aspect_ratio_ext] = mscope.get_roi_stats(coord_ext)

            # Get raw trace from EXTRACT ROIs
            roi_list = list(df_extract.columns[2:])
            trial_length = mscope.trial_length(df_extract)
            ext_trace_trials = []
            for t in trials:
                idx_trial = np.where(trials == t)[0][0]
                tiff_stack = tiff.imread(
                    os.path.join(mscope.path, 'Registered video') + mscope.delim + 'T' + str(t) + '_reg.tif')  # read tiffs
                ext_trace = np.zeros((int(trial_length[idx_trial]), np.shape(df_extract.iloc[:, 2:])[1]))
                ext_trace[:] = np.nan
                for c in range(len(coord_ext)):
                    ext_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
                    for f in range(np.shape(tiff_stack)[0]):
                        ext_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                            np.round(coord_ext[c][:, 1] * mscope.pixel_to_um)), np.int64(
                            np.round(coord_ext[c][:, 0] * mscope.pixel_to_um))]) / np.shape(coord_ext[c])[0]
                    ext_trace[:len(ext_trace_tiffmean), c] = ext_trace_tiffmean
                ext_trace_trials.append(ext_trace)
            ext_trace_arr = np.transpose(np.vstack(ext_trace_trials))  # trace as dataframe
            trial_ext = []
            frame_time_ext = []
            for t in trials:
                idx_trial = np.where(trials == t)[0][0]
                trial_ext.extend(np.repeat(t, len(frame_time[idx_trial])))
                frame_time_ext.extend(frame_time[idx_trial])
            dict_ext = {'trial': trial_ext, 'time': frame_time_ext}
            df_ext1 = pd.DataFrame(dict_ext)
            df_ext2 = pd.DataFrame(np.transpose(ext_trace_arr), columns=roi_list)
            df_extract_rawtrace = pd.concat([df_ext1, df_ext2], axis=1)

            # ROI spatial stats
            [width_roi_rois_nomotion, height_roi_rois_nomotion, aspect_ratio_rois_nomotion] = mscope.get_roi_stats(coord_ext)
            # ROI curation
            trial_curation = 1
            [coord_ext_curated, df_extract_rawtrace_curated] = mscope.roi_curation(ref_image, df_extract_rawtrace, coord_ext, aspect_ratio_rois_nomotion, trial_curation)

            no_rois = 0
            if no_rois:
                data = float('NaN')
                np.save(os.path.join(save_path, dil + '_day_' + day + '_roinr'), data)
                np.save(os.path.join(save_path, dil + '_day_' + day + '_fr'), data)
                np.save(os.path.join(save_path, dil + '_day_' + day + '_cv'), data)
                np.save(os.path.join(save_path, dil + '_day_' + day + '_cv2'), data)
                np.save(os.path.join(save_path, dil + '_day_' + day + '_perc95'), data)
                np.save(os.path.join(save_path, dil + '_day_' + day + '_skew'), data)

            # Get events
            detrend_bool = 1
            csv_name = 'df_events_extract_rawtrace'
            roi_trace = np.array(df_extract_rawtrace_curated.iloc[:, 2:])
            roi_list = list(df_extract_rawtrace_curated.columns[2:])
            trial_ext = list(df_extract_rawtrace_curated['trial'])
            trials = np.unique(trial_ext)
            frame_time_ext = list(df_extract_rawtrace_curated['time'])
            data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
            df_dFF1 = pd.DataFrame(data_dFF1)
            df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
            df_events_extract_rawtrace = pd.concat([df_dFF1, df_dFF2], axis=1)
            roi_list = df_extract_rawtrace_curated.columns[2:]
            count_r = 0
            for r in roi_list:
                print('Processing events of ' + r)
                count_t = 0
                for t in trials:
                    data = np.array(df_extract_rawtrace_curated.loc[df_extract_rawtrace_curated['trial'] == t, r])
                    events_mat = np.zeros(len(data))
                    if len(data) == len(np.where(np.isnan(data))[0]):
                        Ev_Onset = []
                    elif t == 5 or t == 6:
                        Ev_Onset = []
                    else:
                        [Ev_Onset, IncremSet] = mscope.compute_events_onset(data, mscope.sr, detrend_bool)
                    if len(Ev_Onset) > 0:
                        events = mscope.event_detection_calcium_trace(data, Ev_Onset, IncremSet, 3)
                        if detrend_bool == 0:
                            events_new = []
                            for e in events:
                                if data[e] >= np.nanpercentile(data, 75):
                                    events_new.append(e)
                            events = events_new
                        events_mat[events] = 1
                    else:
                        print('No events for ' + r + ' trial ' + str(t))
                    df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, r] = events_mat
                    count_t += 1
                count_r += 1
            if len(csv_name) > 0:
                if not os.path.exists(os.path.join(mscope.path, 'processed files')):
                    os.mkdir(os.path.join(mscope.path, 'processed files'))
                df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', csv_name + '.csv'), sep=',', index=False)

            roi_plot = np.int64(np.random.choice(roi_list)[3:])
            trial_plot = np.random.choice(trials)
            mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace_curated, traces_type, df_events_extract_rawtrace, trials, plot_data, print_plots)

            # Detrend calcium trace
            df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace_curated, 'df_extract_rawtrace_detrended')

            # Save data
            if not os.path.exists(mscope.path + 'processed files'):
                os.mkdir(mscope.path + 'processed files')
            df_extract_rawtrace_curated.to_csv(os.path.join(mscope.path, 'processed files', 'df_extract_raw.csv'), sep=',',
                                       index=False)
            df_extract_rawtrace_detrended.to_csv(
                os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'), sep=',',
                index=False)
            df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'),
                                              sep=',', index=False)
            np.save(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), coord_ext, allow_pickle=True)
            np.save(os.path.join(mscope.path, 'processed files', 'trials.npy'), trials)
            np.save(os.path.join(mscope.path, 'processed files', 'reg_th.npy'), th)
            np.save(os.path.join(mscope.path, 'processed files', 'frames_to_exclude.npy'), idx_to_nan)

        if refine_events:
            df_extract_rawtrace = pd.read_csv(
                os.path.join(mscope.path, 'processed files', 'df_extract_raw.csv'))
            # Get events
            detrend_bool = 1
            csv_name = 'df_events_extract_rawtrace'
            roi_trace = np.array(df_extract_rawtrace.iloc[:, 2:])
            roi_list = list(df_extract_rawtrace.columns[2:])
            trial_ext = list(df_extract_rawtrace['trial'])
            trials = np.unique(trial_ext)
            frame_time_ext = list(df_extract_rawtrace['time'])
            data_dFF1 = {'trial': trial_ext, 'time': frame_time_ext}
            df_dFF1 = pd.DataFrame(data_dFF1)
            df_dFF2 = pd.DataFrame(np.zeros(np.shape(roi_trace)), columns=roi_list)
            df_events_extract_rawtrace = pd.concat([df_dFF1, df_dFF2], axis=1)
            roi_list = df_extract_rawtrace.columns[2:]
            count_r = 0
            for r in roi_list:
                print('Processing events of ' + r)
                count_t = 0
                for t in trials:
                    data = np.array(df_extract_rawtrace.loc[df_extract_rawtrace['trial'] == t, r])
                    events_mat = np.zeros(len(data))
                    if len(data) == len(np.where(np.isnan(data))[0]):
                        Ev_Onset = []
                    elif t == 5 or t == 6:
                        Ev_Onset = []
                    else:
                        [Ev_Onset, IncremSet] = mscope.compute_events_onset(data, mscope.sr, detrend_bool)
                    if len(Ev_Onset) > 0:
                        events = mscope.event_detection_calcium_trace(data, Ev_Onset, IncremSet, 3)
                        if detrend_bool == 0:
                            events_new = []
                            for e in events:
                                if data[e] >= np.nanpercentile(data, 75):
                                    events_new.append(e)
                            events = events_new
                        events_mat[events] = 1
                    else:
                        print('No events for ' + r + ' trial ' + str(t))
                    df_events_extract_rawtrace.loc[df_events_extract_rawtrace['trial'] == t, r] = events_mat
                    count_t += 1
                count_r += 1
            if len(csv_name) > 0:
                if not os.path.exists(os.path.join(mscope.path, 'processed files')):
                    os.mkdir(os.path.join(mscope.path, 'processed files'))
                df_events_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', csv_name + '.csv'), sep=',', index=False)

        if load_data:
            df_extract_rawtrace_detrended = pd.read_csv(
                os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
            df_events_extract_rawtrace = pd.read_csv(
                os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'))
            coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
            trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
            ref_image = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)

            isi_events = mscope.compute_isi(df_events_extract_rawtrace, traces_type, 'isi_events')
            [isi_events_cv, isi_events_cv2] = mscope.compute_isi_cv(isi_events, trials)
            roi_list = mscope.get_roi_list(df_extract_rawtrace_detrended)
            fr_roi = np.zeros(len(roi_list))
            for count_r, r in enumerate(roi_list):
                fr_roi[count_r] = 1/isi_events.loc[isi_events['roi'] == r, 'isi'].mean()
            cv_roi = np.zeros(len(roi_list))
            for count_r, r in enumerate(roi_list):
                cv_roi[count_r] = isi_events_cv.loc[isi_events_cv['roi'] == r, 'isi_cv'].mean(skipna=True)
            cv2_roi = np.zeros(len(roi_list))
            for count_r, r in enumerate(roi_list):
                cv2_roi[count_r] = isi_events_cv2.loc[isi_events_cv2['roi'] == r, 'cv2'].mean(skipna=True)
            perc90_roi = np.zeros(len(roi_list))
            for count_r, r in enumerate(roi_list):
                perc90_roi[count_r] = np.nanpercentile(isi_events.loc[isi_events['roi'] == r, 'isi'], 90)
            np.save(os.path.join(save_path, dil + '_day_' + day + '_roinr'), len(roi_list))
            np.save(os.path.join(save_path, dil + '_day_' + day + '_fr'), fr_roi)
            np.save(os.path.join(save_path, dil + '_day_' + day + '_cv'), cv_roi)
            np.save(os.path.join(save_path, dil + '_day_' + day + '_cv2'), cv2_roi)
            np.save(os.path.join(save_path, dil + '_day_' + day + '_perc90'), perc90_roi)

            skewness_rois = list(df_extract_rawtrace_detrended.skew(axis=0, skipna=True).iloc[2:])
            np.save(os.path.join(save_path, dil + '_day_' + day + '_skew'), skewness_rois)


