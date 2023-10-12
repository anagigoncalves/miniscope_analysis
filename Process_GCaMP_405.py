#THIS SCRIPT TAKES ROIS FROM CORRESPONDING 480 SESSION AND PLOTS ACTIVITY ON THE 405 SESSION
# %% Inputs
import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt

# path inputs
# path = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast 405\\MC13420\\2022_05_31\\'
# path_loco = 'D:\\Miniscopes\\TM TRACKING FILES\\split contra fast 405nm S2 310522\\'
# path_480 = 'D:\\Miniscopes\\TM RAW FILES\\split contra fast 480\\MC13420\\2022_05_31\\'
path = 'D:\\Miniscopes\\TM RAW FILES\\tied baseline 405\\MC13420\\2022_05_30\\'
path_loco = 'D:\\Miniscopes\\TM TRACKING FILES\\tied baseline 405nm S2 300522\\'
path_480 = 'E:\\Miniscopes\\TM RAW FILES\\tied baseline 480\\MC13420\\2022_05_30\\'
session_type = path.split('\\')[-4].split(' ')[0]
version_mscope = 'v4'
plot_data = 1
load_data = 0
print_plots = 1
save_data = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
fsize = 24

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
mscope = miniscope_session_class.miniscope_session(path)
import locomotion_class
loco = locomotion_class.loco_class(path_loco)

# create plots folders
path_images = os.path.join(path, 'images')
path_events = os.path.join(path, 'images', 'events')
if not os.path.exists(path_images):
    os.mkdir(path_images)
if not os.path.exists(path_events):
    os.mkdir(path_events)



# Trial structure, reference image and triggers
animal = mscope.get_animal_id()
session = loco.get_session_id()
trials = mscope.get_trial_id()
strobe_nr_txt = loco.bcam_strobe_number()
trial_start_blip_nr = loco.trial_start_blips()
ops_s2p = mscope.get_s2p_parameters()
print(ops_s2p)
colors_session = mscope.colors_session(animal, session_type, trials, 1)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
traces_type = 'raw'

if load_data == 0:
    ref_image = mscope.get_ref_image()
    frames_dFF = mscope.get_black_frames()  # black frames removed before ROI segmentation
    frame_time = mscope.get_miniscope_frame_time(trials, frames_dFF, version_mscope)  # get frame time for each trial
    trial_length_cumsum = mscope.cumulative_trial_length(frame_time)
    [trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials, trial_start, strobe_nr, bcam_time, colors_session, frame_time, frames_dFF, frames_loco,
     del_trials_index] = mscope.correct_for_deleted_trials(trials, trial_start, strobe_nr, bcam_time, colors_session,
                                                           frame_time, frames_dFF, frames_loco)
    # Load ROIs from corresponding 480 session
    coord_ext_480 = np.load(os.path.join(path_480, 'processed files', 'coord_ext.npy'), allow_pickle=True)
    df_extract_rawtrace_detrended_480 = pd.read_csv(
        os.path.join(path_480, 'processed files', 'df_extract_rawtrace_detrended.csv'))

    #Compute traces from ROIs
    roi_list = list(df_extract_rawtrace_detrended_480.columns[2:])
    ext_trace_trials = []
    for t in trials:
        idx_trial = np.where(trials == t)[0][0]
        tiff_stack = tiff.imread(
            os.path.join(mscope.path, 'Registered video') + mscope.delim + 'T' + str(t) + '_reg.tif')  # read tiffs
        ext_trace = np.zeros((np.shape(tiff_stack)[0], len(roi_list)))
        for c in range(len(coord_ext_480)):
            ext_trace_tiffmean = np.zeros((np.shape(tiff_stack)[0]))
            for f in range(np.shape(tiff_stack)[0]):
                ext_trace_tiffmean[f] = np.nansum(tiff_stack[f, np.int64(
                    np.round(coord_ext_480[c][:, 1] * mscope.pixel_to_um)), np.int64(
                    np.round(coord_ext_480[c][:, 0] * mscope.pixel_to_um))]) / np.shape(coord_ext_480[c])[0]
            ext_trace[:, c] = ext_trace_tiffmean
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
    df_traces = pd.concat([df_ext1, df_ext2], axis=1)

    # Good periods after motion correction
    th = 0.01 # change with the notes from EXCEL
    [x_offset, y_offset, corrXY] = mscope.get_reg_data()  # registration bad moments
    if len(del_trials_index)>0:
        trial_beg = np.insert(trial_length_cumsum[:-1], 0, 0)
        trial_end = trial_length_cumsum[1:]
        for t in del_trials_index:
            x_offset = np.delete(x_offset, np.arange(trial_beg[t], trial_end[t]))
            y_offset = np.delete(y_offset, np.arange(trial_beg[t], trial_end[t]))
            corrXY = np.delete(corrXY, np.arange(trial_beg[t], trial_end[t]))
            np.save(os.path.join(mscope.path, 'processed files', 'x_offsets.npy'), x_offset)
            np.save(os.path.join(mscope.path, 'processed files', 'y_offsets.npy'), y_offset)
            np.save(os.path.join(mscope.path, 'processed files', 'corrXY_frames.npy'), corrXY)
    [idx_to_nan, df_extract] = mscope.corr_FOV_movement(th, df_traces, corrXY)
    [width_roi_ext, height_roi_ext, aspect_ratio_ext] = mscope.get_roi_stats(coord_ext_480)
    [coord_ext, df_extract] = mscope.rois_larger_motion(df_extract, coord_ext_480, idx_to_nan, x_offset, y_offset, width_roi_ext, height_roi_ext, 1)
    corr_rois_motion = mscope.correlation_signal_motion(df_extract, x_offset, y_offset, trials_baseline[-1], idx_to_nan, traces_type, plot_data, print_plots)

    # ROI spatial stats
    [width_roi_rois_nomotion, height_roi_rois_nomotion, aspect_ratio_rois_nomotion] = mscope.get_roi_stats(coord_ext)
    # ROI curation
    [coord_ext_curated, df_extract_curated] = mscope.roi_curation(ref_image, df_extract, coord_ext, aspect_ratio_rois_nomotion, trials_baseline[-1])

    # Get raw trace from EXTRACT ROIs
    roi_list = list(df_extract_curated.columns[2:])
    df_extract_rawtrace = mscope.compute_extract_rawtrace(coord_ext_curated, df_extract_curated, roi_list, trials, frame_time)
    # Find calcium events - label as synchronous or asynchronous
    df_events_extract_rawtrace = mscope.get_events(df_extract_rawtrace, 1, 'df_events_extract_rawtrace')  # 1 for detrending"
    roi_plot = np.int64(np.random.choice(roi_list)[3:])
    trial_plot = np.random.choice(trials)
    mscope.plot_events_roi_trial(trial_plot, roi_plot, frame_time, df_extract_rawtrace, traces_type, df_events_extract_rawtrace, trials, plot_data, print_plots)

    # Detrend calcium trace
    df_extract_rawtrace_detrended = mscope.compute_detrended_traces(df_extract_rawtrace, 'df_extract_rawtrace_detrended')

    # Save data
    if not os.path.exists(mscope.path + 'processed files'):
        os.mkdir(mscope.path + 'processed files')
    df_extract_rawtrace.to_csv(os.path.join(mscope.path, 'processed files', 'df_extract_raw.csv'), sep=',',
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

    corr_rois_motion = mscope.correlation_signal_motion(df_extract_rawtrace, x_offset, y_offset, trials_baseline[-1],
                                                        idx_to_nan, 'raw', plot_data, print_plots)

