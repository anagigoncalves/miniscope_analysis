# %% Inputs
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\miniscope_analysis\\')
import miniscope_session_class
import locomotion_class
path_session_data = 'J:\\Miniscope processed files'
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\STA paw spatial diff\\split contra fast S1\\'
sta_type = 'paw_diff'
window = np.arange(-330, 330 + 1)  # Samples
iter_n = 100 # Number of iterations of CS timestamps random shuffling

# path inputs
path = 'J:\\Miniscope processed files\\TM RAW FILES\\split contra fast\\MC13420\\2022_05_31\\'
path_loco = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split contra fast S1 310522\\'
protocol = 'split contra fast'
# path = 'J:\\Miniscope processed files\\TM RAW FILES\\split contra fast\\MC13419\\2022_05_31\\'
# path_loco = 'J:\\Miniscope processed files\\TM TRACKING FILES\\split contra fast S1 310522\\'
# protocol = 'split contra fast'
session_type = path.split('\\')[-4].split(' ')[0]
mscope = miniscope_session_class.miniscope_session(path)
loco = locomotion_class.loco_class(path_loco)

# Session data and inputs
animal = mscope.get_animal_id()
session = loco.get_session_id()
df_extract_rawtrace_detrended = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_extract_rawtrace_detrended.csv'))
df_events_extract_rawtrace = pd.read_csv(
    os.path.join(mscope.path, 'processed files', 'df_events_extract_rawtrace.csv'))
coord_ext = np.load(os.path.join(mscope.path, 'processed files', 'coord_ext.npy'), allow_pickle=True)
trials = np.load(os.path.join(mscope.path, 'processed files', 'trials.npy'))
ref_image = np.load(os.path.join(mscope.path, 'processed files', 'ref_image.npy'), allow_pickle=True)
frames_dFF = np.load(os.path.join(mscope.path, 'processed files', 'black_frames.npy'), allow_pickle=True)
colors_session = np.load(os.path.join(mscope.path, 'processed files', 'colors_session.npy'), allow_pickle=True)
colors_session = colors_session[()]
[trigger_nr, strobe_nr, frames_loco, trial_start, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
[trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(trials, session_type, animal, session)
centroid_ext = mscope.get_roi_centroids(coord_ext)

# Load behavioral data and get acceleration
filelist = loco.get_track_files(animal, session)
final_tracks_trials = []
bodyacc = []
bodycenter = []
bodyspeed = []
FR_X_excursion = []
FL_X_excursion = []
HR_X_excursion = []
HL_X_excursion = []
st_strides_trials = []
sw_strides_trials = []
for count_trial, f in enumerate(filelist):
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter_DLC] = loco.read_h5(f, 0.9, int(
        frames_loco[count_trial]))
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    bodycenter_trial = sp.medfilt(loco.compute_bodycenter(final_tracks, 'X'), 25) #filter for tracking errors
    bodyspeed_trial = loco.compute_bodyspeed(bodycenter_trial)
    bodyacc_trial = loco.compute_bodyacc(bodycenter_trial)
    final_tracks_trials.append(final_tracks)
    st_strides_trials.append(st_strides_mat)
    sw_strides_trials.append(sw_pts_mat)
    bodyacc.append(bodyacc_trial)
    bodycenter.append(bodycenter_trial)
    bodyspeed.append(bodyspeed_trial)
    FR_X_excursion.append((final_tracks[0, 0, :]*loco.pixel_to_mm)-(np.nanmean(final_tracks[0, :4, :], axis=0)*loco.pixel_to_mm))
    HR_X_excursion.append((final_tracks[0, 1, :]*loco.pixel_to_mm)-(np.nanmean(final_tracks[0, :4, :], axis=0)*loco.pixel_to_mm))
    FL_X_excursion.append((final_tracks[0, 2, :]*loco.pixel_to_mm)-(np.nanmean(final_tracks[0, :4, :], axis=0)*loco.pixel_to_mm))
    HL_X_excursion.append((final_tracks[0, 3, :]*loco.pixel_to_mm)-(np.nanmean(final_tracks[0, :4, :], axis=0)*loco.pixel_to_mm))
final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, 'st-sw-st')

if sta_type == 'bodyvars':
    # Dictionary of all the independent variables on which computing the STA
    ind_vars = {'Body position': bodycenter, 'Body speed': bodyspeed, 'Body acceleration': bodyacc}
    keys=list(ind_vars.keys())
if sta_type == 'paw_diff':
    # Get phase and displacement difference
    displ_diff_diag = mscope.paw_diff(final_tracks_trials, 0, 3)
    displ_diff_homo = mscope.paw_diff(final_tracks_trials, 0, 1)
    displ_diff_front = mscope.paw_diff(final_tracks_trials, 0, 2)
    ind_vars = {'FR-HL': displ_diff_diag, 'FR-HR': displ_diff_homo, 'FR-FL': displ_diff_front}
    keys=list(ind_vars.keys())
if sta_type == 'paws':
    ind_vars = {'FR': FR_X_excursion, 'FL': FL_X_excursion, 'HR': HR_X_excursion, 'HL': HL_X_excursion}
    keys=list(ind_vars.keys())
if sta_type == 'phase_diff':
    paw_diff_fr_hl = loco.phase_diff(final_tracks_phase, 'FR', 'HL', 'X')
    paw_diff_fl_hl = loco.phase_diff(final_tracks_phase, 'FL', 'HL', 'X')
    paw_diff_hr_hl = loco.phase_diff(final_tracks_phase, 'HR', 'HL', 'X')
    paw_diff_fl_fr = loco.phase_diff(final_tracks_phase, 'FL', 'FR', 'X')
    paw_diff_hr_fr = loco.phase_diff(final_tracks_phase, 'HR', 'FR', 'X')
    paw_diff_hl_fr = loco.phase_diff(final_tracks_phase, 'HL', 'FR', 'X')
    ind_vars = {'FL-FR-phase': paw_diff_fl_fr, 'HR-FR-phase': paw_diff_hr_fr, 'HL-FR-phase': paw_diff_hl_fr,
                'FR-HL-phase': paw_diff_fr_hl, 'FL-HL-phase': paw_diff_fl_hl, 'HR-HL-phase': paw_diff_hr_hl}
    keys=list(ind_vars.keys())

# Loop through independent variables to compute and plot STAs of each one
for var in range(len(ind_vars)):
    var_name = keys[var]
    variable = ind_vars[var_name]
    # Compute spike-triggered average (STA) of kinematic variables
    sta_allrois, signal_chunks_allrois = mscope.sta(df_events_extract_rawtrace, variable, bcam_time, window, trials)
    # Standardize observed STA on STA computed with shuffled data
    # Shuffle CS timestamps
    shuffled_spikes_ts = mscope.shuffle_spikes_ts(df_events_extract_rawtrace, iter_n)
    # Compute STA for shuffled data
    sta_shuffled_ts = np.array(mscope.sta_shuffled(shuffled_spikes_ts, variable, bcam_time, window, trials))
    # Standardize STA
    mean_chance = np.nanmean(sta_shuffled_ts, axis=2)
    sd_chance = np.nanstd(sta_shuffled_ts, axis=2)
    sta_zs = np.zeros((len(sta_allrois), len(trials), len(window)))
    sta = np.zeros((len(sta_allrois), len(trials), len(window)))
    sta_shuffled = np.zeros((len(sta_allrois), len(trials), len(window)))
    for n in range(len(sta_allrois)):
        for tr in range(len(trials)):
            sta_zs[n, tr] = (sta_allrois[n][tr] - mean_chance[n][tr]) / sd_chance[n][tr]
            sta[n, tr] = sta_allrois[n][tr]
            sta_shuffled[n, tr] = mean_chance[n][tr]
    if not os.path.exists(os.path.join(save_path, animal + ' ' + protocol)):
        os.mkdir(os.path.join(save_path, animal + ' ' + protocol))
    np.save(os.path.join(save_path, animal + ' ' + protocol,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_zscored.npy'), sta_zs)
    np.save(os.path.join(save_path, animal + ' ' + protocol,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '_shuffled.npy'), sta_shuffled)
    np.save(os.path.join(save_path, animal + ' ' + protocol,
                         'sta_bodyvars_' + var_name.replace(' ', '_') + '.npy'), sta)
    np.save(os.path.join(save_path, animal + ' ' + protocol,
        'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses.npy'), trials_ses)
    np.save(os.path.join(save_path, animal + ' ' + protocol,
        'sta_bodyvars_' + var_name.replace(' ', '_') + '_trials_ses_name.npy'), trials_ses_name)