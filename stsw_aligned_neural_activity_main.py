import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import classes
os.chdir('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
import miniscope_session_class
import locomotion_class
import df_behav_class
import behav_locked_neural_activity_class
nxb = df_behav_class.df_behav_analysis('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
blna = behav_locked_neural_activity_class.behav_locked_neural_activity('C:\\Users\\User\\Documents\\LocalRepo\\miniscope_analysis')
    
path_session_data = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files'
session_data = pd.read_excel('C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\session_data_split_S1.xlsx')
save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'

save_plot = True
save_data = True
align = 'sw' 
bins_phase = np.arange(0, 1.01, 0.05) # 5 degrees # bins for sw-locked phase raster; change to np.arange(-0.5, 0.51, 0.05) for st-locked phase raster
bins_time = np.arange(-0.125, 0.126, 0.01) # 10 ms
stride_norm = 'st-st'
sort_data = True
param_name = 'step_length'

# REMOVE
p = 0
p2 = 2
paw ='FR'
paw2 = 'FL'
align ='sw'
t_dim='phase'

for s in range(len(session_data)-1):
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
    session_type = path.split('\\')[-4].split(' ')[0]
    session_id = session_type + '_' + ses_info[2]
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)

    # Load ession data
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [_, _, _, df_extract_rawtrace_detrended, df_events_extract_rawtrace, _, _, _, trials,
     clusters_rois, colors_cluster, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
    [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
    [trials_ses, trials_ses_name, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)

    # Load behavioral data
    filelist = loco.get_track_files(animal, session)
    st_strides_trials = []
    sw_strides_trials = []
    final_tracks_trials = []
    param_trials = []
    for count_trial, f in enumerate(filelist):
        [final_tracks, _, _, _, _, bodycenter] = loco.read_h5(f, 0.9, int(frames_loco[count_trial]))
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        final_tracks_trials.append(final_tracks)
        st_strides_trials.append(st_strides_mat)
        sw_strides_trials.append(sw_pts_mat)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_name))
    final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials,
                                                        sw_strides_trials, stride_norm)
    [sl_idx_all, sl_time_all_array, sl_sym_all_array] = loco.param_continuous_sym(param_trials, st_strides_trials, trials, paw, paw2, sym = 1, remove_nan = 1)  # SL symmetry for each stride
    body_position, body_speed, body_acc = nxb.body_kinematic(final_tracks_trials, trials, win_len = 81, polyorder = 3)
    paw_position, paw_speed, paw_acc = nxb.paw_kinematic(final_tracks_trials, p, 0, win_len = 81, polyorder = 3)

    # Sort ROIs in the dataframe of neural activity by cluster
    df_spikes, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois)
    # flatten the list of ROIs 'clusters_rois'
    rois_sorted = []
    for i in range(len(clusters_rois)):
        rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
        
    # Sub-divide experimental blocks
    if session_type == 'split':
        split_blocks = nxb.split_expblocks(trials_ses)
    else:
        split_blocks = trials_ses
            
    # Get timestamps of st and sw onset and offset
    st_on_ts, st_on_idx, st_off_ts, st_off_idx, sw_on_ts, sw_on_idx = blna.get_stsw(st_strides_trials, sw_strides_trials, trials, p)
    
    # Find indexes of spikes for each stride cycle and get the indexes of strides in which a spike occurs
    spikes_idx_behav_allrois = []
    strides_spike_idx_allrois = []
    for n in range(2, df_spikes.shape[1]):
        if align == 'st':
            spikes_idx_behav, strides_spike_idx = blna.get_spikes_behav(df_spikes.iloc[:, [0,1,n]], st_on_ts, st_off_ts, st_on_ts, bins_time, bcam_time, trials, t_dim)
        else:
            spikes_idx_behav, strides_spike_idx = blna.get_spikes_behav(df_spikes.iloc[:, [0,1,n]], st_on_ts, st_off_ts, sw_on_ts, bins_time, bcam_time, trials, t_dim)
        spikes_idx_behav_allrois.append(spikes_idx_behav)
        strides_spike_idx_allrois.append(strides_spike_idx)
        
    # Find indexes and timestamps of st and sw just for strides in which a spike occurs            
    st_on_ts_spike, st_on_idx_spike, st_off_ts_spike, st_off_idx_spike, sw_on_ts_spike, sw_on_idx_spike = blna.get_stsw_spike(st_on_ts, st_on_idx, st_off_ts, st_off_idx, sw_on_ts, sw_on_idx, strides_spike_idx_allrois, trials)
   
    # Find time or paw phase at which each spike occurred in the stride cycle
    if t_dim == 'phase':
        temporal_dataset = [final_tracks_trials_phase[tr_idx][p, 0, :] for tr_idx, _ in enumerate(trials)]
        bins = bins_phase
    elif t_dim == 'time':
        temporal_dataset = bcam_time
        bins = bins_time
    if align == 'st':
        align_ts = st_on_ts_spike
    elif align == 'sw' or align == 'stride':
        align_ts = sw_on_ts_spike 
    spikes_timing_allrois = []
    for n in range(len(spikes_idx_behav_allrois)):
        spikes_timing = blna.get_spikes_timing(spikes_idx_behav_allrois[n], temporal_dataset, align_ts[n], trials, t_dim)
        spikes_timing_allrois.append(spikes_timing)
    
    # Save spikes timing data
    file_path = os.path.join(path_session_data, 'Stride-locked neural activity', f'{animal}_{session_id}_{paw}_{align}-locked_spikes_{t_dim}.npy')
    if save_data:
        np.save(file_path, spikes_timing_allrois)
                
    # Bin spikes in time or phase # DO THIS ON A MOVING WINDOW
    spikes_count_allrois = []
    for n in range(len(spikes_timing_allrois)):
        spikes_count = blna.bin_spikes(spikes_timing_allrois[n], bins, trials)
        spikes_count_allrois.append(spikes_count)
    
    # Find st/sw timestamps/phase relative to the choosen alignment point (sw/st)
    stsw_rel_t = []
    # if align == 'stride' and t_dim == 'phase': # find sw phase in the stride cycle for strides with CS
    for n in range(len(strides_spike_idx_allrois)):
        stsw_rel_t.append([final_tracks_trials_phase[tr_idx][p, 0, sw_on_idx_spike[n][tr_idx].astype(int)] for tr_idx, _ in enumerate(trials)])
    # ### ISSUES HERE!
    # elif align == 'sw' and t_dim == 'time': # find st stimestamps relative to sw for strides with CS
    #     for n in range(len(st_on_ts_spike)):
    #         on_rel_t = [(st_on_ts_spike[n][tr_idx] - sw_on_ts_spike[n][tr_idx]) for tr_idx, _ in enumerate(trials)]
    #         off_rel_t = [(st_off_ts_spike[n][tr_idx] - sw_on_ts_spike[n][tr_idx]) for tr_idx, _ in enumerate(trials)]
    #         stsw_rel_t.append([(st_on_ts_spike[n][tr_idx] - sw_on_ts_spike[n][tr_idx]) for tr_idx, _ in enumerate(trials)])
    # elif align == 'st' and t_dim == 'time': # find sw stimestamps relative to st for strides with CS
    #     for n in range(len(sw_on_ts_spike)):                
    #         stsw_rel_t.append([(sw_on_ts_spike[n][tr_idx] - st_on_ts_spike[n][tr_idx]) for tr_idx, _ in enumerate(trials)])

    # Plot raster, normalized spike count and P(CS) in time or phase for each ROI and cluster
    spikes_count_tr_allrois = []
    cs_prob_tr_allrois = []
    for n in range(len(spikes_timing_allrois)): 
        spikes_count_tr, cs_prob_tr = blna.plot_behav_locked_activity_rois(spikes_timing_allrois[n], spikes_count_allrois[n], stsw_rel_t[n], bins, trials, trials_ses, colors_session, animal, session_id, paw, rois_sorted[n], align, save_plot, t_dim)
        spikes_count_tr_allrois.append(spikes_count_tr)
        cs_prob_tr_allrois.append(cs_prob_tr)
    spikes_count_clust = blna.plot_behav_locked_activity_clust(spikes_count_tr_allrois, bins, trials, trials_ses, colors_session, animal, session_id, paw, cluster_transition_idx, colors_cluster, align, save_plot, t_dim)        
    blna.plot__behav_locked_activity_popul(cs_prob_tr_allrois, split_blocks)
    
    # Plot raster, normalized spike count and P(CS) in time or phase for each ROI and cluster for sorted strides
    behavior = body_acc ############# CHANGE HERE!
    sorted_by = 'body acceleration'  ############# CHANGE HERE!
    if sort_data:
        sorted_behav_stride_allrois = []
        sorted_spikes_timing_allrois = []    
        for n in range(5): #### len(spikes_timing_allrois)
            # if sorted_by == 'steph-length symmetry':
            # else:
            #     behav_stride = blna.get_mean_behav_stride(behavior, st_on_idx_spike[n], st_off_idx_spike[n], trials) # Average of a behavioral variable (e.b.: paw acceleration) for each stride
            behav_stride = blna.get_mean_behav_stride(behavior, st_on_idx_spike[n], st_off_idx_spike[n], trials) # Average of a behavioral variable (e.b.: paw acceleration) for each stride
            spikes_t = blna.sublist_arrays2tuples(spikes_timing_allrois[n]) # Transform arrays with spike timing for each stride in tuples to preserve structure when concatenating for sorting
            n_strides_tr = [len(spikes_t[tr_idx]) for tr_idx, _ in enumerate(trials)] 
            # Sort strides within each experimental block
            sorted_spikes_timing, sorted_behav_stride = blna.sort_variable(spikes_timing, behav_stride)  
            sorted_behav_stride_allrois.append(sorted_behav_stride)
            sorted_spikes_timing_allrois.append(sorted_spikes_timing) # List of sorted strides for each experimental block for all ROIs
            blna.plot_behav_locked_activity_sorted_rois(sorted_spikes_timing_allrois[n], bins, trials, sorted_by, animal, session_id, paw, rois_sorted[n], align, save_plot, t_dim)
