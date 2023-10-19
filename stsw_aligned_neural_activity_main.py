import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import pickle
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

plot_data = True
save_plot = True
save_data = True
bins_phase = np.arange(0, 1.01, 0.05) # 5 deg
bins_time = np.arange(-0.125, 0.126, 0.0125) # 12.5 ms
paws = ['FR', 'HR', 'FL', 'HL']

align = 'sw'
temporal_dimension = 'time'
s = 1
roi = 70

for align, temporal_dimension in [['sw', 'phase'], ['stride', 'phase'], ['sw', 'time']]:
    for s in range(len(session_data)-1):
    # for s in [0,1,3,7]: ## For tide session
        # Load animal info for a session
        ses_info = session_data.iloc[s, :]
        print(ses_info)
        date = ses_info[3]
        path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
        path_loco = os.path.join(path_session_data, 'TM TRACKING FILES', ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1]+date.split('_')[-2]+date.split('_')[-3][2:] + '\\')
        session_type = path.split('\\')[-4].split(' ')[0]
        session_id = session_type + '_' + ses_info[2]
        
        # Import classes
        mscope = miniscope_session_class.miniscope_session(path)
        loco = locomotion_class.loco_class(path_loco)
    
        # Load session data
        animal = mscope.get_animal_id()
        session = loco.get_session_id()
        [_, _, _, df_extract_rawtrace_detrended, df_events_extract_rawtrace, _, _, _, trials,
          clusters_rois, colors_cluster, colors_session, _, _, frames_dFF] = mscope.load_processed_files()
        [_, _, frames_loco, _, bcam_time] = loco.get_tdms_frame_start(animal, session, frames_dFF)
        [trials_ses, trials_ses_name, _, _, _, _] = mscope.get_session_data(trials, session_type, animal, session)
        # Sort ROIs in the dataframe of neural activity by cluster
        df_spikes, cluster_transition_idx = nxb.sort_rois_clust(df_events_extract_rawtrace, clusters_rois)
        # Flatten the list of ROIs 'clusters_rois'
        rois_sorted = []
        for i in range(len(clusters_rois)):
            rois_sorted = np.hstack((rois_sorted, clusters_rois[i]))
    
        # Load behavioral data
        filelist = loco.get_track_files(animal, session)
        
        # Compute gait parameters
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
            param_trials.append(loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length'))
        # Transform paw position into phase        
        if temporal_dimension == 'phase':
            if align == 'stride':
                stride_norm = 'st-st'
            else:
                stride_norm = 'st-sw-st'
            final_tracks_trials_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials, stride_norm)
                
        # Get timestamps of st and sw onset and offset for each paw
        stsw = {}
        for p, paw in enumerate(paws):
            stsw[paw] = blna.get_stsw(st_strides_trials, sw_strides_trials, p)

        # In the behavioral dataset, find index of spikes occurring within each stride cycle
        spikesIdx_behavData = {}
        for paw in paws:
            on_ts = stsw[paw]['st onset ts']
            off_ts = stsw[paw]['st offset ts']
            spikesIdx_behavData_allrois = []
            if align == 'st':
                align_ts = stsw[paw]['st onset ts']
            else:
                align_ts = stsw[paw]['sw onset ts']            
            for n in range(2, df_spikes.shape[1]):
                spikesIdx_behavData_roi = blna.get_spikes_behav(df_spikes.iloc[:, [0,1,n]], on_ts, off_ts, align_ts, bins_time, bcam_time, temporal_dimension)
                spikesIdx_behavData_allrois.append(spikesIdx_behavData_roi)
            spikesIdx_behavData[paw] = spikesIdx_behavData_allrois
           
        # Find time or stride phase at which each spike occurred in the stride cycle
        spikes_timing = {}
        for p, paw in enumerate(paws):
            if temporal_dimension == 'phase':
                temporal_dataset = [final_tracks_trials_phase[tr_idx][0, p, :] for tr_idx, _ in enumerate(trials)]
                bins = bins_phase
            elif temporal_dimension == 'time':
                temporal_dataset = bcam_time
                bins = bins_time
            align_ts = stsw[paw]['sw onset ts'] 
            spikes_timing_allrois = []
            for n in range(len(spikesIdx_behavData[paw])):
                spikes_timing_roi = blna.get_spikes_timing(spikesIdx_behavData[paw][n], temporal_dataset, align_ts, temporal_dimension)
                spikes_timing_allrois.append(spikes_timing_roi)
            spikes_timing[paw] = spikes_timing_allrois
            
        # Save data
        file_path = os.path.join(path_session_data, 'Stride-locked neural activity', f'{animal}_{session_id}_{align}-locked_spikes_{temporal_dimension}.npy')
        if save_data:
            dataset = {'stsw idx': stsw, 'spikes timing': spikes_timing}
        with open(file_path, 'wb') as file:
            pickle.dump(dataset, file)
                    
        # Bin spikes in time or phase 
        spikes_count = {}
        for p, paw in enumerate(paws):    
            spikes_count_allrois = []
            for n in range(len(spikes_timing[paw])):
                spikes_count_roi = blna.bin_spikes(spikes_timing[paw][n], bins)
                spikes_count_allrois.append(spikes_count_roi)
            spikes_count[paw] = spikes_count_allrois

        # Compute firing rate and spike probability
        firing_rate_allpaws = {}
        for p, paw in enumerate(spikes_count.keys()):
            firing_rate_allrois = []
            for n in range(len(spikes_count[paw])):
                firing_rate_roi, _ = blna.neural_activity_bin(spikes_count[paw][n], temporal_dataset, bins)
                firing_rate_allrois.append(firing_rate_roi)
            firing_rate_allpaws[paw] = firing_rate_allrois
        
        # Plot raster, firing rate and P(CS) in time or phase for each ROI
        if plot_data:
            print(f'Plotting {align}-locked neural activity {temporal_dimension} {animal} {session_id}')
            for n in range(len(spikes_timing['FR'])): 
            # for n in [roi]:
                spikes_timing_roi_allpaws = {key: spikes_timing[key][n] for key in spikes_timing.keys()}
                firing_rate_roi_allpaws = {key: firing_rate_allpaws[key][n] for key in firing_rate_allpaws.keys()}
                blna.plot_behav_locked_activity_rois(spikes_timing_roi_allpaws, firing_rate_roi_allpaws, bins, trials_ses, colors_session, animal, session_id, rois_sorted[n], align, save_plot, temporal_dimension)