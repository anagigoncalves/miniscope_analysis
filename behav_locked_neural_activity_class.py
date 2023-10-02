# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:42:15 2023

@author: User
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import SlopeThreshold as ST


class behav_locked_neural_activity:  
    
    
    def __init__(self, path):
        self.save_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Figures\\'
        self.sr_cam = 330 #sampling rate of behavior camera for treadmill
        self.sr = 30
        self.my_dpi = 128 #resolution for plotting
        self.font_size = 15

    
    def get_stsw(self, st_strides_trials, sw_strides_trials, p):
        ''' Get stance and swing onset/offset indexes and timestamps'''
        trials = np.arange(1, len(st_strides_trials)+1)
        st_on_ts = [st_strides_trials[tr_idx][p][:, 0, 0]/1000 for tr_idx, _ in enumerate(trials)]
        st_on_idx = [st_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        sw_on_ts = [sw_strides_trials[tr_idx][p][:, 0, 0]/1000 for tr_idx, _ in enumerate(trials)]
        sw_on_idx = [sw_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        st_off_ts = [st_strides_trials[tr_idx][p][:, 1, 0]/1000 for tr_idx, _ in enumerate(trials)]
        st_off_idx = [st_strides_trials[tr_idx][p][:, 1, 4] for tr_idx, _ in enumerate(trials)]
        stsw_dict = {'st onset ts' : st_on_ts, 'st onset idx' : st_on_idx, 'st offset ts' : st_off_ts, 
                       'st offset idx' : st_off_idx, 'sw onset ts' : sw_on_ts, 'sw onset idx' : sw_on_idx}
        return stsw_dict
    
    
    def get_spikes_behav(self, df_spikes, on_ts, off_ts, align_ts, window, bcam_time, temporal_dimension):
        ''' In the behavioral dataset, find matching indices and timestamps of spikes occurring within a behavioral event 
        (e.g.: between the onset and offset of a stride cycle)'''
        trials = np.unique(df_spikes['trial'])
        spikeIdx_behavData = []
        for tr_idx, tr in enumerate(trials):
            spikeIdx_behavData_tr = []
            spikes_tr = df_spikes[df_spikes['trial'] == tr]
            spikes_tr = spikes_tr.reset_index(drop=True)
            for i in range(len(on_ts[tr_idx])): # Loop through each behavioral event (e.g.: each stride)
                if temporal_dimension == 'phase':
                    start = on_ts[tr_idx][i]
                    end = off_ts[tr_idx][i]
                    signal_chunck = spikes_tr[(spikes_tr['time'] >= start) & (spikes_tr['time'] < end)].iloc[:, 2]
                elif temporal_dimension == 'time':
                    start = align_ts[tr_idx][i] + window[0]
                    end = align_ts[tr_idx][i] + window[-1]
                    signal_chunck = spikes_tr[(spikes_tr['time'] > start) & (spikes_tr['time'] < end)].iloc[:, 2]
                spike_idx = signal_chunck.index[signal_chunck == 1].tolist()
                spike_ts = np.array(spikes_tr['time'].iloc[spike_idx])
                if len(spike_ts) > 0:
                     spikeIdx_behavData_tr.append(np.array([np.where(bcam_time[tr_idx] == bcam_time[tr_idx][np.abs(bcam_time[tr_idx] - t).argmin()])[0][0] for t in spike_ts]))
                else:
                    spikeIdx_behavData_tr.append([np.nan]) 
            spikeIdx_behavData.append(spikeIdx_behavData_tr)
        return spikeIdx_behavData


    def get_spikes_timing(self, spikesIdx_behavData, temporal_dataset, align_ts, temporal_dimension): 
        ''' Find paw phase during each spike
        Inputs:
            temporal_dataset: bcam_time or final_tracks_trial_phase'''
        trials = np.arange(1, len(spikesIdx_behavData)+1)
        spikes_timing_tr = []
        for tr_idx, tr in enumerate(trials):
            spikes_timing_all = []
            for i in range(len(spikesIdx_behavData[tr_idx])): 
                if np.isnan(spikesIdx_behavData[tr_idx][i]).any() == False:
                    spikes_timing = temporal_dataset[tr_idx][spikesIdx_behavData[tr_idx][i]]
                    if temporal_dimension == 'time':
                        spikes_timing = spikes_timing - align_ts[tr_idx][i]
                else:
                    spikes_timing = np.array([10000.1]) # Place holder for NaN = 10000.1
                spikes_timing_all.append(spikes_timing)
            spikes_timing_tr.append(spikes_timing_all)
        return spikes_timing_tr


    def bin_spikes(self, dataset, bins): 
        ''' Compute spike count for each bin'''
        trials = np.arange(1, len(dataset)+1)
        spikes_count_tr = []
        bin_count = np.zeros((len(bins)-1))
        for tr_idx, tr in enumerate(trials):
            spikes_count = []
            for value in dataset[tr_idx]: 
                for idx, _ in enumerate(value):
                    if value[idx] >= bins[0] and value[idx] <= bins[-1] and np.isnan(value[idx]) == False:
                        bin_idx = np.digitize(value[idx], bins, right=True) - 1
                        bin_count[bin_idx] +=1
                spikes_count.append(bin_count)
                bin_count = np.zeros((len(bins)-1))
            spikes_count_tr.append(spikes_count) 
        return spikes_count_tr


    def neural_activity_bin(self, spikes_mat, phase, bins):
        ''' Compute firing rate and spike probability for each bin'''
        trials = np.arange(0, len(spikes_mat))
        firing_rate = np.zeros((len(trials), len(bins)-1))
        spike_prob = np.zeros((len(trials), len(bins)-1))
        for tr in range(len(trials)):       
             # Sum spikes in each bin
            spikes_count = np.sum(np.vstack(spikes_mat[tr]), axis = 0)
            # Compute time spent in each bin
            frames_bin, _ = np.histogram(phase[tr][~np.isnan(phase[tr])], bins=len(bins)-1)
            time_bin = frames_bin*(1/self.sr_cam)
            # Compute firing rate
            firing_rate[tr] = spikes_count/time_bin
            # Compute spike probability (normalized by time)
            n_strides = len(spikes_mat[tr])
            spike_prob[tr] = firing_rate[tr]/n_strides
        return firing_rate, spike_prob
    
        
    def plot_behav_locked_activity_rois(self, spikes_timing, firing_rate, spike_prob, bins, trials_ses, colors_session, animal, session_id, roi, align, save_plot, temporal_dimension):
        ''' Plot st/sw-aligned neural activity: rasters, firing rate heatmap, P(CS)'''

        trials = np.arange(0, trials_ses[-1, -1])
        
        # Create tick labels for x-axis
        if temporal_dimension == 'phase':
            if align == 'st':
                x_tick_labels = ['sw', 'st', 'sw']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'stride':
                x_tick_labels = ['st', 'st']
                x_ticks = [0, len(bins)-1]
        elif temporal_dimension == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]        
                
        # Rasterplot divided by block and sorted by time for each ROI
        fig, axs = plt.subplots(3, 4, figsize = (30, 15))
        for p, paw in enumerate(spikes_timing.keys()):
            axs[0, p].set_title(paw, fontsize = self.font_size)
            cumul_strides = 0
            for tr in range(len(trials)):
                if tr in trials_ses[0:2,1]:
                    axs[0, p].axhline(y=cumul_strides, color='crimson')
                for _, timing in enumerate(spikes_timing[paw][tr]):
                    axs[0, p].scatter(timing, np.ones(len(timing))*cumul_strides, c = 'dimgrey', marker = '.', s = 15)
                    cumul_strides += 1
            axs[0, p].set_ylabel('Strides', fontsize = self.font_size)
            axs[0, p].set(xticklabels=[])
            axs[0, p].tick_params(bottom=False)
            axs[0, p].spines['top'].set_visible(False)
            axs[0, p].spines['right'].set_visible(False)
            axs[0, p].set_xlim(bins[0], bins[-1])
            if temporal_dimension == 'phase':
                if align == 'sw':
                    axs[0, p].axvline(x = (bins[-1]+0.01)/2, color='crimson', linestyle='--')
                elif align == 'st':
                    axs[0, p].axvline(x = 0, color='crimson', linestyle='--')     
            elif temporal_dimension == 'time':
                axs[0, p].axvline(x = 0, color='crimson', linestyle='--')
        
            # HM count and P(CS) by trial for each ROI
            for tr in range(len(trials)):       
                if np.any(trials_ses-1 == tr): # Maybe better block mean
                    axs[2, p].plot(spike_prob[paw][tr], c = colors_session[tr+1], linewidth = 2.5)
            sns.heatmap(np.flipud(firing_rate[paw]), cmap = 'viridis', cbar = False, vmin = 0, vmax = 6, ax = axs[1, p])
            for i in range(len(trials_ses)-1):
                axs[1, p].axhline(y=trials_ses[-1, 1]-trials_ses[i, 1], color='white')
            axs[1, p].set_ylabel('Trials', fontsize = self.font_size)
            axs[2, p].set_ylabel('P(CS)', fontsize = self.font_size)
            axs[1, p].set(yticklabels=[])
            axs[1, p].tick_params(left=False)
            axs[1, p].set(xticklabels=[])
            axs[1, p].tick_params(bottom=False)
            axs[2, p].spines['top'].set_visible(False)
            axs[2, p].spines['right'].set_visible(False)
            axs[2, p].set_ylim(0, 0.05)
            axs[2, p].set_xlim(0, len(bins)-1)
            axs[2, p].set_xticks(x_ticks)
            axs[2, p].set_xticklabels(x_tick_labels, fontsize = self.font_size)
            axs[2, p].tick_params(axis='y', labelsize=self.font_size)
            if not align == 'stride':
                axs[1, p].axvline(x=(len(bins)-1)/2, color='white', linestyle='--')
                axs[2, p].axvline(x=(len(bins)-1)/2, color='k', linestyle='--')
        fig.suptitle(f'{align}-locked neuronal activity {animal} {roi} {session_id}', fontsize = self.font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
                os.mkdir(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
            plt.savefig(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} {roi} {animal} {session_id}.png'), dpi=self.my_dpi)
        plt.close()


    def get_mean_behav_stride(self, behavior, on_idx, off_idx, trials):
        ''' Get mean value of a variable (e.g.: acceleration) during each stride '''
        mean_behav_stride_tr = []
        for tr_idx, _ in enumerate(trials):
            mean_behav_stride = []
            for i in range(len(on_idx[tr_idx])):
                start = on_idx[tr_idx][i].astype(int)
                end = off_idx[tr_idx][i].astype(int)
                mean_behav_stride.append(np.mean(behavior[tr_idx][start:end]))
            mean_behav_stride_tr.append(np.array(mean_behav_stride))
        return mean_behav_stride_tr
    
    
    def sublist_arrays2tuples(self, main_list):
        ''' Transform sublists of arrays into sublists of tuples '''
        for sublist_idx, _ in enumerate(main_list):
            for idx in range(len(main_list[sublist_idx])):
                main_list[sublist_idx][idx] = tuple(main_list[sublist_idx][idx])
        return main_list


    def sort_variable(self, variable1, variable2):
        ''' Sort variable1 by the values of variable2 '''
        sort = np.argsort(variable2)
        sorted_variable1 = [variable1[i] for i in sort]
        sorted_variable2 = np.array(variable2)[sort]
        return sorted_variable1, sorted_variable2
    
    
    def plot_behav_locked_activity_sorted_rois(self, spikes_timing, sorted_spikes_timing, bins, trials, sorted_by, animal, session_id, paw, roi, align, save_plot, temporal_dimension):
        ''' Plot sorted st/sw-aligned rasters'''      
        # Rasterplot divided by block and sorted by time for each ROI
        fig, axs = plt.subplots(2,1, sharex = True, sharey = True, figsize=(7,10))       
        for stride, timing in enumerate(sorted_spikes_timing):
            axs[1].scatter(timing, np.ones(len(timing))*stride, c = 'dimgray', marker = '.', s = 15)
        cumul_strides = 0
        for tr_idx, _ in enumerate(trials):
            for _, timing in enumerate(spikes_timing[tr_idx]):
                axs[0].scatter(timing, np.ones(len(timing))*cumul_strides, c = 'dimgray', marker = '.', s = 15)
                cumul_strides += 1
        plt.ylabel('Strides', fontsize = self.font_size)
        plt.tick_params(bottom=False)
        plt.xlim(bins[0], bins[-1])
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(bins[0], bins[-1])
        if temporal_dimension == 'phase':
            if align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [bins[0], 0.5, bins[-1]]
                plt.axvline(x = 0.5, color='crimson', linestyle='--')
            elif align == 'stride':
                x_tick_labels = ['st', 'st']
                x_ticks = [bins[0], bins[-1]]
        elif temporal_dimension == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [bins[0], 0, bins[-1]]
                plt.axvline(x = 0, color='crimson', linestyle='--')  
        ax.set_xticks(x_ticks, fontsize = self.font_size)
        ax.set_xticklabels(x_tick_labels, fontsize = self.font_size)
        plt.suptitle('Sorted by ' + sorted_by + ' ' + paw + ' ' + roi, fontsize = self.font_size)
        axs[0].set_title('Unsorted', fontsize = self.font_size)
        axs[1].set_title('Sorted', fontsize = self.font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id)):
                os.mkdir(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id))
            plt.savefig(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id + '\\', paw + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted by ' + sorted_by +  ' ' + animal + ' ' + roi + ' ' + session_id + '.png'), dpi=self.my_dpi)
        plt.close()


    def peak_detection(self, bodycenter, ampl, TimePntThres, trials):
        '''Use the derivative method to detect body position peaks, onsets and offsets for each trial and plot them
        Inputs:
            - bodycenter: 
            - ampl: minimal amplitude change considered as a valid event
            - TimePntThres: number of points that a change is expected to happen
            - trials = array with trial numbers
        '''
        onsets_tr = []
        peaks_tr = []
        offsets_tr =[]
        for tr_idx, tr in enumerate(trials):
            data = -bodycenter[tr_idx]
            # Derive noise amplitude
            TrueStd, deriv_mean, deriv_std = ST.DerivGauss_NoiseEstim(data, thres=2)
            AmpPntThres = TrueStd*ampl
            # Use that noise amplitude to define regions of slope change / stability
            IncremSet, DecremSet, F_Values = ST.SlopeThreshold(data, AmpPntThres, TimePntThres, CollapSeq=True, acausal=True, graph=None)
            # Detect onsets and peaks with the derivative method
            peaks = []
            onsets_pos = []
            if len(IncremSet) > 0:
                if type(IncremSet[0]) is tuple:
                    for i in range(len(IncremSet)):
                        onsets_pos.append(IncremSet[i][0])
                        if IncremSet[i][0] + TimePntThres >= len(data):
                            values_idx = np.arange(IncremSet[i][0], len(data))
                        else:
                            values_idx = np.arange(IncremSet[i][0], IncremSet[i][0] + TimePntThres)
                        peak_idx = np.argmax(data[values_idx])
                        peaks.append(IncremSet[i][0] + peak_idx)
            peaks_tr.append(peaks)
            onsets_tr.append(onsets_pos)
            # Detect offsets with the derivative method
            onsets_neg = []
            if len(DecremSet) > 0:
                if type(DecremSet[0]) is tuple:
                    for i in range(len(DecremSet)):        
                        onsets_neg.append(DecremSet[i][1]) # DecremSet[i][1] because we want the offset of the peak
            offsets_tr.append(onsets_neg)
            # Plot peaks by trial
            plt.figure()
            plt.plot(data)
            plt.scatter(onsets_pos, data[onsets_pos], s=80, marker = '.', c='g')
            plt.ylabel('- Body position', fontsize = 18)
            plt.xlabel('Samples', fontsize = 18)
            plt.scatter(peaks, data[peaks], s=80, marker = '.', c='k')
            plt.scatter(onsets_neg, data[onsets_neg], s=80, marker = '.', c='r')
            plt.title('Trial ' + str(tr))
        return peaks_tr, onsets_tr, offsets_tr