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
    
    
    def split_expblocks(self, trials_ses):
        '''Sub-divide experimental blocks'''
        block_halflen = (trials_ses[1][1] - trials_ses[1][0]+1)//2
        split_blocks = np.array(([trials_ses[0][0]-1, trials_ses[0][1]], 
                                 [trials_ses[1][0]-1, trials_ses[1][0]-1 + block_halflen], 
                                 [trials_ses[1][0]-1 + block_halflen, trials_ses[1][1]], 
                                 [trials_ses[2][0]-1, trials_ses[2][0]-1 + block_halflen], 
                                 [trials_ses[2][0]-1 + block_halflen, trials_ses[2][1]]))
        return split_blocks

    
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
            # Find spikes timestamps
            spikes_tr = df_spikes[df_spikes['trial'] == tr].reset_index(drop=True)
            spikes_idx = spikes_tr.index[spikes_tr.iloc[:, 2] == 1].tolist() 
            spikes_ts = np.array(spikes_tr['time'].iloc[spikes_idx])
            # Find matching timestamps in behavioral dataset
            mapped_spikes_idx = np.array([np.argmin(np.abs(bcam_time[tr_idx] - t)) for t in spikes_ts])
            mapped_spikes_ts = bcam_time[tr_idx][mapped_spikes_idx]
            # Find corresponding strides
            if temporal_dimension == 'phase':
                start = on_ts[tr_idx]
                end = off_ts[tr_idx]
            elif temporal_dimension == 'time':
                start = align_ts[tr_idx] + window[0]
                end = align_ts[tr_idx] + window[-1]
            spikes_ts_matrix = np.tile(mapped_spikes_ts, (len(start), 1)).T # For broadcasting
            onset_matrix = np.tile(start, (len(mapped_spikes_ts), 1)) # For broadcasting
            offset_matrix = np.tile(end, (len(mapped_spikes_ts), 1)) # For broadcasting
            mask = (spikes_ts_matrix >= onset_matrix) & (spikes_ts_matrix <= offset_matrix) # Mask for spikes falling within each stride
            spikeIdx_behavData.append([mapped_spikes_idx[np.where(mask[:, col])[0]] for col in range(mask.shape[1])])
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
                if np.any(spikesIdx_behavData[tr_idx][i]):
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
            spikes_count = np.sum(np.vstack(spikes_mat[tr]), axis = 0) # Sum spikes in each bin
            frames_bin, _ = np.histogram(phase[tr][~np.isnan(phase[tr])], bins=len(bins)-1) # Compute time spent in each bin
            time_bin = frames_bin*(1/self.sr_cam)
            firing_rate[tr] = spikes_count/time_bin # Compute firing rate
            spike_prob[tr] = (spikes_count/np.sum(spikes_count))/time_bin # Compute spike probability (normalized by time)
        return firing_rate, spike_prob


    def plot_behav_locked_activity_rois(self, spikes_timing, firing_rate, bins, trials_ses, colors_session, animal, session_id, roi, align, save_plot, temporal_dimension):
        ''' Plot st/sw-aligned neural activity for each ROI: rasters, firing rate heatmap, P(CS)'''

        trials = np.arange(0, trials_ses[-1, -1])
        if session_id == 'tied':
            split_blocks = trials_ses
        else:
            split_blocks = self.split_expblocks(trials_ses)
        
        # Create tick labels for x-axis
        if temporal_dimension == 'phase':
            if align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            else:
                x_tick_labels = ['st', 'st']
                x_ticks = [0, len(bins)-1]
        elif temporal_dimension == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]        
                
        # Rasterplot divided by block and sorted by time for each ROI
        fig, axs = plt.subplots(3, 4, figsize = (30, 15))
        for p, paw in enumerate(spikes_timing.keys()):
            axs[0, p].set_title(paw, fontsize = self.font_size)
            spikes_timing_allstrides = [ts for tr in spikes_timing[paw] for ts in tr]
            stride_idx = [np.full_like(ts, idx) for idx, ts in enumerate(spikes_timing_allstrides)]
            axs[0, p].scatter(np.concatenate(spikes_timing_allstrides), np.concatenate(stride_idx), c = 'dimgrey', marker = '.', s = 15)
            axs[0, p].set_ylabel('Strides', fontsize = self.font_size)
            axs[0, p].set(xticklabels=[])
            axs[0, p].tick_params(bottom=False)
            axs[0, p].spines['top'].set_visible(False)
            axs[0, p].spines['right'].set_visible(False)
            axs[0, p].set_xlim(bins[0], bins[-1])
            for b in trials_ses[:-1, 1]:
                axs[0, p].axhline(sum(len(tr) for tr in spikes_timing[paw][:b]), color='crimson')
            if temporal_dimension == 'phase' and align == 'sw':
                axs[0, p].axvline(x = (bins[-1]+0.01)/2, color='crimson', linestyle='--')  
            elif temporal_dimension == 'time':
                axs[0, p].axvline(x = 0, color='crimson', linestyle='--')
        
            # Firing rate heatmap and P(CS) by trial for each ROI
            for tr in range(len(trials)): 
                axs[2, p].plot(firing_rate[paw][tr], c = 'gray', linewidth = 1, alpha = 0.3)
            for start, end in split_blocks:
                axs[2, p].plot(np.mean(firing_rate[paw][start:end], axis = 0), c = colors_session[start+1], linewidth = 3)
            sns.heatmap(np.flipud(firing_rate[paw]), cmap = 'viridis', cbar = False, vmin = 0, vmax = 6, ax = axs[1, p])
            for i in range(len(trials_ses)-1):
                axs[1, p].axhline(y=trials_ses[-1, 1]-trials_ses[i, 1], color='white')
            axs[1, p].set_ylabel('Trials', fontsize = self.font_size)
            axs[2, p].set_ylabel('Firing rate (Hz)', fontsize = self.font_size)
            axs[1, p].set(yticklabels=[])
            axs[1, p].tick_params(left=False)
            axs[1, p].set(xticklabels=[])
            axs[1, p].tick_params(bottom=False)
            axs[2, p].spines['top'].set_visible(False)
            axs[2, p].spines['right'].set_visible(False)
            axs[2, p].set_ylim(0, 7)
            axs[2, p].set_xlim(0, len(bins)-1)
            axs[2, p].set_xticks(x_ticks)
            axs[2, p].set_xticklabels(x_tick_labels, fontsize = self.font_size)
            axs[2, p].tick_params(axis='y', labelsize=self.font_size)
            if not align == 'stride':
                axs[1, p].axvline(x=(len(bins)-1)/2, color='white', linestyle='--')
                axs[2, p].axvline(x=(len(bins)-1)/2, color='k', linestyle='--')
        fig.suptitle(f'{align}-locked neuronal activity {animal} {roi} {session_id}', fontsize = self.font_size)
        
        # Save
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}')):
                os.mkdir(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}'))
            plt.savefig(os.path.join(self.save_path, f'{align}-locked neural activity {temporal_dimension} {animal} {session_id}\\', f'{align}-locked neural activity {temporal_dimension} {roi} {animal} {session_id}.png'), dpi=self.my_dpi)
        plt.close()


    # def plot__behav_locked_activity_popul(self, firing_rate, trials_ses):
    #     ''' Plot heatmap of behavior-locked firing rate for the whole population '''
    #     fig, axs = plt.subplots(len(trials_ses.flatten()), 4)
    #     for p, paw in enumerate(firing_rate.keys()):
    #         axs[0, p].set_title(paw, fontsize = self.font_size)
    #         # for tr, _ in enumerate(trials_ses.flatten()-1):
    #         for tr_idx, tr in enumerate([0, 4, 6, 14, 16, 24]):
    #             firing_rate_block = []
    #             for n in range(len(firing_rate[paw])):
    #                 # firing_rate_block.append(firing_rate[paw][n][tr])
    #                 firing_rate_block.append(np.mean(firing_rate[paw][n][tr:tr+1], axis = 0))
    #             # sns.heatmap(firing_rate_block, cmap = 'viridis', cbar = False, vmin = 0, vmax = 6, ax = axs[tr, p])
    #             sns.heatmap(firing_rate_block, cmap = 'viridis', cbar = False, vmin = 0, vmax = 6, ax = axs[tr_idx, p])
    #             # axs[tr, p].axvline(x=len(firing_rate[paw][0][0])/2, c = 'white', linestyle = '--')
    #             axs[tr_idx, p].axvline(x=len(firing_rate[paw][0][0])/2, c = 'white', linestyle = '--')


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


    def compute_quartiles(self, vector):
        """ Calculate quartile indices for a sorted vector """
        q1_idx = int(0.25 * (len(vector) - 1))
        q2_idx = int(0.50 * (len(vector) - 1))
        q3_idx = int(0.75 * (len(vector) - 1))
        Q = [0, q1_idx, q2_idx, q3_idx, len(vector)]
        return Q


    def plot_behav_locked_activity_sorted_rois(self, sorted_spikes_timing, quartile_firing_rate, behavior_stride, bins, trials, sorted_by, animal, session_id, p1, p2, roi, align, save_plot, temporal_dimension):
        ''' Plot sorted st/sw-aligned rasters'''      
        colors = ['peachpuff', 'lightcoral', 'crimson', 'brown']
        fig, axs = plt.subplots(2,2, figsize=(14,10))       
        for stride, timing in enumerate(sorted_spikes_timing):
            axs[0, 0].scatter(timing, np.ones(len(timing))*stride, c = 'dimgray', marker = '.', s = 15)
        if sorted_by in ['sw phase', f'{p2}-st phase']:
            axs[0, 0].scatter(behavior_stride, np.arange(0, len(behavior_stride)), c = 'crimson', marker = '|', s = 5)
        axs[0, 1].plot(behavior_stride, np.arange(1, len(behavior_stride)+1), c = 'crimson', linewidth = 2.5)
        axs[0, 0].set_xlim(bins[0], bins[-1])
        for q, q_fr in enumerate(quartile_firing_rate):
            axs[1, 0].plot(q_fr, linewidth = 2.5, c = colors[q])
        x_tick_labels = ['st', 'st']
        axs[0, 0].set_xlabel('Stride phase', fontsize=self.font_size)
        axs[0, 0].set_ylabel('Strides', fontsize=self.font_size)
        axs[0, 1].set_xlabel(sorted_by, fontsize=self.font_size)
        axs[0, 1].set_ylabel('Strides', fontsize=self.font_size)
        axs[1, 0].set_xlabel('Stride phase', fontsize=self.font_size)
        axs[1, 0].set_ylabel('Firing Rate (Hz)', fontsize=self.font_size)
        axs[0, 0].set_xticks([bins[0], bins[-1]], fontsize = self.font_size)
        axs[0, 0].set_xticklabels(x_tick_labels, fontsize = self.font_size)
        axs[1, 0].set_xticks([0, len(bins)-1], fontsize = self.font_size)
        axs[1, 0].set_xticklabels(x_tick_labels, fontsize = self.font_size)
        axs[1, 0].set_xlim(0, len(bins)-1)
        axs[1, 0].legend(['1st qrt', '2nd qrt', '3rd qrt', '4th qrt'])
        fig.delaxes(axs[1, 1])
        for ax in axs.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, p1 + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id)):
                os.mkdir(os.path.join(self.save_path, p1 + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id))
            plt.savefig(os.path.join(self.save_path, p1 + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted ' + animal + ' ' + session_id + '\\', p1 + ' ' + align + '-locked neural activity ' + temporal_dimension + ' sorted by ' + sorted_by +  ' ' + animal + ' ' + roi + ' ' + session_id + '.png'), dpi=self.my_dpi)
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