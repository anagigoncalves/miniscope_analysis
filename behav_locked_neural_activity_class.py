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

    
    def get_stsw(self, st_strides_trials, sw_strides_trials, trials, p):
        ''' Get stance and swing onset/offset indexes and timestamps'''
        st_on_ts = [st_strides_trials[tr_idx][p][:, 0, 0]/1000 for tr_idx, _ in enumerate(trials)]
        st_on_idx = [st_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        sw_on_ts = [sw_strides_trials[tr_idx][p][:, 0, 0]/1000 for tr_idx, _ in enumerate(trials)]
        sw_on_idx = [sw_strides_trials[tr_idx][p][:, 0, 4] for tr_idx, _ in enumerate(trials)]
        st_off_ts = [st_strides_trials[tr_idx][p][:, 1, 0]/1000 for tr_idx, _ in enumerate(trials)]
        st_off_idx = [st_strides_trials[tr_idx][p][:, 1, 4] for tr_idx, _ in enumerate(trials)]
        return st_on_ts, st_on_idx, st_off_ts, st_off_idx, sw_on_ts, sw_on_idx
    
    
    def get_spikes_behav(self, df_spikes, on_ts, off_ts, align_ts, window, bcam_time, trials, t_dim):
        ''' Find matching indexes and timestamps of spikes occurring in a defined time window in the behavior dataset)'''
        spike_idx_behav_tr = []
        strides_spike_idx_tr = []
        for tr_idx, tr in enumerate(trials):
            spike_idx_behav = []
            strides_spike_idx = []
            spikes_tr = df_spikes[df_spikes['trial'] == tr]
            spikes_tr = spikes_tr.reset_index(drop=True)
            for i in range(len(on_ts[tr_idx])): 
                if t_dim == 'phase':
                    start = on_ts[tr_idx][i]
                    end = off_ts[tr_idx][i]
                    signal_chunck = spikes_tr[(spikes_tr['time'] >= start) & (spikes_tr['time'] < end)].iloc[:, 2]
                elif t_dim == 'time':
                    start = align_ts[tr_idx][i] + window[0]
                    end = align_ts[tr_idx][i] + window[-1]
                    signal_chunck = spikes_tr[(spikes_tr['time'] > start) & (spikes_tr['time'] < end)].iloc[:,2]
                spike_idx = signal_chunck.index[signal_chunck == 1].tolist()
                spike_ts = np.array(spikes_tr['time'].iloc[spike_idx])
                if len(spike_ts) > 0:
                     spike_idx_behav.append(np.array([np.where(bcam_time[tr_idx] == bcam_time[tr_idx][np.abs(bcam_time[tr_idx] - t).argmin()])[0][0] for t in spike_ts]))
                     strides_spike_idx.append(i) 
            spike_idx_behav_tr.append(spike_idx_behav)
            strides_spike_idx_tr.append(np.array(strides_spike_idx))
        return spike_idx_behav_tr, strides_spike_idx_tr


    def get_spikes_timing(self, spike_idx_behav, temporal_dataset, align_ts, trials, t_dim): 
        ''' Find paw phase during each spike
        Inputs:
            temporal_dataset: bcam_time or final_tracks_trial_phase'''
        spikes_timing_tr = []
        for tr_idx, tr in enumerate(trials):
            spikes_timing_all = []
            for i in range(len(spike_idx_behav[tr_idx])): 
                spikes_timing = temporal_dataset[tr_idx][spike_idx_behav[tr_idx][i]]
                # if np.isnan(spikes_timing).any() == False:
                if t_dim == 'time':
                    spikes_timing = spikes_timing - align_ts[tr_idx][i]
                spikes_timing_all.append(spikes_timing) ### INDENT TILL HERE IS ISNAN IS UNCOMMENTED
            spikes_timing_tr.append(spikes_timing_all)
        return spikes_timing_tr


    def get_stsw_spike(self, st_on_ts, st_on_idx, st_off_ts, st_off_idx, sw_on_ts, sw_on_idx, strides_spike_idx_allrois, trials):
        ''' Find the timestamps and indexes of st and sw just for strides in which a spike occurs '''         
        st_on_ts_spike = []
        st_off_ts_spike = []
        sw_on_ts_spike = []
        st_on_idx_spike = []
        st_off_idx_spike = []
        sw_on_idx_spike = []
        for n in range(len(strides_spike_idx_allrois)):
            st_on_ts_spike.append([st_on_ts[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
            st_off_ts_spike.append([st_off_ts[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
            sw_on_ts_spike.append([sw_on_ts[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
            st_on_idx_spike.append([st_on_idx[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
            st_off_idx_spike.append([st_off_idx[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
            sw_on_idx_spike.append([sw_on_idx[tr_idx][strides_spike_idx_allrois[n][tr_idx]] for tr_idx, _ in enumerate(trials)])
        return st_on_ts_spike, st_on_idx_spike, st_off_ts_spike, st_off_idx_spike, sw_on_ts_spike, sw_on_idx_spike


    def bin_spikes(self, dataset, bins, trials): 
        ''' Compute spike count for each bin'''
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
    
        
    def plot_behav_locked_activity_rois(self, t_values, spikes_mat, stsw_rel_t, bins, trials, trials_ses, colors_session, animal, session_id, paw, roi, align, save_plot, t_dim):
        ''' Plot st/sw-aligned neural activity: rasters, spikes count heatmap, P(CS) 
        Inputs:
            - t_values = phase or timestamps'''
        if t_dim == 'phase':
            if align == 'st':
                x_tick_labels = ['sw', 'st', 'sw']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'stride':
                x_tick_labels = ['st', 'st']
                x_ticks = [0, len(bins)-1]
        elif t_dim == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]        

        # Rasterplot divided by block and sorted by time for each ROI
        fig, axs = plt.subplots(3, 1, figsize = (7, 13))
        # length_stsw_rel_t = len([stsw_rel_t[0][0]])
        cumul_strides = 0
        for tr in range(len(trials)):
            if tr in trials_ses[0:2,1]:
                axs[0].axhline(y=cumul_strides, color='crimson')
            for idx, t_val in enumerate(t_values[tr]):
                axs[0].scatter(t_val, np.ones(len(t_val))*cumul_strides, c = 'gray', marker = '.', s = 10)
                # axs[0].scatter(stsw_rel_t[tr][idx], np.ones(length_stsw_rel_t)*cumul_strides, c = 'red', marker = '|', s = 5, alpha = 0.1)
                if align == 'stride':
                    axs[0].scatter(stsw_rel_t[tr][idx], cumul_strides, c = 'red', marker = '|', s = 5, alpha = 0.2)
                cumul_strides += 1
        axs[0].set_ylabel('Strides', fontsize = self.font_size)
        axs[0].set(xticklabels=[])
        axs[0].tick_params(bottom=False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].set_xlim(bins[0]-0.01, bins[-1]+0.01)
        if t_dim == 'phase':
            if align == 'sw':
                axs[0].axvline(x = (bins[-1]+0.01)/2, color='crimson', linestyle='--')
            elif align == 'st':
                axs[0].axvline(x = 0, color='crimson', linestyle='--')     
        elif t_dim == 'time':
            axs[0].axvline(x = 0, color='crimson', linestyle='--')     
    
        # HM count and P(CS) by trial for each ROI
        spikes_count_tr = []
        total_spikes_tr = np.zeros((len(trials)))
        cs_prob_tr = np.zeros((len(trials), len(bins)-1))
        for tr in range(len(trials)):       
            spikes_count_tr.append(np.sum(np.vstack(spikes_mat[tr]), axis = 0))
            total_spikes_tr[tr] = np.sum(spikes_count_tr[tr])
            cs_prob_tr[tr] = spikes_count_tr[tr] / total_spikes_tr[tr]
            if np.any(trials_ses-1 == tr):
                axs[2].plot(cs_prob_tr[tr], c = colors_session[tr+1], linewidth = 2.5)
        sns.heatmap(spikes_count_tr, cmap = 'viridis', cbar = False, ax = axs[1])
        axs[1].set_ylabel('Trials', fontsize = self.font_size)
        axs[2].set_ylabel('P(CS)', fontsize = self.font_size)
        axs[1].set(yticklabels=[])
        axs[1].tick_params(left=False)
        axs[1].set(xticklabels=[])
        axs[1].tick_params(bottom=False)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].set_ylim(0, 0.25)
        axs[2].set_xlim(0, len(bins)-1)
        axs[2].set_xticks(x_ticks)
        axs[2].set_xticklabels(x_tick_labels, fontsize = self.font_size)
        axs[2].tick_params(axis='y', labelsize=self.font_size)
        if not align == 'stride':
            axs[1].axvline(x=(len(bins)-1)/2, color='white', linestyle='--')
            axs[2].axvline(x=(len(bins)-1)/2, color='k', linestyle='--')
        plt.tight_layout
        fig.suptitle(paw + ' ' + align + '-locked neuronal activity ' + animal + ' ' + roi + ' ' + session_id, fontsize = self.font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id)):
                os.mkdir(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id))
            plt.savefig(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id + '\\', paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + roi + ' ' + session_id + '.png'), dpi=self.my_dpi)
        plt.close()
        return spikes_count_tr, cs_prob_tr
    
    
    def plot__behav_locked_activity_popul(self, cs_prob, split_blocks):
        fig, axs = plt.subplots(1, len(split_blocks), figsize = (18, 5))
        for b_idx, b in enumerate(split_blocks):
            cs_prob_block = []
            for n in range(len(cs_prob)):
                start = b[0]
                end = b[1]
                cs_prob_block.append(np.mean(cs_prob[n][start:end], axis = 0))
            sns.heatmap(cs_prob_block, cmap = 'viridis', cbar= False, ax = axs[b_idx], vmin = 0, vmax = 0.15)
            axs[b_idx].axvline(x=len(cs_prob[0][0])/2, c = 'white', linestyle = '--')
            
            
    def plot_behav_locked_activity_clust(self, spikes_count_tr_allrois, bins, trials, trials_ses, colors_session, animal, session_id, paw, cluster_transition_idx, colors_cluster, align, save_plot, t_dim):
        # HM count and P(CS) by trial for each cluster    
        if t_dim == 'phase':
            if align == 'st':
                x_tick_labels = ['sw', 'st', 'sw']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]
            elif align == 'stride':
                x_tick_labels = ['st', 'st']
                x_ticks = [0, len(bins)-1]
        elif t_dim == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [0, (len(bins)-1)/2, len(bins)-1]    
        spikes_count_tr_allrois_sorted = [[sta_roi[tr_idx] for sta_roi in spikes_count_tr_allrois] for tr_idx, _ in enumerate(trials)] 
        fig, axs = plt.subplots(2, len(cluster_transition_idx), figsize = (15,8))
        if len(cluster_transition_idx) == 1:
            fig, axs = plt.subplots(2, len(cluster_transition_idx)+1, figsize = (15,8))
            fig.delaxes(axs[0, 1])
            fig.delaxes(axs[1, 1])
        else:
            fig, axs = plt.subplots(2, len(cluster_transition_idx), figsize = (15,8))
        for c_idx, c in enumerate(cluster_transition_idx):
            spikes_count_tr_clust = []
            total_spikes_tr_clust = np.zeros((len(trials)))
            cs_prob_tr_clust = np.zeros((len(trials), len(bins)-1))
            for tr in range(len(trials)):
                if c_idx == 0:
                    if len(cluster_transition_idx) > 1:
                        spikes_count_tr_clust.append(np.sum(np.vstack(spikes_count_tr_allrois_sorted[tr][0:c]), axis = 0)) #maybe c+1?
                    else:
                        spikes_count_tr_clust.append(np.sum(np.vstack(spikes_count_tr_allrois_sorted[tr]), axis = 0)) #maybe c+1?
                else:
                    spikes_count_tr_clust.append(np.sum(np.vstack(spikes_count_tr_allrois_sorted[tr][cluster_transition_idx[c_idx-1]:c]), axis = 0))              
                total_spikes_tr_clust[tr] = np.sum(spikes_count_tr_clust[tr])
                cs_prob_tr_clust[tr] = spikes_count_tr_clust[tr] / total_spikes_tr_clust[tr]
                if np.any(trials_ses-1 == tr):
                    axs[1, c_idx].plot(cs_prob_tr_clust[tr], c = colors_session[tr+1])
                sns.heatmap(spikes_count_tr_clust, cmap = 'viridis', cbar = False, ax = axs[0, c_idx]) 
            axs[0, c_idx].set_ylabel('Trials', fontsize = self.font_size)
            axs[1, c_idx].set_ylabel('P(CS)', fontsize = self.font_size)
            axs[0, c_idx].set(yticklabels=[])
            axs[0, c_idx].tick_params(left=False, bottom=False)
            axs[0, c_idx].set(xticklabels=[])
            axs[1, c_idx].spines['top'].set_visible(False)
            axs[1, c_idx].spines['right'].set_visible(False)
            axs[1, c_idx].set_ylim(0, 0.25)
            axs[1, c_idx].set_xlim(0, len(bins)-1)
            axs[1, c_idx].set_xticks(x_ticks)
            axs[1, c_idx].set_xticklabels(x_tick_labels, fontsize = self.font_size)
            axs[0, c_idx].set_title('Cluster ' + str(c_idx+1), c = colors_cluster[c_idx], fontsize = self.font_size)
            if not align == 'stride':
                axs[0, c_idx].axvline(x=(len(bins)-1)/2, color='white', linestyle='--')
                axs[1, c_idx].axvline(x=(len(bins)-1)/2, color='k', linestyle='--')
            plt.tight_layout()
            fig.suptitle(paw + ' ' + align + '-locked neuronal activity clusters ' + animal + ' ' + session_id, fontsize = self.font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id)):
                os.mkdir(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id))
            plt.savefig(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' ' + session_id + '\\', paw + ' ' + align + '-locked neural activity ' + t_dim + ' ' + animal + ' clusters ' + session_id + '.png'), dpi=self.my_dpi)
        plt.close() 
        return spikes_count_tr_clust


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
    
    
    def plot_behav_locked_activity_sorted_rois(self, t_values, bins, trials, sorted_by, animal, session_id, paw, roi, align, save_plot, t_dim):
        ''' Plot sorted st/sw-aligned neural activity: rasters, spikes count heatmap, P(CS) '''      
        # Rasterplot divided by block and sorted by time for each ROI
        plt.figure()        
        cumul_strides = 0
        for stride, spikes_t in enumerate(t_values):
            plt.scatter(spikes_t, np.ones(len(spikes_t))*cumul_strides, c = 'gray', marker = '.', s = 10)
            cumul_strides += 1
        plt.ylabel('Strides', fontsize = self.font_size)
        plt.tick_params(bottom=False)
        plt.xlim(bins[0]-0.01, bins[-1]+0.01)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(bins[0] - 0.01, bins[-1] + 0.01)
        if t_dim == 'phase':
            if align == 'st':
                x_tick_labels = ['sw', 'st', 'sw']
                x_ticks = [bins[0], 0, bins[-1]]
                plt.axvline(x = 0, color='crimson', linestyle='--')     
            elif align == 'sw':
                x_tick_labels = ['st', 'sw', 'st']
                x_ticks = [bins[0], 0.5, bins[-1]]
                plt.axvline(x = 0.5, color='crimson', linestyle='--')
            elif align == 'stride':
                x_tick_labels = ['st', 'st']
                x_ticks = [bins[0], bins[-1]]
        elif t_dim == 'time':
                x_tick_labels = [str(round(bins[0], 3)), str(0), str(round(bins[-1], 3))]
                x_ticks = [bins[0], 0, bins[-1]]
                plt.axvline(x = 0, color='crimson', linestyle='--')  
        ax.set_xticks(x_ticks, fontsize = self.font_size)
        ax.set_xticklabels(x_tick_labels, fontsize = self.font_size) 
        plt.title('Sorted by ' + sorted_by + ' ' + paw + ' ' + roi, fontsize = self.font_size)
        if save_plot:
            if not os.path.exists(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' sorted ' + animal + ' ' + session_id)):
                os.mkdir(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' sorted ' + animal + ' ' + session_id))
            plt.savefig(os.path.join(self.save_path, paw + ' ' + align + '-locked neural activity ' + t_dim + ' sorted ' + animal + ' ' + session_id + '\\', paw + ' ' + align + '-locked neural activity ' + t_dim + ' sorted by ' + sorted_by +  ' ' + animal + ' ' + roi + ' ' + session_id + '.png'), dpi=self.my_dpi)
        plt.close()
        
    
    def phasemap(self, final_tracks_trials_phase, df_events_trace_clusters, bcam_time, p1, colors_clusters, st_align, save_plot):
        '''Plot the phasemap for couple of paws and overimpose spikes. CODE NOT OPTIMIZED YET!!!
        Inputs:
            - final_tracks_trials_phase
            - df_events_trace_clusters
            - bcam_time
            - p1
            - colors_clusters = list of cluster colors
            - save_plot
            * Add sw-st-sw option
        '''
        p = np.array([0, 1, 2, 3])
        p = np.delete(p, np.where(p == p1))
        paws = ['FR','HR','FL','HL']
        for n in range(0,len(p)):
            if st_align == False:
                for tr in range(0,23):
                    fig, ax = plt.subplots()
                    plt.scatter(final_tracks_trials_phase[tr][0, p[n], :], final_tracks_trials_phase[tr][0, p1, :], s=3, c="gray", marker=".")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(1.2)
                    ax.spines['left'].set_linewidth(1.2)
                    ax.set_xlabel(paws[p[n]] + ' St-St Phase', fontsize=18)
                    ax.set_ylabel(paws[p1] + ' St-St Phase', fontsize=18)
                    fig.suptitle('Trial ' + str(tr+1), fontsize=18)
                    plt.show()
                    df_tr = df_events_trace_clusters.loc[df_events_trace_clusters['trial'] == tr+1].iloc[:, 2:]
                    event_idx = []
                    event_clust = []
                    for i in range(len(df_tr.index)):
                          for j in range(len(df_tr.columns)):
                              if df_tr.iloc[i, j] == 1:
                                event_idx.append(i)
                                event_clust.append(j)
                    event_ts = [df_events_trace_clusters['time'][i] for i in event_idx]
                    ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in event_ts])
                    colors = [colors_clusters[cluster] for cluster in event_clust]
                    plt.scatter(final_tracks_trials_phase[tr][0, p[n], ts_idx], final_tracks_trials_phase[tr][0,  p1, ts_idx], s=5, c=colors, marker="o")
                    if save_plot:
                        if not os.path.exists(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1])):
                            os.mkdir(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1]))
                        plt.savefig(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1] + '\\','trial' + str(tr+1) + '.png'), dpi=self.my_dpi)
            else:    
                st_pts_trials = []
                for tr in range (0,23):
                    st_pts = np.where(final_tracks_trials_phase[tr][0, p[n], :] == 0)[0]
                    st_pts_trials.append(st_pts)
                    # plt.figure()
                    # plt.plot(final_tracks_trials_phase[tr][0, 0, :])
                    # plt.scatter(st_pts, final_tracks_trials_phase[tr][0, 0, st_pts], color='red', marker='o')
                    
                st_centered_phase = []
                for tr in range(0,23):
                    center_on_st = [val - 1 if val > 0.5 else val for val in final_tracks_trials_phase[tr][0, p[n], :]]
                    st_centered_phase.append(center_on_st)
                st_centered_phase = [np.array(l) for l in st_centered_phase]
                
                for tr in range(0,23):
                    fig, ax = plt.subplots()
                    plt.scatter(st_centered_phase[tr], final_tracks_trials_phase[tr][0, p1, :], s=3, c="gray", marker=".")
                    ax.set_xlim(-0.5, 0.5)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([-0.5, 0, 0.5])
                    ax.set_yticks([0, 1])
                    ax.tick_params(axis='both', which='major', labelsize=12)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_linewidth(1.2)
                    ax.spines['left'].set_linewidth(1.2)
                    ax.axvline(x=0, color='red', linestyle = '--')
                    ax.text(0, 1.02, 'St', ha='center', fontsize=12, color='red')
                    ax.set_xlabel(paws[p[n]] + ' Stance Phase', fontsize=18)
                    ax.set_ylabel(paws[p1] + ' St-St Phase', fontsize=18)
                    fig.suptitle('Trial ' + str(tr+1), fontsize=18)
                    plt.show()
                    df_tr = df_events_trace_clusters.loc[df_events_trace_clusters['trial'] == tr+1].iloc[:, 2:]
                    event_idx = []
                    event_clust = []
                    for i in range(len(df_tr.index)):
                          for j in range(len(df_tr.columns)):
                              if df_tr.iloc[i, j] == 1:
                                event_idx.append(i)
                                event_clust.append(j)
                    event_ts = [df_events_trace_clusters['time'][i] for i in event_idx]
                    ts_idx = np.array([np.where(bcam_time[tr] == bcam_time[tr][np.abs(bcam_time[tr] - t).argmin()])[0][0] for t in event_ts])
                    colors = [colors_clusters[cluster] for cluster in event_clust]
                    plt.scatter(st_centered_phase[tr][ts_idx], final_tracks_trials_phase[tr][0,  p1, ts_idx], s=5, c=colors, marker="o")
                    if save_plot:
                        if not os.path.exists(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1])):
                            os.mkdir(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1]))
                        plt.savefig(os.path.join(self.save_path, 'phasemaps_' + paws[p[n]] + 'x' + paws[p1] + '\\','trial' + str(tr+1) + '.png'), dpi=self.my_dpi)                


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
    