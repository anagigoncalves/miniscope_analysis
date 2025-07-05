import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Input data
load_path = 'J:\\Miniscope processed files\\Analysis on population data\\Rasters st-sw-st\\split contra fast S1\\'
path_session_data = 'J:\\Miniscope processed files'
save_path = 'J:\\Miniscope processed files\\Analysis on population data\\Calcium modulation\\split contra fast S1\\'
session_data = pd.read_excel(os.path.join(path_session_data, 'session_data_split_S2.xlsx'))
animals = ['MC8855', 'MC9194', 'MC9226', 'MC9513', 'MC10221']
protocol = 'split contra fast'
paw = 'FR'
phase_to_plot = 'full' #options: full, transition
bs_trial = 'baseline_block' #options: first_trial, baseline_block
trials_plot = 'blocks' #options: trials, blocks
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paws = ['FR', 'HR', 'FL', 'HL']

def calcium_rate_modulation_baseline(firing_rate_animal, paw, phase_plot, bs_trial, trials_plot, trials_baseline, trials_split, trials_washout):
    """Compute calcium rate modulation with stride aligned calcium events (in phase). It computes the modulation in relation to the baseline trials
    or the whole baseline period. The output is the modulated calcium rate for the transition trials or blocks. The transition trials are the first baseline
    trial, first split, last split, first washout and last washout. The transition blocks are the whole baseline period, 1st 3 split, last 3 split,
    1st 3 washout, last 3 washout
    Inputs:
        firing_rate_animal: ndarray (rois x paws x trials x phase bins)
        paw: (str) FR, HR, FL, HL
        phase_plot: (str) st_full, sw_full, st_transition, sw_transition
        bs_trial: (str) first_trial, whole_baseline
        trials_plot: (str) trials, blocks
        trials_baseline: (list)
        trials_split: (list)
        trials_washout: (list)"""
    if paw == 'FR':
        p = 0
    if paw == 'HR':
        p = 1
    if paw == 'FL':
        p = 2
    if paw == 'HL':
        p = 3
    if phase_plot == 'st_full': #0 to 50% phase
        phase_idx = np.array([0, 1, 2, 3, 4])
    if phase_plot == 'sw_full': #50 to 100% phase
        phase_idx = np.array([5, 6, 7, 8, 9])
    if phase_plot == 'st_transition': #80% to 20% phase
        phase_idx = np.array([0, 1, 8, 9])
    if phase_plot == 'sw_transition': #30% to 70% phase
        phase_idx = np.array([3, 4, 5, 6])
    if bs_trial == 'first_trial':
        firing_rate_bs = np.nanmedian(firing_rate_animal[:, p, 0, phase_idx], axis=-1)
    if bs_trial == 'baseline_block':
        firing_rate_1 = np.nanmedian(firing_rate_animal[:, p, :, phase_idx], axis=0)
        trials_selected_bs = np.array([trials_baseline[-2], trials_baseline[-1]])
        trials_idx_selected_bs = np.zeros(len(trials_selected_bs))
        for count_t, t in enumerate(trials_selected_bs):
            trials_idx_selected_bs[count_t] = np.where(t == trials)[0][0]
        firing_rate_bs = np.nanmedian(firing_rate_1[:, np.int64(trials_idx_selected_bs)], axis=-1)
    firing_rate = np.nanmedian(firing_rate_animal[:, p, :, phase_idx], axis=0)
    firing_rate_bs_rep = np.swapaxes(np.tile(firing_rate_bs, (np.shape(firing_rate)[1], 1)), 0, 1)
    firing_rate_mod_all = (firing_rate-firing_rate_bs_rep)*100
    # match trials to plot to indexes (to account for missing trials)
    if trials_plot == "trials":
        trials_selected = np.array([trials_baseline[0], trials_split[0], trials_split[-1], trials_washout[0], trials_washout[-1]])
        trials_idx_selected = np.zeros(len(trials_selected))
        for count_t, t in enumerate(trials_selected):
            trials_idx_selected[count_t] = np.where(t==trials)[0][0]
        firing_rate_mod = firing_rate_mod_all[:, np.int64(trials_idx_selected)]
    if trials_plot == "blocks":
        trials_selected_bs = np.array([trials_baseline[-2], trials_baseline[-1]])
        trials_idx_selected_bs = np.zeros(len(trials_selected_bs))
        for count_t, t in enumerate(trials_selected_bs):
            trials_idx_selected_bs[count_t] = np.where(t == trials)[0][0]
        trials_selected_early_split = np.array([trials_split[0], trials_split[1]])
        trials_idx_selected_es = np.zeros(len(trials_selected_early_split))
        for count_t, t in enumerate(trials_selected_early_split):
            trials_idx_selected_es[count_t] = np.where(t==trials)[0][0]
        trials_selected_late_split = np.array([trials_split[-2], trials_split[-1]])
        trials_idx_selected_ls = np.zeros(len(trials_selected_late_split))
        for count_t, t in enumerate(trials_selected_late_split):
            trials_idx_selected_ls[count_t] = np.where(t==trials)[0][0]
        trials_selected_early_washout = np.array([trials_washout[0], trials_washout[1]])
        trials_idx_selected_ew = np.zeros(len(trials_selected_early_washout))
        for count_t, t in enumerate(trials_selected_early_washout):
            trials_idx_selected_ew[count_t] = np.where(t==trials)[0][0]
        trials_selected_late_washout = np.array([trials_washout[-2], trials_washout[-1]])
        trials_idx_selected_lw = np.zeros(len(trials_selected_late_washout))
        for count_t, t in enumerate(trials_selected_late_washout):
            trials_idx_selected_lw[count_t] = np.where(t==trials)[0][0]
        firing_rate_mod_bs = np.nanmedian(firing_rate_mod_all[:, np.int64(trials_idx_selected_bs)], axis=1)
        firing_rate_mod_es = np.nanmedian(firing_rate_mod_all[:, np.int64(trials_idx_selected_es)], axis=1)
        firing_rate_mod_ls = np.nanmedian(firing_rate_mod_all[:, np.int64(trials_idx_selected_ls)], axis=1)
        firing_rate_mod_ew = np.nanmedian(firing_rate_mod_all[:, np.int64(trials_idx_selected_ew)], axis=1)
        firing_rate_mod_lw = np.nanmedian(firing_rate_mod_all[:, np.int64(trials_idx_selected_lw)], axis=1)
        firing_rate_mod = np.squeeze(np.dstack((firing_rate_mod_bs, firing_rate_mod_es, firing_rate_mod_ls, firing_rate_mod_ew, firing_rate_mod_lw)))
    return firing_rate_mod

firing_rate_FR_st_mod_list_all = []
firing_rate_FR_sw_mod_list_all = []
for animal in animals:
    firing_rate_animal = np.load(os.path.join(load_path, animal + ' ' + protocol, 'raster_firing_rate_ro'
                                                                                  'is.npy'))
    s = np.where(session_data['animal'] == animal)[0][0]
    ses_info = session_data.iloc[s, :]
    print(ses_info)
    date = ses_info[3]
    path = os.path.join(path_session_data, 'TM RAW FILES', ses_info[0], ses_info[1], date + '\\')
    path_loco = os.path.join(path_session_data, 'TM TRACKING FILES',
                             ses_info[0] + ' ' + ses_info[2] + ' ' + date.split('_')[-1] + date.split('_')[-2] +
                             date.split('_')[-3][2:] + '\\')

    import miniscope_session_class
    import locomotion_class
    mscope = miniscope_session_class.miniscope_session(path)
    loco = locomotion_class.loco_class(path_loco)
    animal = mscope.get_animal_id()
    session = loco.get_session_id()
    [df_extract, df_events_extract, df_extract_rawtrace, df_extract_rawtrace_detrended, df_events_extract_rawtrace,
     coord_ext, reg_th, reg_bad_frames, trials,
     clusters_rois, colors_cluster, colors_session, idx_roi_cluster_ordered, ref_image,
     frames_dFF] = mscope.load_processed_files()
    [trials_ses, trials_ses_name, cond_plot, trials_baseline, trials_split, trials_washout] = mscope.get_session_data(
        trials, protocol.split(' ')[0], animal, session)

    if phase_to_plot == 'full':
        firing_rate_FR_st_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'st_full', bs_trial, trials_plot, trials_baseline, trials_split, trials_washout)
        firing_rate_FR_sw_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'sw_full', bs_trial, trials_plot, trials_baseline, trials_split, trials_washout)
    if phase_to_plot == 'transition':
        firing_rate_FR_st_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'st_transition', bs_trial, trials_plot, trials_baseline, trials_split, trials_washout)
        firing_rate_FR_sw_mod = calcium_rate_modulation_baseline(firing_rate_animal, paw, 'sw_transition', bs_trial, trials_plot, trials_baseline, trials_split, trials_washout)

    # Modulation
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    for r in range(np.shape(firing_rate_FR_st_mod)[0]):
        ax.plot(firing_rate_FR_st_mod[r, :], color='orange', linewidth=0.5, alpha=0.2)
        ax.plot(firing_rate_FR_sw_mod[r, :], color='green', linewidth=0.5, alpha=0.2)
    ax.plot(np.nanmedian(firing_rate_FR_st_mod, axis=0), color='orange', linewidth=2, marker='o')
    ax.plot(np.nanmedian(firing_rate_FR_sw_mod, axis=0), color='green', linewidth=2, marker='o')
    ax.set_title(animal, fontsize=20)
    ax.axhline(y=0, linestyle='dashed', color='darkgray')
    ax.set_xticks(np.arange(np.shape(firing_rate_FR_st_mod)[1]))
    ax.set_xticklabels(['baseline', 'early split', 'late split', 'early washout', 'late washout'], rotation=45)
    ax.set_ylabel('Calcium rate change (%)', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(os.path.join(save_path, 'calcium_modulation_' + animal + '_' + phase_to_plot + '_' + trials_plot), dpi=mscope.my_dpi)
    plt.savefig(os.path.join(save_path, 'calcium_modulation_' + animal + '_' + phase_to_plot + '_' + trials_plot + '.svg'), dpi=mscope.my_dpi)

    #Depth of modulation
    fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
    for r in range(np.shape(firing_rate_FR_st_mod)[0]):
        ax.plot(firing_rate_FR_st_mod[r, :]-firing_rate_FR_sw_mod[r, :], color='darkgray', linewidth=0.5, alpha=0.8)
    ax.plot(np.nanmedian(firing_rate_FR_st_mod-firing_rate_FR_sw_mod, axis=0), color='black', linewidth=2, marker='o')
    ax.set_title(animal, fontsize=20)
    ax.axhline(y=0, linestyle='dashed', color='darkgray')
    ax.set_xticks(np.arange(np.shape(firing_rate_FR_st_mod)[1]))
    ax.set_xticklabels(['baseline', 'early split', 'late split', 'early washout', 'late washout'], rotation=45)
    ax.set_ylabel('Calcium rate change (%)', fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(os.path.join(save_path, 'calcium_modulation_depth_' + animal + '_' + phase_to_plot + '_' + trials_plot), dpi=mscope.my_dpi)
    plt.savefig(os.path.join(save_path, 'calcium_modulation_depth_' + animal + '_' + phase_to_plot + '_' + trials_plot + '.svg'), dpi=mscope.my_dpi)

    #Append to list, to later concatenate animals
    firing_rate_FR_st_mod_list_all.append(firing_rate_FR_st_mod)
    firing_rate_FR_sw_mod_list_all.append(firing_rate_FR_sw_mod)
    plt.close('all')

firing_rate_FR_st_mod_all = np.vstack(firing_rate_FR_st_mod_list_all)
firing_rate_FR_sw_mod_all = np.vstack(firing_rate_FR_sw_mod_list_all)

fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
for r in range(np.shape(firing_rate_FR_st_mod)[0]):
    ax.plot(firing_rate_FR_st_mod_all[r, :], color='orange', linewidth=0.5, alpha=0.2)
    ax.plot(firing_rate_FR_sw_mod_all[r, :], color='green', linewidth=0.5, alpha=0.2)
ax.plot(np.nanmedian(firing_rate_FR_st_mod_all, axis=0), color='orange', linewidth=2, marker='o')
ax.plot(np.nanmedian(firing_rate_FR_sw_mod_all, axis=0), color='green', linewidth=2, marker='o')
ax.set_title('All animals', fontsize=20)
ax.axhline(y=0, linestyle='dashed', color='darkgray')
ax.set_xticks(np.arange(np.shape(firing_rate_FR_st_mod)[1]))
ax.set_xticklabels(['baseline', 'early split', 'late split', 'early washout', 'late washout'], rotation=45)
ax.set_ylabel('Calcium rate change (%)', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(os.path.join(save_path, 'calcium_modulation_allanimals_' + phase_to_plot + '_' + trials_plot),
            dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'calcium_modulation_allanimals_' + phase_to_plot + '_' + trials_plot + '.svg'),
            dpi=mscope.my_dpi)

fig, ax = plt.subplots(figsize=(7, 7), tight_layout=True)
for r in range(np.shape(firing_rate_FR_st_mod)[0]):
    ax.plot(firing_rate_FR_st_mod_all[r, :]-firing_rate_FR_sw_mod_all[r, :], color='darkgray', linewidth=0.5, alpha=0.8)
ax.plot(np.nanmedian(firing_rate_FR_st_mod_all-firing_rate_FR_sw_mod_all, axis=0), color='black', linewidth=2, marker='o')
ax.set_title('All animals', fontsize=20)
ax.axhline(y=0, linestyle='dashed', color='darkgray')
ax.set_xticks(np.arange(np.shape(firing_rate_FR_st_mod)[1]))
ax.set_xticklabels(['baseline', 'early split', 'late split', 'early washout', 'late washout'], rotation=45)
ax.set_ylabel('Calcium rate change (%)', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(os.path.join(save_path, 'calcium_modulation_allanimals_depth_' + phase_to_plot + '_' + trials_plot),
            dpi=mscope.my_dpi)
plt.savefig(os.path.join(save_path, 'calcium_modulation_allanimals_depth_' + phase_to_plot + '_' + trials_plot + '.svg'),
            dpi=mscope.my_dpi)