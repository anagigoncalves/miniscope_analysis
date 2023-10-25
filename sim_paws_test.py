import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib as mp
import random

#Inputs
sr = 330
t_duration = 60 #seconds
prob = 1  #probability of CS appearing in that stride phase
CS_event_rel = 'st' #simulate CS in relation to stance or swing
phase_CS = 0
st_sw_duration = (1/5) #duration of st or sw - TODO consider do sw shorter than stance and speed related
time = np.arange(0, t_duration, np.round(1/sr, 3))

# Data from miniscope animals
paw_amp_fr_animal_mean = np.array([33.14353913, 32.7693747 , 31.1336233 , 31.8352444 , 32.14147608,
       32.17336272, 36.83117169, 37.08166025, 35.40465603, 37.10860484,
       37.93540391, 37.72864757, 36.51560441, 36.66108305, 36.31308615,
       35.63183443, 31.22621373, 30.78860108, 29.6791804 , 30.64607374,
       30.3253551 , 31.08695024, 30.24916718, 29.88490831, 29.97444866,
       30.24144683])
paw_amp_fl_animal_mean = np.array([32.48254839, 32.23607944, 31.71351701, 32.11623361, 32.12278663,
       31.89553014, 26.11320378, 24.99771588, 23.8317844 , 24.30616269,
       24.27708467, 24.64195775, 23.95726699, 24.14325718, 24.23943008,
       23.22991251, 30.18924305, 29.76931249, 30.06515096, 30.01651684,
       30.56391221, 30.5349457 , 29.31976452, 30.20453002, 29.1101885 ,
       30.16071586])
phase_diff_fl_fr_animal_mean = np.array([181.27298684, 180.46992716, 182.56703643, 181.18436688,
       180.94044709, 182.03996527, 172.27802148, 174.51538417,
       173.80888164, 173.27711463, 175.44406935, 176.14188906,
       176.18106094, 175.63966681, 175.01313556, 175.0216668 ,
       186.72161682, 185.93987271, 184.41329946, 183.21268408,
       184.47233419, 185.18845036, 183.91960814, 183.48726242,
       182.47289277, 181.16993711])
stride_duration_FR_animal_mean = np.array([262.59966213, 258.83056846, 250.35115508, 245.48481809,
       252.10526452, 246.77460565, 253.24943937, 249.65872664,
       243.21659825, 247.94367525, 255.6984879 , 253.69733965,
       242.4141943 , 244.95247433, 241.49739555, 238.30842356,
       247.76400132, 246.2284467 , 241.81067067, 240.33743843,
       239.02831916, 240.96793861, 238.71959802, 238.33637308,
       238.36555365, 241.69790416])/1000
stride_duration_FL_animal_mean = np.array([259.74207514, 255.74025122, 251.42505321, 250.266686  ,
       252.11203934, 248.47164422, 261.86130181, 256.55147511,
       253.70455189, 257.02428366, 259.08048576, 258.27457992,
       249.56143705, 251.58238765, 251.5433237 , 247.80578048,
       246.86977707, 243.86014852, 244.47132958, 241.21778191,
       242.53594099, 240.87716117, 232.51720829, 243.06868127,
       236.95934412, 243.06940889])/1000

trial_name = ['baseline', 'early split', 'late split', 'early washout', 'late washout']
greys = mp.cm.get_cmap('Greys', 14)
reds = mp.cm.get_cmap('Reds', 23)
blues = mp.cm.get_cmap('Blues', 23)
colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: reds(23), 8: reds(21),
                  9: reds(19), 10: reds(17), 11: reds(15), 12: reds(13),
                  13: reds(11), 14: reds(9), 15: reds(7), 16: reds(5), 17: blues(23), 18: blues(21), 19: blues(19), 20: blues(17),
                  21: blues(15), 22: blues(13),
                  23: blues(11), 24: blues(9), 25: blues(7), 26: blues(5)}

# simulate paw positions
phase_bs = np.nanmean(phase_diff_fl_fr_animal_mean[:6])
phase = np.deg2rad(np.array([phase_bs, phase_diff_fl_fr_animal_mean[6], phase_diff_fl_fr_animal_mean[15],
                  phase_diff_fl_fr_animal_mean[16], phase_diff_fl_fr_animal_mean[25]]))
amp_FR_bs = np.nanmean(paw_amp_fr_animal_mean[:6])
amp_FR = np.array([amp_FR_bs, paw_amp_fr_animal_mean[6], paw_amp_fr_animal_mean[15],
                  paw_amp_fr_animal_mean[16], paw_amp_fr_animal_mean[25]])
amp_FL_bs = np.nanmean(paw_amp_fl_animal_mean[:6])
amp_FL = np.array([amp_FL_bs, paw_amp_fl_animal_mean[6], paw_amp_fl_animal_mean[15],
                  paw_amp_fl_animal_mean[16], paw_amp_fl_animal_mean[25]])
freq_FR_bs = np.nanmean(stride_duration_FR_animal_mean[:6]) 
freq_FR = np.array([1/freq_FR_bs, 1/stride_duration_FR_animal_mean[6], 1/stride_duration_FR_animal_mean[15],
                    1/stride_duration_FR_animal_mean[16], 1/stride_duration_FR_animal_mean[25]])
freq_FL_bs = np.nanmean(stride_duration_FL_animal_mean[:6]) 
freq_FL = np.array([1/freq_FR_bs, 1/stride_duration_FL_animal_mean[6], 1/stride_duration_FL_animal_mean[15],
                    1/stride_duration_FL_animal_mean[16], 1/stride_duration_FL_animal_mean[25]])
paw_diff = np.zeros((len(phase), len(time)))
FR = np.zeros((len(phase), len(time)))
FL = np.zeros((len(phase), len(time)))
paw_diff_mean = np.zeros((len(phase)))
for t in range(len(phase)):
    FR[t, :] = amp_FR[t]*np.sin(2*np.pi*freq_FR[t]*time)
    FL[t, :] = amp_FL[t]*np.sin(2*np.pi*freq_FL[t]*time+phase[t])
    paw_diff[t, :] = FR[t, :]-FL[t, :]
    paw_diff_mean[t] = np.mean(paw_diff[t, :])

fig, ax = plt.subplots(1, 5, figsize=(20, 5), tight_layout=True, sharey=True)
ax = ax.ravel()
for t in range(len(phase)):
    ax[t].plot(time, FR[t, :], color='red' , label = 'FR')
    ax[t].plot(time, FL[t, :], color='blue', label = 'FL')
    ax[t].plot(time, paw_diff[t, :], color='black', label = 'FR-FL')
    ax[t].set_xlim([10, 10.5])
    ax[t].set_title('\n' + trial_name[t])
    ax[t].set_xlabel('Time (s)', fontsize=14)
    ax[t].set_ylabel('Paw position (mm)', fontsize=14)
    ax[t].spines['right'].set_visible(False)
    ax[t].spines['top'].set_visible(False)
    ax[t].tick_params(axis='both', which='major', labelsize=12)
ax[t].legend(frameon=False, fontsize=12)
fig.suptitle('Simulated paws\n', fontsize=16)


window = 50
sta_paw_diff_mean = np.zeros((len(phase), 2*window))
event_cs_trials = []
for t in range(len(phase)):
    # find stance and swing points
    st_FR = find_peaks(FR[t, :])[0]
    sw_FR = find_peaks(-FR[t, :])[0]
    st_FL = find_peaks(FL[t, :])[0]
    sw_FL = find_peaks(-FL[t, :])[0]
    
    # simulate CS
    if CS_event_rel == 'st':
        event_cs_initial = random.sample(list(st_FR), np.int64(len(st_FR)*prob))
    if CS_event_rel == 'sw':
        event_cs_initial = random.sample(list(st_FR), np.int64(len(sw_FR)*prob))
    event_cs = event_cs_initial + np.int64((phase_CS*st_sw_duration)*330)
    event_cs_trials.append(event_cs)
    
    # do STA
    sta_paw_diff = np.zeros((len(event_cs), 2*window))
    sta_paw_diff[:] = np.nan
    for count_e, e in enumerate(event_cs):
        if e > window and e < len(paw_diff[t, :])-window:
            sta_paw_diff[count_e, :] = paw_diff[t, :][e-window:e+window]
        
    sta_paw_diff_mean[t, :] = np.nanmean(sta_paw_diff, axis=0)

fig, ax = plt.subplots(1, 5, figsize=(20, 5), tight_layout=True, sharey=True)
ax = ax.ravel()
for t in range(len(phase)):
    ax[t].plot(time, FR[t, :], color='red' , label = 'FR')
    ax[t].plot(time, FL[t, :], color='blue', label = 'FL')
    ax[t].plot(time, paw_diff[t, :], color='darkgray', label = 'FR-FL')
    for e in event_cs_trials[t]:
        ax[t].axvline(time[e], color='black')
    ax[t].set_xlim([10, 10.5])
    ax[t].set_title('\n' + trial_name[t])
    ax[t].set_xlabel('Time (s)', fontsize=14)
    ax[t].set_ylabel('Paw position (mm)', fontsize=14)
    ax[t].spines['right'].set_visible(False)
    ax[t].spines['top'].set_visible(False)
    ax[t].tick_params(axis='both', which='major', labelsize=12)
ax[t].legend(frameon=False, fontsize=12)
fig.suptitle('Simulated paws with CS\n', fontsize=16)

colors_trials = [colors_session[1], colors_session[7], colors_session[16], colors_session[17], colors_session[26]]
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for t in range(len(phase)):
    ax.plot(sta_paw_diff_mean[t, :], color=colors_trials[t], label=trial_name[t])
ax.axvline(x = 50, color='black', linestyle='dashed')
ax.set_xlabel('Time relative to CS (s)', fontsize=14)
ax.set_ylabel('Paw position (mm)', fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(frameon=False, fontsize=12)
ax.set_title('STA paw difference CS relative to ' +  CS_event_rel + 
   '\nphase '+ str(np.round(phase_CS, 1)) + ' probability ' + str(prob))


