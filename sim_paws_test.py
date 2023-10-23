import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib as mp

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
paw_diff_fr_fl_animal_mean = np.array([-1.33533531, -0.83067111, -0.65100418, -0.57497338, -0.60834155,
       -0.53332645, -2.63434305, -2.96666364, -2.76432649, -3.04492336,
       -3.16987661, -3.06997423, -3.42015566, -3.20212595, -3.10109727,
       -3.0882217 , -0.64162562, -0.35874707, -0.52619771, -0.89590468,
       -0.4654575 , -0.88780266, -0.96191711, -0.71500119, -1.1047874 ,
       -0.84939971])
phase_diff_fl_fr_animal_mean = np.array([181.27298684, 180.46992716, 182.56703643, 181.18436688,
       180.94044709, 182.03996527, 172.27802148, 174.51538417,
       173.80888164, 173.27711463, 175.44406935, 176.14188906,
       176.18106094, 175.63966681, 175.01313556, 175.0216668 ,
       186.72161682, 185.93987271, 184.41329946, 183.21268408,
       184.47233419, 185.18845036, 183.91960814, 183.48726242,
       182.47289277, 181.16993711])

greys = mp.cm.get_cmap('Greys', 14)
reds = mp.cm.get_cmap('Reds', 23)
blues = mp.cm.get_cmap('Blues', 23)
colors_session = {1: greys(14), 2: greys(12), 3: greys(10), 4: greys(8), 5: greys(6), 6: greys(4), 7: reds(23), 8: reds(21),
                  9: reds(19), 10: reds(17), 11: reds(15), 12: reds(13),
                  13: reds(11), 14: reds(9), 15: reds(7), 16: reds(5), 17: blues(23), 18: blues(21), 19: blues(19), 20: blues(17),
                  21: blues(15), 22: blues(13),
                  23: blues(11), 24: blues(9), 25: blues(7), 26: blues(5)}

sr = 330
t_duration = 60 #seconds
time = np.arange(0, t_duration, np.round(1/sr, 3))

freq = 5 #Hz, stride frequency

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
paw_diff = np.zeros((len(phase), len(time)))
FR = np.zeros((len(phase), len(time)))
FL = np.zeros((len(phase), len(time)))
for t in range(len(phase)):
    FR[t, :] = amp_FR[t]*np.sin(2*np.pi*freq*time)
    FL[t, :] = amp_FL[t]*np.sin(2*np.pi*freq*time+np.pi+phase[t])
    paw_diff[t, :] = FR[t, :]-FL[t, :]

fig, ax = plt.subplots(1, 5, figsize=(20, 5), tight_layout=True, sharey=True)
ax = ax.ravel()
for t in range(len(phase)):
    ax[t].plot(time, FR[t, :], color='red')
    ax[t].plot(time, FL[t, :], color='blue')
    ax[t].plot(time, paw_diff[t, :], color='darkgray')
    ax[t].set_xlim([10, 11])


window = 50
sta_paw_diff_mean = np.zeros((len(phase), 2*window))
for t in range(len(phase)):
    # find stance and swing points
    st_FR = find_peaks(FR[t, :])[0]
    sw_FR = find_peaks(-FR[t, :])[0]
    st_FL = find_peaks(FL[t, :])[0]
    sw_FL = find_peaks(-FL[t, :])[0]
    
    # simulate CS
    event_cs = st_FR
    
    # do STA
    sta_paw_diff = np.zeros((len(event_cs), 2*window))
    sta_paw_diff[:] = np.nan
    for count_e, e in enumerate(event_cs):
        if e > window and e < len(paw_diff[t, :])-window:
            sta_paw_diff[count_e, :] = paw_diff[t, :][e-window:e+window]
        
    sta_paw_diff_mean[t, :] = np.nanmean(sta_paw_diff, axis=0)

colors_trials = [colors_session[1], colors_session[7], colors_session[16], colors_session[17], colors_session[26]]
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for t in range(len(phase)):
    ax.plot(sta_paw_diff_mean[t, :], color=colors_trials[t])
ax.axvline(x = 50, color='black', linestyle='dashed')



 
