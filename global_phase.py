os.chdir('C:\\Users\\User\\Documents\\Phaser\\gait_signatures-main')


import fourierseries
import util
import phaser
import dataloader
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D


def find_phase(k):
    """
    Detrend and compute the phase estimate using Phaser
    INPUT:
      k -- dataframe
    OUTPUT:
      k -- dataframe
    """
    #l = ['hip_flexion_l','hip_flexion_r'] # Phase variables = hip flexion angles
    y = np.array(k)
    print(y.shape)
    y = util.detrend(y.T).T
    print(y.shape)
    phsr = phaser.Phaser(y=y)
    k[:] = phsr.phaserEval(y)[0,:]
    return k

def find_peaks_troughs(array, peak_threshold=None, trough_threshold=None):
    peaks = []
    troughs = []
    current_state = 'peak'  
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] > array[i + 1] and array[i] > peak_threshold:
            if current_state == 'trough':
                peaks.append(i)
                current_state = 'peak'
        elif array[i - 1] > array[i] < array[i + 1] and array[i] < trough_threshold:
            if current_state == 'peak':
                troughs.append(i)
                current_state = 'trough'
    return peaks, troughs

def interpolate_arrays(list_of_arrays, target_length=200):
    interpolated_arrays = []
    for array in list_of_arrays:
        # Assuming each array has shape Nx3
        N, _ = array.shape
        indices = np.linspace(0, N - 1, num=target_length)
        # Interpolate each column separately
        interpolated_columns = [interp1d(np.arange(N), array[:, i])(indices) for i in range(3)]
        # Combine the interpolated columns into an Mx3 array
        interpolated_array = np.column_stack(interpolated_columns)
        interpolated_arrays.append(interpolated_array)
    return interpolated_arrays

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for tr_idx, tr in enumerate(trials):
    global_phase = pc[f'{tr}']
    data = paw_displ.iloc[np.where(trial_id == tr)[0]].T
    phi = find_phase(data)
    
    phase= np.mod(phi.iloc[0, :], 2 * np.pi).values
    
    # plt.figure()
    # plt.plot(np.arange(len(phase[200:2000]-1))/330, phase[200:2000], c = 'dimgray')
    # plt.xlabel('Time (s)', fontsize = 18)
    # plt.ylabel('Global phase (rad)', fontsize = 18)
    
    # plt.figure()
    # plt.hist(diff(troughs)*(1/330), bins = 20, color = 'magenta')
    # plt.axvline(mean(diff(troughs)*(1/330)), color = 'b')
    # plt.xlabel('Cycle duration (s)', fontsize = 18)
    # plt.ylabel('Count', fontsize = 18)
    
    peaks, troughs = find_peaks_troughs(phase, peak_threshold=6.1, trough_threshold=0.3)
    # plt.figure()
    # plt.plot(phase)
    # plt.scatter(peaks, [phase[i] for i in peaks], color='red')
    # plt.scatter(troughs, [phase[i] for i in troughs], color='blue')
    # plt.show()
    
    cycles = []
    for i in range(len(peaks)):
        onset = troughs[i]
        offset = peaks[i]
        cycles.append(global_phase[onset:offset+1, :])
    
    interp_cycles = interpolate_arrays(cycles, 200)
    
    avg_traj = [np.mean([cycle[:, col] for cycle in interp_cycles], axis = 0) for col in range(3)]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # plt.plot(pc[f'{tr}'][:, 0], pc[f'{tr}'][:, 1], pc[f'{tr}'][:, 2], color = 'dimgray', alpha = 0.3)
    ax.plot(avg_traj[0], avg_traj[1], avg_traj[2], color = colors_session[tr], linewidth = 2)
    
    
