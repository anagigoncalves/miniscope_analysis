import os
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import hankel
import statsmodels.api as sm
warnings.filterwarnings('ignore')


def load_data(file_path, file_name):
    path = os.path.join(file_path, file_name)
    with open(path, 'rb') as file:
        return np.load(file, allow_pickle=True)

# Load data
file_path = 'C:\\Users\\User\\Carey Lab Dropbox\\Rotation Carey\\Francesco and Ana G\\Miniscope processed files\\GLM'    
neural_data = load_data(file_path, 'spike_trains.npy')
behavior_ts = load_data(file_path, 'timestamps_behavior.npy') 
body_kinematic = load_data(file_path, 'body_kinematic.npy') 
stride_phase = load_data(file_path, 'stride_phase.npy')

# Select data for one trial
trial = 4; trial_idx = 3
neural_data_trial = neural_data[neural_data['trial'] == trial]
spike_trains = neural_data_trial.iloc[:, 2:40] # use just n cells
trial_id = neural_data_trial['trial'].values
spikes_ts = neural_data_trial['time'].values
behav_ts = behavior_ts[trial_idx]
mapped_idx = [np.where(behav_ts == behav_ts[np.abs(behav_ts - t).argmin()])[0][0] for t in spikes_ts]
behav_ts = behav_ts[mapped_idx]
acceleration = body_kinematic['acceleration'][1][mapped_idx]
acceleration = (acceleration - np.nanmean(acceleration))/np.nanstd(acceleration)
speed = body_kinematic['speed'][1][mapped_idx]
speed = (speed - np.nanmean(speed))/np.nanstd(speed)
stride_phase_FR = stride_phase[1][0, 0, mapped_idx]

# Get some statistics
dt_stim = behav_ts[1] - behav_ts[0]
num_time_bins = acceleration.size # number of time bins in stimulus
num_cells = spike_trains.shape[1]
    
# # Visualize the spike-train auto and cross-correlations
# fig = plt.figure(figsize=[12,8])
# num_lags = 8 # number of time-lags to use 
# for i in range(num_cells):
#     for j in range(i,num_cells):
#         plt.subplot(num_cells, num_cells, i*num_cells+j+1)
#         plt.xcorr(spike_trains.iloc[:,i], spike_trains.iloc[:,j], maxlags=num_lags, usevlines=False, marker='.', linestyle='-')
#         plt.title(f'cells ({i},{j})')
#         plt.tight_layout()
# plt.xlabel('time shift (s)')


################## Build the design matrix ##################
# Let's work with the i cell for now
cell_idx = 7
spike_train = spike_trains.iloc[:,cell_idx].values

# Number of time bins of regressors to use for predicting spikes
ntfilt = 8 # regressors
nthist = 8 # spike history

# Design matrix for body kinematic
padded_acceleration = np.hstack((np.zeros(ntfilt-1), acceleration)) # pad early bins of stimulus with zero
design_mat_acceleration = hankel(padded_acceleration[:-ntfilt+1], acceleration[-ntfilt:])
# padded_speed = np.hstack((np.zeros(ntfilt-1), speed)) 
# design_mat_speed = hankel(padded_acceleration[:-ntfilt+1], speed[-ntfilt:])

# Design matrix for spike history and other neurons activity
design_mat_all_spikes = np.zeros((num_time_bins,nthist,num_cells)) # allocate space
for j in np.arange(num_cells): # Loop over neurons to build design matrix
    padded_spikes = np.hstack((np.zeros(nthist), spike_trains.iloc[:-1,j]))
    design_mat_all_spikes[:,:,j] = hankel(padded_spikes[:-nthist+1], padded_spikes[-nthist:])
design_mat_all_spikes = np.reshape(design_mat_all_spikes, (num_time_bins,-1), order='F') # Reshape it to be a single matrix

# Combine everything into a single design matrix
design_mat_all = np.concatenate((design_mat_acceleration, design_mat_all_spikes), axis=1) # full design matrix (with all 4 neuron spike hist)
design_mat_all_offset = np.hstack((np.ones((num_time_bins,1)), design_mat_all)) # add a column of ones


################## Fit autoregressive coupled GLM on a single neuron ##################
print('Now fitting Poisson GLM with spike-history and coupling...');
cGLM_model = sm.GLM(endog=spike_train, exog=design_mat_all_offset,
                         family=sm.families.Poisson()) # assumes 'log' link.
cGLM_results = cGLM_model.fit(max_iter=100, tol=1e-6, tol_criterion='params', method="lbfgs")

cGLM_const = cGLM_results.params[0]
cGLM_accel_filt = cGLM_results.params[1:ntfilt+1] # acceleration filter
# cGLM_speed_filt = cGLM_results.params[ntfilt+1:(2*ntfilt)+1] # speed filter
# cGLM_hist_filt = cGLM_results.params[(2*ntfilt)+1:] # all cells spike history filter
cGLM_hist_filt = cGLM_results.params[ntfilt+1:] # all cells spike history filter
cGLM_hist_filt = np.reshape(cGLM_hist_filt, (nthist,num_cells), order='F')


################## Plot results ##################
# Plot acceleration and speed filters
ttk = np.arange(-1*ntfilt+1,1)*dt_stim # time bins for stim filter
tth = np.arange(-1*nthist,0)*dt_stim # time bins for spike-history filter
plt.clf()
fig = plt.figure(figsize=[12,8])
plt.subplot(221)
plt.plot(ttk,cGLM_accel_filt, 'o-', c='blue', label='coupled-GLM', markerfacecolor='none')
# plt.plot(ttk,cGLM_speed_filt, 'o-', c='hotpink', label='coupled-GLM', markerfacecolor='none')
# plt.legend(['acceleration', 'speed'], loc='upper left')
plt.title(f'Body kinematic filter: cell {str(cell_idx+1)}')
plt.ylabel('weight');
plt.xlabel('time before spike (s)')

# Plot spike history filter
plt.subplot(222)
plt.plot(tth,tth*0,'k--')
for i in np.arange(num_cells):
    plt.plot(tth,cGLM_hist_filt[:,i], label='from ' + str(i+1))
# plt.legend(loc='upper left')
plt.title(f'coupling filters: into cell {str(cell_idx+1)}')
plt.xlabel('time before spike (s)')
plt.ylabel('weight')

# Compute and plot predicted spike rate on training data
rate_pred_all = np.exp(cGLM_const + design_mat_all @ cGLM_results.params[1:])
iiplot = np.arange(580)
ttplot = iiplot*dt_stim
plt.subplot(212)
markerline,_,_ = plt.stem(ttplot, spike_train[iiplot], linefmt='k-', basefmt='k-', label='spikes')
plt.setp(markerline, 'markerfacecolor', 'none')
plt.setp(markerline, 'markeredgecolor', 'k')
plt.plot(ttplot, rate_pred_all[iiplot], c='purple', label='coupled-GLM')
plt.legend(loc='upper left')
plt.xlabel('time (s)')
plt.title('spikes and rate predictions')
plt.ylabel('spike count / bin')
plt.tight_layout()
plt.show()


################## Model comparison: log-likelihoood and AIC ##################
# Compute loglikelihood (single-spike information) and AIC
LL_cGLM = spike_train.T @ np.log(rate_pred_all) - np.sum(rate_pred_all)

# log-likelihood for homogeneous Poisson model
spikes_bin_sum = np.sum(spike_train)
rate_pred_const = spikes_bin_sum/num_time_bins  # mean number of spikes / bin
LL0 = spikes_bin_sum * np.log(rate_pred_const) - num_time_bins * np.sum(rate_pred_const)

# Report single-spike information (bits / sp)
SSinfo_cGLM = (LL_cGLM - LL0)/spikes_bin_sum/np.log(2)

print('\n empirical single-spike information:\n ----------------------')
print(f'coupled-GLM: {SSinfo_cGLM:.2f} bits/sp')

# Compute AIC
AIC = -2*LL_cGLM + 2*(1+ntfilt+num_cells*nthist)

print('\n AIC:\n ----------------------')
print(f'coupled-GLM: {AIC:.1f}')