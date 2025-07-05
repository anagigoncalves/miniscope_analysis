import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from numpy.fft import fft, ifft
from scipy.signal import correlate


def detrend(X):
    for k in range(np.shape(X)[1]):
        h = X[:,k]
        b, a = signal.butter(2, .01)
        h = signal.filtfilt(b, a, h.T).T
        X[:,k] -= h
    return X


def periodic_corr(x, y, method='fft'):
    """Periodic correlation, implemented using the FFT or scipy 'correlate'.

    x and y must be real sequences with the same length.
    """
    if method == 'fft':
        return ifft(fft(x) * fft(y).conj()).real
    else:
        return np.correlate(x, np.hstack((y[1:], y)), mode='valid')


def findpeaks(array, peak_threshold=None, trough_threshold=None):
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


def interpolate(X, n_samples):
    X_interp = []
    for x in X:
        size = x.shape
        idx = np.linspace(0, size[0] - 1, num=n_samples)
        if len(size) == 1:
            x_interp = interp1d(np.arange(size[0]), x)(idx)
        else:
            x_interp = np.column_stack([interp1d(np.arange(size[0]), x[:, col])(idx) for col in range(size[1])])
        X_interp.append(x_interp)
    return X_interp


def image_saver(save_path, folder_name, file_name): ################## REMOVE
    if not os.path.exists(os.path.join(save_path, folder_name)):
        os.mkdir(os.path.join(save_path, folder_name))
    plt.savefig(os.path.join(save_path, folder_name + '\\', file_name + '.png'))


def load_data(directory, filename, folder=None): ######################## REMOVE
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    if folder:
        directory = os.path.join(directory, folder)
    else:
        directory = directory
    
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    path = os.path.join(directory, filename)
    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")
    
    data = np.load(path, allow_pickle=True)
    
    return data
    
    
def inpaint_nans(A):
    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)
    return A


def normalize(X, norm_method='z-score', ax=-1):
    if norm_method == 'z-score':
        mean = np.nanmean(X, axis=ax, keepdims=True)
        std_dev = np.nanstd(X, axis=ax, keepdims=True)
        X_norm = (X - mean) / std_dev
        
    elif norm_method == 'centering':
        mean = np.nanmean(X, axis=ax, keepdims=True)
        X_norm = X - mean
        
    elif norm_method == 'min-max':
        min_vals = np.nanmin(X, axis=ax, keepdims=True)
        max_vals = np.nanmax(X, axis=ax, keepdims=True)
        X_norm = (X - min_vals) / (max_vals - min_vals)
        
    elif norm_method == '-1to1':
        min_vals = np.nanmin(X, axis=ax, keepdims=True)
        max_vals = np.nanmax(X, axis=ax, keepdims=True)
        X_norm = 2*((X - min_vals) / (max_vals - min_vals)) - 1
        
    elif norm_method == 'max scaling':
        max_vals = np.nanmax(X, axis=ax, keepdims=True)
        X_norm = X / max_vals
        
    else:
        raise ValueError("Invalid normalization method. Choose from: 'z-score', 'centering', 'min-max', '-1to1', 'max scaling'")
    
    return X_norm


def map_timestamps(t1, t2):
    """
    Maps each timestamp in t1 to its closest timestamp in t2.
    
    Params:
        t1 (array): array of timestamps.
        t2 (array): array of timestamps to which t1 will be mapped.
    
    Returns:
        ndarray: array of indices indicating the closest timestamp in t2 for each timestamp in t1.
    """
    return np.array([np.where(t2 == t2[np.abs(t2 - t).argmin()])[0][0] for t in t1])


def sort_by(a, b):
    """
    Sorts `a` based on the values in `b`.
    
    Params:
    a (list): list to be sorted.
    b (array): array which determines the sorting order of `a`.
    
    Returns:
        - sorted_a (list): sorted `a`.
        - sorted_a (array): sorted `b`.
        - idx: indices of the original positions of the sorted elements.
    """
    if len(a) != len(b):
        raise ValueError("Input data must be of the same length.")
        
    combined = sorted(zip(a, b, range(len(b))), key=lambda x: x[1])
    a_sorted, b_sorted, idx_sorted = zip(*combined)
    
    return list(a_sorted), np.array(b_sorted), np.array(idx_sorted)


def shuffle_ts(ts, n_iter=1000):
    """
    Shuffle the given time series array randomly for a specified number of iterations.

    Parameters:
    - ts: numpy array, the original time series of shape (N,)
    - n_iter: int, number of times to shuffle the time series (default: 1000)

    Returns:
    - ts_shuffled: numpy array of shape (num_iterations, N), 
      where each row is a shuffled version of the time series
    """
    ts_shuffled = np.zeros((n_iter, ts.shape[0]), dtype=ts.dtype)
    for i in range(n_iter):
        shift = np.random.randint(6)
        ts_shuffled[i] = np.roll(ts, shift)
    return ts_shuffled


def select_kclust(data, max_clusters=20):
    # Elbow
    inertia = []
    silhouette_scores = [np.nan]
    K = range(1, max_clusters+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    fig, ax1 = plt.subplots(figsize=(6, 6))
    
    # Elbow
    ax1.plot(K, inertia, 'bo-', linewidth=2, c='darkviolet')
    ax1.set_xlabel('Cluster number', fontsize=20)
    ax1.set_ylabel('Inertia', fontsize=20, color='darkviolet')
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.spines['top'].set_visible(False)
    ax1.set_xticks([1, 5, 10, 15])
    
    # Silhouette
    ax2 = ax1.twinx()
    ax2.plot(K, silhouette_scores, 'go-', linewidth=2, c='g')
    ax2.set_ylabel('Silhouette score', fontsize=20, color='g')
    ax2.tick_params(axis='y', labelsize=15)
    ax2.spines['top'].set_visible(False)
    
    fig.tight_layout()
    plt.show()


def explode_dict(mydict):
    return [value for value in mydict.values()]