import numpy as np
import matplotlib.pyplot as plt
from statsmodels import robust
from scipy import signal, stats
from scipy.interpolate import interp1d
import math as mt

import SlopeThreshold as st

def Load_RawROI():
    import FileLoader as fl
    filepath = fl.OpenExplorer(filetype=[('NPY files',"*.npy")])
    return np.load(filepath)

def Detect_PosEvents_ROI(dff_signal, acq_fq, rtau, graph=None):
    # Function to detect the fast rising events in a fluorescent signal.
    # dff_signal: Normalized fluorescent signal in ΔF/F.
    # acq_fq: Sampling frequency of values in Hz.
    # rtau: Approximated tau constant with the rise of the signal (the fastest process, in seconds)
    # graph: If True, a plot will be made to visualize the thresholding process.

    # Find the most approximated noise amplitude estimation using the 'gaussian derivative' trick:
    TrueStd, deriv_mean, deriv_std = DerivGauss_NoiseEstim(dff_signal, thres=2, graph=0)

    # Use that noise amplitude to define regions of slope change / stability
    IncremSet, DecremSet, F_Values = st.SlopeThreshold(dff_signal, TrueStd*2, int(np.ceil(rtau*acq_fq)),
                                                       CollapSeq=False, acausal=True, graph=graph)

    Ev_Onset = list(map(lambda x : x[0], IncremSet)); Ev_ApproxPeak = list(map(lambda x : x[1], IncremSet))

    return Ev_Onset, Ev_ApproxPeak, TrueStd

def Estim_Baseline_PosEvents(rawdata, acq_fq, dtau=0.2, bmax_tslope=3, filtcut=None, graph=False):
    # Function to approximate a minimal baseline of a signal with positive rising events
    # using changepoint detection algorithms. Events are assumed to be very fast rising,
    # and possibly slow decaying.
    #
    # Inputs:
    # rawdata: Vector of raw values with the signals to detrend.
    # acq_freq: Sampling frequency of values in Hz.
    # dtau: Approximated tau constant with the decay of the signal (the slowest process, in seconds)
    # bmax_tslope: Time parameter to adjust the sensitivity of the baseline to positive slopes.
    #              Bigger values makes the baseline flatter when upward deflections occur.
    # filtcut: (Optional) If not 'None', its value defines the lowpass cut-off filtering of the Baseline.
    # graph: If True, a plot will be made to visualize the calculation process.
    #
    # Output:
    # Baseline estimation and Noise floor estimation.

    Time = np.arange(rawdata.size)/acq_fq

    # Find the most approximated noise amplitude estimation using the 'gaussian derivative' trick:
    TrueStd, deriv_mean, deriv_std = DerivGauss_NoiseEstim(rawdata, thres=2, graph=0)

    # Use that noise amplitude to define regions of slope change / stability
    IncremSet, DecremSet, F_Values = st.SlopeThreshold(rawdata, TrueStd*3, max(2, int(dtau*acq_fq)),
                                                       CollapSeq=False, acausal=True, graph=None)

    # Extract only regions without significant change in slope
    Mask = np.ones_like(F_Values).astype(int)
    for ini, end in IncremSet:
        Mask[ini:end+1] = 0
    for ini, end in DecremSet:
        Mask[ini:end+1] = 0

    # Extract the minimum value on stability regions of more than one point.
    Runs = find_runs(Mask, 1); MinPnts = []; SlopePnts = []; IdxRun= []; prev = 0
    for idx, inter in enumerate(Runs):
        if inter[1]-inter[0] == 1: continue # Remove single points
        pnt = np.argmin(rawdata[inter[0]:inter[1]]) + inter[0]
        slope = (rawdata[pnt]-rawdata[prev])/((pnt-prev)/acq_fq)
        MinPnts.append(pnt); SlopePnts.append(slope); IdxRun.append(idx)
        prev = pnt

    # Cleaning minima that, between pairs, have positive slopes too big to be baseline
    idx = 0; valid_num = len(MinPnts); MinSel = MinPnts.copy()
    while True:
        if SlopePnts[idx] > TrueStd*2/bmax_tslope:
            if idx == 0: prev = 0
            else: prev = MinSel[idx-1]
            if idx != len(MinSel) - 1:
                pnt = MinSel[idx+1]
                SlopePnts[idx+1] = (rawdata[pnt]-rawdata[prev])/((pnt-prev)/acq_fq)
            MinSel.pop(idx); SlopePnts.pop(idx); IdxRun.pop(idx)
            idx -= 1
        idx += 1
        if idx > len(MinSel) - 1:
            if valid_num != len(MinSel):
                valid_num = len(MinSel); idx = 0
            else:
                break

    # Get full lenght of points in regions selected by their minima:
    Concat = np.zeros(0).astype(int)
    for idx in IdxRun:
        Concat = np.concatenate((Concat, np.arange(Runs[idx][0], Runs[idx][1]).astype(int)))
    BaseTime = np.take(Time, Concat); BaseRaw = np.take(rawdata, Concat)

    # Fixing initial and ending boundaries:
    if Concat[0] != 0:
        ini_val = np.amin(rawdata[:Concat[0]+1])
        BaseTime = np.insert(BaseTime, 0, Time[0])
        BaseRaw = np.insert(BaseRaw, 0, ini_val)
    if Concat[-1] + 1 != rawdata.size:
        end_val = np.amin(rawdata[MinSel[len(MinSel)-1]:])
        BaseTime = np.append(BaseTime, Time[-1])
        BaseRaw = np.append(BaseRaw, end_val)

    # Interpolate all time points from template of minima, and smooth result with filter
    f = interp1d(BaseTime, BaseRaw, kind='linear', fill_value='extrapolate')
    Baseline = f(Time)
    if not filtcut is None:
        Baseline = BesselFilter(Baseline, [filtcut], order=4, type='lowpass', filter='filtfilt', Acq_Freq=acq_fq)

    if graph:
        Mask = Mask.astype(bool)
        F_valNan = F_Values.copy(); F_valNan[~Mask] = np.nan

        fig, axes = plt.subplots(nrows=3, num="Detrending", sharex=True)
        axes[0].set_ylabel("Raw Data\nand\nMinima Selection")
        axes[0].plot(Time, rawdata, color="gray")
        axes[0].plot(np.take(Time, MinPnts), np.take(rawdata, MinPnts), color="red", lw=0, marker="o", ms=10)
        axes[0].plot(np.take(Time, MinSel), np.take(rawdata, MinSel), color="blue", lw=0, marker="o", ms=10)
        axes[0].plot(Time, Baseline, color="green")
        axes[1].set_ylabel("F-Values &\nStability Selection")
        axes[1].plot(Time, F_Values, color="purple", alpha=0.25)
        axes[1].plot(Time, F_valNan, color="purple", lw=3)
        axes[2].set_ylabel("Detrended Signal\n(S - B + sigma*2)")
        axes[2].plot(Time, rawdata - Baseline + TrueStd*2, color="black")

        plt.show()

    return Baseline, TrueStd

def DerivGauss_NoiseEstim(data, thres=2):
    # Based on the assumption that the noise is iid, gaussian and stationary.
    # By differencing the data, it is possible to clean the features and keep only the noise.
    # Then, is possible to estimate the noise by finding the joint distribution of sequential
    # points which corresponds to the multiplication of the gaussians noise distribution
    # (by applying Bayes Theorem).
    # Validated by: https://stats.stackexchange.com/a/186545

    diff = np.diff(data)

    # Find outliers using MAD and neutralize them:
    MAD = robust.mad(diff, c=0.3)
    median = np.median(diff)
    condition = np.abs(diff-median)/MAD <= thres
    diff2 = np.extract(condition, diff)

    # Fit a normal distribution to the data and obtain the estimated standard deviation of the true noise
    deriv_mean, deriv_std = stats.norm.fit(diff2)
    TrueStd = mt.sqrt((deriv_std**2)/2) # Based on the product of two equal gaussians

    return TrueStd, deriv_mean, deriv_std

def BesselFilter(data, Wn, order=4, type='lowpass', filter='lfilter', Acq_Freq=0):
    # Function to filter datasets using a Bessel kernel
    # data: Array with the data to filter
    # Wn: List of parameters from 0 to 1 (circular displacement) to set the filter
    #     Note: changed if Acq_Freq is not default (see below)
    # order: The order of the filter
    # type: The type of filter, between 'lowpass', 'highpass', or 'bandpass'
    # Acq_Freq: Optional. Allows to put direct frequency values in the Wn
    #           parameter by using the frequency of acquisition of the data

    if Acq_Freq != 0:
        Wn = [i/(Acq_Freq/2) for i in Wn]

    if isinstance(Wn, list) or isinstance(Wn, tuple):
        Padlenght = data.size - int((1-Wn[0])*data.size)
    else:
        Padlenght = data.size - int((1-Wn)*data.size)

    Init = data[0]
    TempData = data - Init
    b, a = signal.bessel(order,Wn,type,analog=False)
    if filter == 'lfilter':
        return signal.lfilter(b, a, TempData) + Init
    elif filter == 'filtfilt':
        return signal.filtfilt(b, a, TempData, padtype='constant', padlen=Padlenght) + Init

def find_runs(data, elem):
    # Modified from https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array/24892274#24892274
    # data: 1-D numpy array
    # elem: The element repeated in the runs

    if np.isnan(elem):
        iselem = np.concatenate(([0], np.isnan(data).view(np.int8), [0]))
    elif np.isinf(elem):
        iselem = np.concatenate(([0], np.isinf(data).view(np.int8), [0]))
    else:
        iselem = np.concatenate(([0], np.equal(data, elem).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iselem))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def Interpolate_Runs(Arr, elem):
    # Function to replace sequences of repeated values in a 1-D numpy array
    # with a linear interpolation from points before and after
    # Arr: 1-D numpy array to adjust.
    # elem: The element repeated in the runs

    Data = Arr.copy()
    Ranges = find_runs(Data, elem)
    ErrorCheck = [False, False]
    for run in Ranges:
        lenght = run[1] - run[0]; PrevPt = run[0] - 1; AftPt = run[1]
        if PrevPt < 0:
            PrevPt = AftPt; ErrorCheck[0] = True
        if AftPt > Data.size - 1:
            AftPt = PrevPt; ErrorCheck[1] = True
        if all(ErrorCheck):
            raise ValueError('Interpolate_Runs Error -> All Data is a run sequence')
        PrevVal = Data[PrevPt]; AftVal = Data[AftPt]
        LinSlice = np.linspace(PrevVal, AftVal, lenght+2)
        if ErrorCheck[0]:
            LinSlice = np.delete(LinSlice, 0); PrevPt = run[0]
        if ErrorCheck[1]:
            LinSlice = np.delete(LinSlice, LinSlice.size-1); AftPt = run[1] - 1
        Data[PrevPt:AftPt+1] = LinSlice

    return Data

if __name__ == '__main__':
    rawdata = Load_RawROI() # LOAD HERE THE NPY DATA!
    acq_fq = 30 # in Hz

    # Estimate Baseline
    Baseline, Est_Std = Estim_Baseline_PosEvents(rawdata, acq_fq, dtau=0.2, bmax_tslope=3, filtcut=1, graph=False)

    # Calculate dF/F0:
    F0 = Baseline - Est_Std*2
    dff = (rawdata - F0) / F0
    dff = np.where(dff < 0, np.zeros_like(dff), dff) # Remove negative dff values

    Ev_Onset, Ev_ApproxPeak, dff_std = Detect_PosEvents_ROI(dff, acq_fq, rtau=0.02, graph=None)

    fig, axes = plt.subplots(nrows=2, num="Event Detection", sharex=True)
    Time = np.arange(rawdata.size)/acq_fq
    fig.suptitle("Global frequency of events: %.03f per second" % (len(Ev_Onset)/Time.max()))
    axes[0].set_ylabel("ΔF/F Signal")
    axes[0].plot(Time, dff, color="green", alpha=0.75, label="ΔF/F")
    axes[0].plot(Time, dff+dff_std*2, color="lightgreen", alpha=0.75, label="+Std")
    axes[0].plot(Time, dff-dff_std*2, color="lightgreen", alpha=0.75, label="-Std")
    axes[0].plot(np.take(Time, Ev_Onset), np.take(dff, Ev_Onset), color="blue", lw=0, marker=6, ms=7, label="Onsets")
    axes[0].plot(np.take(Time, Ev_ApproxPeak), np.take(dff, Ev_ApproxPeak), color="red", lw=0, marker=7, ms=7, label="Approx. Peaks")
    axes[1].set_ylabel("Raw Signal")
    axes[1].plot(Time, rawdata, color="black")
    axes[1].plot(np.take(Time, Ev_Onset), np.take(rawdata, Ev_Onset), color="blue", lw=0, marker=6, ms=7, label="Onsets")
    axes[1].plot(np.take(Time, Ev_ApproxPeak), np.take(rawdata, Ev_ApproxPeak), color="red", lw=0, marker=7, ms=7, label="Approx. Peaks")

    plt.show()
