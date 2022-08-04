import numpy as np
from numba import jit
import time
import operator
from scipy import signal
import NearestN as nn

@jit(nopython=True, cache=True)
def SlopeLoop(Data, PosEnvel, NegEnvel):

    # Calculating the derivative info about the Data signal for posterior use
    Diff = np.diff(Data)
    DiffAmpl = np.absolute(Diff)
    DiffSign = np.sign(Diff)

    F_Crossings = []
    numpnts = len(Data)
    movPnt = 0
    for refPnt in range(numpnts-1):
        ValRef = Data[refPnt]
        for movPnt in range(refPnt, numpnts-1):
            PosVal = PosEnvel[movPnt+1]
            NegVal =  NegEnvel[movPnt+1]
            Tvalue = movPnt-refPnt+1
            Self_Prox = Data[movPnt]
            CrossAmpl = np.absolute(ValRef - Self_Prox)
            SignCross = np.sign(ValRef - Self_Prox)
            SignDeriv = DiffSign[movPnt]
            DerivAmpl = DiffAmpl[movPnt]
            if SignCross == SignDeriv and DerivAmpl >= CrossAmpl: # In case the search intersects the original data itself
                F_Crossings.append((0, refPnt, 0))
                break
            elif ValRef > PosVal: # In case the search intersects the positive envelope, is a negative changepoint
                F_Crossings.append((-1, refPnt, Tvalue))
                break
            elif ValRef < NegVal: # In case the search intersects the negative envelope, is a positive changepoint
                F_Crossings.append((1, refPnt, Tvalue))
                break
            else:
                pass

    # Adding additional points to complete F_Values time serie up to the original's size.
    if len(F_Crossings) != 0:
        Delta = numpnts - len(F_Crossings)
        if Delta > 0:
            for i in range(Delta):
                F_Crossings.append((0, len(F_Crossings)-1, 0))

    return F_Crossings

def Translate_SlopeLoop_Output(F_Crossings):

    # Extract the arrays and complete the calculations on them
    extrc = lambda tupl:tupl[0]
    extrc1 = lambda tupl:tupl[1]
    extrc2 = lambda tupl:tupl[2]
    Val_Cross = np.asarray(list(map(extrc2, F_Crossings)))
    mask = (Val_Cross != 0)
    F_Times = np.asarray(list(map(extrc1, F_Crossings)))
    F_Sign = np.asarray(list(map(extrc, F_Crossings)))
    F_Values = np.zeros(np.size(F_Sign))

    # Calculate F-values
    np.divide(F_Sign, Val_Cross, out=F_Values, where=mask)

    return F_Values, F_Sign

def LocalCumSum(Array, reset_val = 0):

    LocalCumSum = Array.copy(); Prev = 0
    with np.nditer(LocalCumSum, op_flags=['readwrite']) as it:
        for x in it:
            if not x is reset_val:
                x += Prev; Prev = x
            else:
                Prev = 0

    return LocalCumSum

def Acausal_Integration(F_Sense, F_Anti):

    # Extract the F-Values of both sets
    F_Val_Sense, F_Sign = Translate_SlopeLoop_Output(F_Sense)
    F_Val_Anti, F_Sign = Translate_SlopeLoop_Output(F_Anti)

    F_Val_Anti = -np.flip(F_Val_Anti) # Transform the anti-sense values to match the sense

    # Extract the positive and negative portions of both detections
    Pos_Sense = np.where(F_Val_Sense > 0, F_Val_Sense, np.zeros(F_Val_Sense.size))
    Neg_Sense = np.where(F_Val_Sense < 0, F_Val_Sense, np.zeros(F_Val_Sense.size))
    Pos_Anti = np.where(F_Val_Anti > 0, F_Val_Anti, np.zeros(F_Val_Anti.size))
    Neg_Anti = np.where(F_Val_Anti < 0, F_Val_Anti, np.zeros(F_Val_Anti.size))

    # Join each corresponding portion of the F-Values
    if Pos_Sense.size == Pos_Anti.size:
        Pos_Join = np.where(Pos_Sense >= Pos_Anti, Pos_Sense, Pos_Anti)
    elif Pos_Sense.size == 0:
        Pos_Join = Pos_Anti
    elif Pos_Anti.size == 0:
        Pos_Join = Pos_Sense
    else:
        raise('Acausal_Integration Error -> Number mismatch of detected Sense and Antisense changepoints (Positives)')

    if Neg_Sense.size == Neg_Anti.size:
        Neg_Join = np.where(Neg_Sense <= Neg_Anti, Neg_Sense, Neg_Anti)
    elif Neg_Sense.size == 0:
        Neg_Join = Neg_Anti
    elif Neg_Anti.size == 0:
        Neg_Join = Neg_Sense
    else:
        raise('Acausal_Integration Error -> Number mismatch of detected Sense and Antisense changepoints (Negatives)')

    # # Calculate a cumulative-reset series to resolve conflicts when joining portions:
    # Pos_LocalCumSum = LocalCumSum(Pos_Join)
    # Neg_LocalCumSum = np.abs(LocalCumSum(Neg_Join))

    # Join the two portions into one single vector:
    #F_Vals_Joined = np.where(Pos_LocalCumSum >= Neg_LocalCumSum, Pos_Join, Neg_Join)
    F_Vals_Joined = np.where(np.abs(Pos_Join) >= np.abs(Neg_Join), Pos_Join, Neg_Join)

    return F_Vals_Joined

def SlopeThreshold(Data, AmplThres, TimePntThres, CollapSeq=True, acausal=False, verbose=0, graph=None):
    # Data: numpy array containing the y-data of the time series to be analyzed.
    # AmplThres: the minimal amplitude change considered as a valid event.
    #            Can be a numeric value, or an array with the same size as Data.
    # TimePntThres: the number of points that a change is expected to happen
    #               (used as a threshold to detect significant changes)
    # CollapSeq: (Optional) If False, the horizontal time threshold won't be used,
    #            so no sequential changes will be considered the same one.
    # acausal: (Optional) If True, two passes of the algorithm will be done with
    #          one in sense and other anti-sense of time, so that both onsets and
    #          offsets will be as accurate as possible.
    # verbose: (Optional) If verbose == 1, shows progress on F_value extraction.
    #          if verbose == 2, all messages of the process will be printed.
    # graph: (Optional) If not default, a plot showing the results will be shown.

    # In case the selected Time Threshold for the change is too low,
    # limit it to the Acq_Freq
    if TimePntThres < 0:
        TimePntThres = 1

    # Check the datatypes of the inputs and adjust them or abort the program
    try:
        if isinstance(AmplThres, float) or isinstance(AmplThres, int):
            PosEnvel = np.copy(Data)
            PosEnvel += np.float_(AmplThres)
            NegEnvel = np.copy(Data)
            NegEnvel -= np.float_(AmplThres)
        elif isinstance(AmplThres, np.ndarray) and np.size(AmplThres) == np.size(Data):
            if AmplThres.dtype == Data.dtype:
                PosEnvel = np.add(Data,AmplThres)
                NegEnvel = np.substract(Data, AmplThres)
            else:
                raise TypeError('SlopeThreshold Abort: AmplThres and Data arrays must have the same datatype')
    except:
        raise TypeError('SlopeThreshold Abort: AmplThres was not recognized as a valid datatype')

    if verbose > 0:
        print('SlopeThreshold START ---->')
        print('Calculating F values')

    start = time.time()

    if not acausal:
        F_Crossings = SlopeLoop(Data, PosEnvel, NegEnvel)
    else:
        F_Cross_Sense = SlopeLoop(Data, PosEnvel, NegEnvel)
        F_Cross_Antisense = SlopeLoop(np.flip(Data).copy(),
                                      np.flip(PosEnvel).copy(),
                                      np.flip(NegEnvel).copy())
        F_Crossings = Acausal_Integration(F_Cross_Sense, F_Cross_Antisense)


    end = time.time()
    if verbose > 0:
        print('Time elapsed: %.03f seconds' %(end - start))
        print('----------')
        print('Preparing the F values for changepoint extraction...')

    if len(F_Crossings) == 0:
        if verbose > 0:
            print('SlopeThreshold Alert: Any changepoint was detected in the data')
        return [], [], np.zeros_like(Data)

    if not acausal:
        JoinedPosSet, JoinedNegSet, F_Values =\
        F_val_Ch_detect(F_Crossings, TimePntThres, CollapSeq=CollapSeq, verbose=verbose)
        # JoinedPosSet, JoinedNegSet, F_Values = F_Val_Select(F_Crossings, TimePntThres, verbose=verbose)
    else:
        JoinedPosSet, JoinedNegSet, F_Values =\
        F_val_Ch_detect(F_Crossings, TimePntThres,
                        CollapSeq=CollapSeq, translate=False, verbose=verbose)

    if verbose > 0:
        print('Time threshold: %d points' % TimePntThres)
        print('Selected Positive Changes: %d' % len(JoinedPosSet))
        print('Selected Negative Changes: %d' % len(JoinedNegSet))
        print('----------')

    # Plot the analysis and its results if this option is selected.
    if not graph is None:
        if isinstance(graph, list):
            addgraphs = graph
        else:
            addgraphs = []
        Plot_SlopeTh_Output(Data, PosEnvel, NegEnvel,
                            JoinedPosSet, JoinedNegSet,
                            F_Values, addgraphs=addgraphs)

    return JoinedPosSet, JoinedNegSet, F_Values

def TS_ChPoints_SlopeThres(DataArray, TimeArray, AmplThres, TimeThres, Acq_Freq, acausal=False):
    # Wrapper of the slope threshold function to work with Time Series.
    # DataArray: Data values in a 1-D numpy vector.
    # TimeArray: Data time values in a 1-D numpy vector.
    # AmplThres: The amplitude threshold used to calculate the Slope Threshold.
    # TimeThres: The time threshold (in seconds) used for the Slope Threshold.
    # Acq_Freq: Sampling frequency of the time serie (in Hz)
    # acausal: (Optional) Perform the Slope Threshold two times, forward and backward,
    #          and select changepoints in a more direction-indepedent process. IN DEVELOPMENT!!!

    TimePntThres = int(TimeThres*Acq_Freq)

    JoinedPosSet, JoinedNegSet, F_Values = SlopeThreshold(DataArray, AmplThres, TimePntThres)
    if acausal:
        Rev_JoinedPosSet, Rev_JoinedNegSet, Rev_F_Values = SlopeThreshold(np.flip(DataArray).copy(), AmplThres, TimePntThres)
        Acausal_F = np.subtract(F_Values, Rev_F_Values).tolist()
        Acausal_F_Crossings = list(zip(np.sign(F_Values).tolist(), np.arange(F_Values.size).tolist(), Acausal_F))
        JoinedPosSet, JoinedNegSet, F_Values = F_Val_Select(Acausal_F_Crossings, TimePntThres)

    tupleextrac = lambda elem : elem[0]
    IncrPos = list(map(tupleextrac, JoinedPosSet))
    DecrPos = list(map(tupleextrac, JoinedNegSet))

    IncremPnts = np.stack((np.take(TimeArray, IncrPos), np.take(DataArray, IncrPos)), axis=1)
    DecremPnts = np.stack((np.take(TimeArray, DecrPos), np.take(DataArray, DecrPos)), axis=1)

    return IncremPnts, DecremPnts

def ExtremePairs_Fvals_Rejection(Trajectory, F_vals, AmpThres=0.99, WinThres=10):
    # Function to reject intervals with extreme slopes.
    # Trajectory: 1-D numpy array with the trajectory
    # F_vals: 1-D numpy array with the F-values calculated from the Slope Threshold procedure
    # AmpThres: Amplitude threshold that the extreme events surpass.
    # WinThres: Maximum window of points connecting extreme events with different polarity.

    # Detect the artifacts of the recordings by the amplitude of the change in the F-values
    PosExtrPnts = np.where(F_vals > AmpThres)[0].copy(); NegExtrPnts = np.where(-F_vals > AmpThres)[0].copy()

    if len(PosExtrPnts)*len(NegExtrPnts) > 100000:
        NumbaOptim = True
    else:
        NumbaOptim = False

    if PosExtrPnts.size > 0 and NegExtrPnts.size > 0:
        # Extract Rejections shifts (pairs of sudden changes of different polarity):
        # Pos2Neg_Pairs, Pos2Neg_Pnts =\
        # tl.Find_AproxPairs(PosExtrPnts, NegExtrPnts, oper='<', maxdelta=WinThres, mindelta=1)
        # Neg2Pos_Pairs, Neg2Pos_Pnts =\
        # tl.Find_AproxPairs(NegExtrPnts, PosExtrPnts, oper='<', maxdelta=WinThres, mindelta=1)
        PosNegDict = nn.NearestNeigbourgh_1D(PosExtrPnts, NegExtrPnts, type='<', MinMaxDiff=(1, WinThres), NumbaOptim=NumbaOptim)
        Pos2Neg_Pairs = PosNegDict['Pairs']; Pos2Neg_Pnts = PosNegDict['Pairs Points']
        NegPosDict = nn.NearestNeigbourgh_1D(NegExtrPnts, PosExtrPnts, type='<', MinMaxDiff=(1, WinThres), NumbaOptim=NumbaOptim)
        Neg2Pos_Pairs = NegPosDict['Pairs']; Neg2Pos_Pnts = NegPosDict['Pairs Points']

        # Join and sort the pairs based on their first point:
        extrc_Pnt = lambda elem: elem[0]
        All_Pairs = Pos2Neg_Pairs + Neg2Pos_Pairs; All_Pairs.sort(key=extrc_Pnt)

        # Detect artifacts from drifting away of local baselines.
        Traj_Rej = Baseline_based_Rejection(Trajectory, All_Pairs, basewin=5)
    else:
        Traj_Rej = Trajectory.copy()

    # Interpolate the rejected values.
    Mask = np.invert(np.isnan(Traj_Rej))
    if np.any(np.isnan(Traj_Rej)): NewTraj = Interpolate_Runs(Traj_Rej, np.nan)
    else: NewTraj = Traj_Rej.copy()

    return NewTraj, Mask

def Baseline_based_Rejection(Trajectory, RejPairs, basewin=5):
    # Function to remove points based on choosing a previous baseline.
    # Trajectory: 1-D numpy array of the trajectory
    # RejPairs: List of tuple pairs of sorted rejection shifts.

    TempTraj = Trajectory.copy()
    for pair in RejPairs:
        IniPnt = min(pair); SecPnt = max(pair)
        BaseList = []
        for n in reversed(range(0, IniPnt)):
            if not np.isnan(TempTraj[n]):
                BaseList.append(TempTraj[n])
            if len(BaseList) == basewin:
                break
        if len(BaseList) < 1:
            BaseList.append(np.nanmedian(TempTraj))
        RefPt = np.median(BaseList)
        Pt1 = TempTraj[IniPnt]; Pt2 = TempTraj[SecPnt]
        if not np.isnan(Pt1) and not np.isnan(Pt2):
            if np.abs(RefPt - Pt1) < np.abs(RefPt - Pt2):
                TempTraj[IniPnt+1:SecPnt+1] = np.nan
            elif np.abs(RefPt - Pt1) > np.abs(RefPt - Pt2):
                TempTraj[IniPnt] = np.nan
            else:
                print('Baseline_based_Rejection Error -> Undefined condition 1 met. Points %d and %d' % (IniPnt, SecPnt))
        elif np.isnan(Pt1):
            if np.abs(RefPt - TempTraj[SecPnt]) < np.abs(RefPt - TempTraj[SecPnt+1]):
                TempTraj[SecPnt+1] = np.nan
            elif np.abs(RefPt - TempTraj[SecPnt]) > np.abs(RefPt - TempTraj[SecPnt+1]):
                TempTraj[IniPnt+1:SecPnt+1] = np.nan
            elif np.isnan(TempTraj[SecPnt]) or np.isnan(TempTraj[SecPnt+1]):
                pass
            else:
                print('Baseline_based_Rejection Error -> Undefined condition 2 met. Points %d and %d' % (IniPnt, SecPnt))
        elif np.isnan(Pt2):
            if not all(np.isnan(TempTraj[IniPnt+1:SecPnt+1])):
                Level_Inter = np.nanmedian(TempTraj[IniPnt+1:SecPnt+1])
                if np.abs(RefPt - Level_Inter) < np.abs(RefPt - Pt1):
                    TempTraj[IniPnt+1] = np.nan
                elif np.abs(RefPt - Level_Inter) > np.abs(RefPt - Pt1):
                    TempTraj[IniPnt+1:SecPnt+1] = np.nan
                else:
                    print('Baseline_based_Rejection Error -> Undefined condition 3 met. Points %d and %d' % (IniPnt, SecPnt))

    return TempTraj

def TS_Reconstruct_from_Fvals(Traj, F_vals, AmplThres):
    # Function to make an approximated time series from calculated F values,
    # and allows to create a mask to see the changing values
    # Traj: Original Trajectory of the signal processed
    # F_vals: F-values of the calculated Slope threshold
    # AmplThres: The amplitude threshold used in the Slope threshold calculation

    # Iterate over F-values to reconstruct the time serires and extract the mask
    TS_Pnts = np.zeros(F_vals.size); Mask = np.zeros(F_vals.size).astype(bool)
    TempVals = F_vals * AmplThres
    Zero_Run = 0
    it = np.nditer(TempVals, flags=['f_index'])
    while not it.finished:
        if it[0] != 0:
            if Zero_Run > 0:
                TS_Pnts[it.index - Zero_Run:it.index] = Traj[it.index] #np.median(Traj[it.index - Zero_Run:it.index])
                Mask[it.index - Zero_Run:it.index] = True
                Zero_Run = 0
            if it.index > 0:
                TS_Pnts[it.index] += TS_Pnts[it.index-1] + it[0]
            else:
                TS_Pnts[it.index] = Traj[0]
        else:
            Zero_Run += 1
            if it.index == TempVals.size - 1:
                TS_Pnts[it.index - Zero_Run:it.index+1] = Traj[it.index] #np.median(Traj[it.index - Zero_Run:it.index+1])
                Mask[it.index - Zero_Run:it.index+1] = True
        it.iternext()

    # Make masked arrays:
    Recons_TS = np.ma.masked_where(Mask, np.ma.array(TS_Pnts))
    Changing_Traj = np.ma.masked_where(Mask, np.ma.array(Traj))
    Flat_Traj = np.ma.masked_where(np.invert(Mask), np.ma.array(Traj))

    # Adjust Reconstructed Time series with a roll needed to match the original after processing
    Recons_TS = np.roll(Recons_TS, 1)
    Recons_TS[0] = Recons_TS[1]

    # plt.plot(Changing_Traj, color='red', alpha=1)
    # plt.plot(Flat_Traj, color='grey', alpha=1)
    # plt.show()

    return Recons_TS, Changing_Traj, Flat_Traj, Mask

def Remove_pairs_containing_pnts(Pairs, PntPairs, TestPnts):
    # Pairs, PntPairs: Output from Find_AproxPairs function
    # TestPnts: List of points to test if they are inside a given interval of pairs

    ListPairs = list(zip(Pairs, PntPairs)); Temp = TestPnts.copy()
    for elem in reversed(range(len(ListPairs))):
        if len(Temp) == 0:
            break
        LowBound = ListPairs[elem][0][0]
        HighBound = ListPairs[elem][0][1]
        for idx in reversed(range(len(Temp))):
            if Temp[idx] > HighBound:
                continue
            elif LowBound <= Temp[idx] <= HighBound:
                ListPairs.pop(elem)
                Temp.pop(idx)
                break
            elif Temp[idx] < LowBound:
                break


    extrc_0 = lambda elem : elem[0]; extrc_1 = lambda elem : elem[1]
    NewPairs = list(map(extrc_0, ListPairs))
    NewPntsPairs = list(map(extrc_1, ListPairs))

    return NewPairs, NewPntsPairs

def Mark_intervals_from_Thres_cross(Data, Intervals, Thres):

    InterList = [elem.tolist() for elem in list(Intervals)]
    NewInterList = []

    for idx in range(len(InterList)):
        LowBound = InterList[idx][0]; HighBound = InterList[idx][1]
        if np.amax(np.abs(Data[LowBound:HighBound])) < Thres:
            NewInterList.append([LowBound, HighBound-1, False])
        else:
            NewInterList.append([LowBound, HighBound-1, True])

    return NewInterList

def Join_Valid_Onset_Offset(Pairs, Intervals):

    extrc_0 = lambda elem : elem[0]; extrc_1 = lambda elem : elem[1]
    OnsetTemp = list(map(extrc_0, Intervals))
    OffsetTemp = list(map(extrc_1, Intervals))

    JoinedSet = []; StartPoint = np.nan; Prev_p = 0; CurrEnd = np.nan
    if len(Intervals) > 0:
        idx = -1
        while idx < len(Intervals)- 1:
            idx += 1
            if not Intervals[idx][2]:
                continue

            if np.isnan(StartPoint):
                StartPoint = Intervals[idx][0]
                if len(JoinedSet) > 0:
                    condition1 = (JoinedSet[len(JoinedSet)-1][1] < np.asarray(OnsetTemp))
                    condition2 = (np.asarray(OnsetTemp) < StartPoint)
                    condition = condition1 & condition2
                    JumpedOnsets = np.where(condition)[0]
                    if JumpedOnsets.size > 0:
                        for ons in JumpedOnsets:
                            if Intervals[ons][2]:
                                JoinedSet.append((OnsetTemp[ons], OffsetTemp[ons]))

            if np.isnan(CurrEnd):
                CurrEnd = Intervals[idx][1]

            if len(Pairs) > 0:
                next_idx = idx
                for p in range(Prev_p, len(Pairs)):
                    if Pairs[p][0] == CurrEnd:
                        next_idx = OnsetTemp.index(Pairs[p][1])
                        CurrEnd = OffsetTemp[next_idx]
                    else:
                        Prev_p = p
                        next_idx = OffsetTemp.index(Pairs[p][0]) - 1
                        JoinedSet.append((StartPoint, CurrEnd))
                        StartPoint = np.nan; CurrEnd = np.nan
                        break
                idx = next_idx

                if p == len(Pairs) - 1 and not np.isnan(StartPoint):
                    JoinedSet.append((StartPoint, CurrEnd))
                    break

            else:
                JoinedSet.append((StartPoint, CurrEnd))
                StartPoint = np.nan; CurrEnd = np.nan

    if len(JoinedSet) > 0:
        condition = (JoinedSet[len(JoinedSet)-1][1] < np.asarray(OnsetTemp))
        TrailingOnsets = np.where(condition)[0]
        if TrailingOnsets.size > 0:
            for ons in TrailingOnsets:
                if Intervals[ons][2]:
                    JoinedSet.append((OnsetTemp[ons], OffsetTemp[ons]))

    return JoinedSet

def F_val_Ch_detect(F_Crossings, TimePntThres, CollapSeq=True, translate=True, verbose=0):

    # Find the pointwise detection of horizontal crossings:
    T_Thres = 1/TimePntThres

    if verbose > 1:
        print('F_val_Ch_detect => Translating the ouput of the SlopeLoop')
    # Translate the output of the SlopeLoop
    if translate:
        F_Values, F_Sign = Translate_SlopeLoop_Output(F_Crossings)
    else:
        F_Values = np.asarray(F_Crossings)
        F_Sign = np.sign(F_Values)

    OnlyPosSigns = np.where(F_Sign>0, F_Sign, np.zeros(F_Sign.size))
    OnlyNegSigns = np.where(F_Sign<0, -F_Sign, np.zeros(F_Sign.size))

    if verbose > 1:
        print('F_val_Ch_detect => Finding the continuos runs of Pos. or Neg. events')

    Pos_Intervals = find_runs(OnlyPosSigns, 1)
    Neg_Intervals = find_runs(OnlyNegSigns, 1)

    ### Time Vertical Threshold ###
    # Extract slope changes that surpass the time threshold criterion:

    if verbose > 1:
        print('F_val_Ch_detect => Applying Time Vertical Threshold')

    Pos_Intervals = Mark_intervals_from_Thres_cross(F_Values, Pos_Intervals, T_Thres)
    Neg_Intervals = Mark_intervals_from_Thres_cross(F_Values, Neg_Intervals, T_Thres)

    OnsetPosValid = np.asarray([elem[0] for elem in Pos_Intervals if elem[2]])
    OffsetPosValid = np.asarray([elem[1] for elem in Pos_Intervals if elem[2]])
    OnsetNegValid = np.asarray([elem[0] for elem in Neg_Intervals if elem[2]])
    OffsetNegValid = np.asarray([elem[1] for elem in Neg_Intervals if elem[2]])

    ### Time Horizontal Threshold ###
    # Extract sequences of changes of the same polarity that fulfill the criterions:
    # 1) A valid change starts after another valid change ends plus the time treshold.
    #    If not, the succesive changes are collapsed as a single one starting from the earliest.
    # 2) The valid change finishes always before an opposite change occurs (valid or not)
    #
    # If all changes are required to be detected, this horizontal threshold can be neglected

    if verbose > 1:
        print('F_val_Ch_detect => Applying Time Horizontal Threshold')

    if CollapSeq:

        if verbose > 1:
            print('\t\tCollapsing Sequences -> Finding Aproximate Pairs (NN)')
            verboseNN = True
        else:
            verboseNN = False

        # Criteria 1)
        PosDict = nn.NearestNeigbourgh_1D(OffsetPosValid.tolist(), OnsetPosValid.tolist(), type='<', MinMaxDiff=(1, TimePntThres), verbose=verboseNN)
        Pos_Pairs = PosDict['Pairs']; Pos_Pnts = PosDict['Pairs Points']
        NegDict = nn.NearestNeigbourgh_1D(OffsetNegValid.tolist(), OnsetNegValid.tolist(), type='<', MinMaxDiff=(1, TimePntThres), verbose=verboseNN)
        Neg_Pairs = NegDict['Pairs']; Neg_Pnts = NegDict['Pairs Points']
        # Pos_Pairs, Pos_Pnts = tl.Find_AproxPairs(OffsetPosValid, OnsetPosValid, oper='<',
        #                       maxdelta=TimePntThres, mindelta=1)
        # Neg_Pairs, Neg_Pnts = tl.Find_AproxPairs(OffsetNegValid, OnsetNegValid, oper='<',
        #                       maxdelta=TimePntThres, mindelta=1)

        # Criteria 2)
        OnsetTemp = [elem[0] for elem in Neg_Intervals]
        Pos_Pairs, Pos_Pnts = Remove_pairs_containing_pnts(Pos_Pairs, Pos_Pnts, OnsetTemp)
        OnsetTemp = [elem[0] for elem in Pos_Intervals]
        Neg_Pairs, Neg_Pnts = Remove_pairs_containing_pnts(Neg_Pairs, Neg_Pnts, OnsetTemp)

        if verbose > 1:
            print('\t\tJoining Onsets & Ends')

        # Totalize the two criterias and find valid onsets and offsets
        JoinedPosSet = Join_Valid_Onset_Offset(Pos_Pairs, Pos_Intervals)
        JoinedNegSet = Join_Valid_Onset_Offset(Neg_Pairs, Neg_Intervals)

    else:

        if verbose > 1:
            print('\t\tJoining Onsets & Ends')

        JoinedPosSet = Join_Valid_Onset_Offset([], Pos_Intervals)
        JoinedNegSet = Join_Valid_Onset_Offset([], Neg_Intervals)

    return JoinedPosSet, JoinedNegSet, F_Values

def F_Val_Select(F_Crossings, TimePntThres, verbose=0): # DEPRECATED. IN CASE OF BACKWARDS COMPATIBILITY

    # Find the pointwise detection of horizontal crossings:
    T_Thres = 1/TimePntThres

    # Translate the output of the SlopeLoop
    F_Values, F_Sign = Translate_SlopeLoop_Output(F_Crossings)

    # Find the leftmost/rightmost point on the series of changepoints with the same sign for exact onset/end detection.
    F_Deriv = np.diff(F_Sign)
    F_DerivRev = np.diff(np.flip(F_Sign))
    F_Deriv = np.insert(F_Deriv, 0, 0) * np.abs(F_Sign)
    F_DerivRev = np.flip(np.insert(F_DerivRev, 0, 0) * np.abs(np.flip(F_Sign)))
    EdgePnts = np.arange(np.size(F_Deriv))
    EdgePnts2 = np.arange(np.size(F_DerivRev))
    OnsetPnts = np.extract(F_Deriv != 0, EdgePnts)
    EndPnts = np.extract(F_DerivRev != 0, EdgePnts2)
    if np.abs(F_Values[0]) > 0:
        OnsetPnts = np.insert(OnsetPnts, 0, 0)
    if OnsetPnts[0] == 0 and EndPnts[0] == 0:
        OnsetPnts = np.delete(OnsetPnts, 0)
        EndPnts = np.delete(EndPnts, 0)
    OnsetValues = np.zeros(np.size(OnsetPnts))

    if verbose > 0:
        print('Total Onset Points: %d, Total End Points: %d' %(np.size(OnsetPnts), np.size(EndPnts)))
        print('----------')

    if np.size(OnsetPnts) == 0:
        if verbose > 0:
            print('SlopeThreshold Alert: Any changepoint was detected in the data')
        return [], [], []

    # Find the extrema of the changes in order to identify the series of changepoints.
    for i in range(0, np.size(EndPnts)):
        if F_Sign[OnsetPnts[i]] == 1:
            if OnsetPnts[i] != EndPnts[i]:
                ExtrChange = np.amax(F_Values[OnsetPnts[i]:EndPnts[i]])
            else:
                ExtrChange = np.amax(F_Values[OnsetPnts[i]])
        elif F_Sign[OnsetPnts[i]] == -1:
            if OnsetPnts[i] != EndPnts[i]:
                ExtrChange = np.amin(F_Values[OnsetPnts[i]:EndPnts[i]])
            else:
                ExtrChange = np.amin(F_Values[OnsetPnts[i]])
        OnsetValues[i] = ExtrChange

    # Extract changepoints given their operation
    PosChPnts = np.extract(OnsetValues > 0, OnsetPnts)
    NegChPnts = np.extract(OnsetValues < 0, OnsetPnts)

    if verbose == 2:
        print('Selecting the valid changes...')

    # Select the valid changepoints with the given threshold (vertical)
    Sel_PosChPnts = np.extract(OnsetValues >= T_Thres, OnsetPnts)
    Sel_NegChPnts = np.extract(-OnsetValues >= T_Thres, OnsetPnts)

    if len(Sel_PosChPnts) == 0 and len(Sel_NegChPnts) == 0:
        if verbose > 0:
            print('SlopeThreshold Alert: No changepoint above the vertical time threshold was detected in the data')
        return [], [], []

    # Select the valid changepoints with the given threshold (horizontal)
    # To begin, extract the sequential changes based on two exclusion principles:
    # 1) The valid change has to finish always before another valid change starts
    # 2) The valid change finishes always before an opposite change occurs (valid or not)
    ZipEndPnts = list(zip(EndPnts, [2]*len(EndPnts)))

    JoinedPosSet = []
    if len(Sel_PosChPnts) != 0:
        Sel_Pos_List = list(zip(Sel_PosChPnts, [0]*Sel_PosChPnts.size)) +\
        list(zip(NegChPnts, [1]*NegChPnts.size)) + ZipEndPnts
        Sel_Pos_List.sort(key = operator.itemgetter(0,1))

        ListElems = len(Sel_Pos_List)
        Ref_Sel_Pos = -1
        Ref_EndPnt = -1
        BetweenChange = -1
        for i in range(ListElems):
            RefType = Sel_Pos_List[i][1]
            if RefType == 0 and Ref_Sel_Pos == -1:
                Ref_Sel_Pos = Sel_Pos_List[i][0]
                continue
            elif i == ListElems - 1:
                if Ref_EndPnt != -1 and RefType == 2:
                    JoinedPosSet.append((Ref_Sel_Pos, Sel_Pos_List[i][0]))
            elif RefType == 2 and Ref_Sel_Pos != -1:
                Ref_EndPnt = Sel_Pos_List[i][0]
                continue
            elif RefType == 0 and Ref_EndPnt != -1:
                if Sel_Pos_List[i][0] - Ref_EndPnt > TimePntThres:
                    JoinedPosSet.append((Ref_Sel_Pos, Ref_EndPnt))
                    Ref_Sel_Pos = -1
                    Ref_EndPnt = -1
                continue
            elif RefType == 1 and Ref_EndPnt != -1:
                JoinedPosSet.append((Ref_Sel_Pos, Ref_EndPnt))
                Ref_Sel_Pos = -1
                Ref_EndPnt = -1
                continue

    JoinedNegSet = []
    if len(Sel_NegChPnts) != 0:
        Sel_Neg_List = list(zip(Sel_NegChPnts, [0]*Sel_NegChPnts.size)) +\
        list(zip(PosChPnts, [1]*PosChPnts.size)) + ZipEndPnts
        Sel_Neg_List.sort(key = operator.itemgetter(0,1))

        ListElems = len(Sel_Neg_List)
        Ref_Sel_Neg = -1
        Ref_EndPnt = -1
        for i in range(ListElems):
            RefType = Sel_Neg_List[i][1]
            if RefType == 0 and Ref_Sel_Neg == -1:
                Ref_Sel_Neg = Sel_Neg_List[i][0]
                continue
            elif i == ListElems - 1:
                if Ref_Sel_Neg != -1 and RefType == 2:
                    JoinedNegSet.append((Ref_Sel_Neg, Sel_Neg_List[i][0]))
            elif RefType == 2 and Ref_Sel_Neg != -1:
                Ref_EndPnt = Sel_Neg_List[i][0]
                continue
            elif RefType == 0 and Ref_EndPnt != -1:
                if Sel_Neg_List[i][0] - Ref_EndPnt > TimePntThres:
                    JoinedNegSet.append((Ref_Sel_Neg, Ref_EndPnt))
                    Ref_Sel_Neg = -1
                    Ref_EndPnt = -1
                continue
            elif RefType == 1 and Ref_EndPnt != -1:
                JoinedNegSet.append((Ref_Sel_Neg, Ref_EndPnt))
                Ref_Sel_Neg = -1
                Ref_EndPnt = -1
                continue

    return JoinedPosSet, JoinedNegSet, F_Values

def Plot_SlopeTh_Output(Data, PosEnvel, NegEnvel, JoinedPosSet, JoinedNegSet, F_Values, addgraphs=[]):
    # addgraphs: List of graphs to add that contain elements in the follwing structure:
    #            [numpy.array(2-D), color string, alpha value, zorder value]

    import matplotlib.pyplot as plt
    from matplotlib.widgets import MultiCursor
    from mplcursors import cursor

    tuple_extrc = lambda pair:pair[0]
    tuple_extrc2 = lambda pair:pair[1]

    Sel_PosChPnts = list(map(tuple_extrc, JoinedPosSet))
    PosSelEnds = list(map(tuple_extrc2, JoinedPosSet))
    Sel_NegChPnts = list(map(tuple_extrc, JoinedNegSet))
    NegSelEnds = list(map(tuple_extrc2, JoinedNegSet))

    plt.close('Slope Thres. Visualization')
    fig, (ax1, ax2) = plt.subplots(num='Slope Thres. Visualization', nrows=2, sharex=True, figsize=(16,12), dpi=80, facecolor='w', edgecolor='k')
    ax1.set_title('Visualization in Normal Ordinates')
    ax2.set_title('Visualization in Transformed Ordinates')
    fig.canvas.set_window_title('Full Slope Threshold Visualization')
    x = np.arange(np.size(PosEnvel))

    if len(addgraphs) > 0:
        minExtra = 0; maxExtra = 0; count = -1
        for graph in addgraphs:
            count +=1
            ax1.plot(graph[0][:,0], graph[0][:,1], color=graph[1],
                     alpha=graph[2], zorder=graph[3], label=('graph_%d' % count))
            if minExtra > np.amin(graph[0][:,1]): minExtra = np.amin(graph[0][:,1])
            if maxExtra < np.amax(graph[0][:,1]): maxExtra = np.amax(graph[0][:,1])

    Original, = ax1.plot(x, Data, color='black', alpha=0.5, linewidth=3,
                         zorder=0.8, label='Data+noise')
    ax1.plot(x, PosEnvel, color='orange', linestyle='dashed',
             zorder=0.2, label='Envelope_Pos')
    ax1.plot(x, NegEnvel, color='orange', linestyle='dashed',
             zorder=0.2, label='Envelope_Neg')
    Sel_PosChPnts_Vals = np.take(Data, Sel_PosChPnts)
    ax1.plot(Sel_PosChPnts, Sel_PosChPnts_Vals, marker='<', markersize=5, color='red',
             linewidth=0, alpha=0.75,zorder=0.9, label='Pos_onsets')
    Sel_NegChPnts_Vals = np.take(Data, Sel_NegChPnts)
    ax1.plot(Sel_NegChPnts, Sel_NegChPnts_Vals, marker='<', markersize=5, color='blue',
             linewidth=0, alpha=0.75,zorder=0.9, label='Neg_onsets')
    PosSelEnds_Vals = np.take(Data, PosSelEnds)
    ax1.plot(PosSelEnds, PosSelEnds_Vals, marker='>', markersize=5, color='red',
             linewidth=0, alpha=0.75,zorder=0.9, label='Pos_offsets')
    NegSelEnds_Vals = np.take(Data, NegSelEnds)
    ax1.plot(NegSelEnds, NegSelEnds_Vals, marker='>', markersize=5, color='blue',
             linewidth=0, alpha=0.75,zorder=0.9, label='Neg_offsets')

    RangeFvals = np.amax(F_Values) - np.amin(F_Values)
    diffData = Normalize(np.append(np.diff(Data), 0),
               min=np.amin(F_Values), max=np.amax(F_Values)) +\
               RangeFvals + RangeFvals/20
    ax2.plot(x, diffData, color='green', linewidth=3, zorder=0.5, label='Derivative')
    FVals, = ax2.plot(x, F_Values, color='purple', linewidth=3,
                      zorder=0.8, label='F-values')

    multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1, horizOn=False)
    dc1 = cursor([Original, FVals])

    @dc1.connect("add")
    def on_add(sel):
        global Xpoint, Ypoint
        Location = sel.target
        Xpoint = Location[0]
        Ypoint = Location[1]

    @dc1.connect('add')
    def _(sel):
        str = 'X: %.3f\nY: %.03f' % (Xpoint, Ypoint)
        sel.annotation.set_text(str)
        sel.annotation.arrow_patch.set(arrowstyle="simple", fc="orange", alpha=1)

    plt.show()

def Normalize(data, min=0, max=1):

    if data.size < 2:
        return data

    datamin = np.nanmin(data)
    datamax = np.nanmax(data)
    norm = np.zeros(np.size(data))
    if datamin == datamax:
        return norm
    temp = data[~np.isnan(data)]
    norm = data.copy()
    norm[~np.isnan(data)] = ((max-min)*(temp - datamin)/(datamax - datamin))+min

    return norm

def find_runs(data, elem):
    # Modified from https://stackoverflow.com/questions/24885092
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

def spTest(AmplThres, TimePntThres, MaxNoiseAmpl, seed=1, type='sine', noise='norm', filter=0, acausal=False, CollapSeq=True):
    from scipy import signal
    import Filterolls as fr

    np.seterr(all='warn')

    T = 1000
    Mult = 8
    np.random.seed(seed)

    if isinstance(type, str):
        y, x = TestData(type, T, Mult)
    elif isinstance(type, np.ndarray):
        y = type
        x = np.arange(0,np.size(y))
    elif isinstance(type, list) or isinstance(type, tuple):
        y = np.asarray(type)
        x = np.arange(0,np.size(y))
    else:
        print('No datatype recognized. Using default: ' + type)
        y= np.sin(Mult*np.pi*x/T)

    if np.size(y) < 3:
        print('Test can not be done with the data passed to the function')
        return

    if type != 'data':
        if noise == 'norm':
            y_noise = np.random.normal(y, MaxNoiseAmpl)
        elif noise == 'lognorm':
            y_noise = y + np.random.lognormal(0, MaxNoiseAmpl, np.size(y)) -1
        elif noise == 'uniform':
            y_noise = y + np.random.uniform(high=MaxNoiseAmpl, size=np.size(y)) - (MaxNoiseAmpl/2)
        else:
            print('Noise type not recognized. Using normal (gaussian) noise')
            y_noise = np.random.normal(y, MaxNoiseAmpl)
    else:
        y_noise = y

    if filter != 0:
        y_final = fr.BesselFilter(y_noise, [filter], order=4, type='lowpass', filter='filtfilt', Acq_Freq=1)
    else:
        y_final = y_noise

    # try:
    #     noise_std, yout, FiltWin = fr.NoiseEstimation(y) #autofilt=filter)
    # except ValueError:
    #     print('fr.WienerOptimize Abort: No detected features to extract in the input data')
    #     print('Stopping Test')
    #     return
    #
    # print('Wiener Filter window = ' + str(FiltWin))
    # print('----------')

    addgraphs = []
    addgraphs.append([np.stack((np.arange(y_noise.size), y_noise), axis=1), 'grey', 0.5, 0.1])
    addgraphs.append([np.stack((np.arange(y.size), y), axis=1), 'black', 1, 2])

    SlopeThreshold(y_final, AmplThres, TimePntThres,
                   CollapSeq=CollapSeq, acausal=acausal,
                   verbose=True, graph=addgraphs)

def TestData(type, T, Mult):

    spike = [ 0.33447684,  0.30703258,  0.29674099,  0.29331045,  0.31389365,
    0.20754716,  0.15265865,  0.01886792,  0.00171526, -0.03259006,
   -0.11492282, -0.43396227, -1.        , -0.97598628, -0.0806175 ,
    0.57118353,  1.        ,  0.90737564,  1.        ,  0.73584905,
    0.50257289,  0.28644939,  0.39622641,  0.33447684,  0.28987992,
    0.31046311]
    spike = np.asarray(spike)

    x = np.arange(0,T)

    if type == 'sine':
        y= np.sin(Mult*np.pi*x/T)
    elif type == 'square':
        y = signal.square(Mult*np.pi*x/T)
    elif type == 'sawtooth':
        y = signal.sawtooth(Mult*np.pi*x/T)
    elif type == 'isawtooth':
        y = signal.sawtooth(Mult*np.pi*x/T)
        y = np.flip(y)
    elif type == 'peaks':
        y = np.zeros(T)
        for i in range(int(T/4)-1, T, int(T/4)-1):
            y[i] = -1
            y[i+1] = 1
    elif type == 'x0':
        y = np.zeros(T)
    elif type == 'x1':
        y = np.zeros(T)
        it = np.nditer(x, flags=['f_index'])
        while not it.finished:
            y[it.index] = (it[0]/T) - 0.5
            it.iternext()
    elif type == 'x0+x1':
        y = np.zeros(T)
        it = np.nditer(x, flags=['f_index'])
        while not it.finished:
            if it.index > T/2:
                y[it.index] = 2*(it[0]/T) - 1
            it.iternext()
    elif type == 'x1+x0+x1':
        y = np.zeros(T)
        it = np.nditer(x, flags=['f_index'])
        while not it.finished:
            if it.index < T/3:
                y[it.index] = 2*(it[0]/T) - (2/3)
            elif it.index > 2*T/3:
                y[it.index] = 2*(it[0]/T) - (4/3)
            it.iternext()
    elif type == 'x2':
        y = np.zeros(T)
        it = np.nditer(x, flags=['f_index'])
        while not it.finished:
            y[it.index] = (it[0]/T)*(it[0]**2)
            it.iternext()
        y = Normalize(y)
    elif type == 'spikes':
        Acq_Freq = 25000
        Sep = int(T/10)-1
        Size = int(np.size(spike)/2)
        Scale = [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.075]
        y = np.zeros(T)
        for i in range(Sep, T-Sep, Sep):
            Turn = int(i/Sep)-1
            ScaledSpike = spike*Scale[Turn]
            Init = ScaledSpike[0]
            ScaledSpike -= Init
            y[i-Size:i+Size] = ScaledSpike
    elif type == 'spikes2':
        Acq_Freq = 25000
        Sep = [0, 5, 10, 25, 50, 100]
        Size = int(np.size(spike)/2)
        ScaledSpike = spike - spike[0]
        y = np.zeros(T)
        LastEnd = 0
        for indx, elem in enumerate(Sep):
            i = Size * (indx+1) + LastEnd
            y[i-Size:i+Size] = ScaledSpike
            y[i+elem+Size:i+elem+3*Size] = ScaledSpike
            LastEnd = i+elem+3*Size
    elif type == 'simul':
        y =np.asarray([6.51052533e-02, 6.45239950e-02, 6.38948755e-02, 6.32187646e-02,
                       6.24971066e-02, 6.17318872e-02, 6.09255813e-02, 6.00810860e-02,
                       5.92016418e-02, 5.82907480e-02, 5.73520773e-02, 5.63893943e-02,
                       5.54064819e-02, 5.44070785e-02, 5.33948276e-02, 5.23732379e-02,
                       5.13456538e-02, 5.03152343e-02, 4.92849367e-02, 4.82575042e-02,
                       4.72354556e-02, 4.62210762e-02, 4.52164099e-02, 4.42232537e-02,
                       4.32431556e-02, 4.22774164e-02, 4.13270980e-02, 4.03930364e-02,
                       3.94758592e-02, 3.85760067e-02, 3.76937509e-02, 3.68292133e-02,
                       3.59823769e-02, 3.51530944e-02, 3.43410911e-02, 3.35459668e-02,
                       3.27671989e-02, 3.20041500e-02, 3.12560810e-02, 3.05221721e-02,
                       2.98015498e-02, 2.90933190e-02, 2.83965965e-02, 2.77105439e-02,
                       2.70343969e-02, 2.63674878e-02, 2.57092607e-02, 2.50592788e-02,
                       2.44172250e-02, 2.37828984e-02, 2.31562084e-02, 2.25371707e-02,
                       2.19259060e-02, 2.13226436e-02, 2.07277275e-02, 2.01416244e-02,
                       1.95649294e-02, 1.89983657e-02, 1.84427751e-02, 1.78990952e-02,
                       1.73683227e-02, 1.68514622e-02, 1.63494620e-02, 1.58631427e-02,
                       1.53931220e-02, 1.49397455e-02, 1.45030275e-02, 1.40826104e-02,
                       1.36777452e-02, 1.32872966e-02, 1.29097715e-02, 1.25433692e-02,
                       1.21860502e-02, 1.18356188e-02, 1.14898147e-02, 1.11464096e-02,
                       1.08033031e-02, 1.04586134e-02, 1.01107595e-02, 9.75852901e-03,
                       9.40112991e-03, 9.03822251e-03, 8.66993089e-03, 8.29683361e-03,
                       7.91993573e-03, 7.54062538e-03, 7.16062024e-03, 6.78191029e-03,
                       6.40670463e-03, 6.03739059e-03, 5.67651317e-03, 5.32678226e-03,
                       4.99111292e-03, 4.67270248e-03, 4.37514485e-03, 4.10258062e-03,
                       3.85987888e-03, 3.65284503e-03, 3.48844752e-03, 3.37505544e-03,
                       3.32267805e-03, 3.34319687e-03, 3.45058027e-03, 3.66107014e-03,
                       3.99332991e-03, 4.46854301e-03, 5.11045094e-03, 5.94532077e-03,
                       7.00183245e-03, 8.31087818e-03, 9.90526771e-03, 1.18193366e-02,
                       1.40884577e-02, 1.67484591e-02, 1.98349588e-02, 2.33826266e-02,
                       2.74243914e-02, 3.19906151e-02, 3.71082568e-02, 4.28000559e-02,
                       4.90837655e-02, 5.59714697e-02, 6.34690203e-02, 7.15756275e-02,
                       8.02836352e-02, 8.95785037e-02, 9.94390094e-02, 1.09837657e-01,
                       1.20741283e-01, 1.32111811e-01, 1.43907130e-01, 1.56082026e-01,
                       1.68589151e-01, 1.81379969e-01, 1.94405660e-01, 2.07617944e-01,
                       2.20969825e-01, 2.34416232e-01, 2.47914549e-01, 2.61425046e-01,
                       2.74911206e-01, 2.88339943e-01, 3.01681737e-01, 3.14910676e-01,
                       3.28004417e-01, 3.40944079e-01, 3.53714075e-01, 3.66301889e-01,
                       3.78697815e-01, 3.90894669e-01, 4.02887476e-01, 4.14673155e-01,
                       4.26250210e-01, 4.37618427e-01, 4.48778605e-01, 4.59732317e-01,
                       4.70481703e-01, 4.81029310e-01, 4.91377963e-01, 5.01530680e-01,
                       5.11490612e-01, 5.21261014e-01, 5.30845237e-01, 5.40246730e-01,
                       5.49469059e-01, 5.58515921e-01, 5.67391157e-01, 5.76098741e-01,
                       5.84642765e-01, 5.93027392e-01, 6.01256792e-01, 6.09335057e-01,
                       6.17266113e-01, 6.25053613e-01, 6.32700856e-01, 6.40210707e-01,
                       6.47585549e-01, 6.54827267e-01, 6.61937264e-01, 6.68916509e-01,
                       6.75765623e-01, 6.82484980e-01, 6.89074820e-01, 6.95535373e-01,
                       7.01866961e-01, 7.08070100e-01, 7.14145566e-01, 7.20094441e-01,
                       7.25918130e-01, 7.31618358e-01, 7.37197139e-01, 7.42656746e-01,
                       7.47999657e-01, 7.53228506e-01, 7.58346040e-01, 7.63355072e-01,
                       7.68258440e-01, 7.73058976e-01, 7.77759475e-01, 7.82362674e-01,
                       7.86871226e-01, 7.91287687e-01, 7.95614500e-01, 7.99853984e-01,
                       8.04008326e-01, 8.08079577e-01, 8.12069644e-01, 8.15980294e-01,
                       8.19813156e-01, 8.23569724e-01, 8.27251369e-01, 8.30859350e-01,
                       8.34394832e-01, 8.37858906e-01, 8.41252611e-01, 8.44576961e-01,
                       8.47832972e-01, 8.51021687e-01, 8.54144196e-01, 8.57201661e-01,
                       8.60195320e-01, 8.63126498e-01, 8.65996599e-01, 8.68807093e-01,
                       8.71559501e-01, 8.74255370e-01, 8.76896250e-01, 8.79483670e-01,
                       8.82019119e-01, 8.84504027e-01, 8.86939755e-01, 8.89327585e-01,
                       8.91668720e-01, 8.93964277e-01, 8.96215294e-01, 8.98422730e-01,
                       9.00587477e-01, 9.02710363e-01, 9.04792164e-01, 9.06833619e-01,
                       9.08835435e-01, 9.10798301e-01, 9.12722897e-01, 9.14609899e-01,
                       9.16459984e-01, 9.18273834e-01, 9.20052137e-01, 9.21795586e-01,
                       9.23504885e-01, 9.25180744e-01, 9.26823884e-01, 9.28435031e-01,
                       9.30014921e-01, 9.31564291e-01, 9.33083876e-01, 9.34574404e-01,
                       9.36036580e-01, 9.37471081e-01, 9.38878537e-01, 9.40259522e-01,
                       9.41614539e-01, 9.42944006e-01, 9.44248251e-01, 9.45527508e-01,
                       9.46781921e-01, 9.48011554e-01, 9.49216415e-01, 9.50396477e-01,
                       9.51551714e-01, 9.52682132e-01, 9.53787802e-01, 9.54868883e-01,
                       9.55925638e-01, 9.56958439e-01, 9.57967761e-01, 9.58954174e-01,
                       9.59918319e-01, 9.60860897e-01, 9.61782646e-01, 9.62684324e-01,
                       9.63566694e-01, 9.64430503e-01, 9.65276466e-01, 9.66105252e-01,
                       9.66917464e-01, 9.67713633e-01, 9.68494211e-01, 9.69259582e-01,
                       9.70010069e-01, 9.70745956e-01, 9.71467510e-01, 9.72174996e-01,
                       9.72868688e-01, 9.73548876e-01, 9.74215862e-01, 9.74869948e-01,
                       9.75511427e-01, 9.76140577e-01, 9.76757654e-01, 9.77362900e-01,
                       9.77956552e-01, 9.78538859e-01, 9.79110098e-01, 9.79670590e-01,
                       9.80220711e-01, 9.80760894e-01, 9.81291615e-01, 9.81813371e-01,
                       9.82326645e-01, 9.82831860e-01, 9.83329330e-01, 9.83819209e-01,
                       9.84301448e-01, 9.84775766e-01, 9.85241633e-01, 9.85698280e-01,
                       9.86144727e-01, 9.86579841e-01, 9.87002406e-01, 9.87411210e-01,
                       9.87805146e-01, 9.88183304e-01, 9.88545060e-01, 9.88890150e-01,
                       9.89218731e-01, 9.89531410e-01, 9.89829261e-01, 9.90113814e-01,
                       9.90387026e-01, 9.90651231e-01, 9.90909075e-01, 9.91163435e-01,
                       9.91417329e-01, 9.91673812e-01, 9.91935869e-01, 9.92206298e-01,
                       9.92487595e-01, 9.92781833e-01, 9.93090550e-01, 9.93414630e-01,
                       9.93754203e-01, 9.94108539e-01, 9.94475961e-01, 9.94853763e-01,
                       9.95238129e-01, 9.95624073e-01, 9.96005363e-01, 9.96374462e-01,
                       9.96722453e-01, 9.97038964e-01, 9.97312076e-01, 9.97528234e-01,
                       9.97672133e-01, 9.97726615e-01, 9.97672556e-01, 9.97488780e-01,
                       9.97151985e-01, 9.96636720e-01, 9.95915401e-01, 9.94958406e-01,
                       9.93734235e-01, 9.92209759e-01, 9.90350563e-01, 9.88121372e-01,
                       9.85486585e-01, 9.82410877e-01, 9.78859881e-01, 9.74800932e-01,
                       9.70203838e-01, 9.65041663e-01, 9.59291499e-01, 9.52935176e-01,
                       9.45959900e-01, 9.38358775e-01, 9.30131186e-01, 9.21283027e-01,
                       9.11826751e-01, 9.01781254e-01, 8.91171588e-01, 8.80028530e-01,
                       8.68388022e-01, 8.56290517e-01, 8.43780241e-01, 8.30904401e-01,
                       8.17712369e-01, 8.04254837e-01, 7.90582988e-01, 7.76747685e-01,
                       7.62798707e-01, 7.48784050e-01, 7.34749314e-01, 7.20737189e-01,
                       7.06787046e-01, 6.92934646e-01, 6.79211946e-01, 6.65646996e-01,
                       6.52263935e-01, 6.39083042e-01, 6.26120868e-01, 6.13390411e-01,
                       6.00901344e-01, 5.88660289e-01, 5.76671110e-01, 5.64935252e-01,
                       5.53452077e-01, 5.42219223e-01, 5.31232950e-01, 5.20488473e-01,
                       5.09980266e-01, 4.99702337e-01, 4.89648455e-01, 4.79812328e-01,
                       4.70187744e-01, 4.60768656e-01, 4.51549242e-01, 4.42523921e-01,
                       4.33687354e-01, 4.25034425e-01, 4.16560219e-01, 4.08259989e-01,
                       4.00129134e-01, 3.92163174e-01, 3.84357740e-01, 3.76708567e-01,
                       3.69211488e-01, 3.61862444e-01, 3.54657482e-01, 3.47592770e-01,
                       3.40664594e-01, 3.33869365e-01, 3.27203623e-01, 3.20664043e-01,
                       3.14247436e-01, 3.07950763e-01, 3.01771147e-01, 2.95705882e-01,
                       2.89752447e-01, 2.83908517e-01, 2.78171961e-01, 2.72540836e-01,
                       2.67013375e-01, 2.61587961e-01, 2.56263093e-01, 2.51037346e-01,
                       2.45909332e-01, 2.40877647e-01, 2.35940830e-01, 2.31097320e-01,
                       2.26345415e-01, 2.21683244e-01, 2.17108741e-01, 2.12619637e-01,
                       2.08213448e-01, 2.03887491e-01, 1.99638898e-01, 1.95464651e-01,
                       1.91361619e-01, 1.87326615e-01, 1.83356455e-01, 1.79448033e-01,
                       1.75598391e-01, 1.71804796e-01, 1.68064818e-01, 1.64376400e-01,
                       1.60737919e-01, 1.57148238e-01, 1.53606749e-01, 1.50113394e-01,
                       1.46668676e-01, 1.43273655e-01, 1.39929921e-01, 1.36639556e-01,
                       1.33405078e-01, 1.30229373e-01, 1.27115603e-01, 1.24067117e-01,
                       1.21087341e-01, 1.18179662e-01, 1.15347309e-01, 1.12593235e-01,
                       1.09919997e-01, 1.07329647e-01, 1.04823636e-01, 1.02402732e-01,
                       1.00066961e-01, 9.78155704e-02, 9.56470241e-02, 9.35590195e-02,
                       9.15485375e-02, 8.96119162e-02, 8.77449493e-02, 8.59430017e-02,
                       8.42011395e-02, 8.25142647e-02, 8.08772507e-02, 7.92850685e-02,
                       7.77329002e-02, 7.62162339e-02, 7.47309370e-02, 7.32733077e-02,
                       7.18401036e-02, 7.04285520e-02, 6.90363421e-02, 6.76616039e-02,
                       6.63028774e-02, 6.49590750e-02, 6.36294406e-02, 6.23135077e-02,
                       6.10110594e-02, 5.97220903e-02, 5.84467722e-02, 5.71854233e-02,
                       5.59384799e-02, 5.47064702e-02, 5.34899886e-02, 5.22896705e-02,
                       5.11061663e-02, 4.99401158e-02, 4.87921234e-02, 4.76627367e-02,
                       4.65524293e-02, 4.54615902e-02, 4.43905184e-02, 4.33394242e-02,
                       4.23084329e-02, 4.12975917e-02, 4.03068746e-02, 3.93361872e-02,
                       3.83853684e-02, 3.74541920e-02, 3.65423673e-02, 3.56495417e-02,
                       3.47753046e-02, 3.39191919e-02, 3.30806912e-02, 3.22592452e-02,
                       3.14542533e-02, 3.06650713e-02, 2.98910097e-02, 2.91313356e-02,
                       2.83852775e-02, 2.76520375e-02, 2.69308100e-02, 2.62208070e-02,
                       2.55212850e-02, 2.48315724e-02, 2.41510909e-02, 2.34793711e-02,
                       2.28160591e-02, 2.21609157e-02, 2.15138110e-02, 2.08747164e-02,
                       2.02436985e-02, 1.96209156e-02, 1.90066199e-02, 1.84011650e-02,
                       1.78050156e-02, 1.72187582e-02, 1.66431071e-02, 1.60789019e-02,
                       1.55270916e-02, 1.49887035e-02, 1.44647958e-02, 1.39563953e-02,
                       1.34644267e-02, 1.29896363e-02, 1.25325199e-02, 1.20932597e-02,
                       1.16716749e-02, 1.12671925e-02, 1.08788383e-02, 1.05052492e-02,
                       1.01447070e-02, 9.79519159e-03, 9.45444989e-03, 9.12007805e-03,
                       8.78961303e-03, 8.46062906e-03, 8.13083513e-03, 7.79816907e-03,
                       7.46088406e-03, 7.11762330e-03, 6.76747880e-03, 6.41003092e-03,
                       6.04536607e-03, 5.67407148e-03, 5.29720828e-03, 4.91626628e-03,
                       4.53310651e-03, 4.14989932e-03, 3.76906746e-03, 3.39324317e-03,
                       3.02524780e-03, 2.66810076e-03, 2.32506225e-03, 1.99971216e-03,
                       1.69606512e-03, 1.41871965e-03, 1.17303739e-03, 9.65347010e-04,
                       8.03165741e-04, 6.95430588e-04, 6.52730148e-04, 6.87527356e-04,
                       8.14362852e-04, 1.05002824e-03, 1.41369832e-03, 1.92701134e-03,
                       2.61408664e-03, 3.50146977e-03, 4.61799609e-03, 5.99456586e-03,
                       7.66382554e-03, 9.65975335e-03, 1.20171501e-02, 1.47710405e-02,
                       1.79559949e-02, 2.16053847e-02, 2.57505926e-02, 3.04201992e-02,
                       3.56391753e-02, 4.14281103e-02, 4.78025080e-02, 5.47721821e-02,
                       6.23407808e-02, 7.05054620e-02, 7.92567354e-02, 8.85784757e-02,
                       9.84481024e-02, 1.08836910e-01, 1.19710529e-01, 1.31029491e-01,
                       1.42749870e-01, 1.54823975e-01, 1.67201058e-01, 1.79828029e-01,
                       1.92650128e-01, 2.05611566e-01, 2.18656105e-01, 2.31727586e-01,
                       2.44770408e-01, 2.57729968e-01, 2.70553068e-01, 2.83188296e-01,
                       2.95586393e-01, 3.07700587e-01, 3.19486926e-01, 3.30904581e-01,
                       3.41916147e-01, 3.52487925e-01, 3.62590202e-01, 3.72197518e-01,
                       3.81288909e-01, 3.89848124e-01, 3.97863788e-01, 4.05329504e-01,
                       4.12243862e-01, 4.18610368e-01, 4.24437258e-01, 4.29737229e-01,
                       4.34527062e-01, 4.38827186e-01, 4.42661165e-01, 4.46055150e-01,
                       4.49037307e-01, 4.51637243e-01, 4.53885438e-01, 4.55812709e-01,
                       4.57449719e-01, 4.58826524e-01, 4.59972185e-01, 4.60914441e-01,
                       4.61679444e-01, 4.62291557e-01, 4.62773215e-01, 4.63144839e-01,
                       4.63424812e-01, 4.63629486e-01, 4.63773239e-01, 4.63868564e-01,
                       4.63926176e-01, 4.63955142e-01, 4.63963023e-01, 4.63956026e-01,
                       4.63939151e-01, 4.63916343e-01, 4.63890635e-01, 4.63864286e-01,
                       4.63838907e-01, 4.63815577e-01, 4.63794947e-01, 4.63777336e-01,
                       4.63762805e-01, 4.63751230e-01, 4.63742356e-01, 4.63735851e-01,
                       4.63731334e-01, 4.63728415e-01, 4.63726709e-01, 4.63725858e-01,
                       4.63725539e-01, 4.63725470e-01, 4.63725417e-01, 4.63725188e-01,
                       4.63724637e-01, 4.63723659e-01, 4.63722183e-01, 4.63720171e-01,
                       4.63717611e-01, 4.63714514e-01, 4.63710905e-01, 4.63706823e-01,
                       4.63702315e-01, 4.63697434e-01, 4.63692234e-01, 4.63686769e-01,
                       4.63681091e-01, 4.63675249e-01, 4.63669286e-01, 4.63663243e-01,
                       4.63657152e-01, 4.63651042e-01, 4.63644937e-01, 4.63638854e-01,
                       4.63632807e-01, 4.63626805e-01, 4.63620856e-01, 4.63614961e-01,
                       4.63609122e-01, 4.63603337e-01, 4.63597603e-01, 4.63591917e-01,
                       4.63586273e-01, 4.63580667e-01, 4.63575093e-01, 4.63569546e-01,
                       4.63564022e-01, 4.63558515e-01, 4.63553022e-01, 4.63547538e-01,
                       4.63542062e-01, 4.63536590e-01, 4.63531120e-01, 4.63525650e-01,
                       4.63520180e-01, 4.63514709e-01, 4.63509236e-01, 4.63503761e-01,
                       4.63498284e-01, 4.63492806e-01, 4.63487327e-01, 4.63481847e-01,
                       4.63476366e-01, 4.63470886e-01, 4.63465406e-01, 4.63459926e-01,
                       4.63454446e-01, 4.63448965e-01, 4.63443484e-01, 4.63438001e-01,
                       4.63432514e-01, 4.63427021e-01, 4.63421521e-01, 4.63416010e-01,
                       4.63410487e-01, 4.63404946e-01, 4.63399386e-01, 4.63393802e-01,
                       4.63388190e-01, 4.63382547e-01, 4.63376869e-01, 4.63371155e-01,
                       4.63365402e-01, 4.63359610e-01, 4.63353779e-01, 4.63347912e-01,
                       4.63342015e-01, 4.63336096e-01, 4.63330164e-01, 4.63324234e-01,
                       4.63318325e-01, 4.63312456e-01, 4.63306655e-01, 4.63300951e-01,
                       4.63295376e-01, 4.63289966e-01, 4.63284760e-01, 4.63279797e-01,
                       4.63275116e-01, 4.63270754e-01, 4.63266743e-01, 4.63263110e-01,
                       4.63259869e-01, 4.63257022e-01, 4.63254555e-01, 4.63252432e-01,
                       4.63250593e-01, 4.63248948e-01, 4.63247379e-01, 4.63245731e-01,
                       4.63243815e-01, 4.63241406e-01, 4.63238247e-01, 4.63234050e-01,
                       4.63228510e-01, 4.63221310e-01, 4.63212141e-01, 4.63200722e-01,
                       4.63186829e-01, 4.63170326e-01, 4.63151208e-01, 4.63129650e-01,
                       4.63106062e-01, 4.63081154e-01, 4.63056013e-01, 4.63032178e-01,
                       4.63011730e-01, 4.62997389e-01, 4.62992605e-01, 4.63001662e-01,
                       4.63029775e-01, 4.63083187e-01, 4.63169256e-01, 4.63296528e-01,
                       4.63474803e-01, 4.63715166e-01, 4.64030006e-01, 4.64432994e-01,
                       4.64939026e-01, 4.65564135e-01, 4.66325346e-01, 4.67240495e-01,
                       4.68327999e-01, 4.69606580e-01, 4.71094948e-01, 4.72811454e-01,
                       4.74773709e-01, 4.76998197e-01, 4.79499886e-01, 4.82291851e-01,
                       4.85384928e-01, 4.88787411e-01, 4.92504808e-01, 4.96539660e-01,
                       5.00891432e-01, 5.05556477e-01, 5.10528082e-01, 5.15796567e-01,
                       5.21349460e-01, 5.27171710e-01, 5.33245947e-01, 5.39552771e-01,
                       5.46071063e-01, 5.52778317e-01, 5.59650976e-01, 5.66664787e-01,
                       5.73795149e-01, 5.81017471e-01, 5.88307518e-01, 5.95641752e-01,
                       6.02997636e-01, 6.10353919e-01, 6.17690865e-01, 6.24990429e-01,
                       6.32236379e-01, 6.39414362e-01, 6.46511913e-01, 6.53518426e-01,
                       6.60425085e-01, 6.67224772e-01, 6.73911954e-01, 6.80482553e-01,
                       6.86933818e-01, 6.93264175e-01, 6.99473096e-01, 7.05560949e-01,
                       7.11528864e-01, 7.17378595e-01, 7.23112392e-01, 7.28732876e-01,
                       7.34242915e-01, 7.39645512e-01, 7.44943696e-01, 7.50140417e-01,
                       7.55238458e-01, 7.60240351e-01, 7.65148319e-01, 7.69964227e-01,
                       7.74689567e-01, 7.79325456e-01, 7.83872671e-01, 7.88331690e-01,
                       7.92702772e-01, 7.96986032e-01, 8.01181541e-01, 8.05289417e-01,
                       8.09309914e-01, 8.13243493e-01, 8.17090887e-01, 8.20853128e-01,
                       8.24531562e-01, 8.28127833e-01, 8.31643853e-01, 8.35081747e-01,
                       8.38443796e-01, 8.41732366e-01, 8.44949851e-01, 8.48098613e-01,
                       8.51180940e-01, 8.54199012e-01, 8.57154885e-01, 8.60050480e-01,
                       8.62887579e-01, 8.65667832e-01, 8.68392762e-01, 8.71063775e-01,
                       8.73682174e-01, 8.76249178e-01, 8.78765937e-01, 8.81233553e-01,
                       8.83653103e-01, 8.86025653e-01, 8.88352272e-01, 8.90634036e-01,
                       8.92872025e-01, 8.95067316e-01, 8.97220966e-01, 8.99333997e-01,
                       9.01407376e-01, 9.03441997e-01, 9.05438671e-01, 9.07398116e-01,
                       9.09320954e-01, 9.11207711e-01, 9.13058833e-01, 9.14874693e-01,
                       9.16655611e-01, 9.18401874e-01, 9.20113759e-01, 9.21791553e-01,
                       9.23435572e-01, 9.25046183e-01, 9.26623816e-01, 9.28168972e-01,
                       9.29682234e-01, 9.31164263e-01, 9.32615795e-01, 9.34037630e-01,
                       9.35430613e-01, 9.36795618e-01, 9.38133518e-01, 9.39445163e-01,
                       9.40731347e-01, 9.41992783e-01, 9.43230087e-01, 9.44443758e-01,
                       9.45634180e-01, 9.46801626e-01, 9.47946279e-01, 9.49068258e-01,
                       9.50167652e-01, 9.51244546e-01, 9.52299059e-01, 9.53331359e-01,
                       9.54341684e-01, 9.55330343e-01, 9.56297721e-01, 9.57244267e-01,
                       9.58170487e-01, 9.59076928e-01, 9.59964161e-01, 9.60832762e-01,
                       9.61683294e-01, 9.62516277e-01, 9.63332173e-01, 9.64131365e-01,
                       9.64914139e-01, 9.65680672e-01, 9.66431033e-01, 9.67165176e-01,
                       9.67882940e-01, 9.68584051e-01, 9.69268117e-01, 9.69934620e-01,
                       9.70582909e-01, 9.71212195e-01, 9.71821549e-01, 9.72409914e-01,
                       9.72976121e-01, 9.73518929e-01, 9.74037070e-01, 9.74529301e-01,
                       9.74994471e-01, 9.75431572e-01, 9.75839801e-01, 9.76218603e-01,
                       9.76567706e-01, 9.76887141e-01, 9.77177250e-01, 9.77438680e-01])

    return y, x
