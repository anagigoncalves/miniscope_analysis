import sys, os, errno, copy, re, time
import numpy as np
import pandas as pd
from itertools import combinations
from functools import reduce
import operator
from statsmodels import robust
from scipy import signal
from scipy.stats import chi2_contingency
from scipy.interpolate import interp1d
import math as mt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
from tqdm import tqdm

Invalid_Charaters = ["!","@","#","$","%","^","&","*","(",")","+","/","\\","~",
                     ";",",","{","}","[","]","<",">","?","'","|","\""]

# From: https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
def blockPrint(): # Disable Print function
    sys.stdout = open(os.devnull, 'w')

def enablePrint(): # Enable Print function
    sys.stdout = sys.__stdout__

def user_display_time(milliseconds, user_scale=['milli', 'sec']):
    intervals = {'weeks':604800000,  # 60 * 60 * 24 * 7
                 'days':86400000,    # 60 * 60 * 24
                 'hours':3600000,    # 60 * 60
                 'minutes':60000,
                 'seconds':1000,
                 'milliseconds':1}
    # Choose the levels that are masked by user:
    Names = [name for name in intervals.keys()]
    for idx in range(len(user_scale)):
        match = [i for i in Names if user_scale[idx] in i]
        try:
            user_scale[idx] = match[0]
        except:
            raise ValueError("user_display_time Error: user_scale has unmatched time strings")

    result = []
    for name, count in intervals.items():
        value = milliseconds // count
        if value:
            milliseconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            if name in user_scale:
                result.append("{} {}".format(int(value), name))
    return ', '.join(result)

def is_pathname_valid(pathname):

    # Sadly, Python fails to provide the following magic number for us.
    ERROR_INVALID_NAME = 123
    '''
    Windows-specific error code indicating an invalid pathname.

    See Also
    ----------
    https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
        Official listing of all such codes.
    '''
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    # Did we mention this should be shipped with Python already?
        return True

def findcommonstart(strlist, separator=None):

    if not separator is None:

        Splits = [elem.split(separator) for elem in strlist]
        MaxSplits = Splits[np.argmax([len(elem) for elem in Splits])]

        for split in Splits:
            CompSplits = []
            for num in range(len(MaxSplits)):
                if num + 1 > len(split): break
                if split[num] == MaxSplits[num]:
                    CompSplits.append(split[num])
                else:
                    break
            MaxSplits = copy.deepcopy(CompSplits)
            if not len(MaxSplits) > 0:
                break

        return separator.join(MaxSplits)

    else:

        def getcommonletters(strlist):
            return ''.join([x[0] for x in zip(*strlist) \
                             if reduce(lambda a,b:(a == b) and a or None,x)])

        strlist = strlist[:]
        prev = None
        while True:
            common = getcommonletters(strlist)
            if common == prev:
                break
            strlist.append(common)
            prev = common

        return getcommonletters(strlist)

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def dict_recursive_items(dictionary, level=0):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value, level)
            yield from dict_recursive_items(value, level+1)
        else:
            yield (key, value, level)

def check_types_dict(dictionary, flag=None):

    Bool = False
    print('Upper Level Dict:\n')
    for key, val, level in dict_recursive_items(dictionary):
        if not isinstance(val, dict):
            temp = np.asarray(val)
            print('\t', key, temp.dtype, temp.shape)
            if issubclass(temp.dtype.type, np.int32):
                print('True')
            if not flag is None:
                if flag in str(temp.dtype):
                    Bool = True
        else:
            print('\n', key, type(val), '\n')

    return Bool

def Create_Dict_KeyMap(Dict, KeyRejects=[]):

    prevLevel = 0; keyList = [""]; KeyMap = []; MultiLists = []
    for key, val, level in dict_recursive_items(Dict):
        # Create the key map to open specific data:
        if level+1 != len(keyList):
            for i in range(len(keyList) - level):
                keyList.pop()
            keyList.append(key)
        elif prevLevel < level:
            keyList.append(key)
        elif prevLevel == level:
            keyList.pop(); keyList.append(key)
        prevLevel = level

        if any([key in keyList for key in KeyRejects]): continue
        Type = str(type(val))
        if isinstance(val, np.ndarray): Type += " " + str(val.dtype)

        KeyMap.append([keyList.copy(), Type])

    return KeyMap

def Dict_Key_Rename_to_MATLAB(DataDict):
    # Function to change the name of the keys inside data dictionaries,
    # so that they can be read in a MATLAB file (.mat).
    # It works for almost all dictionary entries, except for nested dictionaries
    # in lists or tuples (normal dictionaries in lists or tuples do work!!)

    repl_func = lambda x : x.replace(' ', '_').replace('.',"").replace("-","_")

    KeyMaps = Create_Dict_KeyMap(DataDict)

    NewDataDict = {}
    for num in range(len(KeyMaps)):
        Cont2 = copy.deepcopy(getFromDict(DataDict, KeyMaps[num][0]))
        KeyMaps[num][0] = [repl_func(string) for string in KeyMaps[num][0]]
        if isinstance(Cont2, dict):
            KeyList = list(Cont2.keys())
            Cont2 = {repl_func(k) : Cont2.pop(k) for k in KeyList}
            setInDict(NewDataDict, KeyMaps[num][0], Cont2)
        if isinstance(Cont2, (list, tuple)):
            for ele in range(len(Cont2)):
                if isinstance(Cont2[ele], dict):
                    KeyList = list(Cont2[ele].keys())
                    Cont2[ele] = {repl_func(k) : Cont2[ele].pop(k) for k in KeyList}
            setInDict(NewDataDict, KeyMaps[num][0], Cont2)

    return NewDataDict

def Sort_StrList_by_number(string_list):
# from https://www.tutorialspoint.com/How-to-correctly-sort-a-string-with-a-number-inside-in-Python
    def atoi(text):
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)',text)]

    return sorted(string_list, key=natural_keys)

def Sort_by_Filename(DataFrame, key_of_Filenames, name_splitter='_'):
    # Function to sort Pandas dataframe based on structured Filename inside of it.
    # DataFrame: Full Pandas Dataframe.
    # key_of_Filenames: String of the key locating the column that contains the filenames to sort.
    # name_splitter: (Optional) The character or string that splits the information in the filename.

    TempList = DataFrame[key_of_Filenames].tolist()
    Split_Filenames = [elem.split(name_splitter) for elem in TempList] # Splits the information contained in the filenames
    Tempdf = pd.DataFrame(Split_Filenames).apply(pd.to_numeric, errors='ignore') # Transforms the splittings into numeric values for sorting
    Tempdf.sort_values(list(Tempdf.columns), inplace=True) # Sorting of the File names
    NewDF = DataFrame.reindex(list(Tempdf.index), copy=True)

    return NewDF.reset_index(drop=True)

def find_nearest_pnt(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def ExtractDataTrend(Data, Acq_Freq, HFcut):

    if HFcut > mt.floor((Acq_Freq/2)-(Acq_Freq/200)):
        HFcut = mt.floor((Acq_Freq/2)-(Acq_Freq/200))

    # Calculate simple filter delay
    DelayPnts = int(Acq_Freq/(2*HFcut))

    # Emphasize the trend in the data with a Bessel filter
    Init = Data[0]
    TempData = Data - Init
    Wn = HFcut/(Acq_Freq/2)
    b, a = signal.bessel(4,Wn,'low',analog=False)
    Trend = signal.lfilter(b, a, TempData)

    # Adjust delay, data points, and initial value
    Trend = np.delete(Trend, np.arange(DelayPnts))
    LastVal = Trend[Trend.size-1]
    Trend = np.append(Trend, [LastVal]*DelayPnts)
    LastVal = Trend[Trend.size-1] - Data[Data.size-1]
    Trend = Trend - LastVal

    return Trend

def Mutual_Information(x, y, bins):
    # from: https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    # WARNING: The commentators said it does not work with counts that have zero value
    c_xy = np.histogram2d(x, y, bins)[0]
    g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood", correction = False)
    mi = 0.5 * g / c_xy.sum()
    return mi/np.log(2)

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

def Norm_keepZero(signal, min=-1, max=1):

    Pos_signal = np.where(signal > 0, signal, np.zeros(signal.shape))
    Neg_signal = np.where(signal < 0, signal, np.zeros(signal.shape))
    Pos_Norm = Normalize(Pos_signal, min=0, max=max)
    Neg_Norm = Normalize(Neg_signal, min=min, max=0)

    return Pos_Norm + Neg_Norm

def UniqueListElem(InputList):
    seen = set()
    result = []
    for item in InputList:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result

def CrossingThres(Data, AmplThres, offset=True, subs=False, fordwin=0):
    # Function to extract the values and point locations beyond a threshold.
    # THE MAIN FUNCTION IS THAT The threshold can be varied point by point,
    # by passing an array with size equal to the data.
    #
    # Data: Numpy array containing the one-dimensional data to threshold.
    # AmplThres: single number or numpy array with the threshold value/s.
    #            If it is an array, it must have the same size as Data.
    # offset: Optional. If True, the Data array will be offset by its median.
    # subs: Optional. If True, each value of crossing will have its threshold substracted.
    # fordwin: Optional. Minimal valid lenght in points that two consecutive crossings
    #          can have.

    if offset:
        Data2 = Data.copy() - np.nanmedian(Data)
    else:
        Data2 = Data

    # Check the datatypes of the inputs and adjust them or abort the program
    datatype = str(type(AmplThres))
    if type(AmplThres) is np.ndarray and np.size(AmplThres) == np.size(Data): # AmplThres must have the same size as Data
        PosEnvel = AmplThres.copy()
        NegEnvel = - AmplThres.copy()
    elif isinstance(AmplThres, float) or isinstance(AmplThres, int):
        PosEnvel = np.ones(Data.size)*AmplThres
        NegEnvel = - np.ones(Data.size)*AmplThres
    else:
        raise ValueError('CrossingThres Abort: AmplThres was not recognized as a valid datatype')

    OutVals = []
    OutPnts = []
    it = np.nditer(Data2, flags=['f_index'])
    while not it.finished:
        Pos_Cross = it[0] > PosEnvel[it.index]
        Neg_Cross = it[0] < NegEnvel[it.index]
        if Pos_Cross or Neg_Cross:
            if len(OutPnts) != 0:
                if OutPnts[len(OutPnts)-1] + fordwin < it.index:
                    if subs:
                        if Pos_Cross:
                            OutVals.append(it[0]-PosEnvel[it.index])
                        else:
                            OutVals.append(it[0]-NegEnvel[it.index])
                    else:
                        OutVals.append(it[0])
                    OutPnts.append(it.index)
            else:
                if subs:
                    if Pos_Cross:
                        OutVals.append(it[0]-PosEnvel[it.index])
                    else:
                        OutVals.append(it[0]-NegEnvel[it.index])
                else:
                    OutVals.append(it[0])
                OutPnts.append(it.index)
        it.iternext()

    return OutVals, OutPnts

def SliceTimeSeries(Data, Init, End, Acq_Freq=0):

    TempData = np.asarray(Data)

    if Acq_Freq != 0: # When timing intervals are chosen (based on the Acquisition Freq.)
        Init = int(Init*Acq_Freq)
        End = int(End*Acq_Freq)

    if Init < 0:
        Init = 0

    if End >= Data.size:
        End = Data.size - 1

    return TempData[Init:End]

def FindPntsWin(DataPnts_Ref, DataPnts_Target, WinPnts, wintype='forward', minElems = 2):
    # Function to find the groups of points in a dataset that are contained
    # in a window from a reference dataset of points.
    # DataPnts_Ref, DataPnts_Target: Arrays of integers with the locations of Points
    #                                to group. Ref is the array used as the reference
    #                                for the window, while Target is the array to be grouped.
    # WinPnts: lenght of the window in points.
    # wintype: The type of window for the grouping criteria. Three options:
    #          ['forward', 'backward', 'central']
    # minElems: Optional. Determines the minimum number of elements admitted per groups
    #           (use it to avoid selection of the same points if comparing two equal datasets)

    TempData1 = np.sort(np.asarray(DataPnts_Ref))
    TempData2 = np.sort(np.asarray(DataPnts_Target))

    ListOfGroups = []
    PntsTarget = []
    PntsRef = []
    PntRef = -1
    jump = 0
    for i in range(TempData1.size):
        PntRef += 1
        Group = []
        Pnts = []
        Start_Group = False
        for n in range(jump, TempData2.size):
            Bool_wintype = False
            # Extracting the condition of grouping for the type of window chosen by the user
            if wintype.startswith('fo'):
                Bool_wintype = TempData1[i] <= TempData2[n] and TempData2[n] < TempData1[i] + WinPnts
            elif wintype.startswith('ba'):
                Bool_wintype = TempData1[i] >= TempData2[n] and TempData2[n] > TempData1[i] - WinPnts
            elif wintype.startswith('ce'):
                Bool_wintype = TempData2[n] < TempData1[i] + int(WinPnts/2) and TempData2[n] > TempData1[i] - int(WinPnts/2)

            if Bool_wintype:
                Group.append(TempData2[n])
                Pnts.append(n)
                Start_Group = True
            elif Start_Group:
                jump = n
                break
        if Start_Group and len(Group) >= minElems:
            ListOfGroups.append(Group)
            PntsRef.append(PntRef)
            PntsTarget.append(Pnts)

    return ListOfGroups, PntsTarget, PntsRef

def Find_AproxPairs(Data1, Data2, oper='=', maxdelta=np.inf, mindelta=0): # KEPT IN CASE OF BACKWARD COMPATIBILITY PROBLEMS
    ### Gets the closest pairs of points between two datasets. Data points are only
    ### placed in one pair, without repetition.
    ### Data1, Data2: Datasets containing the points to pair. Can be on lists or numpy arrays.
    ### oper: Optional. If selected, forces a constraint to the pairing by the chosen
    ###        operation. Example: Data1 <= Data2 (oper == '<=').
    ###        Note: For oper '=', values with more than one pair (surjective pairs) are
    ###              resolved by choosing the pair of elements with the lowest difference.
    ### maxdelta: Optional. Define a maximum allowed difference between valid pairs
    ### mindelta: Optional. Define a minimum allowed difference between valid pairs

    # Check if there are repeated values inside each dataset, and if there are,
    # only take one of its ocurrences.
    if np.size(Data1) > 1:
        Temp1 = np.diff(sorted(Data1))
        Pnts1 = np.arange(np.size(Data1))
        Pnts1 = np.extract(Temp1 != 0, Pnts1)
        Temp1 = np.extract(Temp1 != 0, Data1)
        if Temp1[Temp1.size-1] != Data1[Data1.size-1]:
            Temp1 = np.append(Temp1, Data1[Data1.size-1])
            Pnts1 = np.append(Pnts1, Data1.size-1)
    elif np.size(Data1) == 1:
        try: # Checking if Data is in an iterable format
           _ = (e for e in Data1)
           Temp1 = Data1
        except TypeError:
           Temp1 = [Data1]
        Pnts1 = np.arange(np.size(Data1))
    else:
        #print('Find_AproxPairs Warning: Data1 has no values')
        return [], []

    if np.size(Data2) > 1:
        Temp2 = np.diff(sorted(Data2))
        Pnts2 = np.arange(np.size(Data2))
        Pnts2 = np.extract(Temp2 != 0, Pnts2)
        Temp2 = np.extract(Temp2 != 0, Data2)
        if Temp2[Temp2.size-1] != Data2[Data2.size-1]:
            Temp2 = np.append(Temp2, Data2[Data2.size-1])
            Pnts2 = np.append(Pnts2, Data2.size-1)
    elif np.size(Data2) == 1:
        try: # Checking if Data is in an iterable format
           _ = (e for e in Data2)
           Temp2 = Data2
        except TypeError:
           Temp2 = [Data2]
        Pnts2 = np.arange(np.size(Data2))
    else:
        #print('Find_AproxPairs Warning: Data2 has no values')
        return [], []

    Bool_Changer = False
    if oper == '=':
        opernum = 0
    elif oper == '<':
        Bool_Changer = True
        Changer = Temp1.copy(); ChPnts = Pnts1.copy()
        Temp1 = Temp2.copy(); Pnts1 = Pnts2.copy()
        Temp2 = Changer; Pnts2 = ChPnts
        opernum = -1
    elif oper == '<=':
        opernum = 1
    elif oper == '>':
        opernum = -1
    elif oper == '>=':
        Bool_Changer = True
        Changer = Temp1.copy(); ChPnts = Pnts1.copy()
        Temp1 = Temp2.copy(); Pnts1 = Pnts2.copy()
        Temp2 = Changer; Pnts2 = ChPnts
        opernum = 1
    else:
        raise ValueError('Find_AproxPairs Abort -> The operation to perform was not recognized')

    # Sort the concatenated datasets to detect the closest paired points between them.
    # Keep track of the origin of each point with a binary mask (Class1/2)
    Class1 = np.zeros(np.size(Temp1))
    Class2 = np.zeros(np.size(Temp2)) + 1
    if np.size(Temp1) > 1:
        List1 = list(zip(Temp1, Class1, Pnts1))
    else:
        List1 = [(Temp1[0], Class1[0], Pnts1[0])]
    if np.size(Temp2) > 1:
        List2 = list(zip(Temp2, Class2, Pnts2))
    else:
        List2 = [(Temp2[0], Class2[0], Pnts2[0])]
    All_list = List1 + List2
    keyval = lambda elem: elem[0]
    All_list.sort(key=keyval)
    OrderClass = np.asarray([elem[1] for elem in All_list])
    Diff_class = np.diff(OrderClass) # Use the derivative of the Class to detect the edges where pairs are found

    # Extract the Pairs between the datasets
    Pairs = []
    PairsPnts = []
    it = np.nditer(Diff_class, flags=['f_index'])
    while not it.finished:
        if it[0] == 1 and not opernum == -1:
            if len(Pairs) == 0:
                TempVal = np.abs(All_list[it.index][0] - All_list[it.index+1][0])
                if TempVal >= mindelta and TempVal <= maxdelta:
                    Pairs.append((All_list[it.index][0],All_list[it.index+1][0]))
                    PairsPnts.append((All_list[it.index][2],All_list[it.index+1][2]))
            elif Pairs[len(Pairs)-1][0] == All_list[it.index][0]:
                DiffPrev = np.abs(Pairs[len(Pairs)-1][1] - Pairs[len(Pairs)-1][0])
                DiffCurr = np.abs(All_list[it.index+1][0] - Pairs[len(Pairs)-1][0])
                if DiffPrev <= DiffCurr:
                    pass
                else:
                    TempVal = np.abs(All_list[it.index][0] - All_list[it.index+1][0])
                    if TempVal >= mindelta and TempVal <= maxdelta:
                        Pairs[len(Pairs)-1] = (All_list[it.index][0],All_list[it.index+1][0])
                        PairsPnts[len(PairsPnts)-1] = (All_list[it.index][2],All_list[it.index+1][2])
            else:
                if np.abs(All_list[it.index][0] - All_list[it.index+1][0]) <= maxdelta:
                    Pairs.append((All_list[it.index][0],All_list[it.index+1][0]))
                    PairsPnts.append((All_list[it.index][2],All_list[it.index+1][2]))
        elif it[0] == -1 and not opernum == 1:
            if len(Pairs) == 0:
                TempVal = np.abs(All_list[it.index][0] - All_list[it.index+1][0])
                if TempVal >= mindelta and TempVal <= maxdelta:
                    Pairs.append((All_list[it.index+1][0],All_list[it.index][0]))
                    PairsPnts.append((All_list[it.index+1][2],All_list[it.index][2]))
            elif Pairs[len(Pairs)-1][1] == All_list[it.index][0]:
                DiffPrev = np.abs(Pairs[len(Pairs)-1][1] - Pairs[len(Pairs)-1][0])
                DiffCurr = np.abs(All_list[it.index+1][0] - Pairs[len(Pairs)-1][1])
                if DiffPrev <= DiffCurr:
                    pass
                else:
                    TempVal = np.abs(All_list[it.index][0] - All_list[it.index+1][0])
                    if TempVal >= mindelta and TempVal <= maxdelta:
                        Pairs[len(Pairs)-1] = (All_list[it.index+1][0],All_list[it.index][0])
                        PairsPnts[len(PairsPnts)-1] = (All_list[it.index+1][2],All_list[it.index][2])
            else:
                if np.abs(All_list[it.index][0] - All_list[it.index+1][0]) <= maxdelta:
                    Pairs.append((All_list[it.index+1][0],All_list[it.index][0]))
                    PairsPnts.append((All_list[it.index+1][2],All_list[it.index][2]))
        it.iternext()

    tuple_invert = lambda pair:(pair[1], pair[0])
    if not Bool_Changer:
        return Pairs, PairsPnts
    else:
        Pairs = list(map(tuple_invert, Pairs))
        PairsPnts = list(map(tuple_invert, PairsPnts))
        return Pairs, PairsPnts

def IntersectPairs(Pairs1, Pairs2, val1=0, val2=0):
    # Function to find the points in two sets of Pairs that are equal,
    # and extract the others values from the common one:

    # Pairs1, Pairs2: Pairs detected with the function Find_AproxPairs
    # val1, val2: if 0, take the first value of the tuple pair.
    #             If 1, take the second value.

    # Example:  (a, b); (a, c) -> (b, c) for val1=0, val2=0.

    if val1 > 1 or val1 < 0:
        raise ValueError("PairsIntersection -> 'val' optional values are incorrect")
    if val2 > 1 or val2 < 0:
        raise ValueError("PairsIntersection -> 'val' optional values are incorrect")

    tuple_extrc = lambda pair:pair[0]
    tuple_extrc2 = lambda pair:pair[1]

    Pair1_0 = list(map(tuple_extrc, Pairs1))
    Pair2_0 = list(map(tuple_extrc, Pairs2))
    Pair1_1 = list(map(tuple_extrc2, Pairs1))
    Pair2_1 = list(map(tuple_extrc2, Pairs2))

    if val1 == 0:
        Pair1 = Pair1_0
        Pair1opos = Pair1_1
    else:
        Pair1 = Pair1_1
        Pair1opos = Pair1_0

    if val2 == 0:
        Pair2 = Pair2_0
        Pair2opos = Pair2_1
    else:
        Pair2 = Pair2_1
        Pair2opos = Pair2_0

    Intersec, InterPnts = Find_AproxPairs(np.asarray(Pair1), np.asarray(Pair2), maxdelta=0)

    InterPnts1 = list(map(tuple_extrc, InterPnts))
    InterPnts2 = list(map(tuple_extrc2, InterPnts))

    Intersec1 = np.take(np.asarray(Pair1opos), InterPnts1).tolist()
    Intersec2 = np.take(np.asarray(Pair2opos), InterPnts2).tolist()

    if len(Intersec1) != len(Intersec2):
        raise RuntimeError("PairsIntersection -> An error has ocurred on the intersection procedure ")

    InterSet = list(zip(Intersec1, Intersec2))
    Vals_Intersec = list(map(tuple_extrc, Intersec))
    Pnts_Intersec = list(zip(InterPnts1, InterPnts2))

    return InterSet, Vals_Intersec, Pnts_Intersec

def time_to_points_arr(Time_Ref, Target_Arr, verbose=False):
    # Function to transform the timing values of an array into points inside a timing vector
    # Time_Ref: Timing numpy 1-D vector of sequential, increasing values
    # Target_Arr: Numpy 1-D array with the values to transform into Points

    if np.amax(Time_Ref) < np.amax(Target_Arr) or np.amin(Time_Ref) > np.amin(Target_Arr):
        raise TypeError("time_to_points_arr Abort: Target Array has value(s) outside the Timing Vector interval")

    if verbose: Start = time.time()

    Pnts = []; Prev_pnt = 0
    for ev in Target_Arr:
        for pnt in range(Prev_pnt, Time_Ref.size):
            if Time_Ref[pnt] > ev:
                if pnt != 0:
                    if np.abs(Time_Ref[pnt-1]-ev) < np.abs(Time_Ref[pnt]-ev):
                        pnt = pnt - 1
                Pnts.append(pnt)
                Prev_pnt = pnt - 1
                break

    if verbose: End = time.time(); print("Time-to-Points Elapsed time: %.03f" % (End - Start))

    return np.asarray(Pnts)

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return mt.floor(n*multiplier + 0.5) / multiplier

def extract_numbers_from_string(text, as_string=False):

    s = [float(s) for s in re.findall(r'-?\d+\.?\d*', text)]

    if as_string:
        s = [str(num) for num in s]

    return s

def PoolExtrema(Data, segs=100, percen=50):

    PooledArray = np.array_split(Data, segs)
    listMax = []
    listMin = []
    for i in range(segs):
        TempArray = PooledArray[i]
        listMax.append(np.amax(TempArray))
        listMin.append(np.amin(TempArray))

    MaxPool = np.asarray(listMax)
    MinPool = np.asarray(listMin)

    return np.nanpercentile(MaxPool, percen), np.nanpercentile(MinPool, 100-percen)

def ClassicExtrema(Timeseries, graph=0):
    Diffdata = np.diff(Timeseries)
    zero_crossings = np.where(np.diff(np.sign(Diffdata)))[0]

    if graph != 0:

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(np.arange(Timeseries.size), Timeseries, color='black')
        ax2.plot(np.arange(Diffdata.size), Diffdata, color='green')
        ax2.vlines(zero_crossings, np.amin(Diffdata), np.amax(Diffdata), color='red')

        plt.show()

    return zero_crossings

def Base_SavGol(Data, win_pnt, pol=3, rep=5, pad_mode='constant'):

    Temp = Data.copy()
    for i in range(rep):
        Temp = signal.savgol_filter(Temp, win_pnt, pol, mode=pad_mode, cval=np.median(Data))

    return Temp

def FirstCrossVal(data, thres, direc='increm'):
    # Function that returns the position of the first crossing of a threshold value in a data array
    # data: Numpy Array

    if 'in' in direc:
        above_threshold = data > thres
    else:
        above_threshold = data < thres

    react_tms = np.argmax(above_threshold)
    react_tms = react_tms - (~np.any(above_threshold)).astype(float)

    return int(react_tms)

def Random_TimeSampling_No_Replacement(TimeArr, numEvents, TimeExts=[0,1], maxiter=None, graph=False, verbose=False):
    # TimeArr: A 1-D numpy array of sequential integers spanning the time duration (in sampled points)
    #          when the events happens.
    # numEvents: Integer of the number of events to sample randomly
    # TimeExts: List or tuple of two values containing the left and right extension of the point window
    #           where the event rejects the ocurrence of another event (aka. a refractory period).
    # maxiter: Maximum number of iterations allowed in the loop.
    # graph: (Optional). If True, a plot with the resample will be shown.

    if TimeExts[0] == 0 and TimeExts[1] == 0:
        raise ValueError('Random_TimeSampling_No_Replacement Error -> Invalid TimeExts variable')
    if maxiter is None:
        maxiter = 2*TimeArr.size
    if TimeArr.size / (TimeExts[0] + TimeExts[1]) < numEvents:
        raise ValueError('Random_TimeSampling_No_Replacement Error -> Impossible to fit all the windows of events on the TimeArray')
    if np.isclose(TimeArr.size / (TimeExts[0] + TimeExts[1]), numEvents):
        raise ValueError('Random_TimeSampling_No_Replacement Error -> There is only one way to fit the windows of events on the TimeArray')

    if verbose:
        pbar = tqdm(total=numEvents)
    Mask = np.ones(TimeArr.size, dtype=bool)
    NewEvents = []; NewRandom = True; iter = 0
    while numEvents > 0:
        iter += 1
        # Remove chosen places that cannot fit an event:
        RejIndx = np.where(np.invert(Mask))[0]
        if RejIndx.size > 0:
            DiffRej = np.append(np.diff(RejIndx), 0)
            RejIndxIni = RejIndx[DiffRej < (TimeExts[0] + TimeExts[1] + 2)]
            RejIndxEnd = RejIndxIni + DiffRej[DiffRej < (TimeExts[0] + TimeExts[1] + 2)]
            for idx in range(RejIndxIni.size):
                Mask[RejIndxIni[idx]:RejIndxEnd[idx]] = False

        if maxiter < iter and numEvents > 0:
            strError = 'Exceeded the number of iterations allowed (%d). Stopping with %d missing events.' % (maxiter, numEvents)
            raise RuntimeError('Random_TimeSampling_No_Replacement Error -> %s' % strError)
        if np.all(np.invert(Mask)) and numEvents > 0:
            strError = 'No more elements fit on the Time Array. Stopping with %d missing events.' % numEvents
            raise RuntimeError('Random_TimeSampling_No_Replacement Error -> %s' % strError)

        if NewRandom:
            Rnd = np.random.random_integers(0, TimeArr[Mask].size-1)
            NewIdx = TimeArr[Mask][Rnd]
        Ini = max(NewIdx-TimeExts[0], 0)
        End = min(NewIdx+TimeExts[1], Mask.size)
        if np.any(np.invert(Mask[Ini:End])):
            Back_Condition = np.all(Mask[Ini:NewIdx]) or Mask[Ini:NewIdx].size == 0
            Forw_Condition = np.all(Mask[NewIdx:End]) or Mask[NewIdx:End].size == 0
            if not Back_Condition and not Forw_Condition:
                NewRandom = True
                Intv0 = np.argmin(Mask[Ini:End])
                Intv1 = Mask[Ini:End].size - np.argmin(np.flip(Mask[Ini:End]))
                Mask[Ini:End][Intv0:Intv1] = False # Mask inbetween interval of Rejected points
                continue
            elif Back_Condition and not Forw_Condition:
                NewRandom = False
                NewIdx -= 1
                if NewIdx < 0: # Boundary condition
                    NewRandom = True
                    Mask[Ini:End] = False
                continue
            elif not Back_Condition and Forw_Condition:
                NewRandom = False
                NewIdx += 1
                if NewIdx > Mask.size - 1: # Boundary condition
                    NewRandom = True
                    Mask[Ini:End] = False
                continue
        else:
            NewRandom = True
            Mask[Ini:End] = False
            NewEvents.append(TimeArr[NewIdx])
            # print(iter, numEvents)
            numEvents -= 1; iter = 0
            if verbose:
                pbar.update(1)

    if verbose:
        pbar.close()
    NewEvents = np.asarray(sorted(NewEvents))
    if graph:
        Diffs = np.diff(NewEvents)
        fig, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.vlines(NewEvents, 0, 1, color='black')
        ax1.set_xlim((0, TimeArr[TimeArr.size-1]))
        ax2.hist(np.abs(Diffs), bins=int(np.max(np.abs(Diffs))), align='left')
        plt.show()

    return np.asarray(NewEvents)

def GaussPeak_WidthEstim(data, xdata, peakpnt, valley=False):
    # Function to estimate the width (standard deviation) of a Gaussian peak by exploiting
    # its relationship with the Full Width at Half Height (FWHM)

    if valley:
        if peakpnt == 0:
            LBound = False
        else:
            LBound = data[peakpnt-1] <= data[peakpnt]
        if peakpnt == data.size-1:
            RBound = False
        else:
            RBound = data[peakpnt+1] <= data[peakpnt]
        drc = 'increm'
    else:
        if peakpnt == 0:
            LBound = False
        else:
            LBound = data[peakpnt-1] >= data[peakpnt]
        if peakpnt == data.size-1:
            RBound = False
        else:
            RBound = data[peakpnt+1] >= data[peakpnt]
        drc = 'decrem'

    if LBound or RBound:
        return 0 # Data at Peak point is not an extrema

    thres = data[peakpnt]/2

    # Extract the crossings
    TempHalf = data[peakpnt:data.size-1].copy()
    if TempHalf.size < 1:
        RightCross = -1
    else:
        RightCross = FirstCrossVal(TempHalf, thres, direc=drc)

    FlippedHalf = np.flip(data[0:peakpnt+1])
    if FlippedHalf.size < 1:
        LeftCross = -1
    else:
        LeftCross = FirstCrossVal(FlippedHalf, thres, direc=drc)

    # Redefine the crossings in terms of their distance to the peak center to calculate the width
    if LeftCross == -1:
        L_halfWidth = np.inf
    else:
        LeftCross = FlippedHalf.size - LeftCross + 1
        if LeftCross >= xdata.size: # In case the peak center is located at the last point of the data.
            L_halfWidth = 0
        else:
            L_halfWidth = xdata[peakpnt] - xdata[LeftCross]

    if RightCross == -1:
        R_halfWidth = np.inf
    else:
        RightCross = RightCross + (data.size - TempHalf.size) - 1
        R_halfWidth = xdata[RightCross] - xdata[peakpnt]

    FWHM_estim = 2*min(L_halfWidth, R_halfWidth)
    if np.isinf(FWHM_estim):
        Std_estim = 0
    else:
        Std_estim = FWHM_estim/(2*np.sqrt(2*np.log(2)))

    return Std_estim

def CustomColorPalette(info_ref, cmap=None, norm=None, fix_col=None, int_base=False):

    if fix_col is None:
        if cmap is None: cmap = plt.cm.gist_rainbow
        if norm is None: norm = colors.Normalize(vmin=0, vmax=1)
        if isinstance(info_ref, (int, float)): # if it is a number, give the number of colors
            SeqColors = np.linspace(0, 1, info_ref)
        else:
            SeqColors = info_ref
        ValColors = cmap(norm(SeqColors))
    else:
        if isinstance(info_ref, (int, float)): # if it is a number, give the number of colors
            ValColors = np.asarray([fix_col]*int(info_ref))
        else:
            ValColors = np.asarray([fix_col]*len(info_ref))

    if int_base:
        ValColors = (ValColors*255).astype('uint8')

    return ValColors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vcenter <= self.vmin: x, y = [self.vcenter, self.vmax], [0.5, 1]
        if self.vcenter >= self.vmax: x, y = [self.vmin, self.vcenter], [0, 0.5]
        if self.vmin < self.vcenter and self.vcenter < self.vmax:
            x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class PosNegRegNorm(colors.Normalize):
    def __init__(self, reg_list=None, vmin=None, vmax=None, clip=False):
        if not isinstance(reg_list, list):
            raise TypeError("PosNegRegNorm Abort -> Only a list of values can be given as 'reg_list'")
        self.reg_list = reg_list
        if 0 in self.reg_list: self.reg_list.remove(0)
        self.reg_list = np.sort(np.asarray(self.reg_list, dtype=int))
        vmin = np.min(self.reg_list); vmax = np.max(self.reg_list)
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        NumPos = np.count_nonzero(np.sign(self.reg_list) > 0)
        MaxVal = np.asarray(value).max(); MinVal = np.asarray(value).min()
        if NumPos > 0:
            PosVals = np.linspace(1, 0.5, num=NumPos, endpoint=False).tolist()
            PosRegs = self.reg_list[np.sign(self.reg_list) > 0].tolist()
        elif MaxVal > 0:
            PosVals = [1]; PosRegs = [MaxVal]
        else:
            PosVals = [1]; PosRegs = [1]
        NumNeg = np.count_nonzero(np.sign(self.reg_list) < 0)
        if NumNeg > 0:
            NegVals = np.linspace(0.0, 0.5, num=NumNeg, endpoint=False).tolist()
            NegRegs = self.reg_list[np.sign(self.reg_list) < 0].tolist()
        elif MinVal < 0:
            NegVals = [1]; NegRegs = [MinVal]
        else:
            NegVals = [0]; NegRegs = [-1]
        NormVals = sorted(NegVals) + [0.5] + sorted(PosVals)
        Reg_Vals = NegRegs + [0] + PosRegs

        return np.ma.masked_array(np.interp(value, Reg_Vals, NormVals))

def Box_Whisker_Plot(data, whis=1.5, labels=None, title=None, posit=None, widths=None, notch=False, fignum=None, ax_idx=0):
    # data: List of data arrays to plot.
    if not type(data) is list:
        raise TypeError("Box_Whisker_Plot Error -> Data must be a list of arrays")

    if title is None:
        title = 'Box-&-Whiskers Plot'

    if not labels is None and type(labels) is list:
        if len(labels) != len(data):
            raise ValueError("Box_Whisker_Plot Error -> Labels don't match the data")
    elif not labels is None and not type(labels) is list:
        raise TypeError("Box_Whisker_Plot Error -> Labels must be a list")

    if not posit is None:
        try:
            num = len(posit)
        except:
            raise TypeError("Box_Whisker_Plot Error -> Positions must be a list or 1-D array")
        if num != len(data):
            raise ValueError("Box_Whisker_Plot Error -> Number of positions don't match the data")

    if not widths is None:
        try:
            num = len(widths)
        except:
            raise TypeError("Box_Whisker_Plot Error -> Widths must be a list or 1-D array")
        if num != len(data):
            raise ValueError("Box_Whisker_Plot Error -> Number of widths don't match the data")

    if isinstance(fignum, str):
        plt.close(fignum); fig, ax = plt.subplots(num=fignum)
    else:
        fig, ax = plt.subplots()
    ax.set_title(title)
    if labels is None:
        labels = list(range(len(data)))
    if posit is None:
        posit = list(range(1, len(data)+1))
    if widths is None:
        widths = [0.5]*len(data)

    # Enrich Labels with extra statistical information:
    for i in range(len(data)):
        labels[i] = labels[i] + '\n%.03f %s%.03f' % (np.median(data[i]), u"\u00B1", 2*robust.mad(data[i]))

    boxdict = dict(linewidth=2.0, color='black')
    mediandict = dict(linewidth=2.0, color='red')
    flierdict = dict(marker='o', markerfacecolor='orange', markersize=10,
                     linestyle='none')
    whisdict = dict(linewidth=2, color='gray', alpha=0.5)
    capdict = dict(linewidth=2, color='gray')
    ax.boxplot(data, whis=whis, labels=labels, positions=posit, widths=widths,
               notch=notch, boxprops=boxdict, medianprops=mediandict,
               flierprops=flierdict, whiskerprops=whisdict, capprops=capdict)

    plt.show()

def Joined_Points_Plot(data, title=None, labels=None, posit=None, marker_size=None, marker_type=None, marker_color=None, color_lines=None, fignum=None, ax_idx=0, figover=False):
    # data: List of arrays with the same number of elements (paired)
    # labels: Name of the data arrays
    # posit: Positions of the points in the X-axis per data array
    # marker_size: Size of the markers
    # marker_type: Type of the markers, following the matplotlib convention.
    # marker_color: Color of the markers
    # color_lines: Color of the lines that connect points between data arrays. There are N - 1 lines per N data arrays.
    # fignum: Optional identifier of the figure in order to ease its management.
    # ax_idx: Optional index of subplot if needed to be specified.
    # figover: Optional boolean to overlay the data to plot in an already open figure.

    if not type(data) is list:
        raise TypeError("Joined_Points_Plot Error -> Data must be a list of arrays")
    elif len(data) < 2:
        raise ValueError("Joined_Points_Plot Error -> Data must contain at least 2 arrays")
    elif not all([len(arr) == len(data[0]) for arr in data]):
        raise ValueError("Joined_Points_Plot Error -> Data must contain arrays of the same number of elements")

    if title is None:
        title = 'Points Plot'

    if not labels is None and type(labels) is list:
        if len(labels) != len(data):
            raise ValueError("Joined_Points_Plot Error -> Labels don't match the data")
    elif not labels is None and not type(labels) is list:
        raise TypeError("Joined_Points_Plot Error -> Labels must be a list")

    if not posit is None:
        try:
            num = len(posit)
        except:
            raise TypeError("Joined_Points_Plot Error -> Positions must be a list or 1-D array")
        if num != len(data):
            raise ValueError("Joined_Points_Plot Error -> Number of positions don't match the data")

    if fignum is None or not isinstance(fignum, str):
        fignum = "Points Plots"
    if not plt.fignum_exists(fignum):
        fig, ax = plt.subplots(num=fignum)
    elif not figover:
        plt.close(fignum); fig, ax = plt.subplots(num=fignum)
    else:
        fig = plt.figure(fignum); ax = fig.axes[ax_idx]


    ax.set_title(title)
    if labels is None:
        str_labels = ["Array " + str(st+1) for st in list(range(len(data)))]
    else:
        str_labels = labels
    if posit is None:
        posit = np.arange(len(data)).tolist()
    if marker_size is None:
        marker_size = [50]*len(data)
    if marker_type is None:
        marker_type = ['o']*len(data)
    if marker_color is None:
        marker_color = ['black']*len(data)
    if color_lines is None:
        color_lines = ['black']*(len(data)-1)

    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc

    # Create the individual plots from the data
    numDataArr = len(data); numDataElem = len(data[0])
    TempLine_Collecs = []
    for idx in range(numDataElem):
        Related_Pts = []
        for arr in data:
            Related_Pts.append(arr[idx])
        TempY = np.asarray(Related_Pts); TempY = np.stack((TempY, np.roll(TempY,-1)), axis=1)[:-1]
        TempX = np.asarray(posit); TempX = np.stack((TempX, np.roll(TempX,-1)), axis=1)[:-1]
        for n in range(TempY.shape[0]):
            TempLine_Collecs.append(np.stack((TempX[n], TempY[n]), axis=1))

        scatter = mscatter(posit, Related_Pts, c=marker_color, s=marker_size, m=marker_type, ax=ax)

    for idx in range(numDataArr-1):
        TempColl = LineCollection(TempLine_Collecs[idx::numDataArr-1], colors=color_lines[idx], linewidths=1, zorder=0.5)
        ax.add_collection(TempColl)

    if not figover:
        ax.set_xticks(posit)
        ax.set_xticklabels(str_labels)
        plt.draw()
    elif not labels is None:
        ax.set_xticks(posit)
        ax.set_xticklabels(str_labels)

def LinearInterp(Xdata, Ydata, NewXdata, kind='linear', fill_value='extrapolate'):
    # Function to interpolate a function to find some in-between values
    # Xdata: Sorted array of values from an independent variable
    # Ydata: Corresponding values of Xdata in a dependent variable
    # NewXdata: New independent values from the same variable in Xdata
    # kind, fill_value: Check types in
    #       https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html

    f = interp1d(Xdata, Ydata, kind=kind, fill_value=fill_value)
    return f(NewXdata)

def Event_Time_to_Value_Interpol(Timings, Time_Vect, Value_Vect, kind="linear"):
    # Function to transform continuos event timings into corresponding values
    # from a Time Serie (Values per Time pair) by interpolation.
    # Timing: Numpy array with the times when an event happens.
    # Time_Vect: Vector of continuos, sequential time used to time the events.
    # Value_Vect: Vector with the values assigned to each point of the Time_Vect.
    # kind: (Optional) Type of interpolation. See 'LinearInterp' function for more details

    Inter_Points = LinearInterp(Time_Vect, np.arange(Time_Vect.size), Timings, kind=kind)
    Value_Interp = LinearInterp(np.arange(Value_Vect.size), Value_Vect, Inter_Points, kind=kind)

    return Value_Interp

def Interp_SeqSubs_Periods(Series, Ref_Pairs, Order=0):
    # Series: Time series to make the sequential substraction
    # Ref_Pairs: Numpy array of reference points of the periods to be compared contiguosly, with shape (n, 2)
    # Order: If 0, the first period will be replaced with its interpolated version of the substraction.
    #        If 1, the second period will be replaced.

    Ref_Quads = np.hstack((Ref_Pairs, np.roll(Ref_Pairs,-1, axis=0)))
    Ref_Quads = Ref_Quads[:Ref_Quads.shape[0]-1,:] # Remove last invalid quadruplet

    Subst_Series = np.asarray([np.nan]*Series.size)
    for idx in range(Ref_Quads.shape[0]):
        Prev_Y = Series[Ref_Quads[idx][0]:Ref_Quads[idx][1]]; Prev_X = np.linspace(0, 1, num=Prev_Y.size)
        Next_Y = Series[Ref_Quads[idx][2]:Ref_Quads[idx][3]]; Next_X = np.linspace(0, 1, num=Next_Y.size)
        if Order == 0:
            New_Next_Y = LinearInterp(Next_X, Next_Y, Prev_X, kind='linear')
            Subst_Series[Ref_Quads[idx][0]:Ref_Quads[idx][1]] = New_Next_Y - Prev_Y
            if idx == Ref_Quads.shape[0]-1: # Correction at the end to not dismiss the second period
                New_Prev_Y = LinearInterp(Prev_X, Prev_Y, Next_X, kind='linear')
                Subst_Series[Ref_Quads[idx][2]:Ref_Quads[idx][3]] = Next_Y - New_Prev_Y
        else:
            New_Prev_Y = LinearInterp(Prev_X, Prev_Y, Next_X, kind='linear')
            Subst_Series[Ref_Quads[idx][2]:Ref_Quads[idx][3]] = Next_Y - New_Prev_Y
            if idx == 0: # Correction at the beggining to not dismiss the first period
                New_Next_Y = LinearInterp(Next_X, Next_Y, Prev_X, kind='linear')
                Subst_Series[Ref_Quads[idx][0]:Ref_Quads[idx][1]] = New_Next_Y - Prev_Y

    return Subst_Series

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

def Remove_Run_Sequences(Arr, elems, simplefirst=None, Rng_Scores=None):
    # Function to remove all elements of a run except the first or last element.
    # Arr: 1-D numpy array to adjust.
    # elem: List of elements repeated in the runs to remove
    # simplefirst: (Optional) List of Booleans matching the number of element in 'elem'.
    #              If True, only the firt element will be kept. If False, the
    #              selection will be the last element, or in case of providing
    #              Rng_Scores, the maximum value of the external range score.
    #              Default is all True.
    # Rng_Scores: (Optional) 1-D numpy array with the size of Arr that contains
    #             scores to sort the sequences with an external criteria when
    #             simplefirst is False.

    if simplefirst is None:
        simplefirst = [True]*len(elems)
    elif not isinstance(simplefirst, list):
        raise TypeError("Remove_Run_Sequences Abort -> The 'simplefirst' parameter must be a list of booleans")
    elif len(simplefirst) != len(elems):
        raise TypeError("Remove_Run_Sequences Abort -> The 'simplefirst' and 'elems' lists must have the same number of elements")

    if not Rng_Scores is None:
        if Rng_Scores.shape != Arr.shape:
            raise TypeError("Remove_Run_Sequences Abort -> The 'ExtFuncDict['Data']' entry must have the same size as input Array")

    Ranges = np.zeros((0,2)); Directive = np.zeros(0)
    for idx in range(len(elems)):
        Tmp = find_runs(Arr, elems[idx])
        Ranges = np.concatenate((Ranges, Tmp))
        if simplefirst[idx]:
            direc = np.ones(Tmp.shape[0])
        else:
            direc = np.zeros(Tmp.shape[0])
        Directive = np.concatenate((Directive, direc))

    PtsLeft = []
    for idx in range(Ranges.shape[0]):
        run = Ranges[idx]
        if Directive[idx]:
            PtsLeft.append(np.int_(run[0]))
        else:
            if not Rng_Scores is None:
                ini = np.int_(run[0]); end = np.int_(run[1])
                PtsLeft.append(ini + np.argmax(Rng_Scores[ini:end]))
            else:
                PtsLeft.append(np.int_(run[1]-1))

    PtsLeft.sort()

    return np.take(Arr, PtsLeft), PtsLeft

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

def Recursive_DimProjs(dim_names, to1D=False):
    # Function to construct the skeleton for the projection instructions by recursive programming

    dims = len(dim_names)
    if not to1D and dims < 3: return None # Only valid for dimensions higher than 2D
    elif to1D and dims < 2: return None # Only valid for dimensions higher than 1D
    dim_list = list(range(dims))
    list_dim_comp = list(combinations(dim_list, dims-1))
    Dict_Proj = {}
    for plane in list_dim_comp:
        d_names = [dim_names[dim] for dim in plane]
        Dict_Proj["-".join(d_names)] = {}
        temp = Recursive_DimProjs(d_names, to1D=to1D)
        if not temp is None: Dict_Proj["-".join(d_names)].update(temp)

    return Dict_Proj

def Instructions_DimProjs(dim_names, to1D=False):
    # Funtion to create the instruction for parallel projection calculations:
    # dim_names: List with the names of the dimensions (in their order on the original dataset)
    # to1D: (Optional) If True, the end of the projections will be until the 1D variables.

    Dict_Proj = Recursive_DimProjs(dim_names, to1D=to1D)
    KeyMap = Create_Dict_KeyMap(Dict_Proj)
    for key in KeyMap:
        list_d = key[0][-1].split("-")
        if len(key[0]) < 2:
            plane = tuple([dim_names.index(dim) for dim in list_d])
        else:
            list_prev = key[0][-2].split("-")
            plane = tuple([list_prev.index(dim) for dim in list_d])
        Temp = getFromDict(Dict_Proj, key[0])
        Temp.update({"Index":plane})

    return Dict_Proj, KeyMap
