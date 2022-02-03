# Module with functions to solve Nearest Neighbour problems
# on datasets comprising time series

import numpy as np
import numba as nb
from numba.typed import List
from tqdm import tqdm
import time, warnings

@nb.jit(nopython=True)
def NN_Loop(Event1, Event2, MinDiff, MinMaxDiff, type, unique, verbose):

    if type == 0:
        if MinMaxDiff[0] == 0:
            def FuncEdges(x):
                return np.append(np.abs(np.diff(np.asarray([elem[1] for elem in x]))), 0)
        else:
            def FuncEdges(x):
                return np.ones(len(x))
    elif type == 1:
        def FuncEdges(x):
            return np.append(np.diff(np.asarray([elem[1] for elem in x])), 0)
    elif type == 2:
        def FuncEdges(x):
            return np.append(-np.diff(np.asarray([elem[1] for elem in x])), 0)

    Set0 = []
    EventEnum = list(enumerate(Event1))
    for n in range(len(EventEnum)):
        Set0.append((EventEnum[n][1] + MinDiff, 0, EventEnum[n][0]))
    Set1 = []
    EventEnum = list(enumerate(Event2))
    for n in range(len(EventEnum)):
        Set1.append((EventEnum[n][1], 1, EventEnum[n][0]))
    AllSetsTemp = Set0 + Set1

    MaxNum = len(AllSetsTemp); Counter = 0; Sep = MaxNum/100; Part = 1
    Pairs = []; Bool_Loop = True
    while Bool_Loop:
        AllSetsTemp = sorted(AllSetsTemp)
        if MinMaxDiff[0] != 0 and type == 0:
            DiffsPos = []
            for idx in range(len(AllSetsTemp)):
                BackDiff = (np.inf, 0, 0) # Backward distance
                for pos in range(idx, -1, -1):
                    IDset1 = AllSetsTemp[idx][1]; IDset2 = AllSetsTemp[pos][1]
                    Val1 = AllSetsTemp[idx][0]; Val2 = AllSetsTemp[pos][0]
                    Pt2 = AllSetsTemp[pos][2]
                    if IDset1 == IDset2 or np.abs(Val1 - Val2) < MinMaxDiff[0]:
                        continue
                    else:
                         BackDiff = (np.abs(Val1 - Val2), Val2, Pt2)
                         break
                ForwDiff = (np.inf, 0, 0) # Forward distance
                for pos in range(idx+1, len(AllSetsTemp)):
                    IDset1 = AllSetsTemp[idx][1]; IDset2 = AllSetsTemp[pos][1]
                    Val1 = AllSetsTemp[idx][0]; Val2 = AllSetsTemp[pos][0]
                    Pt2 = AllSetsTemp[pos][2]
                    if IDset1 == IDset2 or np.abs(Val1 - Val2) < MinMaxDiff[0]:
                        continue
                    else:
                         ForwDiff = (np.abs(Val1 - Val2), Val2, Pt2)
                         break
                if BackDiff[0] > ForwDiff[0]:
                    DiffsPos.append(ForwDiff)
                else:
                    DiffsPos.append(BackDiff)

            Diffs = np.asarray(list(map(lambda x : x[0], DiffsPos)))
            # if Diffs.size - len(AllSetsTemp) != 0:
            #     print(Diffs.size - len(AllSetsTemp))
            Positions = list(map(lambda x : x[1], DiffsPos))
            Points = list(map(lambda x : x[2], DiffsPos))

        else:
            Diffs = np.append(np.abs(np.diff(np.asarray([elem[0] for elem in AllSetsTemp]))), np.inf)

        # Detection of pairing type
        Edges = FuncEdges(AllSetsTemp)
        if not np.any(Edges > 0):
            Bool_Loop = False
            break

        # Maximum allowed differences
        ValidDiffs = np.where(Diffs < MinMaxDiff[1], Diffs, np.ones(Diffs.size)*np.inf)

        # Selection of minimal pair in iteration
        Valid_Comps = np.where(Edges > 0, ValidDiffs, np.ones(ValidDiffs.size)*np.inf)
        if np.all(np.isinf(Valid_Comps)):
            Bool_Loop = False
            break
        MinPairIdx = np.argmin(Valid_Comps)

        if AllSetsTemp[MinPairIdx][1] == 0:
            if MinMaxDiff[0] != 0 and type == 0:
                Pairs.append((AllSetsTemp[MinPairIdx], (Positions[MinPairIdx], 1, Points[MinPairIdx])))
                PosEvent1 = MinPairIdx
                PosEvent2 = AllSetsTemp.index((Positions[MinPairIdx], 1, Points[MinPairIdx]))
            else:
                Pairs.append((AllSetsTemp[MinPairIdx], AllSetsTemp[MinPairIdx+1]))
                PosEvent1 = MinPairIdx
                PosEvent2 = MinPairIdx + 1
        else:
            if MinMaxDiff[0] != 0 and type == 0:
                Pairs.append(((Positions[MinPairIdx], 0, Points[MinPairIdx]), AllSetsTemp[MinPairIdx]))
                PosEvent2 = MinPairIdx
                PosEvent1 = AllSetsTemp.index((Positions[MinPairIdx], 0, Points[MinPairIdx]))
            else:
                Pairs.append((AllSetsTemp[MinPairIdx+1], AllSetsTemp[MinPairIdx]))
                PosEvent1 = MinPairIdx + 1
                PosEvent2 = MinPairIdx
        if unique:
            AllSetsTemp.pop(max(PosEvent1, PosEvent2))
            AllSetsTemp.pop(min(PosEvent1, PosEvent2))
            Counter += 2
        else:
            AllSetsTemp.pop(PosEvent2)
            Counter += 1
        if len(AllSetsTemp) < 2:
            Bool_Loop = False
        if verbose:
            if Counter > Sep*Part:
                Part += 1
                #print(int(100*Counter/MaxNum))

    return Pairs

def NearestNeigbourgh_1D(Event1, Event2, type='=', MinMaxDiff=(0, np.inf), allow_crosses=False, unique=False, NumbaOptim=False, graph=False, verbose=False):
    # Event1, Event2: List of values (integers or floating) where an event is located
    # type: (Default '='). Define the comparison of the pairs. If '=', the absolute minimal
    #        distance will be taken. If '<' or '>', the minimal distance is chosen with the
    #        direction/order given by the symbol.
    # MinMaxDiff: (Optional) Tuple or List of integers (or np.inf) specifying the minimum and maximum difference allowed on the pairs.
    # allow_crosses: (Default False). Boolean to accept/reject pairs that cross other pairs.
    # unique: (Default False). Boolean to only allow one-to-one correspondance between elements of
    #         the two events. If False, Event1 can have multiple pairs from Event2. If True, the unique pair is chosen based on
    #         the minimal difference and the first minimum point, in that order. Only functionally relevant for type "=".
    # graph: (Optional Boolean). If True, plots the data and the pairs.
    # verbose: (Optional Boolean). If True, prints information of the process.


    if not isinstance(MinMaxDiff, list) and not isinstance(MinMaxDiff, tuple):
        raise TypeError('NearestNeigbourgh_1D Input Error -> MinMaxDiff should be a list or tuple of two integers (or integer and numpy.inf)')
    elif not isinstance(MinMaxDiff[0], int):
        raise TypeError('NearestNeigbourgh_1D Input Error -> MinMaxDiff should be a list or tuple of two integers (or integer and numpy.inf)')
    elif not isinstance(MinMaxDiff[1], int) and not np.isinf(MinMaxDiff[1]):
        raise TypeError('NearestNeigbourgh_1D Input Error -> MinMaxDiff should be a list or tuple of two integers (or integer and numpy.inf)')
    if MinMaxDiff[0] >= MinMaxDiff[1]:
        raise ValueError('NearestNeigbourgh_1D Input Error -> MinMaxDiff first element should be strictly less than the second element')

    if len(Event1) == 0 or len(Event2) == 0:
        Out_Dict = {'Pairs':[], 'Pairs Points':[], 'Unpaired Event 1':Event1,
                    'Unpaired Points 1':list(range(len(Event1))), 'Unpaired Event 2':Event2,
                    'Unpaired Points 2':list(range(len(Event2)))}
        return Out_Dict

    if type in ["<", ">", "="]:
        if type == '=':
            type = 0; MinDiff = 0
        elif type == '<':
            type = 1; MinDiff = MinMaxDiff[0]
        elif type == '>':
            type = 2; MinDiff = - MinMaxDiff[0]
    else:
        raise ValueError('NearestNeigbourgh_1D Input Error -> Type can only be "<", ">" or "="')

    OrgEvent1 = np.asarray(sorted(list(set(Event1))))
    OrgEvent2 = np.asarray(sorted(list(set(Event2))))

    start = time.time()
    if NumbaOptim:
        warnings.filterwarnings("ignore")
        if verbose:
            print('NearestNeigbourgh_1D -> NN_Loop on Numba-optimized code')
        Pairs = NN_Loop(OrgEvent1, OrgEvent2, MinDiff, MinMaxDiff, type, unique, verbose)
        warnings.filterwarnings("default")
    else:
        if verbose:
            print('NearestNeigbourgh_1D -> NN_Loop on normal Python code')
        Pairs = NN_Loop.py_func(OrgEvent1, OrgEvent2, MinDiff, MinMaxDiff, type, unique, verbose)
    end = time.time()

    if verbose:
        print('NearestNeigbourgh Loop time elapsed: %.03f seconds' %(end - start))

    Paired0 = [elem[0][0] - MinDiff for elem in Pairs]
    Paired1 = [elem[1][0] for elem in Pairs]
    Pts0 = [elem[0][2] for elem in Pairs]
    Pts1 = [elem[1][2] for elem in Pairs]
    OrgPairs = sorted(list(zip(Paired0, Paired1)), key=lambda x : x[0])
    OrgPts = sorted(list(zip(Pts0, Pts1)), key=lambda x : x[0])

    if not allow_crosses: # Check for crossings of pairs:
        # Use distance of Pairs to define order of rejection:
        DiffTemp = list(map(lambda x : np.abs(x[0] - x[1]), OrgPairs))
        CountTest = len(DiffTemp)
        while CountTest > 0:
            IdxPair = np.argmax(DiffTemp); SelPair = OrgPairs[IdxPair]
            try:
                PrevPair = OrgPairs[IdxPair-1]
            except:
                PrevPair = [0, 0]
            try:
                PosPair = OrgPairs[IdxPair + 1]
            except:
                PosPair = [np.inf, np.inf]
            # Check crossing possibilities:
            Condition1 = SelPair[1] < PrevPair[1] and SelPair[0] > PrevPair[0]
            Condition2 = SelPair[1] > PrevPair[1] and SelPair[0] < PrevPair[0]
            Condition3 = SelPair[1] < PosPair[1] and SelPair[0] > PosPair[0]
            Condition4 = SelPair[1] > PosPair[1] and SelPair[0] < PosPair[0]
            if any([Condition1, Condition2, Condition3, Condition4]):
                OrgPairs.pop(IdxPair); DiffTemp.pop(IdxPair)
                OrgPts.pop(IdxPair)
            else:
                DiffTemp[IdxPair] = np.NINF
            CountTest -= 1

        Paired0 = [elem[0] for elem in OrgPairs]
        Paired1 = [elem[1] for elem in OrgPairs]
        Pts0 = [elem[0] for elem in OrgPts]
        Pts1 = [elem[1] for elem in OrgPts]

    Unpaired0 = sorted(list(set(Event1).difference(set(Paired0))))
    Unpaired1 = sorted(list(set(Event2).difference(set(Paired1))))
    UnpairedPts0 = sorted(list(set(list(range(len(Event1)))).difference(set(Pts0))))
    UnpairedPts1 = sorted(list(set(list(range(len(Event2)))).difference(set(Pts1))))

    if graph:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        fig, ax = plt.subplots()
        ax.scatter(Event1, np.ones(len(Event1))*0.6, color='red', alpha=0.6)
        ax.scatter(Event2, np.ones(len(Event2))*0.5, color='blue', alpha=0.6)
        ax.scatter(Paired0, np.ones(len(Paired0))*0.6, color='red')
        ax.scatter(Paired1, np.ones(len(Paired1))*0.5, color='blue')
        LinesXJoin = np.stack((np.asarray(Paired0), np.asarray(Paired1)), axis=1)
        LinesYJoin = np.ones(LinesXJoin.shape)
        LinesYJoin[:,0] = LinesYJoin[:,0]*0.6; LinesYJoin[:,1] = LinesYJoin[:,1]*0.5
        LinesAll = np.stack((LinesXJoin, LinesYJoin), axis=2)
        LineColl = LineCollection(LinesAll, linewidths=2, colors='purple', zorder=0.5)
        ax.add_collection(LineColl)

        plt.show()

    Out_Dict = {'Pairs':OrgPairs, 'Pairs Points':OrgPts, 'Unpaired Event 1':Unpaired0,
                'Unpaired Points 1':UnpairedPts0, 'Unpaired Event 2':Unpaired1,
                'Unpaired Points 2':UnpairedPts1}

    return Out_Dict

def Test(function, **kwargs):

    Event1 = [1,4,9,13,15,18,20,23,29,30,34,37,43,55,60]
    Event2 = [2,8,16,22,28,31,35,40,46,59,62]

    Dict = function(*[Event1, Event2], **kwargs)

    print(Dict)

if __name__=='__main__':
    Test(NearestNeigbourgh_1D, graph=True, verbose=True)
