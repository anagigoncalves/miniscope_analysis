import datetime, copy
import numpy as np
import pandas as pd
import h5py
from scipy.io import savemat

def Save_Dict_to_Matlab(Dict, filename):
    # Save simple nested dictionary into a Matlab compatible file
    RenameData = tl.Dict_Key_Rename_to_MATLAB(Dict)
    savemat(filename, RenameData)

def Save_Dict_to_HDF5(Dict, filename):
    # Modified from: https://codereview.stackexchange.com/questions/120802

    def recursively_save_dict_contents_to_group(h5file, path, dic):
        for key, item in dic.items():
            orig_type = type(item)
            if isinstance(item, pd.Series):
                item = item.to_numpy()
            if isinstance(item, datetime.datetime):
                TempList = item.strftime("%Y %m %d %H %M %S %f").split(" ")
                item = np.asarray([int(num) for num in TempList])
            if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, str, bytes, list, tuple)):
                if isinstance(item, (int, float, list, tuple)):
                    # Transforming list into numpy arrays
                    if isinstance(item, (list, tuple)):
                        if all([isinstance(elem, datetime.datetime) for elem in item]):
                            TempList = []
                            for elem in item:
                                Temp = elem.strftime("%Y %m %d %H %M %S %f").split(" ")
                                TempList.append([int(num) for num in Temp])
                            item = TempList
                    item = np.asarray(item)
                    if "U" in str(item.dtype): item = item.astype("S") # Change list of strings into format compatible with HDF5
                elif isinstance(item, (str, bytes)):
                    if "Attrs" in path:
                        prev = "/".join(path[:path.find("Attrs")].split("/")[:-1]) + "/"
                        if not prev in h5file: h5file.create_group(prev)
                        h5file[prev].attrs[key] = item
                    else: h5file[path + key] = item
                    continue
                if issubclass(item.dtype.type, np.integer) and not issubclass(item.dtype.type, np.int64):
                    # Change numpy integers to 64-bits for general compatibility
                    item = item.astype(np.int64)
                try:
                    if "Attrs" in path:
                        prev = "/".join(path[:path.find("Attrs")].split("/")[:-1]) + "/"
                        if not prev in h5file: h5file.create_group(prev)
                        h5file[prev].attrs[key] = copy.deepcopy(item)
                    else: h5file[path + key] = item
                except Exception as e:
                    raise TypeError('Save_Dict_to_HDF5 Error -> Cannot save "%s" with original %s type\n\tPath: %s'% (key,orig_type,path))
            elif isinstance(item, np.bool_):
                item = bool(item)
            elif isinstance(item, dict):
                recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
            else:
                raise TypeError('Save_Dict_to_HDF5 Error -> Cannot identify the object "%s" with original %s type\n\tPath: %s'% (key,orig_type,path))

    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', Dict)

def Load_Dict_from_HDF5(filename, attrs=False):
    # Modified from: https://codereview.stackexchange.com/questions/120802
    # attrs: (Default False) If True, the routine will also load the attributes of each object.
    #        If "Root", it will only load the attributes of the first file node.

    def recursively_load_dict_contents_from_group(h5file, path, attrs):
        ans = {}
        if len(ans) == 0 and not attrs is False:
            ans["Root Attrs"] = {key:(elem if not isinstance(elem, h5py._hl.base.Empty) else []) for key, elem in dict(h5file['/'].attrs).items()}
            if "oo" in attrs: attrs = False
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item[()]
                if attrs: ans["%s_Attrs" % key] = dict(item.attrs)
            elif isinstance(item, h5py._hl.base.Empty):
                ans[key] = []
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', attrs)
                if attrs: ans[key]["Attrs"] = dict(item.attrs)
        return ans

    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/', attrs)

def dict_recursive_items(dictionary, level=0):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, value, level)
            yield from dict_recursive_items(value, level+1)
        else:
            yield (key, value, level)

def check_types_dict(dictionary, flag=None):
    # Function to quickly check the structure of a nested dictionary
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
    # Creates the full map of keys inside a nested dictionary

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
    # It works for almost all dictionary entries, except for dictionaries inside
    # lists or tuples (nested dictionaries with ending branches of lists or tuples do work!!)

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

def getFromDict(dataDict, mapList):
    # Function that gets the data inside of a nested dictionary by
    # opening directly the given keys in a list (ordered by hierarchy)
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    # Function that changes the data inside of a nested dictionary by
    # opening directly the given keys in a list (ordered by hierarchy)
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
