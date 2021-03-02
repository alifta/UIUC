# Import
# ------

import os
import json
import pickle

import numpy as np
import pandas as pd

import sqlite3

import matplotlib.pyplot as plt
from pylab import matplotlib, cm

# Database
# --------


def db_execute(db, query):
    """
    Execute a single query on database
    """
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(query)
        cur.close()
    except sqlite3.Error as e:
        print(e)
    finally:
        if (con):
            con.close()


def db_execute_many(db, query, data):
    """
    Execute a query (e.g. insert many) on database
    """
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.executemany(query, data)
        con.commit()
        cur.close()
    except sqlite3.Error as e:
        print(e)
    finally:
        if (con):
            con.close()


def db_row_count(db, table, output=False):
    """
    Count the number of entries in specified table of database
    """
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('SELECT COUNT(*) FROM {}'.format(table))
        cur.close()
    except sqlite3.Error as e:
        print(e)
    finally:
        if (con):
            con.close()
    count = cur.fetchall()[0][0]
    if output: print(f'{table} has {count} entries.')
    return count


# Files & Folders
# ---------------


def path_edit(file_names, file_label='', folder_name='', folder_label=''):
    """
    Edit paths of a list of file names as the following format
    {folder_name}/{folder_label}{file_name}_{file_label}.{extension}
    """
    # Check if list of file name is empty, return None
    if len(file_names) == 0:
        return

    # If folder name is empty, set it to current folder
    if len(folder_name) == 0:
        folder_name = os.getcwd()

    paths = []
    for file_name in file_names:
        file_name_new = ''
        if len(file_label) != 0:
            file_name_new = file_name.split('.')[
                0] + '_' + file_label + '.' + file_name.split('.')[1]
        else:
            file_name_new = file_name
        # Edit and add the new path to list of file name
        paths.append(os.path.join(folder_name, folder_label, file_name_new))

    return paths


def folder_walk(path, ext='', save=False):
    """
    Iterate inside folder and find all files with specified extension
    """
    f = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # Relative path
            relative_path = os.path.join(root, file)
            # Extention or type of file
            ext_of_file = os.path.splitext(relative_path)[-1].lower()[1:]
            # If extension is set and equal to desired type
            if ext != '' and ext_of_file == ext:
                f.append(os.path.abspath(relative_path))
            # If extension is not set, meaning we want all the files
            else:
                f.append(os.path.abspath(relative_path))
    f.sort()
    if save: np.savetxt('files.csv', f, delimiter=',', fmt='%s')
    return f


def file_line_count(filename):
    """
    Count number of lines in file
    """
    return len(open(filename).readlines())


# Colors
# ------


def colors_create(number_of_colors=1, color_map='Wistia', output=False):
    """
    create a list of colors from the selected spectrum e.g. Wistia, cold or hot
    """
    colors = []
    cmap = cm.get_cmap(color_map, number_of_colors)
    for i in range(cmap.N):
        # Return rgba, but only first 3 is needed, not alpha
        rgb = cmap(i)[:3]
        colors.append(matplotlib.colors.rgb2hex(rgb))
    if output:
        for i, color in enumerate(colors):
            plt.scatter(i, 1, c=color, s=20)
    return colors


# Structure
# ---------


def list_intersection(lst1, lst2):
    """
    Intersection of two list
    In other words, all common elements of two lists
    """
    temp = set(lst2)
    return [value for value in lst1 if value in temp]


def dict_save(data, file='data', method='n', sort=False):
    """
    Save input dictionary using different method:
        n: numpy (npy) -> good for saving the data type
        c: pandas (csv) -> good for looking at data after saving and sorting
        j: json -> also good for looking at data (simple key type) and sorting
        p: pickle -> should be fastest, good for simple data type
    """
    # NPY
    if method == 'n':
        filename = file + '.npy'
        np.save(filename, data)
    # CSV
    elif method == 'c':
        filename = file + '.csv'
        if not sort:
            pd.DataFrame.from_dict(data, orient='index'
                                   ).to_csv(filename, header=False)
        else:
            pd.DataFrame.from_dict(data, orient='index').sort_index(
                axis=0
            ).to_csv(filename, header=False)
    # JSON
    elif method == 'j':
        filename = file + '.json'
        with open(filename, 'w') as fp:
            if not sort:
                json.dump(data, fp)
            else:
                json.dump(data, fp, sort_keys=True, indent=4)
    # Pickle
    elif method == 'p':
        filename = file + '.p'
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def dict_load(file='data', method='n'):
    """
    Load input file into a dictionary using different method:
        n: numpy (npy)
        c: pandas (csv)
        j: json
        p: pickle
    """
    data = {}
    # npy
    if method == 'n':
        filename = file + '.npy'
        data = np.load(filename, allow_pickle='True').item()
    # csv
    elif method == 'c':
        filename = file + '.csv'
        data = pd.read_csv(filename, header=None,
                           index_col=0).T.to_dict('records')[0]
    # json
    elif method == 'j':
        filename = file + '.json'
        with open(filename, 'r') as fp:
            data = json.load(fp)
    # pickle
    elif method == 'p':
        filename = file + '.p'
        with open(filename, 'rb') as fp:
            data = pickle.load(fp)
    return data


def dict_add(dictionary, key, value):
    """
    Add {KEY:VALUE} to dict if key does not already exist
    """
    if key not in dictionary:
        dictionary[key] = value


def dict_lookup(dictionary, key):
    """
    Search KEY in dict
    If found -> return key's value i.e index
    If not found -> add key and return its value which is a new index
    Useful for creating hash table {KEY:INDEX}
    """
    value = 0
    if key not in dictionary:
        value = len(dictionary)
        dictionary[key] = value
    else:
        value = dictionary.get(key)
    return value


def label_amend(label_list, input_label, end=True):
    """
    Add (or remove) an input label to list of labels
    Type of input_label can be integer or string
    List of labels are assume to have a name like:
    ['folder/file.extension','folder/file.extension']
    Output looks like:
    ['folder/file_label.extension','folder/file_label.extension']
    """
    # Fist check to see if list and input_label are not None
    if len(label_list) > 0 and len(input_label) > 0:
        # Then amend all labels in the label list
        if end:
            # Can be use to amend file name (before the file type)
            # By default adds input label to end of all labels in the list
            label_list = [
                label.split('.')[0] + '_' + str(input_label) + '.' +
                label.split('.')[1] for label in label_list
            ]
        else:
            # Can be use to amend folder of a file and adds a pre-fix name to the folder
            # We assume we onle have one sub-folder-level (or character "/") in the name
            label_list_new = []
            root = os.getcwd()
            for label in label_list:
                path = os.path.join(
                    root,
                    label.split('/')[0] + '/' + str(input_label)
                )
                # Making sure the folder exist, otherwise create it
                os.makedirs(path, exist_ok=True)
                label_list_new.append(
                    label.split('/')[0] + '/' + str(input_label) + '/' +
                    label.split('/')[1]
                )
            label_list = label_list_new[:]
    return label_list


def rank(x, return_rank=False):
    """
    Rank items of a list from largest to smallest value
    and return a list of [(index,value,rank)]
    """
    # Input is list
    if isinstance(x, list):
        # Convert to series
        s = pd.Series(x)
    # Input is series
    if isinstance(x, pd.Series):
        # Only sort based on the index
        s = x.sort_index()
    # Input is 2D array
    if isinstance(x, np.ndarray):
        s = pd.Series(x.flatten())
    # Input is dictionary
    if isinstance(x, dict):
        s = pd.Series(x, index=sorted(x.keys()))
    # Rank the data
    ranked = s.rank(method='dense', ascending=False).astype(int).sort_values()
    # If input was 2D array change index to tuple (i,j) of matrix
    if isinstance(x, np.ndarray):
        temp = np.unravel_index(ranked.index, x.shape)
        ranked.index = list(zip(temp[0], temp[1]))
    # If the rank values are needed then return entire series
    if return_rank:
        return ranked
    # Otherwise return ranked index of items
    return list(ranked.index)


# Test rank
assert rank(
    {
        0: 2,
        1: 4,
        2: 6,
        3: 8,
        4: 10,
        5: 9,
        6: 7,
        7: 7,
        8: 7,
        9: 0,
        10: 1,
        11: 2
    }
) == [4, 5, 3, 6, 7, 8, 2, 1, 0, 11, 10, 9]


def breakdown(lst, num):
    """
    Breakdown a list into chunks of sublist of size N
    """
    # pprint(list(range(len(lst))))
    # pprint(lst)

    # Sort the list (high -> low)
    # Rank the sorted list
    ranks = pd.Series(lst).rank(method='dense',
                                ascending=False).astype(int).sort_values()
    # Divide the ranks into chunk of desired size
    chunks = [list(ranks.iloc[i:i + num]) for i in range(0, len(ranks), num)]
    # Dictionary of {rank : indices}
    rank_idx = {i: set() for i in set(ranks)}
    for idx, rank in ranks.items():
        # print(f'{rank} : {idx}')
        rank_idx[rank].add(idx)
    # Create a new chunk, but index of high ranks to low ranks
    bd = []
    for chunk in chunks:
        lst_temp = []
        for rank in chunk:
            # Picl a random index from the selected rank
            idx = rn.sample(rank_idx[rank], 1)[0]
            lst_temp.append(idx)
            rank_idx.get(rank).remove(idx)
        bd.append(lst_temp)
    return bd


# Figure
# ------


def fig_save(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """
    Method for saving figure
    """
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('saving figure ...', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Linear Algebra
# --------------


def top_n(arr, top_N=1, index=True):
    """
    find top 'N' values of 1d numpy array or list
    Return the index of top values (if index == True) or index and value as tuple
    """
    idx = np.argsort(arr)[::-1][:top_N]
    if index:
        return idx
    else:
        return [(e, arr[e]) for e in idx]


def array_top_n(arr, top_N=1):
    """
    find top 'N' values in 2d numpy array
    """
    # (1)
    # idx = np.argpartition(arr, arr.size - top_N, axis=None)[-top_N:][::-1]
    # result = np.column_stack(np.unravel_index(idx, arr.shape))
    # return [(e[0], e[1]) for e in result]
    # (2)
    idx = (-arr).argsort(axis=None, kind='mergesort')
    # idx = (-arr).argsort(axis=None, kind='mergesort')[:top_N]
    result = np.vstack(np.unravel_index(idx, arr.shape)).T
    return [(e[0], e[1]) for e in result]


def matrix_print(M, out_int=True):
    """
    Print input matrix in terminal, without any cut-off
    """
    for i, element in enumerate(M):
        if out_int:
            print(*element.astype(int))
        else:
            print(*element)