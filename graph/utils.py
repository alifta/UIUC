# Import
# ------

import os
# import sys

import numpy as np

import sqlite3

# Database
# --------


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


# System
# ------


def dir_walk(path, ext='', save=False):
    """
    Walk through a directory, find all the file with specified extension
    """
    # Set extension to a specific extension if needed
    f = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # relative path of file
            relative_path = os.path.join(root, file)
            # extention (type of file)
            ext_of_file = os.path.splitext(relative_path)[-1].lower()[1:]
            # if extension is set and equal to what we want
            if ext != '' and ext_of_file == ext:
                f.append(os.path.abspath(relative_path))
            # if extension is not set add the file anyway
            else:
                f.append(os.path.abspath(relative_path))
    f.sort()
    if save:
        np.savetxt('files.csv', f, delimiter=',', fmt='%s')
    return f


def line_count(filename):
    """
    Count the number of lines in the input file
    """
    # count = 0
    # for line in open(filename).xreadlines(  ): count += 1
    # return count
    # OR
    return len(open(filename).readlines())


# --- Colors ---


def colors_create(number_of_colors=1, color_map='Wistia', output=False):
    """
    create a series of colors from the selected spectrum, e.g., Wistia [cold to hot]
    """
    color_list = []
    cmap = cm.get_cmap(color_map, number_of_colors)
    for i in range(cmap.N):
        # will return rgba, we take only first 3 so we get rgb
        rgb = cmap(i)[:3]
        # print(matplotlib.colors.rgb2hex(rgb))
        color_list.append(matplotlib.colors.rgb2hex(rgb))
    if output:
        for i in range(len(color_list)):
            plt.scatter(i, 1, c=color_list[i], s=20)
    return color_list


# --- Data Structure ---


def dict_add(dictionary, key, value):
    """
    Add a key:value to dictionary if key does not already exist
    """
    if key not in dictionary:
        dictionary[key] = value


def dict_lookup(dictionary, key):
    """
    Search the given KEY in dictionary ...
    found -> return its value (could be an index assign to that value)
    not found -> add the key and return its value (which is a new index)
    useful for creating hash table of KEY->INDEX
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


def list_intersection(lst1, lst2, version=4):
    """
    Intersection of two list or all common elements of two lists
    """
    if version == 1:
        return [value for value in lst1 if value in lst2]
    elif version == 2:
        return list(set(lst1) & set(lst2))
    elif version == 3:
        if len(lst1) > len(lst2):
            return set(lst1).intersection(lst2)
        else:
            return set(lst2).intersection(lst1)
    elif version == 4:  # O(n)
        temp = set(lst2)
        return [value for value in lst1 if value in temp]


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
        ranked.index = list(zip(temp[0],temp[1]))
    # If the rank values are needed then return entire series
    if return_rank:
        return ranked
    # Otherwise return ranked index of items
    return list(ranked.index)


# Test for rank
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
    Breakdown a list into chunks of sub-list with size of n
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


# --- Save & load ---


def fig_save(fig_id, tight_layout=True, fig_extension='png', resolution=300):
    """
    Method for saving figure
    """
    path = os.path.join(IMAGES_PATH, fig_id + '.' + fig_extension)
    print('saving figure ...', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def dict_save(data, file='data', method='n', sort=False):
    """
    Save input dictionary using different method:
        n: numpy (npy) -> good for saving the data type
        c: pandas (csv) -> good for looking at data after saving and sorting
        j: json -> also good for looking at data (simple key type) and sorting
        p: pickle -> should be fastest, good for simple data type
    """
    # npy
    if method == 'n':
        filename = file + '.npy'
        np.save(filename, data)
    # csv
    elif method == 'c':
        filename = file + '.csv'
        if not sort:
            pd.DataFrame.from_dict(data, orient='index'
                                   ).to_csv(filename, header=False)
        else:
            pd.DataFrame.from_dict(data, orient='index').sort_index(
                axis=0
            ).to_csv(filename, header=False)
    # json
    elif method == 'j':
        filename = file + '.json'
        with open(filename, 'w') as fp:
            if not sort:
                json.dump(data, fp)
            else:
                json.dump(data, fp, sort_keys=True, indent=4)
    # pickle
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


def list_save(
    input_list,
    file_name='list',
    file_postname='',
    file_extension='csv',
    delimiter=',',
    replace=True,
):
    """
    Method for saving list
    """
    if len(postname) > 0:
        file_name = file_name + '_' + file_postname
    file_path = os.path.join(OUTPUT_PATH, file_name + '.' + file_extension)
    if os.path.exists(file_input):
        print('file already exist.')
        if replace:
            print('overwriting ... ')
            np.savetxt(file_path, input_list, delimiter=delimiter, fmt='%s')
        else:
            print('saving ... ')
            file_path = os.path.join(
                OUTPUT_PATH, file_name + '_new.' + file_extension
            )
            np.savetxt(file_path, input_list, delimiter=delimiter, fmt='%s')
        print('done!')
    else:
        print('saving ... ')
        np.savetxt(file_path, input_list, delimiter=delimiter, fmt='%s')
        print('done!')


# --- Linear Algebra ---

# def array_top_n(arr, top_N=1):
#     """
#     find top 'N' values in 2d numpy array
#     """
#     idx = np.argpartition(arr, arr.size - top_N, axis=None)[-top_N:][::-1]
#     result = np.column_stack(np.unravel_index(idx, arr.shape))
#     return [(e[0], e[1]) for e in result]


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


# --- FOLDERS ---

# --- Common Folders ---
# ROOT_PATH = '.'
ROOT_PATH = os.getcwd()
DATA_PATH = os.path.join(ROOT_PATH, 'data')
os.makedirs(DATA_PATH, exist_ok=True)
IMAGE_PATH = os.path.join(ROOT_PATH, 'image')
os.makedirs(IMAGE_PATH, exist_ok=True)
# --- Project Specific Folders ---
DB_PATH = os.path.join(ROOT_PATH, 'db')
os.makedirs(DB_PATH, exist_ok=True)
FILE_PATH = os.path.join(ROOT_PATH, 'file')
os.makedirs(DB_PATH, exist_ok=True)
NET_PATH = os.path.join(ROOT_PATH, 'network')
os.makedirs(NET_PATH, exist_ok=True)
HITS_PATH = os.path.join(ROOT_PATH, 'hits')
os.makedirs(HITS_PATH, exist_ok=True)