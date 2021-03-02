import numpy as np

# List
# ----

# Save list to CSV file
x = np.arange(0.0, 5.0, 1.0)
np.savetxt('test.csv', x, delimiter=',', fmt='%s')

import os


def path_edit(file_names, file_label='', folder_name='', folder_label=''):
    """
    Add (or remove) an input label to list of labels
    Type of input_label can be integer or string
    List of labels are assume to have a name like:
    ['folder/file.extension','folder/file.extension']
    Output looks like:
    ['folder/file_label.extension','folder/file_label.extension']
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


print(
    path_edit(
        file_names=['data.csv', 'image.png'],
        file_label='1',
        folder_name='project',
        folder_label='current'
    )
)
print(
    path_edit(
        file_names=['data.csv', 'image.png'],
        file_label='1',
        folder_name='',
        folder_label='current'
    )
)
