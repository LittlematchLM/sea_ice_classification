''':key
删除当前dir内的所有包含数值0的npy文件
'''

import os
import numpy as np
import glob

# npy_files[0].split('\\')[-1].split('.')[0]


npy_dir = r'E:\python_workfile\sea_ice_classification\training3\mask\aari\npy'

npy_files = glob.glob(npy_dir + '\*.npy')

del_list = []
for file in npy_files:
    pic = np.load(file)
    if 0 in set(pic.flatten()):
        del_list.append(file)

for file in del_list:
    try:
        os.remove(file)
    except FileNotFoundError:
        print(file)


