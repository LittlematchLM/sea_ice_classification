import glob
import os

dir_small = r'E:\python_workfile\sea_ice_classification\training6\aari\npy'
dir_all = r'E:\python_workfile\sea_ice_classification\training6\split_VV_HH\VV\npy'

small_files = glob.glob(dir_small + '\\*.npy')
big_files = glob.glob(dir_all + '\\*.npy')

small_list = []

for file in small_files:
    # print(file.split('\\')[-1].split('.')[0])
    small_list.append(file.split('\\')[-1].split('.')[0].split('_')[-1])

big_list = []
for file in big_files:
    # print(file.split('\\')[-1].split('.')[0])
    big_list.append(file.split('\\')[-1].split('.')[0].split('_')[-1])

small_list.sort()
big_list.sort()

del_list = []
for i in range(len(big_list)):
    if (big_list[i] in small_list) == False:
        # del_list.append(dir_all + '\\' +'HY2B_sca_sigmod0_'+ big_list[i] + '.png')
        del_list.append(dir_all + '\\' + big_list[i] + '.npy')

for file in del_list:
    try:
        os.remove(file)
    except FileNotFoundError:
        print(file)




