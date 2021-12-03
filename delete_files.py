import glob
import os

dir_small = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real'
dir_all = r'E:\python_workfile\sea_ice_classification\training7\dataset\aari\pic'

small_files = glob.glob(dir_small + '\\*.png')
big_files = glob.glob(dir_all + '\\*.png')

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
        del_list.append(dir_all + '\\' + big_list[i] + '.png')

for file in del_list:
    try:
        os.remove(file)
    except FileNotFoundError:
        print(file)




