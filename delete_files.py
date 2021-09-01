import glob
import os

dir_small = r'E:\python_workfile\sea_ice_classification\training6\train_data\npy'
dir_all = r'E:\python_workfile\sea_ice_classification\training6\train_data'

small_files = glob.glob(dir_small + '\\*.npy')
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

#
del_dir = r'E:\python_workfile\sea_ice_classification\training4\sigmod0\small_size'
sigmod_small_size_file = glob.glob(del_dir + '\*.png')
del_num = ['002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012',
           '021', '022', '023', '024', '025', '026', '027', '028', '029', '059', '060',
           '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071',
           '072', '073', '074', '075', '078', '079', '084', '085', '086', '087', '088',
           '128', '129', '130', '131', '132', '133', '134', '139', '140', '141', '142',
           '143', '144']
for file in sigmod_small_size_file:
    num = file.split('\\')[-1].split('.')[0][-3:]
    num = str(num)
    # print(num)
    if num in del_num:
        print(file)
        os.remove(file)
