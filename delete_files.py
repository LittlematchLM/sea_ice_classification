import  glob
import os


dir_small = r'E:\python_workfile\sea_ice_classification\training3\mask\aari\jet_small_size'
dir_all = r'E:\python_workfile\sea_ice_classification\training3\mask\aari\small_size'

small_files = glob.glob(dir_small + '\\*.png')
big_files= glob.glob(dir_all + '\\*.png')

small_list =[]

for file in small_files:
    # print(file.split('\\')[-1].split('.')[0])
    small_list.append(file.split('\\')[-1].split('.')[0].split('_')[-1])

big_list = []
for file in big_files:
    # print(file.split('\\')[-1].split('.')[0])
    big_list.append(file.split('\\')[-1].split('.')[0].split('_')[-1])

small_list.sort()
big_list.sort()


del_list =[]
for i in range(len(big_list)):
    if (big_list[i] in small_list) == False:
        # del_list.append(dir_all + '\\' +'HY2B_sca_sigmod0_'+ big_list[i] + '.png')
        del_list.append(dir_all + '\\' + big_list[i] + '.png')


for file in del_list:
    try:
        os.remove(file)
    except FileNotFoundError:
        print(file)




#  删除散射计投影中间的圈
sigmod_small_size_file = glob.glob(r'E:\python_workfile\sea_ice_classification\training3\sigmod0\small_size' + '\*.png')
del_num = ['061','062','063','064','065','066','067','068','069','070','071','072','078','079']
for file in sigmod_small_size_file:
    num = file.split('\\')[-1].split('.')[0][-3:]
    num = str(num)
    # print(num)
    if num in del_num:
        print(file)
        os.remove(file)