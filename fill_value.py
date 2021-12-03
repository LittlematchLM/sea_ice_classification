import numpy as np
import glob
import cv2
import io
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 根据AARI的像素位置填充HY-2B 后向散射系数数据
# 再根据
def cut_img(image_arr, cutsize):
    # 从四边剪短cutsize大小
    return image_arr[cutsize - 1:-cutsize, cutsize - 1:-cutsize]

def add_img_color(image_arr):
    new_aari_array = np.full(shape=(image_arr.shape[:2]), fill_value=0)
    for i in range(3):
        new_aari_array += image_arr[:, :, i]
    return new_aari_array

sig_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real'
aari_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\aari'

sig_files = glob.glob(sig_dir + '\\*.png')
aari_files = glob.glob(aari_dir + '\\*.npy')

aari_npy_save_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\aari\npy_real'
sig_npy_save_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real_use'


for aari_file, sig_file in zip(aari_files[:], sig_files[:]):
    day = sig_file.split('\\')[-1].split('.')[0]
    aari_npy = np.load(aari_file)
    sig_npy = cv2.imread(sig_file)

    aari_npy[add_img_color(sig_npy)==765]=0



    # b = cut_img(aari_npy, 28)

    fill_index = np.where(aari_npy == 0)
    sig_npy[fill_index[0], fill_index[1], :] = 255


    cv2.imwrite(sig_npy_save_dir + '\\' + str(day) + '.png', sig_npy)
    np.save(aari_npy_save_dir+'\\'+str(day)+'.npy',aari_npy)

#
# plt.imshow(sig_npy)
# plt.show()
#
# plt.imshow(aari_npy)
# plt.colorbar()
# plt.show()
