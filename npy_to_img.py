'''
将npy转成只有存储原始数组的png图片
'''
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import cv2

npy_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\aari\npy_real\small_size\npy'
npy_files = glob.glob(npy_dir + r'\*.npy')
save_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\aari\npy_real\small_size\pic_gray'

for file in npy_files:

    c = np.load(file)
    cv2.imwrite(save_dir+'\\' + file.split('\\')[-1].split('.')[0] + '.png',c, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

