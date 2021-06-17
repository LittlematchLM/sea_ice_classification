import numpy as np
import glob
import cv2
import io
import cv2
from PIL import Image


def cut_img(image_arr, cutsize):
    # 从四边剪短cutsize大小
    return image_arr[cutsize-1:-cutsize,cutsize-1:-cutsize]

sig_dir = r'E:\python_workfile\sea_ice_classification\training4\sigmod0'
aari_dir = r'E:\python_workfile\sea_ice_classification\training4\mask\npy'
save_dir = r'E:\python_workfile\sea_ice_classification\training4\sigmod0\fill_value'

sig_files = glob.glob(sig_dir+'\\*.png')
aari_files = glob.glob(aari_dir + '\\*.npy')

for i in range(len(sig_files)):

    day = sig_files[i].split('\\')[-1].split('.')[0]
    aari_npy = np.load(aari_files[i])
    sig_file = cv2.imread(sig_files[i])
    b = cut_img(aari_npy,28)

    fill_index = np.where(b == 0)
    sig_file[fill_index[0],fill_index[1],:] = 0

    cv2.imwrite(save_dir+'\\'+str(day)+'.png', sig_file)

aari_files