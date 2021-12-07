from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from skimage import exposure
from RSData import *
from HaiYangData import *
from PIL import Image
import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import io
from sea_ice_model import *


def get_gray_array_from_grid(grid, vmax, vmin):
    fig = plt.figure(figsize=(7, 7))
    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    hy_m.pcolormesh(x_map, y_map, data=grid, cmap=plt.cm.jet, shading='auto', vmax=vmax, vmin=vmin,
                    latlon=True)
    plt.axis('off')
    # 从图像中提取数组，并取灰度值
    plot_img_np = get_img_from_fig(fig)
    grey_img_np = 0.299 * plot_img_np[:, :, 0] + 0.587 * plot_img_np[:, :, 1] + 0.114 * plot_img_np[:, :, 2]
    plt.close()
    return grey_img_np


def composite_img(weight, height, arr1, arr2, arr3):
    rgbArray = np.zeros((weight, height, 3), 'uint8')
    rgbArray[..., 0] = arr1
    rgbArray[..., 1] = arr2
    rgbArray[..., 2] = arr3
    return Image.fromarray(rgbArray)


VV_npy_dir = r'E:\python_workfile\sea_ice_classification\training6\split_VV_HH\VV\npy'
HH_npy_dir = r'E:\python_workfile\sea_ice_classification\training6\split_VV_HH\HH\npy'

VV_npy_files = glob.glob(VV_npy_dir + '\\*.npy')
HH_npy_files = glob.glob(HH_npy_dir + '\\*.npy')

img_save_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real'
satellite = r'HY2B'
sensor = r'SCA'
hy_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=25000)

# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")
transformer = HaiYangData.set_transformer(crs, crs2)
transformer_back = HaiYangData.set_transformer(crs2, crs)
x_map, y_map = hy_sca.get_map_grid(transformer_back)

for VV_npy_file, HH_npy_file in zip(VV_npy_files, HH_npy_files):
    name = VV_npy_file.split('\\')[-1].split('.')[0]
    HH_grid = np.load(HH_npy_file)
    VV_grid = np.load(VV_npy_file)

    # 为消除VV极化方式照到的地方HH极化照不到，用HH极化的grid扣上去晒掉VV极化有HH极化没有的部分
    VV_grid[np.isnan(HH_grid)] = np.nan
    VV_add_HH_grid = HH_grid + VV_grid
    VV_sub_HH_grid = HH_grid - VV_grid
    ratio_grid = VV_grid / HH_grid
    polar_ratio_grid = VV_sub_HH_grid / VV_add_HH_grid

    HH_gray = get_gray_array_from_grid(HH_grid, -5, -25)
    VV_gray = get_gray_array_from_grid(VV_grid, -5, -25)
    polar_ratio_gray = get_gray_array_from_grid(polar_ratio_grid, 0.08, -0.08)
    ratio_gray = get_gray_array_from_grid(ratio_grid, 1.25, 0.75)


    # # 生成lat,lon,VV/HH
    # img1_save_path = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\lat_lon_ratio'
    # img1 = composite_img(1260, 1260, lon_gray, lat_gray, ratio_gray)
    # img1.save(img1_save_path + f'\\{name}.png')
    #
    # # 生成lat_lon_polaration
    # img2_save_path = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\lat_lon_polaration'
    # img2 = composite_img(1260, 1260, lon_gray, lat_gray, polar_ratio_gray)
    # img2.save(img2_save_path + f'\\{name}.png')

    # 生成VV_ratio_polaration
    img3_save_path = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_ratio_polaration'
    img3 = composite_img(1260, 1260, VV_gray, ratio_gray, polar_ratio_gray)
    img3.save(img3_save_path + f'\\{name}.png')
