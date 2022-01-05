# # HY-2B SCA L2A后向散射系数投影

from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from skimage import exposure
from RSData import *
from HaiYangData import *

import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import io
import cv2
import matplotlib.patches as mpatches
from scipy.interpolate import griddata

satellite = r'HY2B'
sensor = r'SCA'
hy_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=60000)
coin_point = 0
# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")
transformer = HaiYangData.set_transformer(crs, crs2)
transformer_back = HaiYangData.set_transformer(crs2, crs)
x_map, y_map = hy_sca.get_map_grid(transformer_back)

'''
获取文件名称
'''
val_dir = r'E:\python_workfile\sea_ice_classification\training6\csv\add_threshold\val'
val_files = glob.glob(val_dir + '\\*.csv')
val_day = [val_file.split('\\')[-1].split('.')[0] for val_file in val_files]

lat_npy_files = glob.glob(r'E:\python_workfile\sea_ice_classification\training7\lat_lon_array\lat_small_size\*.npy')
lon_npy_files = glob.glob(r'E:\python_workfile\sea_ice_classification\training7\lat_lon_array\lon_small_size\*.npy')
pri_mask_files = glob.glob(
    (r'E:\python_workfile\sea_ice_classification\training7\output\VV_HH_polarratio\pridect\npy\*.npy'))
grid_save_path = r'E:\python_workfile\sea_ice_classification\training7\output\VV_HH_polarratio\pridect\grid'
png_save_path = r'E:\python_workfile\sea_ice_classification\training7\output\VV_HH_polarratio\pridect\full_png'
# lat_npy, lon_npy, pri_mask_npy = lat_npy_files[19], lon_npy_files[19], pri_mask_files[49*2:49*3][19]
pri_mask_files_lists = [pri_mask_files[i:i + 49] for i in range(0, len(pri_mask_files), 49)]

FI = mpatches.Patch(color='maroon', label='Fast Ice')
OI = mpatches.Patch(color='orange', label='Old Ice')
FYI = mpatches.Patch(color='lime', label='First Year Ice')
YI = mpatches.Patch(color='dodgerblue', label='Young Ice')
N = mpatches.Patch(color='midnightblue', label='Nilas')

# pri_mask_files_list = pri_mask_files_lists[5]


for i, pri_mask_files_list in enumerate(pri_mask_files_lists):
    day = val_day[i]
    value_array = np.empty(shape=(128, 128, 6))
    grid_array_VV = np.zeros((hy_sca.nlat, hy_sca.nlon))
    grid_num_array_VV = np.zeros((hy_sca.nlat, hy_sca.nlon))

    for lat_npy, lon_npy, pri_mask_file in zip(lat_npy_files, lon_npy_files, pri_mask_files_list):
        lat = np.load(lat_npy)
        lon = np.load(lon_npy)
        pri_mask = np.load(pri_mask_file).reshape((128, 128))

        value_array[:, :, 0] = lat
        value_array[:, :, 1] = lon
        value_array[:, :, 2], value_array[:, :, 3] = transformer.transform(value_array[:, :, 0], value_array[:, :, 1])
        value_array[:, :, 4] = pri_mask

        x = (value_array[:, :, 2] / hy_sca.resolution).astype(np.int16)
        y = (value_array[:, :, 3] / hy_sca.resolution).astype(np.int16)
        grid_array_VV[y, x] = pri_mask

    grid_array_VV[grid_array_VV == 0] = np.nan

    # 画图
    fig = plt.figure(figsize=(9, 9))
    fig.add_subplot(111)
    fig.set_tight_layout(True)  # reduce the spaces from margin outside the axis

    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    hy_m.fillcontinents()
    hy_m.pcolor(x_map, y_map, data=grid_array_VV, cmap=plt.cm.jet, shading='auto', latlon=True)
    plt.legend(loc='upper right', handles=[FI, OI, FYI, YI, N], title='Ice Type')
    # plt.colorbar(location='right', fraction=0.045)
    hy_m.drawparallels(np.arange(-90., 120., 10.), labels=[1, 0, 0, 0])
    hy_m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])
    # you can get a high-resolution image as numpy array!!
    plt.title(f'predict mask {day}')
    plt.savefig(png_save_path + f'\\{day}.png')
    # plt.show()

    # 存数据（1333，1333）
    save_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=30000)
    x_map_t, y_map_t = save_sca.get_map_grid(transformer_back)

    value_array = np.empty(shape=(128, 128, 6))
    grid_array_VV = np.zeros((save_sca.nlat, save_sca.nlon))
    grid_num_array_VV = np.zeros((save_sca.nlat, save_sca.nlon))

    for lat_npy, lon_npy, pri_mask_file in zip(lat_npy_files, lon_npy_files, pri_mask_files_list):
        lat = np.load(lat_npy)
        lon = np.load(lon_npy)
        pri_mask = np.load(pri_mask_file).reshape((128, 128))

        value_array[:, :, 0] = lat
        value_array[:, :, 1] = lon
        value_array[:, :, 2], value_array[:, :, 3] = transformer.transform(value_array[:, :, 0], value_array[:, :, 1])
        value_array[:, :, 4] = pri_mask

        x = (value_array[:, :, 2] / save_sca.resolution).astype(np.int16)
        y = (value_array[:, :, 3] / save_sca.resolution).astype(np.int16)
        grid_array_VV[y, x] = pri_mask
    grid_array_VV[grid_array_VV == 0] = np.nan
    np.save(grid_save_path + f'//{day}.npy', grid_array_VV)

# 记录数据nan的点 插值后用这个扣一遍
grid_nan_mask = np.isnan(grid_array_VV)

grid_nan_mask_flatten = grid_nan_mask.flatten()
grid_nan_mask_index = np.where(grid_nan_mask_flatten == True)[0]

grid_array_VV[grid_nan_mask] = 0

grid_interpolate = griddata(
    np.vstack((np.delete(x_map.flatten(), grid_nan_mask_index), np.delete(y_map.flatten(), grid_nan_mask_index))).T,
    np.delete(grid_array_VV.flatten(), grid_nan_mask_index),
    (x_map, y_map), method='nearest', rescale=True)

grid_array_VV[grid_nan_mask] = np.nan

fig = plt.figure()
hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
hy_m.fillcontinents()
hy_m.pcolor(x_map, y_map, data=grid_interpolate, cmap=plt.cm.jet, shading='auto', latlon=True)

plt.colorbar(location='right', fraction=0.045)
hy_m.drawparallels(np.arange(-90., 120., 10.), labels=[1, 0, 0, 0])
hy_m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])
# you can get a high-resolution image as numpy array!!
plt.savefig('202004222.png')
plt.show()

plt.pcolormesh(grid_interpolate)
plt.colorbar()
plt.show()

plt.pcolormesh(grid_array_VV)
plt.show()

dt_grid_files = glob.glob(r'I:\sea_ice_classification\training6\dt_output\split_VV_HH\VV\WEEK\grid\*.npy')

dt_grid = np.load(dt_grid_files[5])
# 记录数据nan的点 插值后用这个扣一遍
dt_grid_mask = np.isnan(dt_grid)

grid_interpolate[dt_grid_mask] = np.nan
