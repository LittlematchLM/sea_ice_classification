#!/usr/bin/env python
# coding: utf-8

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


def split_file_day(files):
    # 按照天来划分文件，同一天的内容在一个list里面
    file_list = []
    list = []
    for i in range(len(files)):

        if i == 0:
            list.append(files[i])
            continue

        if (files[i].split('_')[8].split('T')[0]) == (files[i - 1].split('_')[8].split('T')[0]):
            list.append(files[i])
        else:
            file_list.append(list)
            list = []
            list.append(files[i])
    file_list.append(list)
    return file_list


# In[3]:


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[4]:


def draw_sigmod_0(x_map, y_map, grid_array, save_path=None):
    fig = plt.figure(figsize=(9, 9))
    fig.add_subplot(111)
    fig.set_tight_layout(True)  # reduce the spaces from margin outside the axis

    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    hy_m.fillcontinents()
    hy_m.pcolormesh(x_map, y_map, data=grid_array, cmap=plt.cm.jet, vmax=-5, vmin=-25, latlon=True)
    # hy_m.pcolormesh(x_map, y_map, data=grid_array, cmap=plt.cm.jet,vmax=0, vmin=3, latlon=True)

    plt.colorbar(location='right')
    hy_m.drawparallels(np.arange(-90., 120., 10.), labels=[1, 0, 0, 0])
    hy_m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])
    # you can get a high-resolution image as numpy array!!
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.close()

    # return fig


# In[5]:


satellite = r'HY2B'
sensor = r'SCA'
hy_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=25000)
coin_point = 15
# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")

dir_path = r"j:\remote_sensing_data\back_scatter\HY-2B"

files = glob.glob(dir_path + '\*_pwp_250_0*.h5')
# files = glob.glob(dir_path + '\*_dps_250_0*.h5')

file_list = split_file_day(files)

# In[6]:


transformer = HaiYangData.set_transformer(crs, crs2)
transformer_back = HaiYangData.set_transformer(crs2, crs)

train_data_dir = r'E:\python_workfile\sea_ice_classification\data\train_data\pic_bar_new'

# In[7]:


for files in file_list[coin_point:]:
    name = files[0].split('_')[8].split('T')[0]
    value_array = np.empty(shape=(1702, 810, 5))
    grid_array = np.zeros((hy_sca.nlat, hy_sca.nlon))
    grid_num_array = np.zeros((hy_sca.nlat, hy_sca.nlon))

    for file in files:
        try:
            with h5py.File(file, mode='r') as f:
                lat = f['cell_lat'][:]
                lon = f['cell_lon'][:]
                sigma0 = f['cell_sigma0'][:]
                surface_flag = f['cell_sigma0_surface_flag'][:]
                qual_flag = f['cell_sigma0_qual_flag'][:]


        except KeyError:
            continue
        except OSError:
            continue

        sigma0 = sigma0 * 0.01
        lat[lat > 90] = 50
        lon[lat > 90] = 0
        lon[lon > 360] = 0
        lat[lon > 360] = 50

        sigma0[surface_flag != 2] = -99999
        sigma0[qual_flag != 0] = -99999
        # sigma0[sigma0 < -300] = 0



        value_array[:, :, 0] = lat
        value_array[:, :, 1] = lon
        value_array[:, :, 2], value_array[:, :, 3] = transformer.transform(value_array[:, :, 0], value_array[:, :, 1])
        value_array[:, :, 4] = sigma0

        x = (value_array[:, :, 2] / hy_sca.resolution).astype(np.int)
        y = (value_array[:, :, 3] / hy_sca.resolution).astype(np.int)
        # x[x>=1600] = 0
        # y[y>=1600] = 0
        grid_array[y, x] += value_array[:, :, 4]
        grid_num_array[y, x] += 1

    grid_array = grid_array / grid_num_array
    grid_array[grid_array >= -5] = np.nan
    grid_array[grid_array <= -300] = np.nan

    x_map, y_map = hy_sca.get_map_grid(transformer_back)
    grid_array[y_map < 60] = np.nan
    draw_sigmod_0(x_map, y_map, grid_array, train_data_dir + '\\' + str(name) + '.png')

    np.save((train_data_dir + r'\\npy\\' + str(name) + '.npy'), grid_array)
    print(name)
