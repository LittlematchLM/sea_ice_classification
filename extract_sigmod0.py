from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from skimage import exposure
from RSData import *
from HaiYangData import *

import datetime
import time
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import io
import cv2


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


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches='tight', pad_inches=0, edgecolor='white')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_sigmod_0(x_map, y_map, grid_array, save_path=None):
    fig = plt.figure(figsize=(9, 9))
    fig.add_subplot(111)
    fig.set_tight_layout(True)  # reduce the spaces from margin outside the axis

    # hy_m.fillcontinents()
    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    hy_m.pcolormesh(x_map, y_map, data=grid_array, cmap=plt.cm.jet, vmax=-5, vmin=-25, latlon=True)
    # you can get a high-resolution image as numpy array!!
    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight', pad_inches=0, edgecolor='white')
    plt.close()

    return fig


satellite = r'HY2B'
sensor = r'SCA'
hy_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=25000)

# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")

training_dir = r'E:\python_workfile\sea_ice_classification\training6\split_VV_HH\VV'
csv_path = r'E:\python_workfile\sea_ice_classification\training6\csv\split_VV_HH\VV'
sig_dir_path = r'E:\python_workfile\sea_ice_classification\training6\split_VV_HH\VV\npy'
aari_dir_path = r'E:\python_workfile\sea_ice_classification\training6\aari\npy'
# 20210608 处理pwp_250_07.h5
sigmod_files = glob.glob(sig_dir_path + '\*.npy')
aari_files = glob.glob(aari_dir_path + '\*.npy')

transformer = HaiYangData.set_transformer(crs, crs2)
transformer_back = HaiYangData.set_transformer(crs2, crs)

sigmod_files.sort()
aari_files.sort()

for sig_file, aari_file in zip(sigmod_files[:], aari_files[:]):
    day_char = sig_file.split('\\')[-1].split('.')[0]
    day = datetime.date(year=int(day_char[:4]), month=int(day_char[4:6]), day=int(day_char[6:8]))
    x_map, y_map = hy_sca.get_map_grid(transformer_back)
    sig_grid = np.load(sig_file)
    aari_grid = np.load(aari_file)

    df = pd.DataFrame(columns=['lon', 'lat', 'sig0', 'time', 'ice_type'])
    df['lon'] = x_map.flatten()
    df['lat'] = y_map.flatten()
    df['sig0'] = sig_grid.flatten()
    df['ice_type'] = aari_grid.flatten()
    df['time'] = day

    df = df.drop((df[df.lat < 60]).index)
    df = df.drop((df[df.ice_type == 0]).index)
    df = df.dropna(axis=0, subset=['sig0'])
    df.to_csv(csv_path + '\\' + day_char + '.csv', index=False)
    print(day_char, len(df))

'''
2021.09.08 
这次给的aari数据可能有问题
临时做一个只有hy的参数的csv看一下
20210420，20210427，20210504
'''

for sig_file in sigmod_files[:]:
    day_char = sig_file.split('\\')[-1].split('.')[0]
    day = datetime.date(year=int(day_char[:4]), month=int(day_char[4:6]), day=int(day_char[6:8]))
    x_map, y_map = hy_sca.get_map_grid(transformer_back)
    sig_grid = np.load(sig_file)

    df = pd.DataFrame(columns=['lon', 'lat', 'sig0', 'time', 'ice_type'])
    df['lon'] = x_map.flatten()
    df['lat'] = y_map.flatten()
    df['sig0'] = sig_grid.flatten()
    df['time'] = day

    df = df.drop((df[df.lat < 60]).index)
    df = df.drop((df[df.ice_type == 0]).index)
    df = df.dropna(axis=0, subset=['sig0'])
    df.to_csv(csv_path + '\\' + day_char + '.csv', index=False)
    print(day_char, len(df))
