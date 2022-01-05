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


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_sigmod_0(x_map, y_map, grid_array, save_path=None):
    fig = plt.figure(figsize=(7, 7))
    # fig.add_subplot(111)
    # fig.set_tight_layout(True)  # reduce the spaces from margin outside the axis

    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    # hy_m.fillcontinents()
    hy_m.pcolormesh(x_map, y_map, data=grid_array, cmap=plt.cm.jet, shading='auto', vmax=-5, vmin=-25, latlon=True)

    # plt.colorbar(location='right',fraction=0.045)
    # hy_m.drawparallels(np.arange(-90., 120., 10.), labels=[1, 0, 0, 0])
    # hy_m.drawmeridians(np.arange(-180., 180., 60.), labels=[0, 0, 0, 1])
    # you can get a high-resolution image as numpy array!!
    if save_path:
        plt.savefig(save_path, dpi=180)
    plt.close()

    # return fig


satellite = r'HY2B'
sensor = r'SCA'
hy_sca = HaiYangData(satellite=satellite, sensor=sensor, resolution=30000)
coin_point = 0
# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")

dir_path = r"I:\remote_sensing_data\back_scatter\HY-2B"

files = glob.glob(dir_path + '\*_pwp_250_*.h5')
# files = glob.glob(dir_path + '\*_dps_250_0*.h5')

file_list = split_file_day(files)
use_type_flag = True

transformer = HaiYangData.set_transformer(crs, crs2)
transformer_back = HaiYangData.set_transformer(crs2, crs)

train_data_dir = r'F:\python_workspace\sea_ice_classification\data\npy\sigmod0\30000_resolution'


for files in file_list[:]:
    name = files[0].split('_')[8].split('T')[0]
    value_array = np.empty(shape=(1702, 810, 6))
    grid_array_VV = np.zeros((hy_sca.nlat, hy_sca.nlon))
    grid_num_array_VV = np.zeros((hy_sca.nlat, hy_sca.nlon))

    grid_array_HH = np.zeros((hy_sca.nlat, hy_sca.nlon))
    grid_num_array_HH = np.zeros((hy_sca.nlat, hy_sca.nlon))

    for file in files:
        try:
            with h5py.File(file, mode='r') as f:
                lat = f['cell_lat'][:]
                lon = f['cell_lon'][:]
                sigma0 = f['cell_sigma0'][:]
                surface_flag = f['cell_sigma0_surface_flag'][:]
                qual_flag = f['cell_sigma0_qual_flag'][:]
                incidence_flag = f['cell_incidence'][:]


        except KeyError:
            continue
        except OSError:
            continue
        incidence_flag = (incidence_flag / 100).astype(np.int16)

        sigma0 = sigma0 * 0.01
        lat[lat > 90] = 50
        lon[lat > 90] = 0
        lon[lon > 360] = 0
        lat[lon > 360] = 50

        if use_type_flag:
            sigma0[surface_flag != 2] = -99999
            sigma0[qual_flag != 0] = -99999

        # sigma0[sigma0 < -300] = 0
        value_array[:, :, 0] = lat
        value_array[:, :, 1] = lon
        value_array[:, :, 2], value_array[:, :, 3] = transformer.transform(value_array[:, :, 0], value_array[:, :, 1])
        value_array[:, :, 4] = sigma0

        sigma0_VV = sigma0[incidence_flag == 48]
        sigma0_HH = sigma0[incidence_flag == 41]
        x_VV = (value_array[:, :, 2][incidence_flag == 48] / hy_sca.resolution).astype(np.int16)
        y_VV = (value_array[:, :, 3][incidence_flag == 48] / hy_sca.resolution).astype(np.int16)
        x_HH = (value_array[:, :, 2][incidence_flag == 41] / hy_sca.resolution).astype(np.int16)
        y_HH = (value_array[:, :, 3][incidence_flag == 41] / hy_sca.resolution).astype(np.int16)

        # x = (value_array[:, :, 2] / hy_sca.resolution).astype(np.int16)
        # y = (value_array[:, :, 3] / hy_sca.resolution).astype(np.int16)
        # x[x>=1600] = 0
        # y[y>=1600] = 0

        # 处理VV极化下的sigma0

        grid_array_VV[y_VV, x_VV] += sigma0_VV
        grid_num_array_VV[y_VV, x_VV] += 1

        # 处理HH极化下的sigma0

        grid_array_HH[y_HH, x_HH] += sigma0_HH
        grid_num_array_HH[y_HH, x_HH] += 1

    grid_array_VV = grid_array_VV / grid_num_array_VV
    grid_array_VV[grid_array_VV >= -5] = np.nan
    grid_array_VV[grid_array_VV <= -300] = np.nan

    grid_array_HH = grid_array_HH / grid_num_array_HH
    grid_array_HH[grid_array_HH >= -5] = np.nan
    grid_array_HH[grid_array_HH <= -300] = np.nan

    x_map, y_map = hy_sca.get_map_grid(transformer_back)

    # 去除60°N以南的点
    grid_array_VV[y_map < 60] = np.nan
    grid_array_HH[y_map < 60] = np.nan

    # draw_sigmod_0(x_map, y_map, grid_array_VV, save_path=train_data_dir + '\\VV\\pic\\' + str(name) + '.png')
    # draw_sigmod_0(x_map, y_map, grid_array_HH, save_path=train_data_dir + '\\HH\\pic\\' + str(name) + '.png')

    np.save((train_data_dir + r'\\VV\\npy\\' + str(name) + '.npy'), grid_array_VV)
    np.save((train_data_dir + r'\\HH\\npy\\' + str(name) + '.npy'), grid_array_HH)

    print(name)


