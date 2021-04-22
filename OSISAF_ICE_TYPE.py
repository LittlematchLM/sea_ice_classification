#!/usr/bin/env python
# coding: utf-8

# # OSI-SAF ICE TYPE 投影

# In[1]:


from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
from RSData import *
from HaiYangData import *

import matplotlib.pyplot as plt
import numpy as np
import glob
import io
import cv2


# In[2]:


satellite = r'ocisaf'
sensor = r'icetype'
osi_i_t = HaiYangData(satellite=satellite, sensor=sensor,resolution=12500)

# 将WGS 84坐标（4326）转化为极射投影
crs = CRS.from_epsg(4326)
crs = CRS.from_string("epsg:4326")
crs = CRS.from_proj4("+proj=latlon")
crs = CRS.from_user_input(4326)
crs2 = CRS(proj="aeqd")


# In[3]:


dir_path = r"H:\remote_sensing_data\sea_ice_type\osisaf"

osi_save_path = r'E:\\python_workfile\\sea_ice_classification\\data\\mask\\osisaf\\'

files = glob.glob(dir_path + '\*.nc')


# In[4]:


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[5]:


def draw_osi_ice_type(x_map, y_map, grid_array, save_path):
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(111)
    fig.set_tight_layout(True)
    hy_m = Basemap(projection='npaeqd', boundinglat=66, lon_0=90., resolution='c')
    hy_m.pcolormesh(x_map, y_map, data=grid_array, cmap=plt.cm.jet,vmax = 4,vmin = 0,latlon = True)
#     plt.show()
    plt.savefig(save_path,dpi=180)   
    plt.close()
    
    return fig


# In[6]:


transformer = HaiYangData.set_transformer(crs,crs2)
transformer_back = HaiYangData.set_transformer(crs2,crs)


# In[ ]:



for file in files[29:]:
    try:
        value_array = np.empty(shape=(1120, 760,5))
        grid_array = np.zeros((osi_i_t.nlat, osi_i_t.nlon))
        grid_num_array = np.zeros((osi_i_t.nlat, osi_i_t.nlon))


        with Dataset(file, mode='r') as f:
            lat = f['lat'][:]
            lon = f['lon'][:]
            ice_type = f['ice_type'][:]

        ice_type = ice_type.reshape(1120,760)

        projlats, projlons = transformer.transform(lat, lon)

        value_array[:,:,0] = lat
        value_array[:,:,1] = lon
        value_array[:,:,2],value_array[:,:,3] = transformer.transform(value_array[:,:,0], value_array[:,:,1])
        value_array[:,:,4] = ice_type

        x = (value_array[:,:,2] / osi_i_t.resolution).astype(np.int)
        y = (value_array[:,:,3] / osi_i_t.resolution).astype(np.int)
        grid_array[y,x] += value_array[:,:,4]
        grid_num_array[y,x] += 1

        grid_array = grid_array / grid_num_array
        x_map, y_map = osi_i_t.get_map_grid(transformer_back)

        day = file.split('_')[-1].split('.')[0]
        fig = draw_osi_ice_type(x_map, y_map,grid_array, osi_save_path+'\\pic\\osi_ice_type'+str(day) + '.png')
        plot_img_np = get_img_from_fig(fig)
        np.save((osi_save_path + 'npy\\' + str(day) + '.npy'), plot_img_np)
        print(day)
    except TypeError:
        print('wenti' + str(day))
        pass


# In[ ]:




