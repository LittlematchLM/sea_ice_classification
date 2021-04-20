import shapefile
import geopandas

file = r'E://python_workfile//sea_ice_classification//data//aari_arc_20210406_pl_a'
#
# shps = shapefile.Reader(file)
# shp = shps.shapeRecord()
shps_pandas = geopandas.read_file(file+'.shp')
# shps.plot()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.patches import Polygon
# 设置地图的坐标系和坐标显示范围
fig = plt.figure(figsize=(16, 9))
ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(projection='npaeqd', boundinglat=40, lon_0=180, resolution='c')
m.fillcontinents()
m.drawmapboundary()

shp_info = m.readshapefile(file,'aari_arc')


names = []
for info, shp in zip(m.aari_arc_info, m.aari_arc):
    cat = info['CD']
    # print(cat)
    if cat == '99':
        # print('cat')
        poly = Polygon(shp, facecolor='g', edgecolor='c', lw=3)  # 绘制广东省区域
        ax1.add_patch(poly)
plt.show()
    # name = shapedict['NAME']
    # if cat in ['H4','H5'] and name not in names:
    #     if name != 'NOT NAMED':  names.append(name)