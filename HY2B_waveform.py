import glob
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt

file_dir = r'E:\python_workfile\sea_ice_classification\waveform'
files = glob.glob(file_dir + '\*.nc')

with Dataset(files[0], 'r') as f:
    waveforms_20hz_ku = f.variables['waveforms_20hz_ku'][:]
    waveforms_20hz_c = f.variables['waveforms_20hz_c'][:]
    width_leading_edge_20hz_ku = f.variables['width_leading_edge_20hz_ku'][:]
    amplitude_20hz_ku = f.variables['amplitude_20hz_ku'][:]
    width_leading_edge_20hz_c = f.variables['width_leading_edge_20hz_c'][:]
    amplitude_20hz_c = f.variables['amplitude_20hz_c'][:]

#乘光速
c = 299792458
width_leading_edge_20hz_ku = width_leading_edge_20hz_ku*c


fig,(ax1,ax2) = plt.subplots((2),figsize=(6,6))

ax1.plot(amplitude_20hz_ku[1216,:])
ax1.set_ylabel('count')
ax1.set_xlabel('amplitude_20hz_ku')

ax2.plot(amplitude_20hz_c[1216,:])
ax2.set_ylabel('count')
ax2.set_xlabel('amplitude_20hz_c')
plt.show()
