import os
import glob
import shutil

path_from = r'I:\remote_sensing_data\back_scatter\11'

files = glob.glob(path_from + r'\*\*.h5')
for file in files:
    shutil.move(file, r'I:\\remote_sensing_data\\back_scatter\\HY-2B\\')
