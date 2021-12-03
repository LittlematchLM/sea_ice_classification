import os
import glob
import shutil

val_dir = r'E:\python_workfile\sea_ice_classification\training6\csv\add_threshold\val'
val_files = glob.glob(val_dir+'\\*.csv')
val_day = [val_file.split('\\')[-1].split('.')[0] for val_file in val_files]

# 被操作的文件夹
dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real_use\small_size'
val_dir = r'E:\python_workfile\sea_ice_classification\training7\dataset\input_value\VV_HH_polarratio_real_use\small_size\val'
files = glob.glob(dir+'\\*.png')

for file in files:
    if file.split('\\')[-1].split('.')[0][:8] in val_day:
        shutil.move(file, val_dir)
