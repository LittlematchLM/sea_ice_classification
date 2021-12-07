import shutil
import glob

files = glob.glob(r'J:\remote_sensing_data\back_scatter\new\H2B\*\*.h5')

for file in files:
    shutil.move(file, r'J:\remote_sensing_data\back_scatter\temp')
