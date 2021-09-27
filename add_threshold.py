import pandas as pd
import numpy as np
import glob
import os

csv_path = r'E:\python_workfile\sea_ice_classification\data\csv\add_threshold'
csv_files = glob.glob(csv_path + '\\*.csv')

save_path = r'E:\python_workfile\sea_ice_classification\data\csv\add_threshold'

for csv_file in csv_files:

    df = pd.read_csv(csv_file)
    df['large_than_threshold'] = df['sig0'] > -14.5
    df['large_than_threshold'] = df['large_than_threshold'].astype(np.int)
    df.to_csv(save_path+'\\'+csv_file.split('\\')[-1])

