import pandas as pd
import numpy as np
import glob
from sea_ice_model import *
import os

def get_range_threshold(sig0, threshold,range):
    if sig0 > threshold+range:
        return 2
    elif sig0 < threshold-range:
        return 0
    else:
        return 1



VV_HH = 'VV'
csv_path = 'd:\python_workfile\sea_ice_classification\data\csv\split_VV_HH\\'+VV_HH
csv_files = glob.glob(csv_path + '\\*.csv')

#只处理两年的10月份
# csv_files = csv_files[15:33] + csv_files[238:257]

save_path = r'd:\python_workfile\sea_ice_classification\training6\csv\split_VV_HH\\'+VV_HH+r'\-10_-20threshold'



lut = threshold_lut(VV_HH)

for csv_file in csv_files:

    df = pd.read_csv(csv_file)
    if df is None:
        continue

    df['year_month_str'] = df['time'].apply(strftime_year_month)
    # df['threshold'] = -15

    df['range_threshold'] = df['sig0'].apply(get_range_threshold, threshold=15, range=5)
    # df['large_than_threshold'] = (df['sig0'] > float(df['threshold'].iloc[0]))
    # df['large_than_threshold'] = df['large_than_threshold'].astype(np.int8)
    df.to_csv(save_path+'\\'+csv_file.split('\\')[-1])
    print(f'processed file {csv_file}')
