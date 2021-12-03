from sea_ice_model import *
from RSData import *
from HaiYangData import *
from scipy.optimize import fmin, fminbound
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


# y是真实的分布hist
def fitting(M=0):
    """
    n 为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(bins[:-1], hist))
    print('Fitting Parameters:', p_lsq[0])

    return p_lsq

VV_HH = r'VV'
csv_dir = r'd:\python_workfile\sea_ice_classification\training6\csv\split_VV_HH\\'+VV_HH
csv_files = glob.glob(csv_dir + '\*.csv')

save_file = r'd:\python_workfile\sea_ice_classification\training6\hist\monthly'

file_lists = HaiYangData.month_split(csv_files)

file_lists_2019 = file_lists[1:8]
file_lists_2020 = file_lists[9:]
# with open('hist_'+VV_HH+'.txt','a') as f:
fig = plt.figure(figsize=(24, 12))
for n, files in enumerate(file_lists_2019):
    name = files[0].split('\\')[-1].split('.')[0][:6]
    dataframe = get_data_from_csv(files)

    # process_sea_ice_train_dataframe(dataframe)

    # dataframe["time"] = pd.to_datetime(dataframe['time'], errors='coerce')

    num_bins = 100

    hist, bins = np.histogram(dataframe['sig0'], num_bins, range=(-25, -5))

    # 用多项式拟合
    f1 = np.polyfit(bins[:-1], hist, 5)
    p1 = np.poly1d(f1)

    # # 也可使用yvals=np.polyval(f1, x)
    yvals = p1(bins[:-1])  # 拟合y值
    p_lsq_12 = fitting(M=12)

    # 导出最小二乘法所得函数
    func = np.poly1d(p_lsq_12[0])
    # 求出导数为0的点
    x_list = np.polyder(func, 1).r
    # 求二阶导数
    fun2d = np.polyder(func, 2)
    x_list = x_list[fun2d(x_list) > 0]
    # 规定阈值只能在-12dB到-16dB之间
    x_list = x_list[(x_list < -12) & (x_list > -16)]

    # 如果没有极小值，则阈值取12
    if len(x_list) < 1:
        x_list = [-12]
    ax1 = fig.add_subplot(2, 4, n + 1)
    # 可视化
    plt.plot(bins[:-1], hist, label='real')
    # plt.plot(bins[:-1], fit_func(p_lsq_12[0], bins[:-1]), label='fitted curve')
    # plt.plot(x, y, 'bo', label='noise')

    plt.ylim(0, 30000)
    plt.xlim(-25, -5,5)
    plt.vlines(x=x_list, ymin=0, ymax=30000,
               lw=1.5,
               colors='r',
               linestyles='--')
    ax1.set_title(name)
    plt.xlabel('Sigma0(dB)')
    plt.ylabel('Number of Pixels')
    # f.writelines([name + ','+ str(x_list[0])])
    # f.write('\n')

plt.savefig(save_file + '\\' + '2019-2020 winter' + '_' + VV_HH)
