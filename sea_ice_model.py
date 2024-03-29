import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
import io
import cv2

HH_threshold_lut = {"201905":"-12",
"201910":"-12",
"201911":"-12",
"201912":"-11.96949",
"202001":"-12.63408",
"202002":"-12.98586",
"202003":"-13.05187",
"202004":"-13.05196",
"202005":"-12.45499",
"202010":"-12",
"202011":"-12",
"202012":"-12.025733",
"202101":"-12.642169",
"202102":"-12.535940",
"202103":"-12.555068",
"202104":"-12.813603"
}

VV_threshold_lut = {
"201905":"-12",
"201910":"-12",
"201911":"-12",
"201912":"-12.438178",
"202001":"-13.278439",
"202002":"-13.628366",
"202003":"-13.422885",
"202004":"-13.522465",
"202005":"-12",
"202010":"-12",
"202011":"-12",
"202012":"-12.696600",
"202101":"-13.315515",
"202102":"-13.144006",
"202103":"-13.153818",
"202104":"-13.254424"
}

def data_split(data_df, object_col):
    Y = data_df[object_col]
    X = data_df.drop(object_col, axis=1)
    # 测试集占比30%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(Y_train)
    train = pd.concat([Y_train, X_train], axis=1)
    test = pd.concat([Y_test, X_test], axis=1)
    return X_train, X_test, Y_train, Y_test


def threshold_lut(VV_HH):
    if VV_HH == 'VV':
        return VV_threshold_lut
    else:
        return HH_threshold_lut
def strftime_week(time):
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    day = datetime.datetime.strftime(day, '%W')
    return int(day)

def strftime_year_month(time):
    return str(time[:4]) + str(time[5:7])

def strftime_day(time):
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    day = datetime.datetime.strftime(day, '%j')
    return int(day)


def strftime_month(time):
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    day = datetime.datetime.strftime(day, '%m')
    return int(day)


def strftime_year(time):
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    day = datetime.datetime.strftime(day, '%Y')
    return int(day)

def get_year_month_str(time):
    year = str(time[:4])
    month = str(time[5:7])
    return int(year+month)

def strftime_julian_week(time):
    # 计算从start_time开始过了多少周
    start_time = datetime.date(2019, 5, 14)
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    delta = day - start_time
    return int(delta.days / 7)


def strftime_julian_day(time):
    start_time = datetime.date(2019, 5, 14)
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    delta = day - start_time
    return int(delta.days)

def strftime_julian_month(time):
    start_time = datetime.date(2019, 5, 14)
    day = datetime.date(int(time[:4]), int(time[5:7]), int(time[8:10]))
    delta = day - start_time
    return int(delta.days)


max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))


def get_data_from_csv(csv_files):
    dl = []
    for f in csv_files:
        dl.append(pd.read_csv(f, index_col=None, encoding='ANSI'))  # 读取每个表格
    data = pd.concat(dl)  # 合并
    return data


def nor_dataframe(dataframe):
    dataframe[['lon_nor', 'lat_nor', 'sig0_nor']] = dataframe[['lon', 'lat', 'sig0']].apply(max_min_scaler)
    dataframe['week_nor'] = (dataframe['week'] - 1) / (52 - 1)
    dataframe['day_nor'] = (dataframe['day'] - 1) / (365 - 1)


def strftime_quarter(time):
    time = str(time)
    quarter = time[-1]
    return quarter

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_confusion_matrix(classes, cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.close()


def strf_icetype(ice_type):
    '''

    :param ice_type:冰型（1-5）
    :return: 一年冰（0），多年冰（1）
    '''
    ice_type = int(ice_type)
    if ice_type <= 3:
        F_M = 0
    else:
        F_M = 1
    return F_M

def process_sea_ice_train_dataframe(dataframe, fyi_myi=True):
    dataframe['week'] = dataframe['time'].apply(strftime_week)

    dataframe['day'] = dataframe['time'].apply(strftime_day)

    dataframe['month'] = dataframe['time'].apply(strftime_month)

    dataframe['year'] = dataframe['time'].apply(strftime_year)
    dataframe['year_month_str'] = dataframe['time'].apply(get_year_month_str)
    dataframe['julian_week'] = dataframe['time'].apply(strftime_julian_week)
    dataframe['julian_day'] = dataframe['time'].apply(strftime_julian_day)

    dataframe.time = pd.to_datetime(dataframe.time)

    dataframe['quarter'] = pd.PeriodIndex(dataframe.time, freq='Q')

    dataframe['quarter1'] = dataframe['quarter'].apply(strftime_quarter)
    if fyi_myi:
        dataframe['fyi_myi'] = dataframe['ice_type'].apply(strf_icetype)



