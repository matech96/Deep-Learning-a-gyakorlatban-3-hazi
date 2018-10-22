import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


class DataProcessing:
    def __init__(self, train_x, train_y):
        self.scaler_x = MinMaxScaler().fit(train_x)
        self.scaler_y = MinMaxScaler().fit(train_y)

    def preprocessInput(self, x):
        return self.scaler_x.transform(x)

    def poestprocessInput(self, x):
        return self.scaler_x.inverse_transform(x)

    def preprocessOutput(self, y):
        return self.scaler_y.transform(y)

    def postProcessOutput(self, y):
        return self.scaler_y.inverse_transform(y)


def getANSZData(n_days=7, y_offest=45, columns=('Temperature')):
    """
    Reads in the data from amsz.hu and converts it to trainable format.
    :param n_days: Number of day to be involved.
    :param y_offest: Number of days between the input and the predicted temperature.
    :param columns: Columns to be involved. See ansz.xlsx.
    :return: train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y,
           test_data_many_day_x, test_data_many_day_y
    """
    y_offest -= 1
    DATA_DIR = 'data'
    ANSZ_DOT_HU_DATA = os.path.join(DATA_DIR, 'ansz.xlsx')
    data = pd.read_excel(ANSZ_DOT_HU_DATA)

    year = data.index.year.tolist()
    month = data.index.month.tolist()
    day = data.index.day.tolist()

    data_one_day_list = [year, month, day]
    data_one_day_list += [data[column].tolist() for column in columns]
    data_one_day = np.stack(data_one_day_list, 1)
    data_many_day_x = np.array(
        [data_one_day[i:i + n_days, :].flatten() for i in range(0, data_one_day.shape[0] - n_days - y_offest)])
    data_many_day_y = np.array([data_one_day[i, 3].flatten() for i in range(n_days + y_offest, data_one_day.shape[0])])
    rest_x = np.array([data_one_day[i:i + n_days, :].flatten() for i in
                       range(data_one_day.shape[0] - n_days - y_offest, data_one_day.shape[0] - n_days)])

    train_data_many_day_x, train_data_many_day_y, \
        dev_data_many_day_x, dev_data_many_day_y, \
        test_data_many_day_x, test_data_many_day_y = _split_data(data_many_day_x, data_many_day_y)

    # train_data_many_day_x, train_data_many_day_y, \
    # dev_data_many_day_x, dev_data_many_day_y, \
    # test_data_many_day_x, test_data_many_day_y = \
    #     _shuffle_data(train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y,
    #                   test_data_many_day_x, test_data_many_day_y)
    return train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y, \
           test_data_many_day_x, test_data_many_day_y, rest_x


def getData(n_days=7):
    DATA_DIR = 'data'
    MET_DOT_HU_DATA = os.path.join(DATA_DIR, 'BP_d.txt')
    data = pd.read_csv(MET_DOT_HU_DATA, sep=';')
    data = _extract_date(data)

    year = np.array(data['year'])
    month = np.array(data['month'])
    day = np.array(data['day'])
    d_tx = np.array(data['d_tx'])

    data_one_day = np.stack([year, month, day, d_tx], 1)
    data_many_day_x = np.array(
        [data_one_day[i:i + n_days, :].flatten() for i in range(0, data_one_day.shape[0] - n_days)])
    data_many_day_y = np.array([data_one_day[i, 3].flatten() for i in range(n_days, data_one_day.shape[0])])

    train_data_many_day_x, train_data_many_day_y, \
    dev_data_many_day_x, dev_data_many_day_y, \
    test_data_many_day_x, test_data_many_day_y = _split_data(data_many_day_x, data_many_day_y)

    # train_data_many_day_x, train_data_many_day_y, \
    # dev_data_many_day_x, dev_data_many_day_y, \
    # test_data_many_day_x, test_data_many_day_y = \
    #     _shuffle_data(train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y,
    #                   test_data_many_day_x, test_data_many_day_y)
    return train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y, \
           test_data_many_day_x, test_data_many_day_y


def _shuffle_data(train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y,
                  test_data_many_day_x, test_data_many_day_y):
    train_data_many_day_x, train_data_many_day_y = shuffle(train_data_many_day_x, train_data_many_day_y,
                                                           random_state=42)
    dev_data_many_day_x, dev_data_many_day_y = shuffle(dev_data_many_day_x, dev_data_many_day_y, random_state=42)
    test_data_many_day_x, test_data_many_day_y = shuffle(test_data_many_day_x, test_data_many_day_y, random_state=42)
    return train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y, \
           test_data_many_day_x, test_data_many_day_y


def _split_data(data_many_day_x, data_many_day_y):
    train_data_many_day_x, dev_test_data_many_day_x, train_data_many_day_y, dev_test_data_many_day_y = \
        train_test_split(data_many_day_x, data_many_day_y, test_size=0.3, shuffle=False, random_state=42)
    dev_data_many_day_x, test_data_many_day_x, dev_data_many_day_y, test_data_many_day_y = \
        train_test_split(dev_test_data_many_day_x, dev_test_data_many_day_y, test_size=0.66, shuffle=False,
                         random_state=42)
    return train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y, test_data_many_day_x, test_data_many_day_y


def _extract_date(data):
    data['year'] = [int(a[0:4]) for a in data['#datum']]
    data['month'] = [int(a[5:7]) for a in data['#datum']]
    data['day'] = [int(a[8:10]) for a in data['#datum']]
    return data
