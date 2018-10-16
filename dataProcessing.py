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

    def preprocessOutput(self, y):
        return self.scaler_y.transform(y)

    def postProcessOutput(self, y):
        return self.scaler_y.inverse_transform(y)


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

    train_data_many_day_x, train_data_many_day_y, \
        dev_data_many_day_x, dev_data_many_day_y, \
        test_data_many_day_x, test_data_many_day_y = \
        _shuffle_data(train_data_many_day_x, train_data_many_day_y, dev_data_many_day_x, dev_data_many_day_y,
                      test_data_many_day_x, test_data_many_day_y)
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
