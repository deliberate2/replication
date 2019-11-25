import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf

""" 
##################################################################
    preprocess data methods
##################################################################
"""

# 1. no special preprocessing
def lagged_val(series=None):
    pass 

# 2. Time series differencing 
def diff(series=None):
    return sm.tsa.statespace.tools.diff(series, k_diff=1)

# 3. moving averages
def mov_avg(series=None):
    window_1 = [1]
    window_2 = [0.5] * 2
    window_4 = [1/4] * 4
    window_8 = [1/8] * 8
    u1 = np.convolve(series, window_1, mode='valid')
    u2 = np.convolve(series, window_2, mode='valid')
    u3 = np.convolve(series, window_4, mode='valid')
    u4 = np.convolve(series, window_8, mode='valid')

    return u1, u2, u3, u4


""" 
####################################################################
    transform series into nparray which can be input into NNmodel
#####################################################################
"""

def create_input_data(series, n_lagged=1):
    # here we just assume k_fold methods is TimeSeriesSplit
    data = series.values
    x_train, y_train = [], []

    for i in range(n_lagged-1,data.size-1):
        # x = data[index[i-n_lagged+1:i]]
        x = data[i-n_lagged+1:i+1]
        y = data[i+1]
        x_train.append(x)
        y_train.append(y)

    return np.array(x_train), np.array(y_train)

def split_array(x, index):
    result = []
    for i in index:
        result.append(x[i])
    return result




""" 
#############################################################
error measure for forecasting comparisons
############################################################### 
"""

def smape(y, y_pred):
    m = tf.size(y)
    return np.sum(1/m * (np.abs(y_pred - y) / 
                ((np.abs(y_pred) - np.abs(y)) * 2)))

def rank_interval(rank):
    pass

def frac_best(rank):
    pass

        
