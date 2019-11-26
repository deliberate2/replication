import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from sklearn.model_selection import ShuffleSplit
import pickle
from methods import (create_input_data, smape)

def build_model(n_input, n_hidden):
    if n_hidden:
        model = keras.Sequential([
            layers.Dense(n_hidden, activation='sigmoid', input_shape=(n_input,)),
            layers.Dense(1, activation='sigmoid'),
            layers.Dense(1)
        ])
    else:
        model = keras.Sequential([
            layers.Dense(1, activation='sigmoid', input_shape=(n_input,)),
            layers.Dense(1, activation='linear')])
# adjust sgd then is cool
    model.compile(optimizer='sgd',loss='mse')
    return model


# params
N = [1, 2, 3, 4, 5]
NH = [0, 1, 3, 5, 7, 9]
with open('./data/data_pre.pickle', 'rb') as f:
    series = pickle.load(f)

# ss = ShuffleSplit(n_splits=10)
for ser in series:
    for n in N:
        x, y = create_input_data(ser, n_lagged=n)
        for nh in NH:
            # mlp_regressor.set_params(hidden_layer_sizes=nh)
            mlp_regressor = build_model(n, nh)
            """ for train_index, test_index in ss.split(x, y):
                x_train = split_array(x, train_index[0])
                y_train = split_array(y, train_index[1])
                x_valid = split_array(x, test_index[0])
                y_valid = split_array(x, test_index[1]) """
            # how to combine fit and k_fold ??? just ignore ???
            mlp_regressor.fit(x, y, epochs=5)
            y_pre = mlp_regressor.predict(x[1:3])
            # y_pre is ndarray(num_inputs*output)!!!
            print(y_pre)


        