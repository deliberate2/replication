import pickle
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# filename = './data/M3C_pro.CSV'
filename = './data/M3C_pro_rec.CSV'
data_raw = pd.read_csv(filename)

# set data index for each row series

# discard the first line
next(data_raw.iterrows())

# save each series in list
series = []
for _, row in data_raw.iterrows():
    # generate date_index
    year = row['Starting Year']
    month = row['Starting Month']
    n_observ = row['N'] - row['NF']
    date_index = pd.date_range(start='{}-{}'.format(year, month),                                           periods=n_observ, freq='M')

    # abstract numeric value
    temp = row.iloc[6:]
    # temp.infer_objects()
    temp.dropna(inplace=True)
    # print(temp)
    data = temp.values
    # print(data)

    # generate Series
    x = pd.Series(data, index=date_index, name=row['Series'])
    # print(x.name)
    series.append(x)

# preprocess training data in following 3 steps
data_pre = []
for ser in series:
    # 1. Log transformation
    ser = pd.to_numeric(ser)
    ser = ser.apply(np.log)

    # 2. Deseasonalization

    #  taking autocorrelation to test 
    #  how to make sure confidence?? 
    #  assume alpha=0.05
    _, _, p = sm.tsa.stattools.acf(ser, nlags=13, qstat=True)
    # when refuse hypothesis
    if p[12] < .05:
        # why only res works?????
        res = sm.tsa.seasonal_decompose(ser, model='additive', freq=12,                                         two_sided=True)
        # print(res.seasonal)
        ser = ser.sub(res.seasonal)


    # 3. Scaling
    ser = ser.div(ser.abs().max())
    data_pre.append(ser)

# save to csv
# data_pre.to_csv('./data/data_preprocessed.csv')

with open('./data/data_pre.pickle', 'wb') as f:
    pickle.dump(data_pre, f)










