import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import statsmodels.api as sm
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import time

path = r'/home/data2/stock_dataset/CSI300'
# path = r'/home/data2/stock_dataset/CSI500'
source_file = os.listdir(path)
source_file.sort()
files = []
mses = []
maes = []
times = []

flag= 0
for file in source_file:
    if os.path.getsize(path + r'/' + file) <= 2:
        continue
    print('filename',file)
    data = pd.read_csv(path + r'/'+ file)
    if len(data) == 0:
        continue
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data.sort_values(by='trade_date',ascending=True,inplace=True)
    df_close = data.loc[:,['trade_date','close']]
    df_close['trend'] = (df_close['close'] / df_close['close'].shift(1)).dropna()
    df_close = df_close.iloc[1:,:]

    train_data = df_close[df_close['trade_date'] <= pd.to_datetime('20191231', format='%Y%m%d')]
    test_data = df_close[df_close['trade_date'] > pd.to_datetime('20200101', format='%Y%m%d')]
    if len(train_data) < 2:
        continue
    train_data.set_index('trade_date', inplace=True)
    test_data.set_index('trade_date', inplace=True)
    train_data = train_data.drop(['close'],axis =1)
    test_data = test_data.drop(['close'],axis =1)

    # Build Model and get best parameter for ARIMA
    model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                                 test='adf',
                                 max_p=3, max_q=3,
                                 m=1,  # frequency of series
                                 d=None,  # let model determine 'd'
                                 seasonal=False,  # No Seasonality
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
    fitted = model_autoARIMA.fit(train_data)

    # Forecast
    start_time = time.time()
    output = fitted.predict(n_periods=len(test_data))
    infer_time = time.time() - start_time
    fc_series = pd.DataFrame(output, columns = ['pred'])
    fc_series['stock'] = file
    fc_series.set_index(test_data.index, inplace=True)

    # report performance
    mse = mean_squared_error(test_data, output)
    mae = mean_absolute_error(test_data, output)

    mses.append(mse)
    maes.append(mae)
    times.append(infer_time)
    test_data['stock'] = file
    fc_series = pd.concat([fc_series, test_data['trend']], axis=1)
    if flag == 0:
        all_fc = fc_series
    else:
        all_fc = pd.concat([all_fc, fc_series])
    files.append(file)
    flag += 1

ic = all_fc.groupby(level=0).apply(lambda all_fc: all_fc["pred"].corr(all_fc["trend"]))
ric = all_fc.groupby(level=0).apply(lambda all_fc: all_fc["pred"].corr(all_fc["trend"], method="spearman"))


# check the perfomance of model
print('metrics:\n')
print('MSE: ' + str(np.mean(mses)))
print('MAE: ' + str(np.mean(maes)))
print('IC: ' + str(ic.mean()))
print('ICIR: ' + str(ic.mean()/ic.std()))
print('Rank IC: ' + str(ric.mean()))
print('Rank ICIR: ' + str(ric.mean()/ric.std()))
print('infer_time: ' + str(np.mean(times)))