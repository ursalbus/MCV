#%%
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
import pickle

from joblib import Parallel, delayed
import forecasting.utils as pu

from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

plt.style.use('dark_background')


#%%

end_time = datetime.now()
start_time = end_time - timedelta(days=200000)

if True:
    data_fresh = yf.download(pu.getTickers(), start=start_time, end=end_time, interval="1d")

#%%
    



#%%
N_DAYS_RANGE = 20000
HORIZON = 60

TICKERS_TO_FIT = pu.getTickers()

DIRECTORY = 'forecast_60'

N = 3 * 52 # number of training weeks

with open('data.pkl', 'rb') as handle:
    data = pickle.load(handle)

X = pu.getSignals(data)

#%%

X_debug = data.Close.rolling(5).mean()
X_debug.SPY[-50:].plot(label='lag')
data.Close.SPY[-50:].plot(label='close')
plt.legend()

#%%

hls = 1e2 * np.log(data.High / data.Low)
hls_debug = hls.apply(lambda x: (x.ewm(20).mean()) / 1)

hls.SPY[-50:].plot(label='close')
hls_debug.SPY[-50:].plot(label='lag')
plt.legend()

#%%



#%%

Y1 = pu.getFwdHighRtn(data, HORIZON, DIRECTORY) 
Y2 = pu.getFwdLowRtn(data, HORIZON, DIRECTORY) 

sotw, eotw = pu.getWeeklyIxs(X)

c = "SPY"

y1 = pu.sortIndex(Y1, c)

y2 = pu.sortIndex(Y2, c)
CUTOFF_DATE = '2019-11-01'
y2_abrev = y2.loc[:pd.to_datetime(CUTOFF_DATE)]


#%%

test_ixs = [i for i in pu.generateIxs(sotw, eotw, N)][1200:1270]

res = Parallel(n_jobs=-1)(delayed(pu.fitPredict)(ixs_tuple, X, y2) for ixs_tuple in test_ixs)
print("######################### NOW ABREV #####################################")
res_abrev = Parallel(n_jobs=-1)(delayed(pu.fitPredict)(ixs_tuple, X, y2_abrev) for ixs_tuple in test_ixs)


#%%

test_res = {}
test_res["low"] = {"SPY": res_abrev}

print(test_res.keys())

#%%

preds = {}
labels = {}
ixs = {}

for k, v in test_res.items():
    preds[k] = {}
    labels[k] = {}
    ixs[k] = {}

    for c, p in v.items():

        preds[k][c] = []
        labels[k][c] = []
        ixs[k][c] = np.array([], dtype=object)

        for r in p:
            if r == None:
                continue
           
            preds[k][c] = np.concatenate([preds[k][c], r["y_hat"]])
            labels[k][c] = np.concatenate([labels[k][c], r["y_test"]])
            ixs[k][c] = np.concatenate([ixs[k][c], r["test_ix"]])

#%%

preds_df = {}
labels_df = {}

for k, v in preds.items():
    preds_df[k] = pd.DataFrame()
    for j, u in v.items():
        preds_df[k] = pd.concat([preds_df[k], pd.Series(preds[k][j], ixs[k][j], name=j)], axis=1)

for k, v in labels.items():
    labels_df[k] = pd.DataFrame()
    for j, u in v.items():
        l = [np.nan if i >= len(labels[k][j]) else labels[k][j][i] for i in range(len(ixs[k][j]))]
        labels_df[k] = pd.concat([labels_df[k], pd.Series(l, ixs[k][j], name=j)], axis=1)

    preds_df[k].set_index(pd.to_datetime(preds_df[k].index), inplace=True)
    labels_df[k].set_index(pd.to_datetime(labels_df[k].index), inplace=True)

    preds_df[k].sort_index(inplace=True)
    labels_df[k].sort_index(inplace=True)

# %%
    
plt.close()
preds_df["low"].SPY.plot(label='pred')
labels_df["low"].SPY.plot(label='label')
plt.legend()
plt.show()

# %%
