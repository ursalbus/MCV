import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
import pickle

from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)


def getTickers():
    EQUITIES = ["SPY", "QQQ", "IWM", "FXI", "EEM", "VGK", "EWJ", "VIXY"]
    BONDS = ["BIL", "SHY", "IEF", "TLT", "TIP"]
    CREDIT = ["HYG", "LQD", "BKLN", "MBB", "VCIT", "VCSH"]
    SECTORS = ["XLF", "XLV", "XLI", "XLU", "XLP", "XLY", "XLE"]
    INDUSTRY = ["KRE", "XOP", "XHB", "IYR"]
    COMMODS = ["USO", "GLD", "UNG", "DBC"]
    
    return EQUITIES + BONDS + CREDIT + SECTORS + INDUSTRY + COMMODS


def getData(n_days, pull_fresh_data):
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=n_days)

    if pull_fresh_data:
        data = yf.download(getTickers(), start=start_time, end=end_time, interval="1d")

        log(f'downloaded data | last start date: {data.Close.notna().idxmax().max()}')
        data.to_pickle('data.pkl')

    else:
        data = pd.read_pickle('data.pkl')
        log(f"using stored data | last start date: {data.Close.notna().idxmax().max()}")
    
    return data


def getSignals(data):
    EMA_DAYS_LIST = [5, 10, 20, 50, 200]
    FAST_EMA = 5
    SLOW_EMA = 20

    signal_df = pd.DataFrame()

    for d in EMA_DAYS_LIST:
        emas = 1e2 * np.log(data.Close / data.Close.ewm(d).mean()) / np.sqrt(d)
        emas = emas.rename(columns={c: f'{c}_{d}' for c in emas.columns})
        signal_df = pd.concat([signal_df, emas], axis=1)

    rtns = 1e2 * np.log(data.Close / data.Open)
    rtns = rtns.rename(columns={c: f'{c}_rtn' for c in rtns.columns})

    ont = 1e2 * np.log(data.Open / data.Close.shift(1))
    ont = ont.rename(columns={c: f'{c}_ont' for c in ont.columns})

    chs = np.log(data.Close / data.High) / np.log(data.High / data.Low)
    chs = chs.apply(lambda x: (x - x.ewm(FAST_EMA).mean()) / x.rolling(FAST_EMA).std())
    chs = chs.rename(columns={c: f'{c}_ch' for c in chs.columns})

    cls = np.log(data.Close / data.Low) / np.log(data.High / data.Low)
    cls = cls.apply(lambda x: (x - x.ewm(FAST_EMA).mean()) / x.rolling(FAST_EMA).std())
    cls = cls.rename(columns={c: f'{c}_cl' for c in cls.columns})

    hls = 1e2 * np.log(data.High / data.Low)
    hls = hls.apply(lambda x: (x - x.ewm(SLOW_EMA).mean()) / x.rolling(SLOW_EMA).std())
    hls = hls.rename(columns={c: f'{c}_hl' for c in hls.columns})

    vlm = data.Volume.transform('log').apply(lambda x: (x - x.ewm(SLOW_EMA).mean()) / x.rolling(SLOW_EMA).std())
    vlm = vlm.rename(columns={c: f'{c}_vlm' for c in vlm.columns})

    log('created signals')

    return pd.concat([signal_df, rtns, ont, chs, cls, hls, vlm], axis=1)


def getFwdRets(data, h, dir):

    fwd_rets = 1e2 * np.log(data.Close.shift(-(1 + h)) / data.Open.shift(-1))

    toPickle(fwd_rets, dir + f"/label_fwd_rets_{h}.pkl")

    return fwd_rets


def getFwdHls(data, h, dir):
    
    fwd_hls = 1e2 * np.log(data.High.rolling(h+1).max() / data.Low.rolling(h+1).min()).shift(-(1 + h))

    # toPickle(fwd_hls, dir + "/label_fwd_hls.pkl")

    return fwd_hls


def getFwdHighRtn(data, h, dir):
    
    fwd_hls = 1e2 * np.log(data.High.rolling(h+1).max().shift(-(h+1)) / data.Open.shift(-1))

    # toPickle(fwd_hls, dir + "/label_fwd_high_rtns.pkl")

    return fwd_hls


def getFwdLowRtn(data, h, dir):
    
    fwd_hls = 1e2 * np.log(data.Low.rolling(h+1).min().shift(-(h+1)) / data.Open.shift(-1))

    # toPickle(fwd_hls, dir + "/label_fwd_low_rtns.pkl")

    return fwd_hls


def sortIndex(Y, c):
    return pd.DataFrame(Y[c]).sort_index()[c]


def getWeeklyIxs(y_raw):
    y = y_raw.copy()

    y["week"] = y.index.isocalendar().week
    y["week"][(y.week == 1) & (y.index.month > 1)] += 52
    y["year"] = y.index.year

    sotw = y.groupby([y.year, y.week]).head(1).index
    eotw = y.groupby([y.year, y.week]).tail(1).index
    
    test_size_mask = (eotw - sotw).days > 0    

    sotw = sotw[test_size_mask]
    eotw = eotw[test_size_mask]
    
    return sotw, eotw


def generateIxs(s, e, l): 
    n = len(e)
    for i in range(n-l-1):
        yield s[i], e[i+l], s[i+l+1], e[i+l+1]

 
def fitPredict(ix_tuple, X, y):
    N_TRAIN_LABLES_REQ = 500
    N_UNIQUE_SIGNALS_REQ = 5

    start_train_ix, end_train_ix, start_test_ix, end_test_ix = ix_tuple

    X_train, X_test = X.loc[start_train_ix:end_train_ix], X.loc[start_test_ix: end_test_ix]
    y_train, y_test = y.loc[start_train_ix:end_train_ix], y.loc[start_test_ix: end_test_ix]

    train_mask =  (X_train.notna().sum(1) > N_UNIQUE_SIGNALS_REQ) & y_train.notna()
    test_mask = X_test.notna().sum(1) > N_UNIQUE_SIGNALS_REQ

    test_ix = X_test[test_mask].index

    X_train, y_train = X_train[train_mask].values, y_train[train_mask].values
    X_test, y_test = X_test[test_mask].values, y_test[test_mask].values

    if (len(X_test) > 0) & (len(y_train) > N_TRAIN_LABLES_REQ):

        r = HistGradientBoostingRegressor(l2_regularization=1).fit(X_train, y_train)

        y_hat = r.predict(X_test)

        print(f'last train date: {end_train_ix} | n train labels used: {len(y_train)}')

        if (end_test_ix.month == 12) & (end_test_ix.day > 23):
           log(f'fit {y.name} | progress: {end_test_ix}') 

        return {"test_ix": test_ix, "y_test": y_test, "y_hat": y_hat}


def getCurrentTime():
    return datetime.strftime(datetime.now(), format="%D %T")

def log(msg):
    print(f'{getCurrentTime()} | ' + msg)

def toPickle(data, filename):
    
    log(f'attempting to pickle: {filename}')

    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)