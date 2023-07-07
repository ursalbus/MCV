import numpy as np
import pandas as pd

from copy import deepcopy
from numba import njit

class Features():
    def __init__(self, config, data):
        self.config = config

        self.closes = data['Close']
        self.volumes = data["Volume"]
        self.opens = data['Open']
        self.highs = data['High']
        self.lows = data['Low']

        self.ticker_names = config["TICKER_NAMES"]
        self.n_tickers = len(self.ticker_names)

        self.features = self.getFeatures()

    def getFeatures(self):

        momo = self.getMomoRtns(self.closes, self.config["DAYS_MA_FAST"], self.config["DAYS_MA_SLOW"])
        momo2 = self.getMomoRtns(self.closes, 1, self.config["DAYS_MA_SLOW"] * 2)
        momo3 = self.getMomoRtns(self.closes, 1, self.config["DAYS_MA_FAST"] * 2)
        momo4 = self.getMomoRtns(self.closes, self.config["DAYS_MA_FAST"] * 2, self.config["DAYS_MA_SLOW"] * 2)
        v_momo = self.getMomoRtns(self.volumes, self.config["DAYS_MA_FAST"], self.config["DAYS_MA_SLOW"])
        v_momo2 = self.getMomoRtns(self.volumes, 1, self.config["DAYS_MA_SLOW"])

        self.adv = (self.volumes * self.closes).rolling(20, min_periods=1).mean() 

        daily_return = lr(self.closes / self.closes.shift(1))
        daily_hl = lr(self.highs / self.lows)

        drawdown_1yr = lr(self.closes.rolling(250).max() / self.closes) / 100
        drawdown_ath = lr(self.closes.cummax() / self.closes) / 100
        drawup_1yr = lr(self.closes / self.closes.rolling(250).min()) / 100

        a = 2 / (self.config["DAYS_ROLLING_CORR"] + 1)
        self.vol = daily_return.transform(lambda x: x**2).ewm(alpha=a, min_periods=self.config["DAYS_MA_FAST"]).mean().transform('sqrt')
        vol = self.windowPercentile(self.vol, 250)
        hl_vol = self.windowPercentile(daily_hl.ewm(alpha=a, min_periods=self.config["DAYS_MA_FAST"]).mean(), 250)
        
        index_corr = self.closes.rolling(self.config["DAYS_ROLLING_CORR"], min_periods = self.config["DAYS_ROLLING_MIN"]).corr(self.closes[self.config["INDEX_NAME"]])
        index_corr_med = self.repeatFeature(index_corr.median(axis=1))
        index_return = self.repeatFeature(daily_return[self.config["INDEX_NAME"]])
        index_momo = self.repeatFeature(momo[self.config["INDEX_NAME"]])
        index_drawdown_1yr = self.repeatFeature(drawdown_1yr[self.config["INDEX_NAME"]])
        index_drawup_1yr = self.repeatFeature(drawup_1yr[self.config["INDEX_NAME"]])
        index_drawdown_ath = self.repeatFeature(drawdown_ath[self.config["INDEX_NAME"]])
        index_momo_corr = momo.rolling(self.config["DAYS_ROLLING_CORR"], min_periods = self.config["DAYS_ROLLING_MIN"]).corr(index_momo)

        rank_momo = self.wideRank(momo)
        rank_v_momo = self.wideRank(momo)
        rank_adv = self.wideRank(self.adv)
        rank_daily_ret = self.wideRank(momo)
        rank_drawdown_1yr = self.wideRank(drawdown_1yr)
        rank_drawdown_ath = self.wideRank(drawdown_ath)
        rank_drawup_1yr = self.wideRank(drawup_1yr)
                
        self.feature_names = ["momo", "momo2", "momo3", "momo4", "v_momo", "v_momo2", "daily_return", \
                                "vol", "daily_hl", "hl_vol", \
                                "drawdown_1yr", "drawup_1yr", "drawdown_ath", \
                                "index_corr", "index_corr_med", "index_return", "index_momo", "index_momo_corr", \
                                "index_drawdown_1yr", "index_drawup_1yr", "index_drawdown_ath", \
                                "rank_momo", "rank_v_momo", "rank_adv", "rank_daily_ret", "rank_drawdown_1yr", "rank_drawdown_ath", "rank_drawup_1yr"]
                
        feaures_list = [momo, momo2, momo3, momo4, v_momo, v_momo2, daily_return, \
                        vol, daily_hl, hl_vol, \
                        drawdown_1yr, drawup_1yr, drawdown_ath,
                        index_corr, index_corr_med, index_return, index_momo, index_momo_corr, \
                        index_drawdown_1yr, index_drawup_1yr, index_drawdown_ath,
                        rank_momo, rank_v_momo, rank_adv, rank_daily_ret, rank_drawdown_1yr, rank_drawdown_ath, rank_drawup_1yr]
        
        for f in feaures_list:
            f[np.isinf(f)] = np.nan
            f.fillna(0)
            if f.shape[1] > self.n_tickers:
                f.drop(self.config["INDEX_NAME"], axis=1, inplace=True)

        return feaures_list
        
    def windowPercentile(self, x, w):
        return 2 * ((x.rolling(w).rank() / w) - 0.5)

    def wideRank(self, df):
        return df.rank(axis=1).apply(lambda x:  x/(~x.isna()).sum(), axis=1)

    def getMomoRtns(self, c, f_ma, s_ma):
        return np.log(c.rolling(f_ma, min_periods = 1).mean() / c.rolling(s_ma, min_periods = self.config["DAYS_ROLLING_MIN"]).mean())

    def repeatFeature(self, f):
        df = pd.concat([f] * self.n_tickers, axis=1)
        df.columns = self.ticker_names
        return df

class Labels():
    def __init__(self, config, features, data, is_buying):
        self.config = config

        self.upside = self.config["UPSIDE"] if is_buying else self.config["DOWNSIDE"]
        self.downside = self.config["DOWNSIDE"] if is_buying else self.config["UPSIDE"]

        self.features = features.features
        self.closes = data['Close']
        self.volumes = data["Volume"]
        self.opens = data['Open']
        self.highs = data['High']
        self.lows = data['Low']
        self.vol = features.vol

        self.ticker_names = config["TICKER_NAMES"]
        self.ix = self.closes.index
        self.ix_safe = self.closes.index
        self.n_tickers = len(self.ticker_names)

        self.nan_features = np.zeros((config["N_SPLITS"], ), dtype=object)

        self.labels = self.getLabels()
 
    def getLabels(self):
        self.opens[self.opens == 0.0] = np.nan

        ud = {s: getUpsideDownside1(self.opens[s].values, 
                                   self.highs[s].values, 
                                   self.lows[s].values, 
                                   self.closes[s].values, 
                                   self.vol[s].values, 
                                   self.upside, self.downside, 
                                   self.config["DAYS_TIL_UPSIDE"]) for s in self.ticker_names}
        
        self.has_upside = pd.DataFrame.from_dict({s: ud[s][0] for s in self.ticker_names}).astype(bool).set_index(self.ix)
        self.has_downside = pd.DataFrame.from_dict({s: ud[s][1] for s in self.ticker_names}).astype(bool).set_index(self.ix)
        self.days_to_upside = pd.DataFrame.from_dict({s: ud[s][2] for s in self.ticker_names}).set_index(self.ix)
        self.days_to_downside = pd.DataFrame.from_dict({s: ud[s][3] for s in self.ticker_names}).set_index(self.ix)
        self.forward_expired_rtn = pd.DataFrame.from_dict({s: ud[s][4] for s in self.ticker_names}).set_index(self.ix)

        self.has_expired = ~self.has_upside * ~self.has_downside

        # upside must materialize before downside
        labels = self.days_to_upside[self.has_upside] < self.days_to_downside.mask(~self.has_downside, np.inf)

        # weights are sided
        self.weights = (~self.has_expired * ((labels * self.vol * self.upside) + \
                                        ((~labels * ~self.has_expired) * self.vol * self.downside * -1))) + \
                       (self.has_expired * self.forward_expired_rtn)
        
        self.weights.drop(self.config["INDEX_NAME"], axis=1, inplace=True)

        return labels
    
    def getXyw(self, ixs):
        labels = self.labels.loc[ixs].to_numpy()
        weights = self.weights.loc[ixs].fillna(0).to_numpy()
        d = len(labels) * len(labels[0])

        nan_features = [f.loc[ixs].isna().all().all() for f in self.features]
        features = [f.loc[ixs] for i, f in enumerate(self.features) if ~nan_features[i]]
        features = [f.to_numpy() for f in features]

        X, y, w = self.njitGetXyw(features, labels, weights, d)

        return X, y, w, nan_features

    # @njit
    def njitGetXyw(self, features, labels, weights, d):

        y = np.reshape(labels, (d,))
        w = np.reshape(weights, (d,))
        xs = [np.reshape(f, (d, 1)) for f in features]
        X = np.hstack(xs)
                
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))

        return X[mask], y[mask], w[mask]


    def getPredictionsByTicker(self, model, ixs, features_to_mask):
        X = self.getFeaturesByTicker(ixs, features_to_mask)
        dummy = pd.DataFrame({"dummy": self.closes.loc[ixs].iloc[:,0]}) * np.nan
        
        return pd.DataFrame.from_dict({s: self.safePredict(model, X[s], ixs, dummy) for s in self.ticker_names}).set_index(ixs)
    
    def getFeaturesByTicker(self, ix, features_to_mask):
        n_features = len(self.features)
        return {s: np.vstack([self.features[i][s][ix] for i in range(n_features) if ~features_to_mask[i]]).T for s in self.ticker_names}

    def safePredict(self, model, X, ixs, dummy):
        mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)

        if sum(mask) == 0:
            return dummy["dummy"] #np.full(len(ix), np.nan)
        
        # 0th column is 0th class = prob label is false | 1st column is 1st class = prob label is true 
        pred = pd.DataFrame({"p": model.predict_proba(X[mask])[:,1]}, index=ixs[mask])
        return dummy.join(pred)["p"]
    

######################
## GLOBAL FUNCTIONS ##
######################
def lr(x):
    return 1e2 * np.log(x)
    
def sr(o, c):
    return 1e2 * (c - o) / o

@njit
def getUpsideDownside(o, h, l, c, v, us, ds, dtu_max):
    n = len(h) 

    max_idx = [min([n-1, i+dtu_max]) for i in range(n)]

    window_high_rtn = [1e2 * np.log(h[i:max_idx[i]] / o[i]) for i in range(n)]
    upside_target_rtn = [np.nan] + [v[i-1] * us for i in range(1,n)]
    is_upside = [window_high_rtn[i] > upside_target_rtn[i] for i in range(n)]
    has_upside = [(is_upside[i]).any() for i in range(n)]
    dtu = [np.nonzero(is_upside[i])[0] + 1 for i in range(n)]
    days_to_upside = [dtu[i][0] if len(dtu[i]) > 0 else np.NaN for i in range(n)]

    window_low_rtn = [1e2 * np.log(l[i:max_idx[i]] / o[i]) for i in range(n)]
    downside_target_rtn = [np.nan] + [v[i-1] * ds for i in range(1,n)]
    is_downside = [window_low_rtn[i] < -1 * downside_target_rtn[i] for i in range(n)]
    has_downside = [(is_downside[i]).any() for i in range(n)]
    dtd = [np.nonzero(is_downside[i])[0] + 1 for i in range(n)]
    days_to_downside = [dtd[i][0] if len(dtd[i]) > 0 else np.NaN for i in range(n)]

    expired_rtn = [1e2*np.log(c[max_idx[i]] / o[i]) for i in range(n)]

    if us > ds:
        return has_upside, has_downside, days_to_upside, days_to_downside, expired_rtn
    else:
        return has_downside, has_upside, days_to_downside, days_to_upside, expired_rtn


@njit
def getUpsideDownside1(o, h, l, c, v, us, ds, dtu_max):
    n = len(h) 

    # use +1 here so we can use range(1,n) below, could equally do +0 here and range(1,n-1) below.
    max_idx = [min([n-1, i+dtu_max+1]) for i in range(n)]

    
    is_upside = [np.array([False]*n)] + [1e2*np.log(h[i+1:max_idx[i]] / o[i]) > v[i-1] * us for i in range(1,n)]
    has_upside = [(is_upside[i]).any() for i in range(n)]
    dtu = [np.nonzero(is_upside[i])[0] + 1 for i in range(n)]
    days_to_upside = [dtu[i][0] if len(dtu[i]) > 0 else np.NaN for i in range(n)]

    is_downside = [np.array([False]*n)] + [1e2*np.log(l[i+1:max_idx[i]] / o[i]) < -1 * v[i-1] * ds for i in range(1,n)]
    has_downside = [(is_downside[i]).any() for i in range(n)]
    dtd = [np.nonzero(is_downside[i])[0] + 1 for i in range(n)]
    days_to_downside = [dtd[i][0] if len(dtd[i]) > 0 else np.NaN for i in range(n)]

    expired_rtn = [1e2*np.log(c[max_idx[i]] / o[i]) for i in range(n)]

    if us > ds:
        return has_upside, has_downside, days_to_upside, days_to_downside, expired_rtn
    else:
        return has_downside, has_upside, days_to_downside, days_to_upside, expired_rtn


