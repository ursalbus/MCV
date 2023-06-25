import numpy as np
import pandas as pd

from copy import deepcopy
from numba import njit

######################
## GLOBAL FUNCTIONS ##
######################
def lr(x):
    return 1e2 * np.log(x)
    
def sr(o, c):
    return 1e2 * (c - o) / o

def highestN(x, n):
    x = x[~np.isnan(x)]
    if n == 0 or len(x) == 0:
        return np.nan
    if n > len(x):
        n = 0

    return np.sort(x)[-n]

def lowestN(x, n):
    if n==0 or nnn(x)==0:
        return np.nan
    x = x[~np.isnan(x)]
    max_ix = np.min([n, len(x)])
    return np.sort(x)[max_ix-1]

def nnn(x):
    # number of non-nans
    return np.count_nonzero(~np.isnan(x))

@njit
def getUpsideDownside(o, h, l, c, v, us, ds, dtu_max):
    n = len(h) 

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

################
## MAIN CLASS ##
################
class FeaturesLabels():
    def __init__(self, config, data, is_buying=True):
        self.config = config
        self.is_buying = is_buying
        self.upside = config["UPSIDE"] if is_buying else config["DOWNSIDE"]
        self.downside = config["DOWNSIDE"] if is_buying else config["UPSIDE"]

        self.closes = data['Close']
        self.volumes = data["Volume"]
        self.opens = data['Open']
        self.highs = data['High']
        self.lows = data['Low']

        self.ticker_names = config["TICKER_NAMES"]
        self.ix = self.closes.index
        self.ix_safe = self.closes.index
        self.n_days = self.closes.shape[1]

        self.nan_features = dict()

        self.features = self.getFeatures()
        self.labels = self.getLabels()

    def getMomoRtns(self, c, f_ma, s_ma):
        return np.log(c.rolling(f_ma, min_periods = 1).mean() / c.rolling(s_ma, min_periods = self.config["DAYS_ROLLING_MIN"]).mean())

    # @njit
    def njitGetXyw(self, features, labels, weights, d):

        y = np.reshape(labels, (d[0],))
        w = np.reshape(weights, (d[0],))
        xs = [np.reshape(f, d) for f in features]
        X = np.hstack(xs)
        
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))

        return X[mask], y[mask], w[mask]


    def getFeatures(self):

        momo = self.getMomoRtns(self.closes, self.config["DAYS_MA_FAST"], self.config["DAYS_MA_SLOW"])
        momo2 = self.getMomoRtns(self.closes, 1, self.config["DAYS_MA_SLOW"] * 2)
        momo3 = self.getMomoRtns(self.closes, 1, self.config["DAYS_MA_FAST"] * 2)
        momo4 = self.getMomoRtns(self.closes, self.config["DAYS_MA_FAST"] * 2, self.config["DAYS_MA_SLOW"] * 2)
        v_momo = self.getMomoRtns(self.volumes, self.config["DAYS_MA_FAST"], self.config["DAYS_MA_SLOW"])
        v_momo2 = self.getMomoRtns(self.volumes, 1, self.config["DAYS_MA_SLOW"])

        daily_return = lr(self.closes / self.closes.shift(1))
        daily_hl = lr(self.highs / self.lows)

        drawdown_1yr = lr(self.closes.rolling(250).max() / self.closes)
        drawup_1yr = lr(self.closes / self.closes.rolling(250).min())
        drawdown_ath = lr(self.closes.cummax() / self.closes)

        a = 2 / (self.config["DAYS_ROLLING_CORR"] + 1)
        vol = daily_return.transform(lambda x: x**2).ewm(alpha=a, min_periods=self.config["DAYS_MA_FAST"]).mean().transform('sqrt')
        hl_vol = daily_hl.ewm(alpha=a, min_periods=self.config["DAYS_MA_FAST"]).mean()
        
        index_corr = self.closes.rolling(self.config["DAYS_ROLLING_CORR"], min_periods = self.config["DAYS_ROLLING_MIN"]).corr(self.closes[self.config["INDEX_NAME"]])
        index_corr_med = pd.concat([index_corr.median(axis=1)] * self.n_days, axis=1)
        index_return = pd.concat([daily_return[self.config["INDEX_NAME"]]] * self.n_days, axis=1)
        index_momo = pd.concat([momo[self.config["INDEX_NAME"]]] * self.n_days, axis=1)
        index_drawdown_1yr = pd.concat([drawdown_1yr[self.config["INDEX_NAME"]]] * self.n_days, axis=1)
        index_drawup_1yr = pd.concat([drawup_1yr[self.config["INDEX_NAME"]]] * self.n_days, axis=1)
        index_drawdown_ath = pd.concat([drawdown_ath[self.config["INDEX_NAME"]]] * self.n_days, axis=1)

        index_corr_med.columns = self.ticker_names
        index_return.columns = self.ticker_names
        index_momo.columns = self.ticker_names
        index_drawdown_1yr.columns = self.ticker_names
        index_drawup_1yr.columns = self.ticker_names
        index_drawdown_ath.columns = self.ticker_names

        index_momo_corr = momo.rolling(self.config["DAYS_ROLLING_CORR"], min_periods = self.config["DAYS_ROLLING_MIN"]).corr(index_momo)

        self.vol = vol
        
        return [momo, momo2, momo3, momo4, v_momo, v_momo2, daily_return, \
                vol, daily_hl, hl_vol, \
                drawdown_1yr, drawup_1yr, drawdown_ath,
                index_corr, index_corr_med, index_return, index_momo, index_momo_corr, \
                index_drawdown_1yr, index_drawup_1yr, index_drawdown_ath]

    def getLabels(self):

        self.opens[self.opens == 0.0] = np.nan

        ud = {s: getUpsideDownside(self.opens[s].values, 
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

        return labels
    
    def getXyw(self, startIx=None, endIx=None):
        labels = self.labels[startIx:endIx].drop(self.config["INDEX_NAME"], axis=1).to_numpy()
        weights = self.weights[startIx:endIx].drop(self.config["INDEX_NAME"], axis=1).fillna(0).to_numpy()
        d = (self.closes[startIx:endIx].drop(self.config["INDEX_NAME"], axis=1).size, 1)

        nan_features = [f[startIx:endIx].isna().all().all() for f in self.features]
        features = [self.features[i][startIx:endIx] for i in range(len(self.features)) if ~nan_features[i]]
        features = [f.drop(self.config["INDEX_NAME"], axis=1).to_numpy() for f in features]

        self.nan_features[endIx] = nan_features

        return self.njitGetXyw(features, labels, weights, d)


    def getPredictionsByTicker(self, model, start_ix, end_ix):
        ix = self.ix[start_ix:end_ix]
        X = self.getFeaturesByTicker(ix, start_ix)
        dummy = pd.DataFrame({"d": self.closes.loc[ix].iloc[:,0]}) * np.nan
        
        return pd.DataFrame.from_dict({s: self.safePredict(model, X[s], ix, dummy) for s in self.ticker_names}).set_index(ix)
    
    def getFeaturesByTicker(self, ix, start_ix):
        n_features = len(self.features)
        nan_features = self.nan_features[start_ix] # start_ix for test data must = end_ix for train data
        return {s: np.vstack([self.features[i][s][ix] for i in range(n_features) if ~nan_features[i]]).T for s in self.ticker_names}

    def safePredict(self, model, X, ix, dummy):
        mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)

        if sum(mask) == 0:
            return dummy["d"] #np.full(len(ix), np.nan)
        
        pred = pd.DataFrame({"p": model.predict_proba(X[mask])[:,1]}, index=ix[mask])
        return dummy.join(pred)["p"]
        

    def getTrades(self, probs, n_max_portfolio, conf=None, upside_mult = 1.0, downside_mult = 1.0, window_mult = 1.0, allow_early_unwind=False):

        if conf == None:
            conf = 0.5

        ix = probs.index
        n_days = len(ix)
   
        upside_target = (self.opens * (1 + self.vol.shift(1) * self.upside * 1e-2 * upside_mult)).loc[ix].to_numpy()
        downside_target = (self.opens * (1 - self.vol.shift(1) * self.downside * 1e-2 * downside_mult)).loc[ix].to_numpy()

        side = 1.0 if self.is_buying else -1.0

        signal = probs.gt(conf, axis=0)
        self.probs = probs.copy()
        self.signal = signal.copy()

        signalWindow = SignalWindow(signal, window_mult, self.config["DAYS_TIL_UPSIDE"]) # not shifted! for simpler indexing later
        signal = signal.shift(1).to_numpy()
        probs = probs.shift(1).to_numpy()

        opens = self.opens.loc[ix].to_numpy()
        highs = self.highs.loc[ix].to_numpy()
        lows = self.lows.loc[ix].to_numpy()
        closes = self.closes.loc[ix].to_numpy()
        
        portfolio = Portfolio(allow_early_unwind = allow_early_unwind)
        self.closed_trades = Blotter(ix, self.ticker_names)

        def closePosition(p, rate, i, level, is_early_close=False):
            p.mark(rate, ix[i], level)
            self.closed_trades.add(p if is_early_close else portfolio.remove(p))
            signalWindow.clear(p.ord, i)

        # no signals on 0th day
        for i in range(1, n_days):

            early_unwinds = portfolio.filter(signal[i], probs[i], n_max_portfolio)

            for p in portfolio.positions.copy():
                s = p.ord

                if p.isNew():
                    p.set(side, downside_target[i][s], upside_target[i][s])
                    p.mark(opens[i][s], ix[i], "OPEN")

                if signalWindow.isActive(s, i):
                    if highs[i][s] > p.upside_target:
                        closePosition(p, p.upside_target, i, "TP" if self.is_buying else "SL")

                    elif lows[i][s] < p.downside_target:
                        closePosition(p, p.downside_target, i, "SL" if self.is_buying else "TP")

                    else:
                        if signalWindow.isExpiring(s, i):
                            closePosition(p, closes[i][s], i, "EXPIRE")
                        else:
                            p.mark(closes[i][s], ix[i], "HOLD")

            for p in early_unwinds.positions.copy():
                closePosition(p, opens[i][p.ord], i, "EARLY", True)

        # close shop end of run
        for p in portfolio.positions.copy():
            closePosition(p, closes[n_days-1][p.ord], n_days-1, "END") #EXPIRE

        return self.closed_trades


#####################
## TRADING CLASSES ##
#####################

class Blotter():
    def __init__(self, ix, ticker_names):
        self.trades = []
        self.ix = ix
        self.ticker_names = ticker_names

    def add(self, p):
        self.trades.append(p)
    
    def getReturns(self, aggregated=True):
        merged_dfs = [pd.DataFrame(index=self.ix)] + \
            [pd.DataFrame({self.ticker_names[p.ord]: np.diff(sr(p.marks[0], p.marks[1:]), prepend=0) * p.side}, index=p.dates[1:]) \
             for p in self.trades]
        merged_df = pd.concat(merged_dfs)

        res = merged_df.groupby(merged_df.index).sum()
        return res.sum(1) if aggregated else res
    
    def getPositions(self, aggregated=True):
        merged_dfs = [pd.DataFrame(index=self.ix)] + \
            [pd.DataFrame({self.ticker_names[p.ord]: (np.array(p.levels[1:]) != "EARLY") * p.side}, index=p.dates[1:]) \
            for p in self.trades]
        merged_df = pd.concat(merged_dfs)

        res = merged_df.groupby(merged_df.index).sum()
        return res.sum(1) if aggregated else res

    def getTurnover(self, aggregated=True):
        merged_dfs = [pd.DataFrame(index=self.ix)] + \
            [pd.DataFrame({self.ticker_names[p.ord]: (np.array(p.levels) != "HOLD") * p.side}, index=p.dates) \
            for p in self.trades]
        merged_df = pd.concat(merged_dfs)

        res = merged_df.groupby(merged_df.index).sum()
        return res.sum(1) if aggregated else res

    def getReturnsByOrd(self, ord):
        res = []
        for t in self.trades:
            if t.ord == ord:
                res.append(t.getReturn())
        return np.array(res)
    

class SignalWindow():
    def __init__(self, signals, mult, days_til_upside):
        self.signalT = signals.to_numpy().T

        self.window_length = int(mult * days_til_upside)
    
    def isActive(self, ord, i):
        min_idx = np.max([0, i-self.window_length])
        return self.signalT[ord][min_idx:i].any()

    def clear(self, ord, i):
        min_idx = np.max([0, i-self.window_length])
        self.signalT[ord][min_idx:i] = False

    def isExpiring(self, ord, i):
        min_idx = np.max([0, i-self.window_length])
        return self.signalT[ord][min_idx+1]


class Portfolio():
    def __init__(self, positions = None, allow_early_unwind=False, age_to_cut=18):
        self.positions = set() if positions is None else positions
        self.allow_early_unwind = allow_early_unwind
        self.age_to_cut = age_to_cut

    def addNew(self, signals, probs, n):

        if not np.any(signals):
            return Portfolio()
        
        candidate_portfolio = Portfolio().create(np.where(signals)[0], probs)

        early_unwinds = Portfolio()

        if len(self.positions) + len(candidate_portfolio.positions) > n:
            self.cutLosers(early_unwinds)

        spacesLeft = n - len(self.positions)

        candidate_threshold = candidate_portfolio.getThreshold(spacesLeft)

        for p in candidate_portfolio.positions:
            # TODO use a better tie-break condition 
            if p.prob >= candidate_threshold and len(self.positions) < n:
                self.add(p)
                
        # early unwinds are only losers that were cut
        return early_unwinds


    def filter(self, signals, probs, n):

        if not np.any(signals):
            return Portfolio()
        
        if not self.allow_early_unwind:
            return self.addNew(signals, probs, n)
        
        candidate_portfolio = Portfolio().create(np.where(signals)[0], probs)

        early_unwinds = Portfolio()

        if len(self.positions) + len(candidate_portfolio.positions) > n:
            self.cutLosers(early_unwinds)

        # if new signal for existing Position, retain existing Position object
        union_portfolio = self.union(candidate_portfolio)
        union_threshold = union_portfolio.getThreshold(n)

        for p in union_portfolio.positions:
            if p.prob >= union_threshold:
                self.add(p)
            elif self.contains(p):
                self.remove(p)
                early_unwinds.add(p)
        
        return early_unwinds
    
    def cutLosers(self, early_unwinds):
        for p in self.positions.copy():
            if p.getRawPnl() < 0 and len(p.dates) > self.age_to_cut:
                self.remove(p)
                early_unwinds.add(p)

    def getThreshold(self, n):
        probs = []
        for p in self.positions:
            probs.append(p.prob)

        return highestN(np.array(probs), n)
    
    def create(self, ords, probs):
        for o in ords:
            self.add(Position(o, probs[o]))
        
        return self

    def union(self, P):
        union_set = self.positions.union(P.positions)
        return Portfolio(union_set)
    
    def remove(self, p):
        self.positions.remove(p)
        return p
    
    def add(self, p):
        self.positions.add(p)

    def contains(self, p):
        return self.positions.__contains__(p)

class Position(object):
    def __init__(self, ord, prob):
        self.ord = ord
        self.prob = prob
        self.side = None

        self.marks = []
        self.dates = []
        self.levels = []

    def __hash__(self):
        return hash(self.ord)
    
    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.ord == other.ord)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def set(self, side, downside_target, upside_target):
        self.side = side
        self.downside_target = downside_target
        self.upside_target = upside_target

    def isNew(self):
        return self.side is None

    def mark(self, rate, date, level):
        self.marks.append(rate)
        self.dates.append(date)
        self.levels.append(level)

    def getRawPnl(self):
        return self.side * (self.marks[-1] - self.marks[0])
    
    def getLogReturn(self):
        return self.side * lr(self.marks[-1] / self.marks[0])
    
    def getReturn(self):
        return self.side * sr(self.marks[0], self.marks[-1])
    
    def getMtM(self, i):
        n = len(self.marks)
        if n == 0 or i >= n:
            return np.nan
        return self.side * sr(self.marks[0], self.marks[i])
    