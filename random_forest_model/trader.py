import numpy as np
import pandas as pd
from copy import deepcopy


class Trader():
    def __init__(self, config, model):
        self.config = config | model.config
        self.model = model

        self.opens = self.model.features.opens
        self.highs = self.model.features.highs
        self.lows = self.model.features.lows
        self.closes = self.model.features.closes
        self.vol = self.model.features.vol

    def trade(self, **args):
        self.sell_trades = self.getTrades(False, self.config["N_MAX_PORTFOLIO"], self.config["SELL_CONF"], **args)
        self.buy_trades = self.getTrades(True, self.config["N_MAX_PORTFOLIO"], self.config["BUY_CONF"], **args)

        return self.sell_trades, self.buy_trades

    def getTrades(self, is_buying, n_max_portfolio, conf=None, upside_mult = 1.0, downside_mult = 1.0, window_mult = 1.0):

        print({"is_buying": is_buying, "conf": conf, "allow_early_unwind": self.config["ALLOW_EARLY_UNDWIND"], "adv_cutoff": self.config["ADV_CUTOFF"]})

        if conf == None:
            conf = 0.5

        if is_buying:
            side = 1.0 
            probs = deepcopy(self.model.buy_probs)
            upside = self.config["UPSIDE"]
            downside = self.config["DOWNSIDE"]
        else:
            side = -1.0 
            probs = deepcopy(self.model.sell_probs)
            upside = self.config["DOWNSIDE"]
            downside = self.config["UPSIDE"]

        probs = probs.reindex(columns=self.config["TICKER_NAMES"])
        cols = probs.columns

        ix = probs.index
        n_days = len(ix)
   
        upside_target = (self.opens * (1 + self.vol.shift(1) * upside * 1e-2 * upside_mult)).loc[ix].to_numpy()
        downside_target = (self.opens * (1 - self.vol.shift(1) * downside * 1e-2 * downside_mult)).loc[ix].to_numpy()

        adv_mask = self.model.features.adv.rank(axis=1).apply(lambda x:  x/(~x.isna()).sum(), axis=1) > self.config["ADV_CUTOFF"]
        
        probs = probs.where(adv_mask, other=0)

        signal = probs.gt(conf, axis=0)

        signalWindow = SignalWindow(signal, window_mult, self.config["DAYS_TIL_UPSIDE"]) # not shifted! for simpler indexing later
        
        # signal fires at close on day t is traded at open on day t+1 => shift signals to align with ohlc
        signal = signal.shift(1).to_numpy()
        probs = probs.shift(1).to_numpy()

        opens = self.opens.loc[ix].to_numpy()
        highs = self.highs.loc[ix].to_numpy()
        lows = self.lows.loc[ix].to_numpy()
        closes = self.closes.loc[ix].to_numpy()

        # only used for logging at this point
        vol = self.vol.shift(1).loc[ix].to_numpy()
        adv = self.model.features.adv.shift(1).loc[ix].to_numpy()
        
        portfolio = Portfolio(allow_early_unwind = self.config["ALLOW_EARLY_UNDWIND"])
        self.closed_trades = Blotter(ix, cols)

        def closePosition(p, rate, i, level, is_early_close=False):
            p.mark(rate, ix[i], level)
            self.closed_trades.add(early_unwinds.remove(p) if is_early_close else portfolio.remove(p))
            signalWindow.clear(p.ord, i, level)

        # signal from day i=0 is traded on day i=1, therefore start loop from i=1
        for i in range(1, n_days-1):

            early_unwinds = portfolio.filter(signal[i], probs[i], n_max_portfolio)

            for p in portfolio.positions.copy():
                s = p.ord

                if p.isNew():
                    p.set(side, downside_target[i][s], upside_target[i][s], vol[i][s], adv[i][s])
                    p.mark(opens[i][s], ix[i], "OPEN")

                # if signalWindow.isActive(s, i):
                if opens[i][s] > p.upside_target:
                    closePosition(p, opens[i][s], i, "TP" if is_buying else "SL")

                elif opens[i][s] < p.downside_target:
                    closePosition(p, opens[i][s], i, "SL" if is_buying else "TP")

                elif highs[i][s] > p.upside_target:
                    rate = closes[i][s] if highs[i][s] == closes[i][s] else p.upside_target
                    closePosition(p, rate, i, "TP" if is_buying else "SL")

                elif lows[i][s] < p.downside_target:
                    rate = closes[i][s] if lows[i][s] == closes[i][s] else p.downside_target
                    closePosition(p, rate, i, "SL" if is_buying else "TP")

                elif signalWindow.isExpiring(s, i):
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
        df = pd.DataFrame(index=self.ix)
        for i, p in enumerate(self.trades):
            # print("rtn {} has {} dates".format(i, len(p.dates)))
            col_name = "{}_{}".format(self.ticker_names[p.ord], i)
            new = df.join(pd.DataFrame({col_name: np.diff(sr(p.marks[0], p.marks[1:]), prepend=0) * p.side}, index=p.dates[1:]))
            df = new
            del(new)
        
        return df.sum(1) if aggregated else df
    
    def getPositions(self, aggregated=True):
        df = pd.DataFrame(index=self.ix)
        for i, p in enumerate(self.trades):
            # print("psn {} has {} dates".format(i, len(p.dates)))
            col_name = "{}_{}".format(self.ticker_names[p.ord], i)
            new = df.join(pd.DataFrame({col_name: (np.array(p.levels[1:]) != "EARLY") * p.side}, index=p.dates[1:]))
            df = new
            del(new)

        return df.sum(1) if aggregated else df

    def getTurnover(self, aggregated=True):
        df = pd.DataFrame(index=self.ix)
        for i, p in enumerate(self.trades):
            # print("trn {} has {} dates".format(i, len(p.dates)))
            col_name = "{}_{}".format(self.ticker_names[p.ord], i)
            df_to_join = pd.DataFrame({col_name: p.side}, index=list(set([p.dates[0],p.dates[-1]])))
            new = df.join(df_to_join)
            df = new
            del(new)

        return df.sum(1) if aggregated else df

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

    def clear(self, ord, i, reason):
        min_idx = np.max([0, i-self.window_length])
        self.signalT[ord][min_idx:i] = False
        # print("{}: cleared {} at ix {}".format(reason, ord, i))

    def isExpiring(self, ord, i):
        min_idx = np.max([0, i-self.window_length])
        return self.signalT[ord][min_idx+1]


class Portfolio():
    def __init__(self, positions = None, allow_early_unwind=False, age_to_cut=30):
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
    
    def set(self, side, downside_target, upside_target, vol, adv):
        self.side = side
        self.downside_target = downside_target
        self.upside_target = upside_target
        self.vol = vol
        self.adv = adv

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
