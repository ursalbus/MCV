#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.style.use('dark_background')

#%%

with open('data.pkl', 'rb') as handle:
    data = pickle.load(handle)


#%%
    

# del(res_fwd_high_SPY)÷

with open('forecast_60/hl_fits/res_fwd_high_SPY.pkl', 'rb') as handle:
    res_fwd_high_SPY = pickle.load(handle)
res_fwd_high_SPY
    

#%%

import os

def list_files(path):
    all_files = os.listdir(path)

    files = [f for f in all_files if os.path.isfile(os.path.join(path, f))]

    return files

directory_path = 'forecast_60/hl_fits'
print(list_files(directory_path))

test_res = {}

for file_name in list_files(directory_path):
    file_path = directory_path + "/" + file_name

    file_name_split = file_name[:-4].split("_")

    stat_name = file_name_split[2]
    ticker_name = file_name_split[3]

    if not test_res.__contains__(stat_name):
        print(f'adding {file_path}' )
        with open(file_path, 'rb') as handle:
            r = pickle.load(handle)

        test_res[stat_name] = {ticker_name: r}
    else:
        if not test_res[stat_name].__contains__(ticker_name):
            print(f'adding {file_path}')
            with open(file_path, 'rb') as handle:
                r = pickle.load(handle)

            test_res[stat_name][ticker_name] = r

#%%

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
        labels_df[k] = pd.concat([labels_df[k], pd.Series(labels[k][j], ixs[k][j], name=j)], axis=1)

    preds_df[k].set_index(pd.to_datetime(preds_df[k].index), inplace=True)
    labels_df[k].set_index(pd.to_datetime(labels_df[k].index), inplace=True)

    preds_df[k].sort_index(inplace=True)
    labels_df[k].sort_index(inplace=True)

    

#%%

class Trade():
    def __init__(self, ts, side, price, size):
        self.ts = ts
        self.side = side
        self.price = price
        self.size = size


class Order():
    def __init__(self, type, side, price, expiry, size):
        self.type = type
        self.side = side
        self.price = price
        self.expiry = expiry
        self.size = size


class ManagedPosition():
    def __init__(self, trade, TPs, SLs):
        self.ts = trade.ts
        self.side = trade.side
        self.price = trade.price
        self.size = trade.size
        
        self.open_orders = {}
        self.order_ID = 0

        self.tp_levels = []
        self.sl_levels = []

        for tp in TPs:
            self.addOrder(tp)
            self.tp_levels.append(tp.price)

        for sl in SLs:
            self.addOrder(sl)
            self.sl_levels.append(sl.price)

    def addOrder(self, order):
        self.order_ID += 1
        self.open_orders[self.order_ID] = order

    def removeOrder(self, order_ID):
        if self.open_orders.__contains__(order_ID):
            self.open_orders.pop(order_ID)


class PositionFIFO():
    def __init__(self, mtl, name):
        self.position_ID = 0
        self.open_positions = {}
        self.position = 0

        self.name = name
        self.mtl = mtl
        self.mtl[name] = []

    def addPosition(self, position):
            self.position_ID += 1
            self.open_positions[self.position_ID] = position
            self.position += position.size * position.side   

            self.mtl[self.name].append(position)

    def reducePositionAndCrystalizePnl(self, ts, order_ID, position_ID, slippage, close=None):
        pnl = 0

        position = self.open_positions[position_ID]
        order = position.open_orders[order_ID]

        trade_price = order.price if close == None else close

        if position.size > order.size:
            pnl += (position.price - trade_price) * order.size * order.side * slippage
            self.position += order.side * order.size
            position.size -= order.size
            position.removeOrder(order_ID)
            self.mtl[self.name].append(Trade(ts, order.side, trade_price, order.size))
        else:
            pnl += (position.price - trade_price) * position.size * order.side * slippage
            self.position += order.side * position.size
            self.open_positions.pop(position_ID)
            position.removeOrder(order_ID)
            self.mtl[self.name].append(Trade(ts, order.side, trade_price, position.size))

        return pnl
    
    def getVwap(self):
        if self.position == 0:
            return np.nan

        px_sum = 0.0
        size_sum = 0.0

        open_positions = self.open_positions.copy()

        for _, p in open_positions.items():
            px_sum += p.price * p.size
            size_sum += p.size
        
        return px_sum / size_sum     


class Instrument():
    def __init__(self, ord, name, mtl):
        self.ord = ord
        self.name = name

        self.fifo = PositionFIFO(mtl, name)

        self.pnl = 0.0
        self.eod_pnl = {}
        self.eod_pos = {}
        self.eod_exposure = {}

        self.last_close = np.nan

    def hasExpiredPositions(self, ts):
        for position_ID, p in self.fifo.open_positions.items():
            if ts > p.ts + HORIZON:
                return True
    
    def hasOpenPositions(self):
        return np.abs(self.fifo.position) > 0
    
    def fillOrders(self, ts, high, low, close, slippage):

        managed_positions = self.fifo.open_positions.copy()
        for position_ID, p in managed_positions.items():

            orders = p.open_orders.copy()
            for order_ID, o in orders.items():

                # another order may have closed the position, while we are still looping over it's copied orders
                if not self.fifo.open_positions.__contains__(position_ID):
                    continue

                if ts >= o.expiry:
                    self.pnl += self.fifo.reducePositionAndCrystalizePnl(ts, order_ID, position_ID, slippage, close)

                elif low <= o.price:
                    if (o.side > 0 and o.type == "TP") or (o.side < 0 and o.type == "SL"):
                        self.pnl += self.fifo.reducePositionAndCrystalizePnl(ts, order_ID, position_ID, slippage)

                elif high >= o.price:
                    if (o.side < 0 and o.type == "TP") or (o.side > 0 and o.type == "SL"):
                        self.pnl += self.fifo.reducePositionAndCrystalizePnl(ts, order_ID, position_ID, slippage)
   

    def newPosition(self, managed_position):
        self.fifo.addPosition(managed_position)

    def getExposure(self):
        e = abs(self.fifo.position) * self.fifo.getVwap() - self.getMtm(self.last_close)
        return 0.0 if np.isnan(e) else e

    def getMtm(self, mark):
        mtm = (mark - self.fifo.getVwap()) * self.fifo.position
        return 0.0 if np.isnan(mtm) else mtm
    
    def mtm(self, ts, close):
        tod_mtm = 0
        position = self.fifo.position
        
        if position != 0:
            tod_mtm += self.getMtm(close)
        
        self.eod_pnl[ts] = tod_mtm + self.pnl
        self.eod_pos[ts] = position
        self.eod_exposure[ts] = self.getExposure()

        self.last_close = close


class Portfolio():
    def __init__(self, names, max_exposure, tp_range_mults, sl_range_mults, horizon, slippage, master_trade_list):
        self.instruments = {ord: Instrument(ord, c, master_trade_list) for ord, c in enumerate(names)}

        self.max_exposure = max_exposure
        self.horizon = horizon
        self.slippage = slippage

        self.tp_range_mults = tp_range_mults
        self.sl_range_mults = sl_range_mults

    def getExposure(self):
        e = 0
        for _, inst in self.instruments.items():
            e += inst.getExposure()

        return e
    
    def manageExposure(self, size, open):
        exposure_remaining = max(0, self.max_exposure - self.getExposure())

        if exposure_remaining == 0:
            return 0
        else:
            return np.floor(exposure_remaining / open) if size * open > exposure_remaining else size
        
    def buyAndPlaceOrders(self, ord, ts, open, size, h_pred, l_pred):
        self.tradeAndPlaceOrders(ord, ts, open, size, h_pred, l_pred, 1.0)

    def sellAndPlaceOrders(self, ord, ts, open, size, h_pred, l_pred):
        self.tradeAndPlaceOrders(ord, ts, open, size, l_pred, h_pred, -1.0)

    def tradeAndPlaceOrders(self, ord, ts, open, size, tp_pred, sl_pred, trade_side):
        size = self.manageExposure(size, open)
        if size == 0:
            return
        
        position = Trade(ts, trade_side, open * (self.slippage ** trade_side), size)

        h = ts + self.horizon
        t = np.ceil(size / len(self.tp_range_mults))
        s = np.ceil(size / len(self.sl_range_mults))
        order_side = trade_side * -1
        
        TPs = [Order("TP", order_side, np.exp(tp_pred * m) * open, h, t) for m in self.tp_range_mults]
        SLs = [Order("SL", order_side, np.exp(sl_pred * m) * open, h, s) for m in self.sl_range_mults]

        self.instruments[ord].newPosition(ManagedPosition(position, TPs, SLs))
            
    def updateOrderBook(self, ts, highs, lows, closes):
        for ord, inst in self.instruments.items():
            if inst.hasOpenPositions():
                inst.fillOrders(ts, highs[ord], lows[ord], closes[ord], 1/self.slippage)
            
            inst.mtm(ts, closes[ord])


#%%

MIN_RANGE = 0
MIN_SIGNAL = 0.55

TP_RANGE_MULTS = [0.75, 1.0, 1.25]
SL_RANGE_MULTS = [2.0, 3.0]

# predict at today's close, trade on tomorrow's open
high_preds = 1e-2 * preds_df["high"].iloc[-252*20:,:].shift(1)
low_preds = 1e-2 * preds_df["low"].iloc[-252*20:,:].shift(1)

signals = ((high_preds + low_preds) / (high_preds - low_preds))

buy_mask = ((signals > MIN_SIGNAL) * (high_preds > MIN_RANGE)).values
sell_mask = ((signals < -MIN_SIGNAL) * (low_preds < -MIN_RANGE)).values

ix = signals.index

NAV = 20e6
MAX_LEVERAGE = 3.0
MAX_EXPOSURE = MAX_LEVERAGE * NAV   
HORIZON = 60
SLIPPAGE = 1 + 25 * 1e-4

names = signals.columns
master_trade_list = {}

portfolio = Portfolio(names, MAX_EXPOSURE, TP_RANGE_MULTS, SL_RANGE_MULTS, HORIZON, SLIPPAGE, master_trade_list)

signals = signals.values
high_preds = high_preds.values
low_preds = low_preds.values

opens = data.Open.loc[ix].values
highs = data.High.loc[ix].values
lows = data.Low.loc[ix].values
closes = data.Close.loc[ix].values

advs = data.Volume.rolling(30).mean().loc[ix].values

n_tickers = signals.shape[1]

# MAX_CONCENTRATION = 0.1
# position_size = NAV / n_tickers

POSITION_ADV_FRACTION = 0.1
MIN_ADV = NAV / 2

n_days = len(signals)

for t in range(n_days):

    is_tradable = (advs[t] > MIN_ADV)
    position_size = np.minimum(advs[t] * POSITION_ADV_FRACTION, NAV / np.sum(is_tradable))
    
    for ord in np.where(buy_mask[t] & is_tradable)[0]:
        portfolio.buyAndPlaceOrders(ord, t, opens[t][ord] * SLIPPAGE, np.floor(position_size[ord]/opens[t][ord]), high_preds[t][ord], low_preds[t][ord])

    for ord in np.where(sell_mask[t] & is_tradable)[0]:
        portfolio.sellAndPlaceOrders(ord, t, opens[t][ord] / SLIPPAGE, np.floor(position_size[ord]/opens[t][ord]), high_preds[t][ord], low_preds[t][ord])

    portfolio.updateOrderBook(t, highs[t], lows[t], closes[t])


#%%
    
pnl_df = pd.DataFrame()
pos_df = pd.DataFrame()
exp_df = pd.DataFrame()

for _, inst in portfolio.instruments.items():
    n = [inst.name]

    d = list(inst.eod_pnl.values())
    pnl_df = pd.concat([pnl_df, pd.DataFrame(data=d, index=ix, columns=n)], axis=1)

    p = list(inst.eod_pos.values())
    pos_df = pd.concat([pos_df, pd.DataFrame(data=p, index=ix, columns=n)], axis=1)

    e = list(inst.eod_exposure.values())
    exp_df = pd.concat([exp_df, pd.DataFrame(data=e, index=ix, columns=n)], axis=1)

cumsum_pnl = pnl_df.sum(1)
(NAV + cumsum_pnl).plot(label='pnl')
(NAV*np.exp(data.Close.SPY.loc[ix].transform('log').diff().cumsum())).plot(label='SPY pnl')
plt.legend(loc='upper left')
ax2 = plt.twinx()
(exp_df.sum(1)/NAV).plot(ax=ax2, c='r', alpha=0.4, label='gross position / NAV')
ax2.legend(loc='upper center')
plt.show()

(pos_df.sum(1)/NAV).plot(label="net position / NAV")
drawdown = (cumsum_pnl.cummax() - cumsum_pnl)
(drawdown/NAV).plot(alpha=0.3, label='drawdown / NAV')
plt.legend()
plt.show()

pd.concat([(data.Close.loc[ix]*pos_df).where(lambda x: x.gt(0)).sum(1).rename('long exposure'), 
           (data.Close.loc[ix]*pos_df).where(lambda x: x.lt(0)).sum(1).abs().rename("short exposure")], axis=1).plot()
plt.show()

daily_pnl = cumsum_pnl.diff()
print(f'years: {n_days/252:.1f} | up days: {100 * (daily_pnl > 0).sum() / n_days :.2f}%')
print(f'final pnl: ${cumsum_pnl[-1]/1e6:.1f}m ({cumsum_pnl[-1]/NAV:.2f} x NAV) | max drawdown: ${np.max(drawdown)/1e6:.2f}m ({np.max(drawdown)/NAV:.2f} x NAV)')

yearly_pnl = cumsum_pnl.groupby([pd.to_datetime(cumsum_pnl.index).year]).first()
yearly_mean = yearly_pnl.diff().mean()
yearly_std =  yearly_pnl.diff().std()
print(f'yearly | avg rtn: ${yearly_mean/1e6:.2f}m | std: ${yearly_std/1e6:.2f}m | sharpe: {yearly_mean/yearly_std:.2f}')
monthly_pnl = cumsum_pnl.groupby([pd.to_datetime(cumsum_pnl.index).year, pd.to_datetime(cumsum_pnl.index).month]).first()
monthly_mean = monthly_pnl.diff().mean()
monthly_std = monthly_pnl.diff().std()
print(f'monthly | avg rtn: ${monthly_mean/1e3:.1f}k | std: ${monthly_std/1e3:.1f}k | sharpe: {np.sqrt(12) * monthly_mean/monthly_std:.2f}')
daily_mean = daily_pnl.mean()
daily_std = daily_pnl.std()
print(f'daily | avg rtn: ${daily_mean/1e3:.1f}k | std: ${daily_std/1e3:.1f}k | sharpe: {np.sqrt(252) * daily_mean/daily_std:.2f}')

#%%

traded_name_mask = np.array([len(v) for k, v in master_trade_list.items()]) > 0

traded_names = np.array(list(master_trade_list.keys()))[traded_name_mask]
n_traded_names = len(traded_names)

_, ax = plt.subplots(nrows=n_traded_names, ncols=1, figsize=(15, 10*n_traded_names), layout='constrained')

for i, (name, trade_list) in enumerate(np.array(list(master_trade_list.items()))[traded_name_mask]):
    buy_pxs = []
    sell_pxs = []
    buy_ixs = []
    sell_ixs = []
    buy_szs = []
    sell_szs = []

    for t in trade_list:

        if t.side > 0:
            buy_pxs.append(t.price)
            buy_ixs.append(t.ts)
            buy_szs.append(t.size)
        if t.side < 0:
            sell_pxs.append(t.price)
            sell_ixs.append(t.ts)
            sell_szs.append(t.size)

    buy_szs /= np.max(np.array(buy_szs))
    sell_szs /= np.max(np.array(sell_szs))

    first_ix = max(np.min([buy_ixs[0], sell_ixs[0]]) - 250, 0)
    
    ax[i].step(ix[first_ix:], data.Close[name].loc[ix[first_ix:]], alpha=0.6, where='post')
    ax[i].scatter(ix[buy_ixs], buy_pxs, marker='^', color='green', s=50*buy_szs)
    ax[i].scatter(ix[sell_ixs], sell_pxs, marker='v', color='red', s=50*sell_szs)
    ax[i].set_yscale('log')
    ax[i].set_title(name)

    ax2 = ax[i].twinx()
    ax2.step(ix[first_ix:], pnl_df.loc[ix[first_ix:]][name]/NAV, alpha=0.8, c='orange', linestyle='--', where='post')
    ax2.axhline(0.0, alpha= 0.3, c='orange', linestyle='--')

    ax3 = ax2.twinx()
    ax3.step(ix[first_ix:], pos_df.loc[ix[first_ix:]][name]/1e6, alpha=0.8, c='red', linestyle=':', where='post')
    ax3.axhline(0.0, alpha= 0.3, c='red', linestyle=':')


#%%

_, ax = plt.subplots(nrows=n_traded_names, ncols=1, figsize=(10, 5*n_traded_names), layout='constrained')

for i, (name, trade_list) in enumerate(np.array(list(master_trade_list.items()))[traded_name_mask]):
    aftermaths = []
    sizes = []

    closes = data.Close[name]

    for t in trade_list:
        
        if not type(t) == ManagedPosition:
            continue

        tp_rtn = ((t.tp_levels[0] / closes.loc[ix[t.ts]]) - 1) * t.side

        if np.isnan(tp_rtn):
            continue

        c = np.array(closes.loc[ix[t.ts:t.ts+HORIZON]].ffill())

        if len(c) < HORIZON:
            c = np.concatenate([c, [c[-1] for _ in range(HORIZON - len(c))]])

        afm = t.side * (c - t.price) / (t.price*tp_rtn)

        if np.isnan(afm[0]):
            print("here")

        aftermaths.append(afm)

        sizes.append(t.size)

        ax[i].plot(afm, alpha=0.3, c='blue')
        ax[i].set_title(name)
        ax[i].axhline(0.0, alpha= 0.5, c='orange', linestyle='--')
        ax[i].axhline(1.0, alpha= 0.5, c='red', linestyle='--')

    if len(aftermaths) < 1:
        continue
    
    avg_afm = np.array(aftermaths).mean(0)
    ax[i].plot(avg_afm, alpha=0.8, c='red', linestyle=':')
    ax[i].set_ylim(min(0, min(avg_afm))-1, max(1, max(avg_afm))+1)

    ax[i].axhline(0.0, alpha= 0.5, c='orange', linestyle='--')
    ax[i].axhline(1.0, alpha= 0.5, c='red', linestyle='--')




#%%





















#%%









