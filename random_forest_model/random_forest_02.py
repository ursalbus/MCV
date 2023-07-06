#%%
## IMPORTS
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sp500 import sp500_tickers

from model import Model
from trader import Trader
from results import Results
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression, SGDClassifier

pd.options.mode.chained_assignment = None # default='warn' # type: ignore 

#%%
## SETUP
INDEX_NAME = "SPY"

model_config = {"INDEX_NAME": INDEX_NAME,
                "TICKER_NAMES": sp500_tickers + [INDEX_NAME], 
                "START_DATE": "2011-01-01", 
                "TRADING_START_DATE": "2013-01-01", 
                "END_DATE": "2018-01-01", 
                "DAYS_TIL_UPSIDE": 60, 
                "DAYS_MA_FAST": 5,
                "TRAIN_SIZE": 120, 
                "TEST_SIZE": 5}

#%% 
## PULL DATA
data_raw = yf.download(model_config["TICKER_NAMES"], 
                       start=model_config["START_DATE"], 
                       end=model_config["END_DATE"], 
                       interval = "1d")

#%%

def spoofData(data, frac_spoofed=0.20, overnight_spoof=True):
    rtns =  1 + np.log((data["Close"].shift(1) if overnight_spoof else data["Close"]) / data["Open"]) 

    dates_to_mask = rtns.sample(frac=1-frac_spoofed).index
    rtns.loc[dates_to_mask,:] = 1 # type: ignore
    rtns[rtns > 1] = 1

    spoof_chg_cuml = rtns.cumprod()
    
    data["High"] = data["High"] * spoof_chg_cuml
    data["Low"] = data["Low"] * spoof_chg_cuml
    data["Close"] = data["Close"] * spoof_chg_cuml
    data["Open"] = data["Open"] * (spoof_chg_cuml if overnight_spoof else spoof_chg_cuml.shift(1))

    return data
        
data_spoofed = spoofData(deepcopy(data_raw), overnight_spoof=True)

data_spoofed["Close"]["AAPL"].plot()
data_raw["Close"]["AAPL"].plot()

#%%

def spoofTickers(data, frac_dates_spoofed=0.10, frac_tickers_spoofed=0.10, overnight_spoof=True):
    rtns =  1 + np.log((data["Close"].shift(1) if overnight_spoof else data["Close"]) / data["Open"]) 

    tickers_to_mask = rtns.sample(frac=1-frac_tickers_spoofed, axis=1).columns
    print(tickers_to_mask)
    dates_to_mask = rtns.sample(frac=1-frac_dates_spoofed).index

    rtns.loc[dates_to_mask, :] = 1 # type: ignore
    rtns.loc[:, tickers_to_mask] = 1 # type: ignore
    rtns[rtns < 1] = 1

    spoof_chg_cuml = rtns.cumprod()
    
    data["High"] = data["High"] * spoof_chg_cuml
    data["Low"] = data["Low"] * spoof_chg_cuml
    data["Close"] = data["Close"] * spoof_chg_cuml
    data["Open"] = data["Open"] * (spoof_chg_cuml if overnight_spoof else spoof_chg_cuml.shift(1))

    return data, set(rtns.columns).difference(set(tickers_to_mask))
    
data_spoofed, tickers_spoofed = spoofTickers(deepcopy(data_raw), overnight_spoof=True)

data_spoofed["Close"]["AAPL"].plot()
data_raw["Close"]["AAPL"].plot()


#%%
## CREATE MODEL
    
import importlib
import features_labels
importlib.reload(features_labels)
from features_labels import Features, Labels

import importlib
import model
importlib.reload(model)
from model import Model

import importlib
import trader
importlib.reload(trader)
from trader import Trader

# FOLDER_PATH = '/Users/coconut/Developer/git/MCV/logistic_model/models/'
FOLDER_PATH = '/Users/coconut/Developer/git/MCV/random_forest_model/models/'

rfc_model = Model(RandomForestClassifier(n_estimators=100, n_jobs=-1), model_config, data_spoofed, FOLDER_PATH)

#%%
## TRAIN, PREDICT AND SAVE

rfc_model.train()
# rfc_model.predict()
model_path = rfc_model.save()

# %%
## RELOAD FIT

# rfc_model.load()

# loaded_file = rfc_model.load(return_list=True)
# rfc_model.config = rfc_model.processConfig(model_config, data_raw, RandomForestClassifier, FOLDER_PATH)
# rfc_model.ixs = loaded_file[1]
# rfc_model.models = loaded_file[2]
# rfc_model.labels = loaded_file[3] 
# rfc_model.sell_probs = loaded_file[4] 
# rfc_model.buy_probs = loaded_file[5]

#%%
## GET TRADES

# import importlib
# import trader
# importlib.reload(trader)
# from trader import Trader

trader_config = {"N_MAX_PORTFOLIO": 100, "CONF": 0.55}
rfc_trader = Trader(trader_config, rfc_model)
rfc_trader.trade()

#%%
## CALC RESULTS

# import importlib
# import results
# importlib.reload(results)
# from results import Results

results_config = {"SLIPPAGE_PER_TRADE_BPS": 10}      
res = Results(results_config, rfc_model, rfc_trader)

#%%
## PLOT RESULTS

res.returns = res.returns_raw - res.fees
# res.returns = 6 * (res.returns * (2/3) +  res.index_returns * (1/3))

res.plotReturns()
res.returnCorrelation()
res.returnCorrelation('W')
res.returnCorrelation('M')

# res.plotSharpe('Y', 250)
# res.plotSharpe('M', 20)
# res.plotReturnsHistogram('W')
# res.plotPricePaths()
# res.tradesSummary()
# res.plotFeaturesByTicker("AAPL")
# res.plotFeatures()

#%%




#%%
## HOW MANY TIMES WERE SPOOFED TICKERS TRADED
rtns = []

trade_count = {}
trade_count_spoofed = {}

for t in rfc_trader.buy_trades.trades + rfc_trader.sell_trades.trades:
    t_name = model_config["TICKER_NAMES"][t.ord]
    if t_name in tickers_spoofed:
        trade_count[t_name] = trade_count.get(t_name, 0) + 1
    else: 
        trade_count_spoofed[t_name] = trade_count.get(t_name, 0) + 1

trade_count


#%%

plt.hist(np.array(rtnss), bins=100)

#%%
np.percentile(rtnss[~np.isnan(rtnss)], 100)

#%%



#%%



#%%

#%%



#%%

#%%



#%%

#%%






