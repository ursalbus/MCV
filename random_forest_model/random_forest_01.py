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

from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression, SGDClassifier

pd.options.mode.chained_assignment = None # default='warn' # type: ignore 

#%%
## SETUP
INDEX_NAME = "SPY"

model_config = {"INDEX_NAME": INDEX_NAME,
                "TICKER_NAMES": sp500_tickers + [INDEX_NAME], 
                "START_DATE": "2016-01-01", 
                "TRADING_START_DATE": "2017-01-01", 
                "END_DATE": "2023-07-01", 
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

rfc_model = Model(RandomForestClassifier(n_estimators=100, n_jobs=-1), model_config, data_raw, FOLDER_PATH)

#%%
## TRAIN AND PREDICT

rfc_model.train()

## TRY TO SAVE
try:
    model_path = rfc_model.save()
except:
    print("nope")


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

import importlib
import trader
importlib.reload(trader)
from trader import Trader

trader_config = {"N_MAX_PORTFOLIO": 100, "SELL_CONF": 0.5, "BUY_CONF": 0.5, "ADV_CUTOFF_PERCENTILE": 0.20, "ALLOW_EARLY_UNDWIND": False}
rfc_trader = Trader(trader_config, rfc_model)
rfc_trader.trade()

#%%
## CALC RESULTS

import importlib
import results
importlib.reload(results)
from results import Results

results_config = {"SLIPPAGE_PER_TRADE_BPS": 10}      
res = Results(results_config, rfc_model, rfc_trader)

#%%
## PLOT RESULTS

res.returns = res.returns_raw - res.fees
# res.returns = 3 * (res.returns * (2/3) +  res.index_returns * (1/3))

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
res.tradesSummary()

#%%

(res.trades_df.join(pd.qcut(res.trades_df.vol, q=5).rename("vol_bucket")).groupby(["vol_bucket", "side", "prob"])["ret"].agg({'mean'}).unstack(level=1) * 100).style.background_gradient(cmap='RdBu', vmax=10, vmin=-5)

#%%

(res.trades_df.join(pd.qcut(1e-6 * res.trades_df.adv, q=5).rename("adv_bucket")).groupby(["adv_bucket", "side", "prob"])["ret"].agg({'mean'}).unstack(level=1) * 100).style.background_gradient(cmap='RdBu', vmax=10, vmin=-5)


#%%

#%%



#%%

#%%






