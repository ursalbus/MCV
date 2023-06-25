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

pd.options.mode.chained_assignment = None # default='warn' # type: ignore 

#%%
## SETUP
INDEX_NAME = "SPY"

model_config = {"INDEX_NAME": INDEX_NAME,
                "TICKER_NAMES": sp500_tickers + [INDEX_NAME], 
                "START_DATE": "2019-01-01", 
                "END_DATE": "2023-06-23", 
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
from features_labels import FeaturesLabels

import importlib
import model
importlib.reload(model)
from model import Model

FOLDER_PATH = '/Users/coconut/Developer/git/MCV/random_forest_model/models/'

rfc_model = Model(RandomForestClassifier, model_config, data_raw, FOLDER_PATH)

#%%
## TRAIN AND PREDICT
# rfc_model.train()
# rfc_model.predict()
# rfc_model.save()

rfc_model.load()

#%%

# loaded_file = rfc_model.load(return_list=True)
# rfc_model.config = rfc_model.processConfig(model_config, data_raw, RandomForestClassifier, FOLDER_PATH)
# rfc_model.ixs = loaded_file[1]
# rfc_model.models = loaded_file[2]
# rfc_model.Xyw = loaded_file[3] 
# rfc_model.sb_FL = loaded_file[4] 
# rfc_model.sell_probs = loaded_file[5] 
# rfc_model.buy_probs = loaded_file[6]

#%%

import importlib
import features_labels
importlib.reload(features_labels)
from features_labels import FeaturesLabels

rfc_model.init()

#%%
## GET TRADES
trader_config = {"N_MAX_PORTFOLIO": 50, "CONF": 0.50}
trader = Trader(trader_config, rfc_model)
trader.trade()

#%%
## RESULTS

# import importlib
# import results
# importlib.reload(results)
# from results import Results

results_config = {"SLIPPAGE_PER_TRADE_BPS": 10}      
res = Results(results_config, rfc_model, trader)

res.plotReturns()

res.plotSharpe('Y', 250)
res.plotSharpe('M', 20)

res.returnCorrelation()
res.returnCorrelation('W')
res.returnCorrelation('M')

res.plotReturnsHistogram('W')

res.plotPricePaths()

res.tradesSummary()

res.plotFeaturesByTicker("AAPL")
res.plotFeatures()


#%%




#%%


