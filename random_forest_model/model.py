
import pandas as pd
import pickle
import glob
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from features_labels import Features, Labels
from joblib import Parallel, delayed
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
from numba import njit

class Model():

    def __init__(self, unfit_model, config, data_raw, folder_path):

        # allow features to burn in before trading start date
        self.data_raw = data_raw.loc[pd.Timestamp(config["TRADING_START_DATE"]):]

        self.config = self.processConfig(config, self.data_raw, unfit_model, folder_path)
        
        self.unfit_model = unfit_model
        self.last_fit_models = np.zeros((2,), dtype=object)

        # train/test | split_i |
        self.ixs = np.zeros((2, self.config["N_SPLITS"], ), dtype=object)

        # sell/buy | split_i
        self.coefs = np.zeros((2, self.config["N_SPLITS"]), dtype = object)

        self.features = Features(self.config, data_raw)
        
        # sell/buy
        self.labels = [Labels(self.config, self.features, self.data_raw, is_buying=False), 
                       Labels(self.config, self.features, self.data_raw, is_buying=True)]


    def processConfig(self, config, data_raw, unfit_model, folder_path):
        config["N_SPLITS"] = int((len(data_raw) - config["TRAIN_SIZE"]) / config["TEST_SIZE"]) 
        config["UNFIT_MODEL"] = unfit_model
        config["FOLDER_PATH"] = folder_path
        
        config["UPSIDE"] = np.sqrt(config["DAYS_TIL_UPSIDE"] * 2)
        config["DOWNSIDE"] = np.sqrt(config["DAYS_TIL_UPSIDE"])

        config["DAYS_MA_SLOW"] = config["DAYS_TIL_UPSIDE"] * 2
        config["DAYS_ROLLING_CORR"] = int(config["DAYS_TIL_UPSIDE"] / 2)
        config["DAYS_ROLLING_MIN"] = config["DAYS_MA_FAST"]
        
        return config


    def train(self):

        tscv = TimeSeriesSplit(n_splits=self.config["N_SPLITS"], max_train_size=self.config["TRAIN_SIZE"], test_size=self.config["TEST_SIZE"])

        k = 0
        for train, test in tscv.split(self.data_raw):   
            self.ixs[0][k] = self.data_raw.index[train[0]:train[0] + self.config["TRAIN_SIZE"]]
            self.ixs[1][k] = self.data_raw.index[test[0]:test[0] + self.config["TEST_SIZE"]]
            k = k + 1

        # sell/buy

        self.sell_probs = pd.DataFrame()
        self.buy_probs = pd.DataFrame()

        for k in range(self.config["N_SPLITS"]): 
            print("fitting fold {} of {}".format(k, self.config["N_SPLITS"]))
            
            fit_models = [deepcopy(self.unfit_model), deepcopy(self.unfit_model)]
            features_to_mask = np.zeros((2, ), dtype=object)
            
            for j in range(2):      
                X, y, _, features_to_mask[j] = self.labels[j].getXyw(self.ixs[0][k]) #type: ignore
        
                if y.any():
                    fit_models[j].fit(X, y)
                    del(X, y,)

            # out-of-sample: predictions on test-data, made by models fit to training-data, mask features based on training-data
            s_probs = self.labels[0].getPredictionsByTicker(fit_models[0], self.ixs[1][k], features_to_mask[0])  #type: ignore
            b_probs = self.labels[1].getPredictionsByTicker(fit_models[1], self.ixs[1][k], features_to_mask[1])  #type: ignore

            self.sell_probs = pd.concat([self.sell_probs, s_probs])
            self.buy_probs = pd.concat([self.buy_probs, b_probs])
            del(s_probs, b_probs, features_to_mask) #type: ignore

            self.coefs[0][k] = fit_models[0].feature_importances_
            self.coefs[1][k] = fit_models[1].feature_importances_

            if k < self.config["N_SPLITS"] - 1:
                del(fit_models)

        self.last_fit_models = fit_models #type: ignore


    def save(self, overwrite_last=False):
    
        if overwrite_last:
            # TODO implement
            list_of_files = glob.glob(self.config["FOLDER_PATH"] + '*.pkl')
            latest_file = max(list_of_files, key=os.path.getctime)
        
        timestamp = self.config["START_DATE"] + "_" + self.config["END_DATE"] + "__" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file_path = self.config["FOLDER_PATH"] + '{}.pkl'.format(timestamp)

        with open(file_path, 'wb') as file:
            pickle.dump((self.config, self.ixs, 
                         self.last_fit_models[0], self.last_fit_models[1], 
                         np.array(self.coefs), 
                         self.sell_probs, self.buy_probs), file)

        return file_path
    
    def load(self, file_path=None, return_list=False):
        
        if file_path is None:    
            list_of_files = glob.glob(self.config["FOLDER_PATH"] + '*.pkl')
            file_path = max(list_of_files, key=os.path.getctime)

        with open(self.config["FOLDER_PATH"] + file_path if file_path is None else file_path, 'rb') as file:
            if return_list:
                return pickle.load(file)
            else:         
                self.config, self.ixs, self.last_fit_models[0], self.last_fit_models[1], self.coefs, self.sell_probs, self.buy_probs = pickle.load(file)                


    def getFeatureLabelCorrelations(self):
        feature_names = self.features.feature_names
        feature_label_corrs = [pd.DataFrame(), pd.DataFrame()]
        percentile_list = np.arange(0.1, 1.01, 0.1)

        for j in range(2):
            corr_dict = {f: 
                            {t: 
                                {int(k*100): v for k,v in zip(percentile_list, 
                                                    getCorrelationByPercentile(self.features.features[i][t].to_numpy(), 
                                                                                self.labels[j].labels[t].to_numpy()))}
                            for t in self.config["TICKER_NAMES"]}
                        for i, f in enumerate(feature_names)}
            
            flat_dict = flattenNestedDictToTupleKeys(corr_dict)

            feature_label_corrs[j] = pd.DataFrame({"corr": flat_dict.values()}, 
                                                index=pd.MultiIndex.from_tuples(flat_dict, names=['feauture', 'ticker', 'p']))

        return feature_label_corrs
    
    def plotFeatureLabelCorrelations(self, sell_buy_ord = 0):
        feature_names = self.features.feature_names
        feature_label_corrs = self.getFeatureLabelCorrelations

        _, ax = plt.subplots(ncols=1, nrows=len(feature_names), figsize=(10,len(feature_names)*7), sharex=True, sharey=True)

        for i, f in enumerate(feature_names):
            for p in [0.2, 0.4, 0.5, 0.6, 0.8]:
                data = feature_label_corrs[sell_buy_ord].xs(f, level=0).droplevel(0).groupby('p').quantile(p)  # type: ignore
                ax[i].plot(data.index, data["corr"], label=int(100*p))
                ax[i].axhline(0, linestyle="--", alpha=0.5, color='grey')
            ax[i].title.set_text(f)
            ax[i].legend(loc='upper left')


######################
## GLOBAL FUNCTIONS ##
######################

@njit
def getCorrelationByPercentile(A, B):
     # take percentiles of A, correlate with B

    quantile_bins = [np.nanquantile(A, q) for q in np.arange(0, 1.1, 0.1)]
    quantile_masks = [np.where((A >= q1) & (A < q2))[0] for q1, q2 in zip(quantile_bins[:-1], quantile_bins[1:])]
    return [np.corrcoef(A[ixs], B[ixs])[0][1] for ixs in quantile_masks]


def flattenNestedDictToTupleKeys(data_dict):
    flattened_data = {}
    for outer_key, outer_value in data_dict.items():
        for inner_key, inner_value in outer_value.items():
            for key, value in inner_value.items():
                flattened_data[(outer_key, inner_key, key)] = value

    return flattened_data