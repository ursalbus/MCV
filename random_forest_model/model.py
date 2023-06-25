
import pandas as pd
import pickle
import glob
import os
from datetime import datetime
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from features_labels import FeaturesLabels

class Model():

    def __init__(self, model_class, config, data_raw, folder_path):

        self.config = self.processConfig(config, data_raw, model_class, folder_path)
        self.data_raw = data_raw

        self.tscv = TimeSeriesSplit(n_splits=self.config["N_SPLITS"], 
                                    max_train_size=self.config["TRAIN_SIZE"], 
                                    test_size=self.config["TEST_SIZE"])

        # test/train | split_i | start/end
        self.ixs = np.zeros((2, self.config["N_SPLITS"], 2), dtype=np.int64)

        # train/test | sell/buy | split_i
        self.models = np.zeros((2, 2, self.config["N_SPLITS"]), dtype = model_class)

        # train/test | sell/buy | split_i | X/y/w
        self.Xyw = np.zeros((2, 2, self.config["N_SPLITS"], 3),dtype=object)

        self.init()

    def processConfig(self, config, data_raw, model_class, folder_path):
        config["N_SPLITS"] = int((len(data_raw) - config["TRAIN_SIZE"]) / config["TEST_SIZE"]) 
        config["MODEL_CLASS"] = model_class
        config["FOLDER_PATH"] = folder_path
        
        config["UPSIDE"] = np.sqrt(config["DAYS_TIL_UPSIDE"] * 2)
        config["DOWNSIDE"] = np.sqrt(config["DAYS_TIL_UPSIDE"])

        config["DAYS_MA_SLOW"] = config["DAYS_TIL_UPSIDE"] * 2
        config["DAYS_ROLLING_CORR"] = int(config["DAYS_TIL_UPSIDE"] / 2)
        config["DAYS_ROLLING_MIN"] = config["DAYS_MA_FAST"]
        
        return config
        
    def init(self):
        # sell/buy
        self.sb_FL = [FeaturesLabels(self.config, self.data_raw, is_buying=False), FeaturesLabels(self.config, self.data_raw, is_buying=True)]

    def save(self, overwrite_last=False):
    
        if overwrite_last:
            # TODO implement
            list_of_files = glob.glob(self.config["FOLDER_PATH"] + '*.pkl') # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
        
        timestamp = self.config["START_DATE"] + "_" + self.config["END_DATE"] + "__" + datetime.now().strftime("%Y-%M-%d_%H-%M-%S")

        file_path = self.config["FOLDER_PATH"] + '{}.pkl'.format(timestamp)

        with open(file_path, 'wb') as file:
            pickle.dump((self.config ,self.ixs, self.models, self.Xyw, self.sb_FL, self.sell_probs, self.buy_probs), file)

    def load(self, file_path=None, return_list=False):
        
        if file_path is None:    
            list_of_files = glob.glob(self.config["FOLDER_PATH"] + '*.pkl') # * means all if need specific format then *.csv
            file_path = max(list_of_files, key=os.path.getctime)

        with open(file_path, 'rb') as file:
            if return_list:
                return pickle.load(file)
            else:         
                self.config, self.ixs, self.models, self.Xyw, self.sb_FL, self.sell_probs, self.buy_probs = pickle.load(file)                

    def train(self, fit_model=True):

        k = 0
        for train, test in self.tscv.split(self.data_raw):   
            # print("%s %s" % (train[0], test[0]))
            self.ixs[0][k][0] = train[0]
            self.ixs[0][k][1] = train[0] + self.config["TRAIN_SIZE"]
            self.ixs[1][k][0] = test[0]
            self.ixs[1][k][1] = test[0] + self.config["TEST_SIZE"]
            k = k + 1

        for k in range(self.config["N_SPLITS"]): 
            print("fitting fold {} of {}".format(k, self.config["N_SPLITS"]))

            for i in range(2):
                for sb in [False, True]:
                    j = int(sb)
                    self.Xyw[i][j][k][0], self.Xyw[i][j][k][1], self.Xyw[i][j][k][2] = self.sb_FL[j].getXyw(self.ixs[i][k][0], self.ixs[i][k][1])
                    if fit_model:
                        self.models[i][j][k] = self.config["MODEL_CLASS"](n_jobs=-1)
                        if self.Xyw[i][j][k][1].any():
                            self.models[i][j][k].fit(self.Xyw[i][j][k][0], self.Xyw[i][j][k][1])

    def predict(self):
        i = 1 # test-data for predictions

        self.sell_probs = pd.DataFrame()
        self.buy_probs = pd.DataFrame()

        for k in range(self.config["N_SPLITS"]):
            print("predicting fold {} of {}".format(k, self.config["N_SPLITS"]))

            # predictions from models fit on training-data
            s_probs = self.sb_FL[0].getPredictionsByTicker(self.models[0][0][k], self.ixs[i][k][0], self.ixs[i][k][1])
            b_probs = self.sb_FL[1].getPredictionsByTicker(self.models[0][1][k], self.ixs[i][k][0], self.ixs[i][k][1])

            self.sell_probs = pd.concat([self.sell_probs, s_probs])
            self.buy_probs = pd.concat([self.buy_probs, b_probs])

        return self.sell_probs, self.buy_probs
    
    def rerunXyw(self):
        # train/test | sell/buy | split_i | X/y/w
        Xyw = np.zeros((2, 2, self.config["N_SPLITS"], 3),dtype=object)

        for k in range(self.config["N_SPLITS"]): 
            print("fitting fold {} of {}".format(k, self.config["N_SPLITS"]))

            for i in range(2):
                for sb in [False, True]:
                    j = int(sb)
                    Xyw[i][j][k][0], Xyw[i][j][k][1], Xyw[i][j][k][2] = self.sb_FL[j].getXyw(self.ixs[i][k][0], self.ixs[i][k][1])

class ModelConfig():
    def __init__(self, c):
        self.config = c