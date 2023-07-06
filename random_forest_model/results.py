import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


class Results():
    def __init__(self, config, model, trader):
        self.config = config | trader.config
        self.trader = trader
        self.model = model

        self.data_raw = model.data_raw
        self.prob_bins = np.arange(0.5, 1.01, 0.1)

        self.returns_raw = ((trader.sell_trades.getReturns() + trader.buy_trades.getReturns()) / self.config["N_MAX_PORTFOLIO"])
        self.ix_range = self.returns_raw.index

        self.index_returns = self.data_raw.loc[self.ix_range]["Close"][self.config["INDEX_NAME"]].apply(lambda x: np.log(x) * 1e2).diff()

        self.n_sell_positions = trader.sell_trades.getPositions()
        self.n_buy_positions = trader.buy_trades.getPositions()

        self.buy_turnover = trader.buy_trades.getTurnover()
        self.sell_turnover = trader.sell_trades.getTurnover()

        self.fees = ((self.buy_turnover - self.sell_turnover) * 1e2 * self.config["SLIPPAGE_PER_TRADE_BPS"] * 1e-4) / self.config["N_MAX_PORTFOLIO"]
        self.returns = self.returns_raw - self.fees
        # self.returns = 1 * (self.returns * (2/3) +  self.index_returns * (1/3))

        self.feature_names = model.features.feature_names


    def plotReturns(self):

        _, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(15,40))

        ax[0].plot(self.returns.cumsum())
        ax[0].plot(self.fees.cumsum())
        ax[0].plot(self.index_returns.cumsum())
        ax[0].axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax0a = ax[0].twinx()
        ax0a.sharey(ax[0])
        (self.returns.cumsum() - self.index_returns.cumsum()).plot(ax=ax0a, color='r')

        ax[1].plot((self.n_buy_positions + self.n_sell_positions))
        ax[1].axhline(0, color='grey', linestyle='--', alpha=0.5)

        ax[2].plot(self.n_buy_positions, color='r')
        ax[2].plot(self.n_sell_positions, color='g')
        ax[2].axhline(0, color='grey', linestyle='--', alpha=0.5)

        ax[3].bar(self.buy_turnover.index, self.buy_turnover, color='r')
        ax[3].bar(self.sell_turnover.index, self.sell_turnover, color='g')
        ax[3].axhline(0, color='grey', linestyle='--', alpha=0.5)

        (self.returns.cumsum().cummax() - self.returns.cumsum()).plot(ax=ax[4])
        (self.index_returns.cumsum().cummax() - self.index_returns.cumsum()).plot(ax=ax[4])
        plt.show()

        ## print out
        n_years = (self.returns.index[-1] - self.returns.index[0]).days / 365 # type: ignore
        total_return = self.returns.cumsum()[-1]
        index_return = self.index_returns.cumsum()[-1]
        print("{:.2f}% total return in {:.1f} years".format(total_return, n_years))
        print("{:.2f}% above index".format(total_return - index_return, n_years))
        print("{:.2f}% CAGR".format((pow(1 + total_return / 100, 1/n_years) - 1) * 100))
        print("{:.2f}% index CAGR".format((pow(1 + index_return / 100, 1/n_years) - 1) * 100))
        print("")
        print("total sharpe: {}".format(self.returns.cumsum()[-1] / (self.returns.std() * np.sqrt(len(self.returns)))))
        print("index sharpe: {}".format(self.index_returns.cumsum()[-1] / (self.index_returns.std() * np.sqrt(len(self.index_returns)))))
        print("total sortino: {}".format(self.returns.cumsum()[-1] / (self.returns[self.returns < 0].std() * np.sqrt(len(self.returns[self.returns < 0])))))
        print("index sortino: {}".format(self.index_returns.cumsum()[-1] / (self.index_returns[self.index_returns < 0].std() * np.sqrt(len(self.index_returns[self.index_returns < 0])))))
        print("")

        ldd = 0
        cdd = 0
        for t in range(len(self.returns)):
            r = ((self.returns.cumsum().cummax() - self.returns.cumsum()) > 0)[t]
            cdd = cdd + 1 if r else 0
            ldd = cdd if cdd > ldd else ldd

        print("longest drawdown (days): " + str(ldd))   
        print("")


    def plotSharpe(self, freq='Y', scale=250):

        first_trade_i = np.where((self.buy_turnover - self.sell_turnover) > 0)[0][0]
        yearly_sharpe = self.returns[first_trade_i:].cumsum().resample(freq).first().diff() / (self.returns[first_trade_i:].resample(freq).std() * np.sqrt(scale))
        yearly_sharpe = yearly_sharpe[np.isfinite(yearly_sharpe)]
        (yearly_sharpe).plot.bar(figsize=(10,5))
        plt.axhline(yearly_sharpe.mean(), color='r', linestyle='--', alpha=0.7)
        print("average {} sharpe: {}".format(freq, yearly_sharpe.mean()))
        plt.show()

    def returnCorrelation(self, freq=None):

        index_name = self.config["INDEX_NAME"]
        
        if freq is None:
            top_10_pctl = self.index_returns > np.nanquantile(self.index_returns.values, 0.90)
            bottom_10_pctl = self.index_returns < np.nanquantile(self.index_returns.values, 0.10)

            print("top10_corr: " + str(pd.DataFrame({index_name + "_top10": self.index_returns, "r": self.returns}).loc[self.ix_range][top_10_pctl].corr().iloc[0,1]))
            print("btm10_corr: " + str(pd.DataFrame({index_name + "_btm10": self.index_returns, "r": self.returns}).loc[self.ix_range][bottom_10_pctl].corr().iloc[0,1]))
            print("all_corr: " + str(pd.DataFrame({index_name: self.index_returns, "r": self.returns}).corr().iloc[0,1]))
            print("")
        else:
            self.index_returns_resampled = self.data_raw["Close"].set_index(self.data_raw.index)[index_name][self.ix_range].transform('log').resample(freq).first().diff() * 1e2
            self.returns_resampled = self.returns.cumsum().resample(freq).first().diff()

            top_10_pctl_resampled = self.index_returns_resampled > np.nanquantile(self.index_returns_resampled.values, 0.90)
            bottom_10_pctl_resampled = self.index_returns_resampled < np.nanquantile(self.index_returns_resampled.values, 0.10)

            print("top10_corr: " + str(pd.DataFrame({index_name + "_top10": self.index_returns_resampled, "r": self.returns_resampled}).loc[top_10_pctl_resampled[top_10_pctl_resampled].index].corr().iloc[0,1]))
            print("btm10_corr: " + str(pd.DataFrame({index_name + "_btm10": self.index_returns_resampled, "r": self.returns_resampled}).loc[bottom_10_pctl_resampled[bottom_10_pctl_resampled].index].corr().iloc[0,1]))
            print("all_corr: " + str(pd.DataFrame({index_name: self.index_returns_resampled, "r": self.returns_resampled}).corr().iloc[0,1]))
            print("")

    def plotReturnsHistogram(self, freq):
        self.returns.cumsum().resample(freq).first().diff().hist(bins=100)
        plt.show()

    def plotPricePaths(self):
        ## CALC PRICEPATHS
        opens =  self.model.features.opens
        closes =  self.model.features.closes
        highs =  self.model.features.highs
        lows =  self.model.features.lows
        vols = self.model.features.vol

        opens = opens.set_index(pd.to_datetime(opens.index))
        closes = closes.set_index(pd.to_datetime(closes.index))
        highs = highs.set_index(pd.to_datetime(highs.index))
        lows = lows.set_index(pd.to_datetime(lows.index))
        vols = vols.set_index(pd.to_datetime(vols.index)).shift(1)

        trades = self.trader.buy_trades.trades + self.trader.sell_trades.trades

        rets = [[] for _ in range(len(self.prob_bins))] 
        ups = [[] for _ in range(len(self.prob_bins))] 
        downs = [[] for _ in range(len(self.prob_bins))] 
        stopped = [[] for _ in range(len(self.prob_bins))] 
        early = [[] for _ in range(len(self.prob_bins))] 

        t_ix = self.config["DAYS_TIL_UPSIDE"] * 3

        data_len = len(closes)

        for t in trades:
            open_ix = opens.index.get_loc(pd.Timestamp(t.dates[0]))
            ord = t.ord
            prob_ord = np.digitize(t.prob, self.prob_bins) - 1
            open_rate = t.marks[0]
            vol_at_open = vols.iloc[open_ix,ord]

            h = highs if t.side > 0 else lows
            l = lows if t.side > 0 else highs

            start_ix = np.max([0, open_ix - t_ix])
            end_ix = np.min([data_len, open_ix + t_ix])

            ret_vec = t.side * ((closes.iloc[start_ix:open_ix+t_ix,ord] - open_rate) / open_rate).values / (vol_at_open * 1e-2) 
            up_vec = t.side * ((h.iloc[start_ix:end_ix,ord] - open_rate) / open_rate).values / (vol_at_open * 1e-2) 
            down_vec = t.side * ((l.iloc[start_ix:end_ix,ord] - open_rate) / open_rate).values / (vol_at_open * 1e-2) 

            if open_ix + t_ix > data_len:
                ret_vec = np.append(ret_vec, [ret_vec[-1]] * (t_ix*2 - len(ret_vec)))
                up_vec = np.append(up_vec, [up_vec[-1]] * (t_ix*2 - len(up_vec)))
                down_vec = np.append(down_vec, [down_vec[-1]] * (t_ix*2 - len(down_vec)))

            if open_ix - t_ix < 0 :
                ret_vec = np.insert(ret_vec, 0, [0.0] * (t_ix*2 - len(ret_vec)))
                up_vec = np.insert(up_vec, 0, [0.0] * (t_ix*2 - len(up_vec)))
                down_vec = np.insert(down_vec, 0, [0.0] * (t_ix*2 - len(down_vec)))

            rets[prob_ord].append(ret_vec)
            ups[prob_ord].append(up_vec)
            downs[prob_ord].append(down_vec)

            if t.levels[-1] == "EARLY": #EXPIRE
                close_ix = closes.index.get_loc(pd.Timestamp(t.dates[-1]))
                close_rate = t.marks[-1]
                start_ix = np.max([0, close_ix - t_ix])
                end_ix = np.min([data_len, close_ix + t_ix])

                early_vec = -1 * t.side * ((closes.iloc[start_ix:end_ix,ord] - close_rate) / close_rate).values / (vol_at_open * 1e-2) 

                if close_ix + t_ix > data_len:
                    early_vec = np.append(early_vec, [early_vec[-1]] * (t_ix*2 - len(early_vec)))

                if close_ix - t_ix < 0 :
                    early_vec = np.insert(early_vec, 0, [0.0] * (t_ix*2 - len(early_vec)))

                early[prob_ord].append(early_vec)


            if t.levels[-1] == "SL": #"TP"
                close_ix = closes.index.get_loc(pd.Timestamp(t.dates[0]))
                close_rate = t.marks[0]
                start_ix = np.max([0, close_ix - t_ix])
                end_ix = np.min([data_len, close_ix + t_ix])

                stopped_vec =  t.side * ((closes.iloc[start_ix:end_ix,ord] - close_rate) / close_rate).values / (vol_at_open * 1e-2) 
        
                if close_ix + t_ix > data_len:
                    stopped_vec = np.append(stopped_vec, [stopped_vec[-1]] * (t_ix*2 - len(stopped_vec)))

                if close_ix - t_ix < 0 :
                    stopped_vec = np.insert(stopped_vec, 0, [0.0] * (t_ix*2 - len(stopped_vec)))

                stopped[prob_ord].append(stopped_vec)


        ## AVERAGE TRADE PRICEPATHS
        us = self.config["UPSIDE"]
        ds = self.config["DOWNSIDE"]

        for p in range(len(self.prob_bins)):
            if len(rets[p]) > 0:
                r = np.array(rets[p])
                r = r[~np.isnan(r).any(axis=1)]

                plt.plot(np.arange(t_ix * 2) - t_ix, np.mean(r, axis=0), label="n={} | p={:.2f}".format(len(r), self.prob_bins[p]))
            
                # itm = r[(r.T[t_ix + 20] > 0).T]
                # otm = r[(r.T[t_ix + 20] < 0).T]
                # plt.plot(np.arange(t_ix * 2) - t_ix, np.mean(itm, axis=0), label="+ | {} | {:.2f}".format(len(itm), self.prob_bins[p]))
                # plt.plot(np.arange(t_ix * 2) - t_ix, np.mean(otm, axis=0), label="- | {} | {:.2f}".format(len(otm), self.prob_bins[p]))
            
                # plt.plot(np.arange(t_ix), np.array(ups[p]).mean(0), alpha=0.5, color='grey')
                # plt.plot(np.arange(t_ix), np.array(downs[p]).mean(0), alpha=0.5, color='grey')
        plt.legend()
        plt.axhline(us, alpha=0.5, linestyle='--', color='grey')
        plt.axhline(0, alpha=0.5, linestyle='--', color='red')
        plt.axhline(-ds, alpha=0.5, linestyle='--', color='grey')
        plt.show()

        ## EARLY CLOSE PRICEPATHS
        for p in range(len(self.prob_bins)):
            if len(early[p]) > 0:
                r = np.array(early[p])
                r = r[~np.isnan(r).any(axis=1)]
                plt.plot(np.arange(t_ix * 2) - t_ix, r.mean(0),label="{} | {:.2f}".format(len(r), self.prob_bins[p]))
                # plt.plot(np.arange(t_ix), np.array(ups[p]).mean(0), alpha=0.5, color='grey')
                # plt.plot(np.arange(t_ix), np.array(downs[p]).mean(0), alpha=0.5, color='grey')
        plt.legend()
        plt.axhline(us, alpha=0.5, linestyle='--', color='grey')
        plt.axhline(0, alpha=0.5, linestyle='--', color='red')
        plt.axhline(-ds, alpha=0.5, linestyle='--', color='grey')
        plt.show()

    
        ## STOP LOSS PRICEPATHS
        non_zero_prob_ords = np.array([len(i) for i in stopped]).nonzero()[0]

        _, axs = plt.subplots(nrows = len(non_zero_prob_ords), ncols=1, sharex=True)

        for p in range(len(non_zero_prob_ords)):
            if len(stopped[non_zero_prob_ords[p]]) > 0:
                for t in stopped[non_zero_prob_ords[p]]:
                    axs[p].plot(np.arange(t_ix * 2) - t_ix, np.array(t), label=self.prob_bins[non_zero_prob_ords[p]], alpha=0.2, color='blue')

                axs[p].axhline(ds, alpha=0.5, linestyle='--', color='grey')
                axs[p].axhline(0, alpha=0.5, linestyle='--', color='red')
                axs[p].axhline(-us, alpha=0.5, linestyle='--', color='grey')
        plt.show()


    def tradesSummary(self):
        ticker_names = self.config["TICKER_NAMES"]
        n_max_portfolio = self.config["N_MAX_PORTFOLIO"]

        trades = self.trader.buy_trades.trades + self.trader.sell_trades.trades

        self.trades_df = pd.DataFrame([{"ticker": ticker_names[t.ord], 
                                "side": t.side, 
                                "open": t.marks[0], 
                                "open_date": t.dates[0], 
                                "close": t.marks[-1], 
                                "ret":t.getReturn() / n_max_portfolio, 
                                "close_date": t.dates[-1], 
                                "age": (t.dates[-1] - t.dates[0]).days,
                                "prob": self.prob_bins[np.digitize(t.prob, self.prob_bins) - 1],
                                "type": t.levels[-1],
                                "vol": t.vol,
                                "adv": t.adv} for t in trades]) #.set_index("close_date")

        grouped_df = self.trades_df.groupby(["type", "prob", "side"])["ret"].agg({'sum', 'median', 'mean', 'count'})\
                                .join(self.trades_df.groupby(["type", "prob", "side"])["age"].agg({'mean'}), on=["type","prob","side"], rsuffix="_age")\
                                .assign(fees = lambda x: x["count"] * self.config["SLIPPAGE_PER_TRADE_BPS"] * 1e-4 * 1e2 / n_max_portfolio)

        print(grouped_df)
        #plt.plot(trades_df.index, trades_df["ret"].cumsum()) 
        #trades_df.groupby(["type"]).apply(lambda x: (x.close_date - x.open_date).mean())


        # (res.trades_df.join(pd.qcut(res.trades_df.vol, q=10).rename("vol_bucket")).groupby(["vol_bucket", "side", "prob"])["ret"].agg({'mean'}).unstack(level=1) * 100).style.background_gradient(cmap='RdBu', vmax=10, vmin=-5)
        # (res.trades_df.join(pd.qcut(1e-6 * res.trades_df.adv, q=10).rename("adv_bucket")).groupby(["adv_bucket", "side", "prob"])["ret"].agg({'mean'}).unstack(level=1) * 100).style.background_gradient(cmap='RdBu', vmax=10, vmin=-5)


    def plotFeaturesByTicker(self, ticker_name):
        ## PLOT FEATURES BY TICKER
        n_features = len(self.model.features.features)

        _, ax = plt.subplots(n_features, 1, sharex=True, figsize=(15, n_features * 4))

        for i in range(n_features):
            self.model.features.features[i][ticker_name].plot(ax=ax[i], label=self.feature_names[i])
            ax[i].axhline(0, linestyle='--', color='grey', alpha=0.5)
            ax[i].legend(loc='upper left')
        plt.show()


    def plotFeatures(self):
        n_features = len(self.model.features.features)
        n_splits = self.config["N_SPLITS"]

        ## CALC FEATURE COEFS
        features_11_ord_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
        features_13_ord_map = np.arange(13)
        features_21_ord_map = np.arange(21)

        feature_coefs = np.zeros((n_features, 2, n_splits)) * np.nan

        for j in range(2):
            for k in range(n_splits):
                # if type(self.model.models[j][k]) is RandomForestClassifier:
                #     coefs = self.model.models[j][k].feature_importances_[0] * self.model.Xyw[i][j][k][0].std(axis=0) 
                # else:
                #     coefs = self.model.models[i][j][k].coef_[0] * self.model.Xyw[i][j][k][0].std(axis=0) 

                coefs = self.model.models[j][k].coef_[0] * self.model.Xyw[0][j][k][0].std(axis=0) 

                n_coefs = len(coefs)
                ord_map = features_11_ord_map if n_coefs == 11 else \
                        features_13_ord_map if n_coefs == 13 else \
                        features_21_ord_map
                
                for c in range(n_coefs):
                    feature_coefs[ord_map[c]][j][k] = coefs[c]

        ## PLOT FEATURES COEFS
        _, ax = plt.subplots(n_features, 1, sharex=True, figsize=(15, n_features * 4))

        x_axis =  self.data_raw.index[np.arange(n_splits)]

        for f in range(n_features):
            ax[f].plot(x_axis, feature_coefs[f][0], label="sell | " + self.feature_names[f])
            ax[f].plot(x_axis, feature_coefs[f][1], label="buy | " + self.feature_names[f])
            ax[f].axhline(0, linestyle='--', color='grey', alpha=0.5)
            ax[f].legend()
        plt.show()

        ## FEATURE-RETURNS CORRELATION BY PERCENTILE
        quantile_list = np.arange(0, 1.01, 0.1)

        feature_return_corrs = dict()

        for f in range(n_features):
                ff = self.model.features.features[f]
                feature_return_corrs[self.feature_names[f]] = {q: ff.quantile(q, axis=1).corr(self.returns) for q in quantile_list}
        print(pd.DataFrame.from_dict(feature_return_corrs))

        ## FEATURE-RETURNS CORRELATION BY TICKER
        feature_return_corrs = dict()

        for f in range(n_features):
                ff = self.model.features.features[f].corrwith(self.returns)
                feature_return_corrs[self.feature_names[f]] = {q: np.nanquantile(ff, q) for q in quantile_list}

        print(pd.DataFrame.from_dict(feature_return_corrs))

