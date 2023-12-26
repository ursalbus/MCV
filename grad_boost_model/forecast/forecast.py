#%%

import forecasting.utils as pu
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

N_DAYS_RANGE = 20000
HORIZON = 60

TICKERS_TO_FIT = pu.getTickers()

DIRECTORY = 'forecast_60'

N = 3 * 52 # number of training weeks

def main(pull_fresh_data):

    data = pu.getData(N_DAYS_RANGE, pull_fresh_data)
    # data["Close"] = data["Adj Close]"]

    X = pu.getSignals(data)

    Y1 = pu.getFwdHighRtn(data, HORIZON, DIRECTORY) 
    Y2 = pu.getFwdLowRtn(data, HORIZON, DIRECTORY) 

    sotw, eotw = pu.getWeeklyIxs(X)

    for c in TICKERS_TO_FIT:

        y1 = pu.sortIndex(Y1, c)
        y2 = pu.sortIndex(Y2, c)

        pu.log(f'attempting to fit highs: {c}')
        res = Parallel(n_jobs=-1)(delayed(pu.fitPredict)(ixs_tuple, X, y1) for ixs_tuple in pu.generateIxs(sotw, eotw, N))
        pu.toPickle(res, DIRECTORY + f"/hl_fits/res_fwd_high_{c}.pkl")
        del(res, y1)

        pu.log(f'attempting to fit lows: {c}')
        res = Parallel(n_jobs=-1)(delayed(pu.fitPredict)(ixs_tuple, X, y2) for ixs_tuple in pu.generateIxs(sotw, eotw, N))
        pu.toPickle(res, DIRECTORY + f"/hl_fits/res_fwd_low_{c}.pkl")
        del(res, y2)

if __name__ == "__main__":
    main(True)