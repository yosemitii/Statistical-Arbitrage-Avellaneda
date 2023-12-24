import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from sklearn.decomposition import PCA


def pca_factorize(ret_table, n_components=15):
    _ret_table = ret_table.fillna(0)
    _ret_mean = _ret_table.mean(axis=0)
    _ret_stddev = _ret_table.std(axis=0)
    _std_ret = (_ret_table - _ret_mean) / _ret_stddev
    pca_model = PCA(n_components=n_components).fit(_std_ret)
    _weights = pd.DataFrame(pca_model.components_, columns=_ret_table.columns) / _ret_stddev
    pca_factors_ret = pd.DataFrame(_ret_table @ _weights.T,index=ret_table.index)
    return pca_factors_ret, _weights


def pca_sscore(residual, keppa=8.4):
    cum_res = residual.cumsum()
    m = pd.Series(index=cum_res.columns)
    sigma_eq = pd.Series(index=cum_res.columns)
    for i in cum_res.columns:
        b = cum_res[i].autocorr()
        if -np.log(b) * 252 > keppa:
            temp = (cum_res[i] - cum_res[i].shift(1) * b)[1:]
            a = temp.mean()
            cosi = temp - a
            m[i] = a / (1 - b)
            sigma_eq[i] = np.sqrt(cosi.var() / (1 - b * b))
    m.dropna(inplace=True)
    m = m - m.mean()
    sigma_eq.dropna(inplace=True)
    s_score = -m / sigma_eq
    return s_score