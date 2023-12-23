import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from sklearn.decomposition import PCA


def pca_factorize(ret_table, n_components=15):
    _ret_table = ret_table.fillna(0, fillna=0)
    _ret_mean = _ret_table.mean(axis=0)
    _ret_stddev = _ret_table.std(axis=0)
    _std_ret = (_ret_table - _ret_mean) / _ret_stddev
    pca_model = PCA(n_components=n_components).fit(_std_ret)
    _weights = pd.DataFrame(pca_model.components_, columns=_ret_table.columns) / _ret_stddev
    pca_factors_ret = pd.DataFrame(np.dot(_ret_table, _weights.T),index=_weights.index)
    return pca_factors_ret


if __name__ == '__main__':
    load_dotenv(".env")
    data_path = os.getenv("STOCK_DATA_PATH")
    print(data_path)