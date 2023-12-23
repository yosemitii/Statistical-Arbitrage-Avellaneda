import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def load_spy(path):
    na_values = ['NA', 'N/A', 'NaN', 'null', 'None']
    _dtypes = {'permno': pd.Int32Dtype(), 'permco': pd.Int32Dtype()}
    spy_raw_df = pd.read_csv(path, na_values=na_values, dtype=_dtypes)
    spy_raw_df.dropna(subset=['permno'], inplace=True)
    result = spy_raw_df.groupby('report_dt')['permno'].agg(set).reset_index()
    spy_dict = {}
    for _, row in result.iterrows():
        spy_dict[row['report_dt']] = row['permno']
    return spy_dict

def preprocess_v2(path, nrows=None):
    raw_df = pd.read_csv(path, nrows=nrows) if ".csv" in path else pd.read_parquet(path, nrows=nrows)
    _required_cols = ['PERMNO', 'Ticker', 'DlyCalDt', 'DlyPrc', 'DlyRet']
    df = raw_df[_required_cols]
    del raw_df
    return df


