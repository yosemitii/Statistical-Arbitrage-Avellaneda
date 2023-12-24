import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import logging
from bisect import bisect
from preprocessing import load_crsp_v2, load_spy_constituents
from pca_method import pca_factorize, pca_sscore
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',)

if __name__ == '__main__':
    # Config
    CORR_WINDOW = 252
    RESIDUAL_WINDOW = 60
    N_COMPONENTS = 15
    KEPPA = 8.4
    S_BUY_OPEN = 1.5
    S_SELL_OPEN = 1.5
    S_BUY_CLOSE = 0.5
    S_SELL_CLOSE = 0.5
    TRANSACTION_COST = 0.0005
    DELTA_T = 1/252
    INTERES_RATE = 0.0
    INITIAL_VALUE = 100
    # Database Related config
    DATE_COL = 'DlyCalDt'
    RET_COL = 'DlyRet'
    ID_COL = 'PERMNO'

    # Loading Files
    load_dotenv(".env")
    stock_path = os.getenv("STOCK_DATA_PATH")
    spy_path = os.getenv("SPY_HOLDINGS_PATH")
    spy_dict = load_spy_constituents(spy_path)

    all_spy_names = set()
    for s in spy_dict.values():
        all_spy_names = all_spy_names.union(s)

    df = load_crsp_v2(stock_path, id_set=all_spy_names, n_stocks=1000)
    logging.info("Loaded stock DataFrame file.")

    spy_change_dates = [s for s in spy_dict.keys()]
    trading_days = [d for d in df[DATE_COL].unique()]
    trading_days.sort()

    # position = pd.DataFrame(columns=df[ID_COL].dropna().unique(), index=df[DATE_COL]).sort_index()

    ret_table = df.pivot_table(index=DATE_COL, columns=ID_COL, values=RET_COL).sort_index()
    stock_universe = df[ID_COL].unique()
    position_matrix = pd.DataFrame(0.0, columns=ret_table.columns,
                                   index=np.array(['stock'] + list(range(N_COMPONENTS))),)
    prev_position = pd.Series(0.0, index=ret_table.columns)
    pnl = pd.Series(INITIAL_VALUE, index=trading_days[CORR_WINDOW-1:])
    details = pd.DataFrame(0.0, index=pnl.index,
                           columns=['lmv', 'long_pnl', 'long_ret',
                                    'smv', 'short_smv', 'short_ret',
                                    'market_neutral_ret', 'dollar_neutral_ret'])
    sscore_table = pd.DataFrame(index=ret_table.index[CORR_WINDOW:], columns=ret_table.columns)
    for dt_index, asof_date in enumerate(trading_days):
        ret_PIT = ret_table.iloc[dt_index].fillna(0.0)

        if dt_index-CORR_WINDOW < 0:
            continue
        # Using SPY 500
        spy_date_index = min(max(0, bisect(spy_change_dates, asof_date)) - 1, len(spy_change_dates) - 1)
        spy_constituents = spy_dict[spy_change_dates[spy_date_index]]

        # Factorize using PCA
        mkt_portfolio_start_date = trading_days[dt_index-CORR_WINDOW]
        mkt_portfolio_end_date = trading_days[dt_index]
        mkt_portfolio_data = df[(df[DATE_COL] >= mkt_portfolio_start_date)
                                & (df[DATE_COL] < mkt_portfolio_end_date)
                                & (df[ID_COL].isin(spy_constituents))
                                ]

        logging.info(f"Building market portfolio during ({mkt_portfolio_start_date}, {mkt_portfolio_end_date}]."
              f"Tracing back {len(mkt_portfolio_data[DATE_COL].unique())} days"
              f"for {len(mkt_portfolio_data[ID_COL].unique())} securities")
        mkt_ret = mkt_portfolio_data.pivot_table(index='DlyCalDt', columns='PERMNO', values='DlyRet')

        factor_ret, weights = pca_factorize(mkt_ret)

        # Calculate residual
        window_start = trading_days[dt_index - RESIDUAL_WINDOW]
        window_end = asof_date
        ret_records = df[(df[DATE_COL] >= window_start) & (df[DATE_COL] < window_end)]

        actively_traded = ret_records.groupby('PERMNO').agg({'Ticker': 'count'}).reset_index()
        actively_traded_names = set(actively_traded[actively_traded['Ticker'] >= 0.9 * RESIDUAL_WINDOW]['PERMNO'])
        _ret = ret_records[ret_records[ID_COL].isin(actively_traded_names)].pivot_table(
            index=DATE_COL, columns=ID_COL, values=RET_COL)

        residual = pd.DataFrame(columns=_ret.columns, index=_ret.index)
        beta = pd.DataFrame(columns=_ret.columns, index=range(15))
        window_factor_ret = factor_ret[(factor_ret.index >= window_start) &
                                       (factor_ret.index < window_end)]
        ols = LinearRegression()
        for sec_id in _ret.columns:
            ols.fit(window_factor_ret, _ret[sec_id].fillna(0))
            beta[sec_id] = ols.coef_
            residual[sec_id] = _ret[sec_id] - ols.intercept_ - np.dot(window_factor_ret, ols.coef_)

        s_score = pca_sscore(residual)
        for sec_id, score in s_score.items():
            if position_matrix[sec_id]['stock'] == 0:
                if score < -S_BUY_OPEN:
                    position_matrix[sec_id]['stock'] = 1
                    position_matrix[sec_id][1:] = -beta[sec_id]
                elif score > S_SELL_OPEN:
                    position_matrix[sec_id]['stock'] = -1
                    position_matrix[sec_id][1:] = beta[sec_id]
            elif position_matrix[sec_id][0] > 0:
                if score > -S_BUY_CLOSE:
                    position_matrix[sec_id] = 0
                # else:
                #     position_matrix[sec_id] = prev_position[sec_id]
            elif position_matrix[sec_id][0] < 0:
                if score < S_SELL_CLOSE:
                    position_matrix[sec_id] = 0
                # else:
                #     position_matrix[sec_id] = prev_position[sec_id]
        fac_sum = position_matrix.sum(axis=1)[1:]
        converted_stock_position = pd.Series(0.0, index=position_matrix.columns)
        for i in weights.columns:
            converted_stock_position[i] = np.inner(weights[i], fac_sum)
        stock_position = converted_stock_position + position_matrix.iloc[0]
        position_change = sum(abs(stock_position - prev_position))

        sscore_table.loc[asof_date] = s_score

        pnl_index = dt_index-CORR_WINDOW+1
        pnl.iloc[pnl_index] = (pnl.iloc[pnl_index-1] +
                               pnl.iloc[pnl_index-1] * INTERES_RATE * DELTA_T +
                               np.inner(stock_position, ret_PIT) -
                               stock_position.sum() * INTERES_RATE * DELTA_T -
                               position_change * TRANSACTION_COST)

        long_position = stock_position[stock_position > 0]
        lmv = sum(long_position)
        long_pnl = np.inner(long_position, ret_PIT[long_position.index])
        long_ret = long_pnl / lmv

        short_position = stock_position[stock_position < 0]
        smv = abs(sum(short_position))
        short_pnl = np.inner(short_position, ret_PIT[short_position.index])
        short_ret = short_pnl / smv
        details.iloc[pnl_index]['lmv'] = lmv
        details.iloc[pnl_index]['long_pnl'] = long_pnl
        details.iloc[pnl_index]['long_ret'] = long_ret
        details.iloc[pnl_index]['smv'] = smv
        details.iloc[pnl_index]['short_pnl'] = short_pnl
        details.iloc[pnl_index]['short_ret'] = short_ret
        details.iloc[pnl_index]['market_neutral_ret'] = (long_pnl+short_pnl)/(lmv+smv)
        details.iloc[pnl_index]['dollar_neutral_ret'] = (long_ret+short_ret)/2
        logging.info(f"{asof_date} pnl={pnl.iloc[pnl_index]} "
                     f"Finishsed {(100*pnl_index/(len(trading_days)-CORR_WINDOW-1)):.2f}%")
        prev_position = stock_position

        long_position = stock_position[stock_position > 0]

    sscore_table.to_csv('s_score.csv')
    details.to_csv('pnl_details.csv')
    pnl.to_csv('pnl.csv')
    # plt.plot(pnl)
    logging.info("Done")