import pandas as pd
import numpy as np
import datetime as dt
from util import get_data


def compute_portvals(orders, sv=100000, commission=9.95, impact=0.005):
    orders.sort_index(ascending=True, inplace=True)
    symbol = np.array(['JPM'])
    start_date = orders.index.min()
    end_date = orders.index.max()
    dates = pd.date_range(start_date, end_date)

    prices = get_data(symbol, dates)
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    prices = prices[symbol]
    prices['Cash'] = pd.Series(1.0, index=prices.index)

    trades = get_data(symbol, dates)
    trades = trades[symbol]
    trades['Cash'] = pd.Series(1.0, index=trades.index)
    trades.iloc[:, :] = np.zeros((prices.shape[0], prices.shape[1]))
    trades.iloc[0, -1] = sv

    for dt_idx, row in orders.iterrows():
        shares = row['JPM']
        share_price = prices.loc[dt_idx, symbol]
        if shares < 0:
            trades.loc[dt_idx, symbol] = trades.loc[dt_idx, symbol] + shares
            share_price = share_price - (share_price * impact)
        else:
            trades.loc[dt_idx, symbol] = trades.loc[dt_idx, symbol] + shares
            share_price = share_price + (share_price * impact)

        cost = trades.loc[dt_idx, 'Cash'] - commission - (share_price * shares)
        trades['Cash'][dt_idx] = cost

    holdings = trades.copy()
    holdings = holdings.cumsum()
    vals = prices * holdings
    portvals = vals.sum(axis=1)

    # Stats
    dailyrets = portvals.copy()
    dailyrets[1:] = (portvals[1:] / portvals[:-1].values) - 1
    dailyrets.iloc[0] = 0
    dailyrets = dailyrets[1:]
    cumrets = (portvals[-1] / portvals[0]) - 1
    avgrets = dailyrets.mean()
    stdrets = dailyrets.std()

    return portvals, cumrets, avgrets, stdrets


def author():
    return "jwilkins36"

