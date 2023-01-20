import numpy as np
import pandas as pd
from util import get_data, symbol_to_path
import datetime as dt
import matplotlib.pyplot as plt


def get_sma(df, lookback, plot=False):
    sma = df.rolling(window=lookback).mean()

    if plot == True:
        figure, axis = plt.subplots()
        axis.set(xlabel='Date', ylabel="Price",
                 title=('Simple Moving Average (SMA) - ' + str(lookback) + ' / ' + str(lookback * 5) + ' Day lookback'))
        axis.plot(df, label="Price")
        axis.plot(sma, label="SMA 10")
        sma5 = df.rolling(window=lookback * 5).mean()
        axis.plot(sma5, label="SMA 50")
        axis.legend()
        axis.grid(True)
        figure = plt.gcf()
        figure.set_size_inches(10, 5, forward=True)
        figure.savefig('.//images//SMA.png', dpi=100)
        plt.clf()

    return sma

def get_bb(df, lookback, plot=False):
    rolling_std = df.rolling(window=lookback, min_periods=lookback).std()
    sma = get_sma(df, lookback)
    top_band = sma + 2 * rolling_std
    bottom_band = sma - 2 * rolling_std
    bb = pd.concat([bottom_band, top_band], axis=1)
    #bb.columns = ['bottom_band', 'top_band']

    bbp = (df - bottom_band) / (top_band - bottom_band)

    if plot == True:
        figure, axis = plt.subplots()
        axis.set(xlabel='Date', ylabel="Price", title=('Bollinger Bands (BB) - ' + str(lookback) + ' Day Lookback'))

        axis.plot(df, label="Price")
        axis.plot(sma, label="SMA", color='orange', linestyle='--')
        axis.plot(top_band, label="Upper Band", color='red')
        axis.plot(bottom_band, label="Bottom Band", color='red')
        axis.grid(True)
        axis.legend()
        figure = plt.gcf()
        figure.set_size_inches(10, 5, forward=True)
        figure.savefig('.//images//BollingerBands.png', dpi=100)
        plt.clf()

    return bbp

def get_momentum(df, lookback, plot=False):
    momentum = (df / df.shift(lookback)) - 1

    if plot == True:
        figure, axis = plt.subplots()
        axis.set(xlabel='Date', ylabel="Price (Normalized)", title=('Momentum - ' + str(lookback) + ' Day Lookback'))
        axis.plot(df, label="Price - 1")
        axis.plot(momentum, label="Momentum")
        axis.grid(True)
        axis.legend()
        figure = plt.gcf()
        figure.set_size_inches(10, 5, forward=True)
        figure.savefig('.//images//Momentum.png', dpi=100)
        plt.clf()

    return momentum

def get_ema(df, lookback, plot=False):
    ema = df.ewm(span=lookback, min_periods=lookback, adjust=False).mean()

    if plot == True:
        figure, axis = plt.subplots()
        axis.set(xlabel='Date', ylabel="Price", title=('Exponential Moving Average (EMA) - '
                                                       + str(lookback) + ' / ' + str(lookback*2.5) + ' / '
                                                       + str(lookback*5) + ' / ' + str(lookback*10) + ' Day Lookback'))
        axis.plot(df, label="Price")
        axis.plot(ema, label="EMA")
        ema2f = df.ewm(span=lookback*2.5, min_periods=lookback*2.5, adjust=False).mean()
        axis.plot(ema2f, label="EMA 25")
        ema5 = df.ewm(span=lookback * 5, min_periods=lookback * 5, adjust=False).mean()
        axis.plot(ema5, label="EMA 50")
        ema10 = df.ewm(span=lookback * 10, min_periods=lookback * 10, adjust=False).mean()
        axis.plot(ema10, label="EMA 100")
        axis.legend()
        axis.grid(True)
        figure = plt.gcf()
        figure.set_size_inches(10, 5, forward=True)
        figure.savefig('.//images//EMA.png', dpi=100)
    return ema

def get_ppo(df, plot=False):
    ppo = (get_ema(df, 12) - get_ema(df, 26)) / get_ema(df, 26)
    ppo[0:26] = np.nan
    signal = get_ema(ppo, 9)
    diff = ppo - signal

    if plot == True:
        figure, axis = plt.subplots()
        axis.set(xlabel='Date', ylabel="Percentage Price",
                 title=('Price Percentage Oscillator (PPO) - (12-26)/26 vs. 9 Day Lookback'))
        axis.plot(ppo, label="PPO")
        axis.plot(signal, label="PPO Signal")
        axis.plot(diff, label="PPO Difference", color='gray', linestyle='--')
        axis.legend()
        axis.grid(True)
        figure = plt.gcf()
        figure.set_size_inches(10, 5, forward=True)
        figure.savefig('.//images//PPO.png', dpi=100)
    return ppo

def test():

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sym = ['JPM']
    dates = pd.date_range(sd, ed)

    df = get_data(sym, dates)
    df = df[sym]

    sma = get_sma(df, 10, True)
    bb = get_bb(df, 20, True)
    momentum = get_momentum(df, 10, True)
    ema = get_ema(df, 10, True)
    ppo = get_ppo(df, True)

    print(sma)
    print(bb)
    print(momentum)
    print(ema)
    print(ppo)

def author():
    return 'jwilkins36'