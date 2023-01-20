import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt
from util import get_data
import marketsimcode
import indicators


def author():
    return 'jwilkins36'


class ManualStrategy(object):
    def __init__(self):
        self.lookback = 14

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv=100000):
        # example usage of the old backward compatible util function
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates)
        prices = prices_all[symbol]

        # indicator vals
        sma = indicators.get_sma(prices, self.lookback)
        smap = prices/sma
        bbp = indicators.get_bb(prices, self.lookback)
        momentum = indicators.get_momentum(prices, self.lookback)

        holdings = pd.DataFrame(np.nan, index=prices.index, columns=['holds'])

        # indicator trading
        for i in range(len(prices.index)):
            if smap.iloc[i] < 0.8 or bbp.iloc[i] < 0.0 or momentum.iloc[i] < -0.2:
                holdings.iloc[i] = 1000
            elif smap.iloc[i] > 1.2 or bbp.iloc[i] > 1.0 or momentum.iloc[i] > 0.2:
                holdings.iloc[i] = -1000

        holdings.ffill(inplace=True)
        holdings.fillna(0, inplace=True)
        trades = holdings.diff()

        if holdings.iloc[0].values == 0:
            trades.iloc[0] = 0
        else:
            trades.iloc[0] = holdings.iloc[0]

        df_trades = pd.DataFrame(data=trades.values, index=trades.index, columns=[symbol])

        return df_trades

    def benchmark(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices_df = get_data([symbol], dates)
        prices_df = pd.DataFrame(index=prices_df.index)
        prices_df[symbol] = 0
        prices_df.iloc[0] = 1000
        prices_df.iloc[-1] = -1000
        return prices_df


def msplots(): #plot all the charts required in the mannual strategy section
    sym = 'JPM'
    is_sd = dt.datetime(2008, 1, 1)
    is_ed = dt.datetime(2009, 12, 31)
    sv = 100000

    # IN SAMPLE ###########
    # manual
    m = ManualStrategy()
    is_trades_df = m.testPolicy(symbol=sym, sd=is_sd, ed=is_ed, sv=sv)

    long = is_trades_df[is_trades_df>0]
    long.dropna(inplace=True)
    short = is_trades_df[is_trades_df<0]
    short.dropna(inplace=True)

    is_ms_pv, is_ms_cumrets, is_ms_avgrets, is_ms_stdr = marketsimcode.compute_portvals(orders=is_trades_df, sv=sv, commission=9.95, impact=0.005)

    # benchmark
    bm_trades = m.benchmark(sym, is_sd, is_ed)
    is_bm_pv, is_bm_cumrets, is_bm_avgrets, is_bm_stdr = marketsimcode.compute_portvals(orders=bm_trades, commission=9.95, impact=0.005, sv=sv)

    # concat for plot
    is_portvals = pd.concat([is_ms_pv, is_bm_pv], axis=1)
    is_portvals.columns = ['Manual Strategy','Benchmark']

    # normalize
    n_is_portvals = is_portvals/is_portvals.iloc[0]

    # InSample Plot
    ax = n_is_portvals.plot(title='Manual Strategy in-sample Portfolio', fontsize=12, color=['red','purple'])
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Price')

    for i in short.index:
        plt.axvline(x=i, color='black')
    for i in long.index:
        plt.axvline(x=i, color='b')

    plt.savefig('.//images//Manual_InSample_plot.png')
    plt.close()

    # OUT SAMPLE ########
    os_sd=dt.datetime(2010, 1, 1)
    os_ed=dt.datetime(2011, 12, 31)

    # manual
    os_trades_df = m.testPolicy(symbol=sym, sd=os_sd, ed=os_ed, sv=sv)

    long = os_trades_df[os_trades_df>0]
    long.dropna(inplace=True)
    short = os_trades_df[os_trades_df<0]
    short.dropna(inplace=True)

    os_ms_pv, os_ms_cumrets, os_ms_avgrets, os_ms_stdr = marketsimcode.compute_portvals(orders=os_trades_df, commission=9.95, impact=0.005, sv=sv)

    # bench
    os_bm_trades = m.benchmark(sym, os_sd, os_ed)
    os_bm_pv, os_bm_cumrets, os_bm_avgrets, os_bm_stdr = marketsimcode.compute_portvals(orders=os_bm_trades, commission=9.95, impact=0.005, sv=sv)

    # concat
    os_portvals = pd.concat([os_ms_pv, os_bm_pv], axis=1)
    os_portvals.columns = ['Manual Strategy','Benchmark']
    n_os_portvals = os_portvals/os_portvals.iloc[0]

    # OutSample Plot
    ax = n_os_portvals.plot(title='Manual Strategy out-of-sample Portfolio', fontsize=12, color=['red', 'purple'])
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Price')

    for i in short.index:
        plt.axvline(x=i, color='black')
    for i in long.index:
        plt.axvline(x=i, color='b')

    plt.savefig('.//images//Manual_OutSample_plot.png')
    plt.close()

    # Stats
    file = open(".//images//Manual_Stats.txt", "w")
    file.write("In Sample: " + str(is_sd) + " - " + str(is_ed) + " for " + sym +
               "\n" +
               "\nManual Strategy - In Sample" +
               "\nCumulative Return: " + str(is_ms_cumrets) +
               "\nStandard Deviation: " + str(is_ms_stdr) +
               "\nAverage Daily Return: " + str(is_ms_avgrets) +
               "\n" +
               "\nBenchmark Strategy - In Sample" +
               "\nCumulative Return: " + str(is_bm_cumrets) +
               "\nStandard Deviation: " + str(is_bm_stdr) +
               "\nAverage Daily Return: " + str(is_bm_avgrets) +
               "\n" +
               "\nOut Of Sample: " + str(os_sd) + " - " + str(os_ed) + " for " + sym +
               "\n" +
               "\nManual Strategy - Out Of Sample" +
               "\nCumulative Return: " + str(os_ms_cumrets) +
               "\nStandard Deviation: " + str(os_ms_stdr) +
               "\nAverage Daily Return: " + str(os_ms_avgrets) +
               "\n" +
               "\nBenchmark Strategy - Out Of Sample" +
               "\nCumulative Return: " + str(os_bm_cumrets) +
               "\nStandard Deviation: " + str(os_bm_stdr) +
               "\nAverage Daily Return: " + str(os_bm_avgrets)
               )
    file.close()


if __name__ == "__main__":
    msplots()