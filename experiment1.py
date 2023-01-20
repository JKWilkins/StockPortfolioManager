import numpy as np
import pandas as pd
import StrategyLearner as sl
import ManualStrategy as ms
import marketsimcode
from util import get_data
import datetime as dt
from matplotlib import pyplot as plt


def author():
    return "jwilkins36"


def experiment1():

    symbol = 'JPM'
    sd = dt.date(2008,1,1)
    ed = dt.date(2009,12,31)
    sv = 100000
    impact = 0.005
    commission = 9.95

    m = ms.ManualStrategy()
    ### IN SAMPLE

    # bench
    bm_is_trades = m.benchmark(symbol, sd, ed)
    bm_is_portval, bm_is_cumrets, bm_is_avgrets, bm_is_stdrets = marketsimcode.compute_portvals(orders=bm_is_trades, commission=commission, impact=impact, sv=sv)

    # manual
    ms_is_trades = m.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    ms_is_portval, ms_is_cumrets, ms_is_avgrets, ms_is_stdrets = marketsimcode.compute_portvals(orders=ms_is_trades, sv=sv, commission=commission, impact=impact)

    # strategy
    strategy = sl.StrategyLearner(impact=impact,commission=commission)
    strategy.add_evidence(symbol, sd=sd, ed=ed, sv=sv)
    sl_is_trades = strategy.testPolicy(symbol, sd=sd, ed=ed, sv=sv)
    sl_is_portval, sl_is_cumrets, sl_is_avgrets, sl_is_stdrets = marketsimcode.compute_portvals(orders=sl_is_trades, sv=sv, commission=commission, impact=impact)

    # concat portvals
    is_portvals = pd.concat([bm_is_portval, ms_is_portval, sl_is_portval], axis = 1)
    is_portvals.columns = ['Benchmark', 'Manual Strategy', 'Strategy Learner']
    n_is_portvals = is_portvals / is_portvals.iloc[0]


    # plot portvals
    ax = n_is_portvals.plot(title='In-Sample Strategy Portfolio Values', fontsize=12,color=['purple','red','green'])
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Price')
    plt.savefig('.//images//Exp1_InSample_plot.png')
    plt.close()

    ### OUT OF SAMPLE #####

    symbol = 'JPM'
    os_sd = dt.date(2010, 1, 1)
    os_ed = dt.date(2011, 12, 31)
    sv = 100000

    # bench
    bm_os_trades = m.benchmark(symbol, os_sd, os_ed)
    bm_os_portval, bm_os_cumrets, bm_os_avgrets, bm_os_stdrets = marketsimcode.compute_portvals(orders=bm_os_trades,commission=commission,impact=impact, sv=sv)

    # manual
    ms_os_trades = m.testPolicy(symbol=symbol, sd=os_sd, ed=os_ed, sv=sv)
    ms_os_portval, ms_os_cumrets, ms_os_avgrets, ms_os_stdrets = marketsimcode.compute_portvals(orders=ms_os_trades,sv=sv,commission=commission,impact=impact)

    # strategy
    os_strategy = sl.StrategyLearner(impact=impact, commission=commission)
    os_strategy.add_evidence(symbol, sd=os_sd, ed=os_ed, sv=sv)
    sl_os_df_trades = os_strategy.testPolicy(symbol, sd=os_sd, ed=os_ed, sv=sv)
    sl_os_portval, sl_os_cumrets, sl_os_avgrets, sl_os_stdrets = marketsimcode.compute_portvals(orders=sl_os_df_trades,commission=commission,impact=impact,sv=sv)

    # concat portvals
    os_portvals = pd.concat([bm_os_portval, ms_os_portval, sl_os_portval], axis=1)
    os_portvals.columns = ['Benchmark', 'Manual Strategy', 'Strategy Learner']
    n_os_portvals = os_portvals / os_portvals.iloc[0]

    # plot portvals
    ax = n_os_portvals.plot(title='Out-Sample Strategy Portfolio Values', fontsize=12, color=['purple', 'red', 'green'])
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Price')
    plt.savefig('.//images//Exp1_OutSample_plot.png')
    plt.close()

    # chart of final portvals / stats
    file = open('.//images//Exp1_Stats.txt', 'w')
    file.write('In Sample: ' + str(sd) + ' - ' + str(ed) + ' for ' + symbol +
               ' with commission = ' + str(commission) + ' & impact = ' + str(impact) +
               '\n' +
               '\nBenchmark - In Sample' +
               '\nCumulative Return: ' + str(bm_is_cumrets) +
               '\nAverage Daily Return: ' + str(bm_is_avgrets) +
               '\nStandard Deviation: ' + str(bm_is_stdrets) +
               '\nFinal Portfolio Value: ' + str(bm_is_portval.iloc[-1]) +
               '\n' +
               '\nManual Strategy - In Sample' +
               '\nCumulative Return: ' + str(ms_is_cumrets) +
               '\nAverage Daily Return: ' + str(ms_is_avgrets) +
               '\nStandard Deviation: ' + str(ms_os_stdrets) +
               '\nFinal Portfolio Value: ' + str(ms_is_portval.iloc[-1]) +
               '\n' +
               '\nStrategy Learner - In Sample' +
               '\nCumulative Return: ' + str(sl_is_cumrets) +
               '\nAverage Daily Return: ' + str(sl_is_avgrets) +
               '\nStandard Deviation: ' + str(sl_is_stdrets) +
               '\nFinal Portfolio Value: ' + str(sl_is_portval.iloc[-1]) +
               '\n' +
               '\nOut Of Sample: ' + str(os_sd) + ' - ' + str(os_ed) + ' for ' + symbol +
               ' with commission = ' + str(commission) + ' & impact = ' + str(impact) +
               '\n' +
               '\nBenchmark - Out Sample' +
               '\nCumulative Return: ' + str(bm_os_cumrets) +
               '\nAverage Daily Return: ' + str(bm_os_avgrets) +
               '\nStandard Deviation: ' + str(bm_os_stdrets) +
               '\nFinal Portfolio Value: ' + str(bm_os_portval.iloc[-1]) +
               '\n' +
               '\nManual Strategy - Out Sample' +
               '\nCumulative Return: ' + str(ms_os_cumrets) +
               '\nAverage Daily Return: ' + str(ms_os_avgrets) +
               '\nStandard Deviation: ' + str(ms_os_stdrets) +
               '\nFinal Portfolio Value: ' + str(ms_os_portval.iloc[-1]) +
               '\n' +
               '\nStrategy Learner - Out Sample' +
               '\nCumulative Return: ' + str(sl_os_cumrets) +
               '\nAverage Daily Return: ' + str(sl_os_avgrets) +
               '\nStandard Deviation: ' + str(sl_os_stdrets) +
               '\nFinal Portfolio Value: ' + str(sl_os_portval.iloc[-1])
               )

    file.close()

if __name__ == '__main__':
    experiment1()
