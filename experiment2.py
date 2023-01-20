import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import StrategyLearner as sl
import marketsimcode


def author():
    return "jwilkins36"


def experiment2():
    symbol ='JPM'
    sd = dt.date(2008,1,1)
    ed = dt.date(2009,12,31)
    commission = 0.00
    sv = 100000

    impacts = [0, 0.001, 0.005, 0.01, 0.05]
    pv = []
    stats = []
    trades = []

    for impact in impacts:
        strategy = sl.StrategyLearner(impact=impact, commission=commission)
        strategy.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sl_df_trades = strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sl_is_portval, sl_is_cumrets, sl_is_avgrets, sl_is_stdrets = marketsimcode.compute_portvals(orders=sl_df_trades,
                                                                                                    sv=sv,
                                                                                                    commission=commission,
                                                                                                    impact=impact)
        pv.append(sl_is_portval)
        stats.append([sl_is_cumrets, sl_is_avgrets, sl_is_stdrets])
        trades.append(sl_df_trades[sl_df_trades != 0].count()[0])

    portvals = pd.concat(pv, axis=1)
    n_portvals = portvals / portvals.iloc[0]
    n_portvals.columns = ['0', '0.001', '0.005', '0.01', '0.05']

    ax = n_portvals.plot(title='In-sample Portfolio Values w/ Varying Impact', fontsize=12)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Price')
    plt.savefig('.//images//Exp2_Impacts_plot.png')
    plt.close()

    file = open('.//images//Exp2_Stats.txt', 'w')
    file.write('In Sample: ' + str(sd) + ' - ' + str(ed) + ' for ' + symbol +
               ' with commission = ' + str(commission) + ' & varying impacts'
               '\nStrategy Learner - Impact = ' + str(impacts[0]) +
               '\nCumulative Return: ' + str(stats[0][0]) +
               '\nAverage Daily Return: ' + str(stats[0][1]) +
               '\nStandard Deviation: ' + str(stats[0][2]) +
               '\nFinal Portfolio Value: ' + str(pv[0].iloc[-1]) +
               '\nTotal # of Trades: ' + str(trades[0]) +
               '\n' +
               '\nStrategy Learner - Impact = ' + str(impacts[1]) +
               '\nCumulative Return: ' + str(stats[1][0]) +
               '\nAverage Daily Return: ' + str(stats[1][1]) +
               '\nStandard Deviation: ' + str(stats[1][2]) +
               '\nFinal Portfolio Value: ' + str(pv[1].iloc[-1]) +
               '\nTotal # of Trades: ' + str(trades[1]) +
               '\n' +
               '\nStrategy Learner - Impact = ' + str(impacts[2]) +
               '\nCumulative Return: ' + str(stats[2][0]) +
               '\nAverage Daily Return: ' + str(stats[2][1]) +
               '\nStandard Deviation: ' + str(stats[2][2]) +
               '\nFinal Portfolio Value: ' + str(pv[2].iloc[-1]) +
               '\nTotal # of Trades: ' + str(trades[2]) +
               '\n' +
               '\nStrategy Learner - Impact = ' + str(impacts[3]) +
               '\nCumulative Return: ' + str(stats[3][0]) +
               '\nAverage Daily Return: ' + str(stats[3][1]) +
               '\nStandard Deviation: ' + str(stats[3][2]) +
               '\nFinal Portfolio Value: ' + str(pv[3].iloc[-1]) +
               '\nTotal # of Trades: ' + str(trades[3]) +
               '\n' +
               '\nStrategy Learner - Impact = ' + str(impacts[4]) +
               '\nCumulative Return: ' + str(stats[4][0]) +
               '\nAverage Daily Return: ' + str(stats[4][1]) +
               '\nStandard Deviation: ' + str(stats[4][2]) +
               '\nFinal Portfolio Value: ' + str(pv[4].iloc[-1]) +
               '\nTotal # of Trades: ' + str(trades[4])
               )
    file.close()


if __name__ == "__main__":
    experiment2()
