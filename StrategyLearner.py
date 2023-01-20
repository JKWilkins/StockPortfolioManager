""""""  		  	   		  	  		  		  		    	 		 		   		 		  
"""  		  	   		  	  		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  	  		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  	  		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  	  		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  	  		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  	  		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  	  		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  	  		  		  		    	 		 		   		 		  
or edited.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  	  		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  	  		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  	  		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  	  		  		  		    	 		 		   		 		  
  		  	   		  	  		  		  		    	 		 		   		 		  
Student Name: Jameson Wilkins		  	   		  	  		  		  		    	 		 		   		 		  
GT User ID: jwilkins36		  	   		  	  		  		  		    	 		 		   		 		  
GT ID: 903564659		  	   		  	  		  		  		    	 		 		   		 		  
"""

import datetime as dt
import numpy as np
import pandas as pd
from util import get_data
import BagLearner as bl
import RTLearner as rt
import indicators
import marketsimcode



def author():
    return 'jwilkins36'


class StrategyLearner(object):


    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):

        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.lookback = 14
        self.learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 5}, bags=25, boost=False)

    # this method should create a QLearner, and train it for trading
    def add_evidence(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """
        prices = get_data([symbol], pd.date_range(sd, ed))

        if 'SPY' not in symbol:
            prices.drop('SPY', axis=1, inplace=True)

        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0, :]

        # indicators
        sma = indicators.get_sma(prices, self.lookback)
        smap = prices/sma
        bbp = indicators.get_bb(prices, self.lookback)
        momentum = indicators.get_momentum(prices, self.lookback)

        # train data
        train_x = pd.concat((smap, bbp, momentum), axis=1)
        train_x.columns = ['SMA', 'BBP', 'Momentum']
        train_x.fillna(0, inplace=True)
        train_x = train_x[:-14].values
        train_y = np.zeros(train_x.shape[0])

        for i in range(prices.shape[0] - self.lookback):
            future_price = prices.loc[prices.index[i + self.lookback], symbol]
            now_price = prices.loc[prices.index[i], symbol]
            impact = self.impact
            threshold = (future_price / now_price) - 1.0
            if threshold > 0.02 + impact:
                train_y[i] = 1
            elif threshold < -0.02 - impact:
                train_y[i] = -1
            else:
                train_y[i] = 0

        self.learner.add_evidence(train_x, train_y)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """
        prices = get_data([symbol], pd.date_range(sd, ed))

        if 'SPY' not in symbol:
            prices.drop('SPY', axis=1, inplace=True)
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices / prices.iloc[0, :]

        sma = indicators.get_sma(prices, self.lookback)
        smap = prices / sma
        bbp = indicators.get_bb(prices, self.lookback)
        momentum = indicators.get_momentum(prices, self.lookback)

        test_x = pd.concat((smap, bbp, momentum), axis=1)
        test_x.columns = ['SMAP', 'BBP', 'Momentum']
        test_x.fillna(0, inplace=True)
        test_x = test_x.values
        test_y = self.learner.query(test_x)

        df_trades = prices.copy()
        df_trades.loc[:] = 0

        f = 0
        dates = pd.Series(prices.index).array

        for i in range(len(dates) - 1):
            test_val = test_y[0][i]
            index = dates[i]

            if f == 1:
                if test_val < 0:
                    f = -1
                    df_trades.loc[index, :] = - 2000
            elif f == -1:
                if test_val > 0:
                    f = 1
                    df_trades.loc[index, :] = 2000
            elif f == 0:
                if test_val > 0:
                    f = 1
                    df_trades.loc[index, :] = 1000
                elif test_val < 0:
                    f = -1
                    df_trades.loc[index, :] = -1000
        return df_trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")
    import StrategyLearner as sl
    learner = sl.StrategyLearner(verbose=False, impact=0.0, commission=0.0) # constructor
    learner.add_evidence(symbol = "AAPL", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000) # testing phase

