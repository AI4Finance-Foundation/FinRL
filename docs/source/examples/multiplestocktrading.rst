:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

DRL Multiple Stocks Trading
===============================

Deep Reinforcement Learning for Stock Trading from Scratch: Multiple Stock Trading


.. tip::

    Run the code step by step at `Google Colab`_.

    .. _Google Colab: https://colab.research.google.com/github/AI4Finance-Foundation/FinRL/blob/master/FinRL_StockTrading_NeurIPS_2018.ipynb



Preparation before running this demonstration
---------------------------------------
As the first step, we install the newest version of FinRL library.

**FinRL installation**：

.. code-block::
    :linenos:

    ## install finrl library
    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git

Then we import the packages needed for this demonstration.

**Import packages**：

.. code-block:: python
    :linenos:

    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    import datetime

    %matplotlib inline
    from finrl.apps import config
    from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
    from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.neo_finrl.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.drl_agents.stablebaselines3.models import DRLAgent

    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    from pprint import pprint

    import sys
    sys.path.append("../FinRL-Library")

    import itertools

Finally, create folders for storage.

**Create folders**：

.. code-block:: python
    :linenos:

    import os
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

Then all the preparation work are done. We can start now!

Download Data
---------------------------------------
Before training our DRL agent, we need to get the historical data of DOW30 stocks first. Here we use the data from Yahoo! Finance.
Yahoo! Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free. yfinance is an open-source library that provides APIs to download data from Yahoo! Finance. We will use this package to download data here.

.. code-block:: python
    :linenos:

    # Dow 30 constituents in 2021/10
    df = YahooDownloader(start_date = config.START_DATE,
                         end_date = config.END_DATE,
                         ticker_list = config.DOW_30_TICKER).fetch_data()

Preprocess Data
---------------------------------------

Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.


**Check missing data**

.. code-block:: python
    :linenos:

    # check missing data
    dow_30.isnull().values.any()



**Add technical indicators**

In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.


.. code-block:: python
    :linenos:

    def add_technical_indicator(df):
            """
            calcualte technical indicators
            use stockstats package to add technical inidactors
            :param data: (df) pandas dataframe
            :return: (df) pandas dataframe
            """
            stock = Sdf.retype(df.copy())
            stock['close'] = stock['adjcp']
            unique_ticker = stock.tic.unique()

            macd = pd.DataFrame()
            rsi = pd.DataFrame()

            #temp = stock[stock.tic == unique_ticker[0]]['macd']
            for i in range(len(unique_ticker)):
                ## macd
                temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
                temp_macd = pd.DataFrame(temp_macd)
                macd = macd.append(temp_macd, ignore_index=True)
                ## rsi
                temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
                temp_rsi = pd.DataFrame(temp_rsi)
                rsi = rsi.append(temp_rsi, ignore_index=True)

            df['macd'] = macd
            df['rsi'] = rsi
            return df


**Add turbulence index**

Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level.

To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

.. code-block:: python
    :linenos:

    def add_turbulence(df):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        turbulence_index = calcualte_turbulence(df)
        df = df.merge(turbulence_index, on='datadate')
        df = df.sort_values(['datadate','tic']).reset_index(drop=True)
        return df



    def calcualte_turbulence(df):
        """calculate turbulence index based on dow 30"""
        # can add other market assets

        df_price_pivot=df.pivot(index='datadate', columns='tic', values='adjcp')
        unique_date = df.datadate.unique()
        # start after a year
        start = 252
        turbulence_index = [0]*start
        #turbulence_index = [0]
        count=0
        for i in range(start,len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index ]]
            cov_temp = hist_price.cov()
            current_temp=(current_price - np.mean(hist_price,axis=0))
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)
            if temp>0:
                count+=1
                if count>2:
                    turbulence_temp = temp[0][0]
                else:
                    #avoid large outlier because of the calculation just begins
                    turbulence_temp=0
            else:
                turbulence_temp=0
            turbulence_index.append(turbulence_temp)


        turbulence_index = pd.DataFrame({'datadate':df_price_pivot.index,
                                         'turbulence':turbulence_index})
        return turbulence_index

Design Environment
---------------------------------------


Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a Markov Decision Process (MDP) problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.

Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.

The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.


**Environment for Training**

.. code-block:: python
    :linenos:

    ## Environment for Training
    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # shares normalization factor
    # 100 shares per trade
    HMAX_NORMALIZE = 100
    # initial amount of money we have in our account
    INITIAL_ACCOUNT_BALANCE=1000000
    # total number of stocks in our portfolio
    STOCK_DIM = 30
    # transaction fee: 1/1000 reasonable percentage
    TRANSACTION_FEE_PERCENT = 0.001

    REWARD_SCALING = 1e-4


    class StockEnvTrain(gym.Env):
        """A stock trading environment for OpenAI gym"""
        metadata = {'render.modes': ['human']}

        def __init__(self, df,day = 0):
            #super(StockEnv, self).__init__()
            self.day = day
            self.df = df

            # action_space normalization and shape is STOCK_DIM
            self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
            # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
            # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
            self.observation_space = spaces.Box(low=0, high=np.inf, shape = (121,))
            # load data from a pandas dataframe
            self.data = self.df.loc[self.day,:]
            self.terminal = False
            # initalize state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()
                          #self.data.cci.values.tolist() + \
                          #self.data.adx.values.tolist()
            # initialize reward
            self.reward = 0
            self.cost = 0
            # memorize all the total balance change
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.rewards_memory = []
            self.trades = 0
            self._seed()

        def _sell_stock(self, index, action):
            # perform sell action based on the sign of the action
            if self.state[index+STOCK_DIM+1] > 0:
                #update balance
                self.state[0] += \
                self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 (1- TRANSACTION_FEE_PERCENT)

                self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                 TRANSACTION_FEE_PERCENT
                self.trades+=1
            else:
                pass

        def _buy_stock(self, index, action):
            # perform buy action based on the sign of the action
            available_amount = self.state[0] // self.state[index+1]
            # print('available_amount:{}'.format(available_amount))

            #update balance
            self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                              (1+ TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] += min(available_amount, action)

            self.cost+=self.state[index+1]*min(available_amount, action)* \
                              TRANSACTION_FEE_PERCENT
            self.trades+=1

        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                plt.plot(self.asset_memory,'r')
                plt.savefig('account_value_train.png')
                plt.close()
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                print("previous_total_asset:{}".format(self.asset_memory[0]))

                print("end_total_asset:{}".format(end_total_asset))
                df_total_value = pd.DataFrame(self.asset_memory)
                df_total_value.to_csv('account_value_train.csv')
                print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- INITIAL_ACCOUNT_BALANCE ))
                print("total_cost: ", self.cost)
                print("total_trades: ", self.trades)
                df_total_value.columns = ['account_value']
                df_total_value['daily_return']=df_total_value.pct_change(1)
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
                print("Sharpe: ",sharpe)
                print("=================================")
                df_rewards = pd.DataFrame(self.rewards_memory)
                df_rewards.to_csv('account_rewards_train.csv')

                return self.state, self.reward, self.terminal,{}

            else:
                actions = actions * HMAX_NORMALIZE

                begin_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))
                #print("begin_total_asset:{}".format(begin_total_asset))

                argsort_actions = np.argsort(actions)

                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._buy_stock(index, actions[index])

                self.day += 1
                self.data = self.df.loc[self.day,:]
                #load next state
                # print("stock_shares:{}".format(self.state[29:]))
                self.state =  [self.state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        list(self.state[(STOCK_DIM+1):61]) + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()

                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))

                #print("end_total_asset:{}".format(end_total_asset))

                self.reward = end_total_asset - begin_total_asset
                self.rewards_memory.append(self.reward)

                self.reward = self.reward * REWARD_SCALING
                # print("step_reward:{}".format(self.reward))

                self.asset_memory.append(end_total_asset)


            return self.state, self.reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []
            #initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()
            return self.state

        def render(self, mode='human'):
            return self.state

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]


**Environment for Trading**

.. code-block:: python
    :linenos:

    ## Environment for Trading
    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # shares normalization factor
    # 100 shares per trade
    HMAX_NORMALIZE = 100
    # initial amount of money we have in our account
    INITIAL_ACCOUNT_BALANCE=1000000
    # total number of stocks in our portfolio
    STOCK_DIM = 30
    # transaction fee: 1/1000 reasonable percentage
    TRANSACTION_FEE_PERCENT = 0.001

    # turbulence index: 90-150 reasonable threshold
    #TURBULENCE_THRESHOLD = 140
    REWARD_SCALING = 1e-4

    class StockEnvTrade(gym.Env):
        """A stock trading environment for OpenAI gym"""
        metadata = {'render.modes': ['human']}

        def __init__(self, df,day = 0,turbulence_threshold=140):
            #super(StockEnv, self).__init__()
            #money = 10 , scope = 1
            self.day = day
            self.df = df
            # action_space normalization and shape is STOCK_DIM
            self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,))
            # Shape = 181: [Current Balance]+[prices 1-30]+[owned shares 1-30]
            # +[macd 1-30]+ [rsi 1-30] + [cci 1-30] + [adx 1-30]
            self.observation_space = spaces.Box(low=0, high=np.inf, shape = (121,))
            # load data from a pandas dataframe
            self.data = self.df.loc[self.day,:]
            self.terminal = False
            self.turbulence_threshold = turbulence_threshold
            # initalize state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()

            # initialize reward
            self.reward = 0
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            # memorize all the total balance change
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[]
            self._seed()


        def _sell_stock(self, index, action):
            # perform sell action based on the sign of the action
            if self.turbulence<self.turbulence_threshold:
                if self.state[index+STOCK_DIM+1] > 0:
                    #update balance
                    self.state[0] += \
                    self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                     (1- TRANSACTION_FEE_PERCENT)

                    self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
                    self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * \
                     TRANSACTION_FEE_PERCENT
                    self.trades+=1
                else:
                    pass
            else:
                # if turbulence goes over threshold, just clear out all positions
                if self.state[index+STOCK_DIM+1] > 0:
                    #update balance
                    self.state[0] += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                                  (1- TRANSACTION_FEE_PERCENT)
                    self.state[index+STOCK_DIM+1] =0
                    self.cost += self.state[index+1]*self.state[index+STOCK_DIM+1]* \
                                  TRANSACTION_FEE_PERCENT
                    self.trades+=1
                else:
                    pass

        def _buy_stock(self, index, action):
            # perform buy action based on the sign of the action
            if self.turbulence< self.turbulence_threshold:
                available_amount = self.state[0] // self.state[index+1]
                # print('available_amount:{}'.format(available_amount))

                #update balance
                self.state[0] -= self.state[index+1]*min(available_amount, action)* \
                                  (1+ TRANSACTION_FEE_PERCENT)

                self.state[index+STOCK_DIM+1] += min(available_amount, action)

                self.cost+=self.state[index+1]*min(available_amount, action)* \
                                  TRANSACTION_FEE_PERCENT
                self.trades+=1
            else:
                # if turbulence goes over threshold, just stop buying
                pass

        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                plt.plot(self.asset_memory,'r')
                plt.savefig('account_value_trade.png')
                plt.close()

                df_date = pd.DataFrame(self.date_memory)
                df_date.columns = ['datadate']
                df_date.to_csv('df_date.csv')


                df_actions = pd.DataFrame(self.actions_memory)
                df_actions.columns = self.data.tic.values
                df_actions.index = df_date.datadate
                df_actions.to_csv('df_actions.csv')

                df_total_value = pd.DataFrame(self.asset_memory)
                df_total_value.to_csv('account_value_trade.csv')
                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                print("previous_total_asset:{}".format(self.asset_memory[0]))

                print("end_total_asset:{}".format(end_total_asset))
                print("total_reward:{}".format(self.state[0]+sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):61]))- self.asset_memory[0] ))
                print("total_cost: ", self.cost)
                print("total trades: ", self.trades)

                df_total_value.columns = ['account_value']
                df_total_value['daily_return']=df_total_value.pct_change(1)
                sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
                      df_total_value['daily_return'].std()
                print("Sharpe: ",sharpe)

                df_rewards = pd.DataFrame(self.rewards_memory)
                df_rewards.to_csv('account_rewards_trade.csv')

                # print('total asset: {}'.format(self.state[0]+ sum(np.array(self.state[1:29])*np.array(self.state[29:]))))
                #with open('obs.pkl', 'wb') as f:
                #    pickle.dump(self.state, f)

                return self.state, self.reward, self.terminal,{}

            else:
                # print(np.array(self.state[1:29]))
                self.date_memory.append(self.data.datadate.unique())

                #print(self.data)
                actions = actions * HMAX_NORMALIZE
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-HMAX_NORMALIZE]*STOCK_DIM)
                self.actions_memory.append(actions)

                #actions = (actions.astype(int))

                begin_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
                #print("begin_total_asset:{}".format(begin_total_asset))

                argsort_actions = np.argsort(actions)
                #print(argsort_actions)

                sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
                buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

                for index in sell_index:
                    # print('take sell action'.format(actions[index]))
                    self._sell_stock(index, actions[index])

                for index in buy_index:
                    # print('take buy action: {}'.format(actions[index]))
                    self._buy_stock(index, actions[index])

                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.turbulence = self.data['turbulence'].values[0]
                #print(self.turbulence)
                #load next state
                # print("stock_shares:{}".format(self.state[29:]))
                self.state =  [self.state[0]] + \
                        self.data.adjcp.values.tolist() + \
                        list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                        self.data.macd.values.tolist() + \
                        self.data.rsi.values.tolist()

                end_total_asset = self.state[0]+ \
                sum(np.array(self.state[1:(STOCK_DIM+1)])*np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))

                #print("end_total_asset:{}".format(end_total_asset))

                self.reward = end_total_asset - begin_total_asset
                self.rewards_memory.append(self.reward)

                self.reward = self.reward * REWARD_SCALING

                self.asset_memory.append(end_total_asset)

            return self.state, self.reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            #self.iteration=self.iteration
            self.rewards_memory = []
            self.actions_memory=[]
            self.date_memory=[]
            #initiate state
            self.state = [INITIAL_ACCOUNT_BALANCE] + \
                          self.data.adjcp.values.tolist() + \
                          [0]*STOCK_DIM + \
                          self.data.macd.values.tolist() + \
                          self.data.rsi.values.tolist()

            return self.state

        def render(self, mode='human',close=False):
            return self.state


        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]


Implement DRL Algorithms
-------------------------------------

The implementation of the DRL algorithms are based on OpenAI Baselines and Stable Baselines. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.


**Training data split**: 2009-01-01 to 2018-12-31

.. code-block:: python
    :linenos:

    def data_split(df,start,end):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df.datadate >= start) & (df.datadate < end)]
        data=data.sort_values(['datadate','tic'],ignore_index=True)
        data.index = data.datadate.factorize()[0]
        return data


**Model training**: DDPG

.. code-block:: python
    :linenos:

    ## tensorboard --logdir ./multiple_stock_tensorboard/
    # add noise to the action in DDPG helps in learning for better exploration
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    # model settings
    model_ddpg = DDPG('MlpPolicy',
                       env_train,
                       batch_size=64,
                       buffer_size=100000,
                       param_noise=param_noise,
                       action_noise=action_noise,
                       verbose=0,
                       tensorboard_log="./multiple_stock_tensorboard/")

    ## 250k timesteps: took about 20 mins to finish
    model_ddpg.learn(total_timesteps=250000, tb_log_name="DDPG_run_1")


**Trading**

Assume that we have $1,000,000 initial capital at 2019-01-01. We use the DDPG model to trade Dow jones 30 stocks.

**Set turbulence threshold**

Set the turbulence threshold to be the 99% quantile of insample turbulence data, if current turbulence index is greater than the threshold, then we assume that the current market is volatile

.. code-block:: python
    :linenos:

    insample_turbulence = dow_30[(dow_30.datadate<'2019-01-01') & (dow_30.datadate>='2009-01-01')]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])

**Prepare test data and environment**

.. code-block:: python
    :linenos:

    # test data
    test = data_split(dow_30, start='2019-01-01', end='2020-10-30')
    # testing env
    env_test = DummyVecEnv([lambda: StockEnvTrade(test, turbulence_threshold=insample_turbulence_threshold)])
    obs_test = env_test.reset()

**Prediction**

.. code-block:: python
    :linenos:

    def DRL_prediction(model, data, env, obs):
        print("==============Model Prediction===========")
        for i in range(len(data.index.unique())):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()


Backtest Our Strategy
---------------------------------

For simplicity purposes, in the article, we just calculate the Sharpe ratio and the annual return manually.

.. code-block:: python
    :linenos:

    def backtest_strat(df):
        strategy_ret= df.copy()
        strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
        strategy_ret.set_index('Date', drop = False, inplace = True)
        strategy_ret.index = strategy_ret.index.tz_localize('UTC')
        del strategy_ret['Date']
        ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
        return ts


**Dow Jones Industrial Average**

.. code-block:: python
    :linenos:

    def get_buy_and_hold_sharpe(test):
        test['daily_return']=test['adjcp'].pct_change(1)
        sharpe = (252**0.5)*test['daily_return'].mean()/ \
        test['daily_return'].std()
        annual_return = ((test['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)

        print("sharpe ratio: ", sharpe)
        #return sharpe


**Our DRL trading strategy**

.. code-block:: python
    :linenos:

    def get_daily_return(df):
        df['daily_return']=df.account_value.pct_change(1)
        #df=df.dropna()
        sharpe = (252**0.5)*df['daily_return'].mean()/ \
        df['daily_return'].std()

        annual_return = ((df['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)
        print("sharpe ratio: ", sharpe)
        return df

**Plot the results using Quantopian pyfolio**

Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

.. code-block:: python
    :linenos:

    %matplotlib inline
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                       benchmark_rets=dow_strat, set_context=False)
