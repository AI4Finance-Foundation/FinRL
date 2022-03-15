:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Portfolio Allocation
===================================

**Our paper**: 
`FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance`_. 

.. _FinRL\: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance: https://arxiv.org/abs/2011.09607

Presented at NeurIPS 2020: Deep RL Workshop.

The Jupyter notebook codes are available on our Github_ and `Google Colab`_.

.. _Github: https://github.com/AI4Finance-LLC/FinRL-Library
.. _Google Colab: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb

.. tip::

    - FinRL `Single Stock Trading`_ at Google Colab.
    
    .. _Single Stock Trading: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb
    
    - FinRL `Multiple Stocks Trading`_ at Google Colab: 

    .. _Multiple Stocks Trading: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb
    
Check our previous tutorials: `Single Stock Trading <https://finrl.readthedocs.io/en/latest/tutorial/SingleStockTrading.html>`_ and `Multiple Stock Trading <https://finrl.readthedocs.io/en/latest/tutorial/MultipleStockTrading.html>`_ for detailed explanation of the FinRL architecture and modules.



Overview
-------------

To begin with, we would like to explain the logic of portfolio allocation using Deep Reinforcement Learning.We use Dow 30 constituents as an example throughout this article, because those are the most popular stocks.

Let’s say that we got a million dollars at the beginning of 2019. We want to invest this $1,000,000 into stock markets, in this case is Dow Jones 30 constituents.Assume that no margin, no short sale, no treasury bill (use all the money to trade only these 30 stocks). So that the weight of each individual stock is non-negative, and the weights of all the stocks add up to one.

We hire a smart portfolio manager- Mr. Deep Reinforcement Learning. Mr. DRL will give us daily advice includes the portfolio weights or the proportions of money to invest in these 30 stocks. So every day we just need to rebalance the portfolio weights of the stocks.The basic logic is as follows.

.. image:: ../image/portfolio_allocation_1.png

Portfolio allocation is different from multiple stock trading because we are essentially rebalancing the weights at each time step, and we have to use all available money.

The traditional and the most popular way of doing portfolio allocation is mean-variance or modern portfolio theory (MPT):

.. image:: ../../image/portfolio_allocation_2.png


However, MPT performs not so well in out-of-sample data. MPT is calculated only based on stock returns, if we want to take other relevant factors into account, for example some of the technical indicators like MACD or RSI, MPT may not be able to combine these information together well.

We introduce a DRL library FinRL that facilitates beginners to expose themselves to quantitative finance. FinRL is a DRL library designed specifically for automated stock trading with an effort for educational and demonstrative purpose.

This article is focusing on one of the use cases in our paper: Portfolio Allocation. We use one Jupyter notebook to include all the necessary steps.




Problem Definition
--------------------------

This problem is to design an automated trading solution for portfolio allocation. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.

The components of the reinforcement learning environment are:

    - **Action**: portfolio weight of each stock is within [0,1]. We use softmax function to normalize the actions to sum to 1.
    
    - **State: {Covariance Matrix, MACD, RSI, CCI, ADX}, **state space** shape is (34, 30). 34 is the number of rows, 30 is the number of columns.
    
    - **Reward function**: r(s, a, s′) = p_t, p_t is the cumulative portfolio value.
    
    - **Environment**: portfolio allocation for Dow 30 constituents.


Covariance matrix is a good feature because portfolio managers use it to quantify the risk (standard deviation) associated with a particular portfolio.

We also assume no transaction cost, because we are trying to make a simple portfolio allocation case as a starting point.



Load Python Packages
--------------------------

Install the unstable development version of FinRL:

.. code-block:: python
   :linenos:

    # Install the unstable development version in Jupyter notebook:
    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
    
    
Import Packages:

.. code-block:: python
   :linenos:

    # import packages
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    import datetime
    
    from finrl import config
    from finrl import config_tickers
    from finrl.marketdata.yahoodownloader import YahooDownloader
    from finrl.preprocessing.preprocessors import FeatureEngineer
    from finrl.preprocessing.data import data_split
    from finrl.env.environment import EnvSetup
    from finrl.env.EnvMultipleStock_train import StockEnvTrain
    from finrl.env.EnvMultipleStock_trade import StockEnvTrade
    from finrl.model.models import DRLAgent
    from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot, backtest_strat, baseline_strat
    from finrl.trade.backtest import backtest_strat, baseline_strat
    
    import os
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)
    
    

Download Data
--------------------------

FinRL uses a YahooDownloader class to extract data.

.. code-block:: python
   
    class YahooDownloader:
        """
        Provides methods for retrieving daily stock data from Yahoo Finance API
        
        Attributes
        ----------
            start_date : str
                start date of the data (modified from config.py)
            end_date : str
                end date of the data (modified from config.py)
            ticker_list : list
                a list of stock tickers (modified from config.py)
                
        Methods
        -------
            fetch_data()
                Fetches data from yahoo API
        """

Download and save the data in a pandas DataFrame:

.. code-block:: python
   :linenos:

    # Download and save the data in a pandas DataFrame:
    df = YahooDownloader(start_date = '2008-01-01',
                         end_date = '2020-12-01',
                         ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
    

Preprocess Data
--------------------------

FinRL uses a FeatureEngineer class to preprocess data.

.. code-block:: python

    class FeatureEngineer:
        """
        Provides methods for preprocessing the stock price data
        
        Attributes
        ----------
            df: DataFrame
                data downloaded from Yahoo API
            feature_number : int
                number of features we used
            use_technical_indicator : boolean
                we technical indicator or not
            use_turbulence : boolean
                use turbulence index or not
                
        Methods
        -------
            preprocess_data()
                main method to do the feature engineering
        """

Perform Feature Engineering: covariance matrix + technical indicators:

.. code-block:: python
   :linenos:

    # Perform Feature Engineering:
    df = FeatureEngineer(df.copy(),
                        use_technical_indicator=True,
                        use_turbulence=False).preprocess_data()
    
    
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]
    
    cov_list = []
    # look back is one year
    lookback=252
    for i in range(lookback,len(df.index.unique())):
      data_lookback = df.loc[i-lookback:i,:]
      price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
      return_lookback = price_lookback.pct_change().dropna()
      covs = return_lookback.cov().values 
      cov_list.append(covs)
      
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    df.head()    

.. image:: ../../image/portfolio_allocation_3.png

Build Environment
--------------------------

FinRL uses a EnvSetup class to setup environment.


.. code-block:: python

    class EnvSetup:
        """
        Provides methods for retrieving daily stock data from
        Yahoo Finance API
        
        Attributes
            ----------
            stock_dim: int
                number of unique stocks
            hmax : int
                maximum number of shares to trade
            initial_amount: int
                start money
            transaction_cost_pct : float
                transaction cost percentage per trade
            reward_scaling: float
                scaling factor for reward, good for training
            tech_indicator_list: list
                a list of technical indicator names (modified from config.py)
        Methods
            -------
            create_env_training()
                create env class for training
            create_env_validation()
                create env class for validation
            create_env_trading()
                create env class for trading
        """


Initialize an environment class:

User-defined Environment: a simulation environment class.The environment for portfolio allocation:

.. code-block:: python
   :linenos:

    import numpy as np
    import pandas as pd
    from gym.utils import seeding
    import gym
    from gym import spaces
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    class StockPortfolioEnv(gym.Env):
        """A single stock trading environment for OpenAI gym
        Attributes
        ----------
            df: DataFrame
                input data
            stock_dim : int
                number of unique stocks
            hmax : int
                maximum number of shares to trade
            initial_amount : int
                start money
            transaction_cost_pct: float
                transaction cost percentage per trade
            reward_scaling: float
                scaling factor for reward, good for training
            state_space: int
                the dimension of input features
            action_space: int
                equals stock dimension
            tech_indicator_list: list
                a list of technical indicator names
            turbulence_threshold: int
                a threshold to control risk aversion
            day: int
                an increment number to control date
        Methods
        -------
        _sell_stock()
            perform sell action based on the sign of the action
        _buy_stock()
            perform buy action based on the sign of the action
        step()
            at each step the agent will return actions, then 
            we will calculate the reward, and return the next observation.
        reset()
            reset the environment
        render()
            use render to return other functions
        save_asset_memory()
            return account value at each time step
        save_action_memory()
            return actions/positions at each time step
            
        """
        metadata = {'render.modes': ['human']}
    
        def __init__(self, 
                    df,
                    stock_dim,
                    hmax,
                    initial_amount,
                    transaction_cost_pct,
                    reward_scaling,
                    state_space,
                    action_space,
                    tech_indicator_list,
                    turbulence_threshold,
                    lookback=252,
                    day = 0):
            #super(StockEnv, self).__init__()
            #money = 10 , scope = 1
            self.day = day
            self.lookback=lookback
            self.df = df
            self.stock_dim = stock_dim
            self.hmax = hmax
            self.initial_amount = initial_amount
            self.transaction_cost_pct =transaction_cost_pct
            self.reward_scaling = reward_scaling
            self.state_space = state_space
            self.action_space = action_space
            self.tech_indicator_list = tech_indicator_list
    
            # action_space normalization and shape is self.stock_dim
            self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
            # Shape = (34, 30)
            # covariance matrix + technical indicators
            self.observation_space = spaces.Box(low=0, 
                                                high=np.inf, 
                                                shape = (self.state_space+len(self.tech_indicator_list),
                                                         self.state_space))
    
            # load data from a pandas dataframe
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs),
                          [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.terminal = False     
            self.turbulence_threshold = turbulence_threshold        
            # initalize state: inital portfolio return + individual stock return + individual weights
            self.portfolio_value = self.initial_amount
    
            # memorize portfolio value each step
            self.asset_memory = [self.initial_amount]
            # memorize portfolio return each step
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]]
    
            
        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)
    
            if self.terminal:
                df = pd.DataFrame(self.portfolio_return_memory)
                df.columns = ['daily_return']
                plt.plot(df.daily_return.cumsum(),'r')
                plt.savefig('results/cumulative_reward.png')
                plt.close()
                
                plt.plot(self.portfolio_return_memory,'r')
                plt.savefig('results/rewards.png')
                plt.close()
    
                print("=================================")
                print("begin_total_asset:{}".format(self.asset_memory[0]))           
                print("end_total_asset:{}".format(self.portfolio_value))
    
                df_daily_return = pd.DataFrame(self.portfolio_return_memory)
                df_daily_return.columns = ['daily_return']
                if df_daily_return['daily_return'].std() !=0:
                  sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                           df_daily_return['daily_return'].std()
                  print("Sharpe: ",sharpe)
                print("=================================")
                
                return self.state, self.reward, self.terminal,{}
    
            else:
                #print(actions)
                # actions are the portfolio weight
                # normalize to sum of 1
                norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
                weights = norm_actions 
                #print(weights)
                self.actions_memory.append(weights)
                last_day_memory = self.data
    
                #load next state
                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.covs = self.data['cov_list'].values[0]
                self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
                # calcualte portfolio return
                # individual stocks' return * weight
                portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
                # update portfolio value
                new_portfolio_value = self.portfolio_value*(1+portfolio_return)
                self.portfolio_value = new_portfolio_value
    
                # save into memory
                self.portfolio_return_memory.append(portfolio_return)
                self.date_memory.append(self.data.date.unique()[0])            
                self.asset_memory.append(new_portfolio_value)
    
                # the reward is the new portfolio value or end portfolo value
                self.reward = new_portfolio_value 
                #self.reward = self.reward*self.reward_scaling
    
    
            return self.state, self.reward, self.terminal, {}
    
        def reset(self):
            self.asset_memory = [self.initial_amount]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            # load states
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.portfolio_value = self.initial_amount
            #self.cost = 0
            #self.trades = 0
            self.terminal = False 
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]] 
            return self.state
        
        def render(self, mode='human'):
            return self.state
        
        def save_asset_memory(self):
            date_list = self.date_memory
            portfolio_return = self.portfolio_return_memory
            #print(len(date_list))
            #print(len(asset_list))
            df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
            return df_account_value
    
        def save_action_memory(self):
            # date and close price length must match actions length
            date_list = self.date_memory
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            return df_actions
    
        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]
                                             

Implement DRL Algorithms
--------------------------


FinRL uses a DRLAgent class to implement the algorithms.

.. code-block:: python

    class DRLAgent:
        """
        Provides implementations for DRL algorithms
        
        Attributes
        ----------
            env: gym environment class
                 user-defined class
        Methods
        -------
            train_PPO()
                the implementation for PPO algorithm
            train_A2C()
                the implementation for A2C algorithm
            train_DDPG()
                the implementation for DDPG algorithm
            train_TD3()
                the implementation for TD3 algorithm 
            DRL_prediction() 
                make a prediction in a test dataset and get results
        """

**Model Training**:

We use A2C for portfolio allocation, because it is stable, cost-effective, faster and works better with large batch sizes.

Trading:Assume that we have $1,000,000 initial capital at 2019/01/01. We use the A2C model to perform portfolio allocation of the Dow 30 stocks.


.. code-block:: python
   :linenos:

    trade = data_split(df,'2019-01-01', '2020-12-01')
    
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                             env_class = StockPortfolioEnv) 
    
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_a2c,
                            test_data = trade,
                            test_env = env_trade,
                            test_obs = obs_trade)
    

.. image:: ../../image/portfolio_allocation_4.png


The output actions or the portfolio weights look like this:

.. image:: ../../image/portfolio_allocation_5.png


Backtesting Performance
--------------------------

FinRL uses a set of functions to do the backtesting with Quantopian pyfolio.

.. code-block:: python
   :linenos:

    from pyfolio import timeseries
    DRL_strat = backtest_strat(df_daily_return)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=DRL_strat, 
                                  factor_returns=DRL_strat, 
                                    positions=None, transactions=None, turnover_denom="AGB")
    print("==============DRL Strategy Stats===========")
    perf_stats_all
    print("==============Get Index Stats===========")
    baesline_perf_stats=BaselineStats('^DJI',
                                      baseline_start = '2019-01-01',
                                      baseline_end = '2020-12-01')
                                      
                                      
    # plot                                
    dji, dow_strat = baseline_strat('^DJI','2019-01-01','2020-12-01')
    import pyfolio
    %matplotlib inline
    with pyfolio.plotting.plotting_context(font_scale=1.1):
            pyfolio.create_full_tear_sheet(returns = DRL_strat,
                                           benchmark_rets=dow_strat, set_context=False)
                                           
The left table is the stats for backtesting performance, the right table is the stats for Index (DJIA) performance.


                 
**Plots**:







