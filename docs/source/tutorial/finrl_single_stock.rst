:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Tutorial for Single Stock Trading
===================================

**Our paper**: 
`FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance`_. 

.. _FinRL\: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance: https://arxiv.org/abs/2011.09607

Presented at NeurIPS 2020: Deep RL Workshop.

The Jupyter notebook codes are available on our Github_ and `Google Colab`_.

.. _Github: https://github.com/AI4Finance-LLC/FinRL-Library
.. _Google Colab: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb

.. tip::

    - FinRL `Single Stock Trading`_ at Google Colab.
    
    .. _Single Stock Trading: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_single_stock_trading.ipynb
    
    - FinRL `Multiple Stocks Trading`_ at Google Colab: 

    .. _Multiple Stocks Trading: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/FinRL_multiple_stock_trading.ipynb

Overview
-------------

As deep reinforcement learning (DRL) has been recognized as an effective approach in quantitative finance, getting hands-on experiences is attractive to beginners. However, to train a practical DRL trading agent that decides where to trade, at what price, and what quantity involves error-prone and arduous development and debugging.

We introduce a DRL library FinRL that facilitates beginners to expose themselves to quantitative finance and to develop their own stock trading strategies. Along with easily-reproducible tutorials, FinRL library allows users to streamline their own developments and to compare with existing schemes easily.

FinRL is a beginner-friendly library with fine-tuned standard DRL algorithms. It has been developed under three primary principles:

    - Completeness: Our library shall cover components of the DRL framework completely, which is a fundamental requirement;
    
    - Hands-on tutorials: We aim for a library that is friendly to beginners. Tutorials with detailed walk-through will help users to explore the functionalities of our library;
    
    - Reproducibility: Our library shall guarantee reproducibility to ensure the transparency and also provide users with confidence in what they have done

This article is focusing on one of the use cases in our paper: Single Stock Trading. We use one Jupyter notebook to include all the necessary steps.

We use Apple Inc. stock: AAPL as an example throughout this article, because it is one of the most popular stocks.

.. image:: ../image/FinRL-Architecture.png


Problem Definition
--------------------------

This problem is to design an automated trading solution for single stock trading. We model the stock trading process as a Markov Decision Process (MDP). We then formulate our trading goal as a maximization problem.

The components of the reinforcement learning environment are:

    - Action: The action space describes the allowed actions that the agent interacts with the environment. Normally, a ∈ A includes three actions: a ∈ {−1, 0, 1}, where −1, 0, 1 represent selling, holding, and buying one stock. Also, an action can be carried upon multiple shares. We use an action space {−k, …, −1, 0, 1, …, k}, where k denotes the number of shares. For example, “Buy 10 shares of AAPL” or “Sell 10 shares of AAPL” are 10 or −10, respectively
    
    - Reward function: r(s, a, s′) is the incentive mechanism for an agent to learn a better action. The change of the portfolio value when action a is taken at state s and arriving at new state s’, i.e., r(s, a, s′) = v′ − v, where v′ and v represent the portfolio values at state s′ and s, respectively
    
    - State: The state space describes the observations that the agent receives from the environment. Just as a human trader needs to analyze various information before executing a trade, so our trading agent observes many different features to better learn in an interactive environment.
    
    - Environment: single stock trading for AAPL


The data of the single stock that we will be using for this case study is obtained from Yahoo Finance API. The data contains Open-High-Low-Close price and volume.



Load Python Packages
--------------------------

Install the unstable development version of FinRL:

.. code-block:: python
   :linenos:

    #Install the unstable development version in Jupyter notebook
    !pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git


Install individual packages in Jupyter notebook if missing:

.. code-block:: python
   :linenos:

    import pkg_resources
    import pip
    installedPackages = {pkg.key for pkg in pkg_resources.working_set}
    required = {'yfinance', 'pandas', 'matplotlib', 'stockstats','stable-baselines','gym','tensorflow'}
    missing = required - installedPackages
    if missing:
        !pip install yfinance
        !pip install pandas
        !pip install matplotlib
        !pip install stockstats
        !pip install gym
        !pip install stable-baselines[mpi]
        !pip install tensorflow==1.15.4
    
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
    from finrl.config import config
    from finrl.marketdata.yahoodownloader import YahooDownloader
    from finrl.preprocessing.preprocessors import FeatureEngineer
    from finrl.preprocessing.data import data_split
    from finrl.env.environment import EnvSetup
    from finrl.env.EnvMultipleStock_train import StockEnvTrain
    from finrl.env.EnvMultipleStock_trade import StockEnvTrade
    from finrl.model.models import DRLAgent
    from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot
    
    

Download Data
--------------------------

`Yahoo Finance`_ is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free. 

`This Medium blog`_ explains how to use Yahoo Finance API to extract data directly in Python.

.. _Yahoo Finance: https://finance.yahoo.com/
.. _This Medium blog: https://towardsdatascience.com/free-stock-data-for-python-using-yahoo-finance-api-9dafd96cad2e

    - FinRL uses a class YahooDownloader to fetch data from Yahoo Finance API
    
    - Call Limit: Using the Public API (without authentication), you are limited to 2,000 requests per hour per IP (or up to a total of 48,000 requests a day).
    
We can either download the stock data like open-high-low-close price manually by entering a stock ticker symbol like AAPL into the website search bar, or we just use Yahoo Finance API to extract data automatically.


FinRL uses a YahooDownloader_ class to extract data.

.. _YahooDownloader: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/marketdata/yahoodownloader.py

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
    df = YahooDownloader(start_date = '2009-01-01', 
                              end_date = '2020-09-30', 
                              ticker_list = config.DOW_30_TICKER).fetch_data()
                              
    print(df.sort_values(['date','tic'],ignore_index=True).head(30))
    

.. image:: ../image/single_1.png



Preprocess Data
--------------------------

Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.

    - FinRL uses a FeatureEngineer class to preprocess the data
    
    - Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc.

**Calculate technical indicators**:

In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc.

    - FinRL uses stockstats to calcualte technical indicators such as Moving Average Convergence Divergence (MACD), Relative Strength Index (RSI), Average Directional Index (ADX), Commodity Channel Index (CCI) and other various indicators and stats.
   
    - stockstats: supplies a wrapper StockDataFrame based on the pandas.DataFrame with inline stock statistics/indicators support.
   
    - we store the stockstats technical indicator column names in config.py
   
    - config.TECHNICAL_INDICATORS_LIST = [‘macd’, ‘rsi_30’, ‘cci_30’, ‘dx_30’]
    
    - User can add more technical indicators, check https://github.com/jealous/stockstats for different names
    
FinRL uses a FeatureEngineer_ class to preprocess data.

.. _FeatureEngineer: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/preprocessing/preprocessors.py

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

Perform Feature Engineering:

.. code-block:: python
   :linenos:

    # Perform Feature Engineering:
    df = FeatureEngineer(df.copy(),
                         use_technical_indicator=True,
                         tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                         use_turbulence=True,
                         user_defined_feature = False).preprocess_data()



Build Environment
--------------------------

Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a Markov Decision Process (MDP) problem. The training process involves observing stock price change, taking an action and reward’s calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.

Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.

Environment design is one of the most important part in DRL, because it varies a lot from applications to applications and from markets to markets. We can’t use an environment for stock trading to trade bitcoin, and vice versa.

The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, “Buy 10 shares of AAPL” or “Sell 10 shares of AAPL” are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

In this article, I set k=200, the entire action space is 200*2+1 = 401 for AAPL.

FinRL uses a EnvSetup_ class to setup environment.

.. _EnvSetup: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/env/environment.py

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
            fetch_data()
                Fetches data from yahoo API
        """


Initialize an environment class:

.. code-block:: python
   :linenos:

    # Initialize env:
    env_setup = EnvSetup(stock_dim = stock_dimension,
                         state_space = state_space,
                         hmax = 100,
                         initial_amount = 1000000,
                         transaction_cost_pct = 0.001,
                         tech_indicator_list = config.TECHNICAL_INDICATORS_LIST)
                         
    env_train = env_setup.create_env_training(data = train, 
                                             env_class = StockEnvTrain)
                                             
                                             

User-defined Environment: a simulation environment class.

FinRL provides blueprint for `single stock trading environment`_.

.. _single stock trading environment: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/env/EnvSingleStock.py

.. code-block:: python

    class SingleStockEnv(gym.Env):
        """
        A single stock trading environment for OpenAI gym
        
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
                we will calculate the reward, and return the next    
                observation.
            reset()
                reset the environment
            render()
                use render to return other functions
            save_asset_memory()
                return account value at each time step
            save_action_memory()
                return actions/positions at each time step
        """
    
Tutorial for how to design a customized trading environment will be pulished in the future soon.


Implement DRL Algorithms
--------------------------

The implementation of the DRL algorithms are based on `OpenAI Baselines`_ and Stable Baselines. `Stable Baselines`_ is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.

.. _OpenAI Baselines: https://github.com/openai/baselines
.. _Stable Baselines: https://github.com/hill-a/stable-baselines

.. tip::
    FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to design their own DRL algorithms by adapting these DRL algorithms.
    
.. image:: ../image/alg_compare.png
    
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

We use 5 DRL models in this article, namely PPO, A2C, DDPG, SAC and TD3. I introduced these models in the previous article. TD3 is an improvement over DDPG.

Tensorboard: reward and loss function plot
We use tensorboard integration for hyperparameter tuning and model picking. Tensorboard generates nice looking charts.

Once the learn function is called, you can monitor the RL agent during or after the training, with the following bash command:


.. code-block:: python
   :linenos:

    # cd to the tensorboard_log folder, run the following command 
    tensorboard --logdir ./A2C_20201127-19h01/
    # you can also add past logging folder
    tensorboard --logdir ./a2c_tensorboard/;./ppo2_tensorboard/
    

Total rewards for each of the algorithm:

.. image:: ../image/single_2.png


total_timesteps (int): the total number of samples to train on. It is one of the most important hyperparameters, there are also other important parameters such as learning rate, batch size, buffer size, etc.

To compare these algorithms, I set the total_timesteps = 100k. If we set the total_timesteps too large, then we will face a risk of overfitting.

By observing the episode_reward chart, we can see that these algorithms will converge to an optimal policy eventually as the step grows. TD3 converges very fast.

actor_loss for DDPG and policy_loss for TD3:

.. image:: ../image/single_3.png

.. image:: ../image/single_4.png


**Picking models**:

We pick the TD3 model, because it converges pretty fast and it’s a state of the art model over DDPG. By observing the episode_reward chart, TD3 doesn’t need to reach full 100k total_timesteps to converge.

**Trading**:

Assume that we have $100,000 initial capital at 2019/01/01. We use the TD3 model to trade AAPL.

.. code-block:: python
   :linenos:

    # create trading env
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                           env_class = StockEnvTrade,
                                            turbulence_threshold=250)
    ## make a prediction and get the account value change
    df_account_value = DRLAgent.DRL_prediction(model=model_sac,
                                               test_data = trade,
                                               test_env = env_trade,
                                               test_obs = obs_trade)
                                               

.. image:: ../image/single_5.png


Backtesting Performance
--------------------------

Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. 
We usually use the `Quantopian pyfolio`_ package to backtest our trading strategies. 
It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

.. _Quantopian pyfolio: https://github.com/quantopian/pyfolio

FinRL uses a `set of functions`_ to do the backtesting with pyfolio.

.. _set of functions: https://github.com/AI4Finance-LLC/FinRL-Library/blob/master/finrl/trade/backtest.py

.. code-block:: python
   :linenos:

    # BackTestStats
    # pass in df_account_value, this information is stored in env class
    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(account_value = df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')
    
    print("==============Get Baseline Stats===========")
    baesline_perf_stats=BaselineStats('^DJI',
                                      baseline_start = '2019-01-01',
                                      baseline_end = '2020-09-30')
    
    # BackTestPlot
    # pass the account value memory into the backtest functions
    # and select a baseline ticker
    print("==============Compare to DJIA===========")
    %matplotlib inline
    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    BackTestPlot(df_account_value, 
                 baseline_ticker = '^DJI', 
                 baseline_start = '2019-01-01',
                 baseline_end = '2020-09-30')
                 
                 
**Plots**:

.. image:: ../image/single_6.0.png
    :scale: 60 %
.. image:: ../image/single_6.png

.. image:: ../image/single_7.png






