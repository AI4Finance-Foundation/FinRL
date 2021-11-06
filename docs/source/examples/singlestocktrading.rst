:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

DRL Single Stock Trading
============================

Deep Reinforcement Learning for Stock Trading from Scratch: Single Stock Trading


.. tip::

    Run the code step by step at `Google Colab`_.
    
    .. _Google Colab: https://colab.research.google.com/github/AI4Finance-LLC/FinRL-Library/blob/master/examples/old/DRL_single_stock_trading.ipynb


Python Package Installation
----------------------------------

As a first step we check if the additional packages needed are present, if not install them.

    - Yahoo Finance API
    - pandas
    - matplotlib
    - stockstats
    - OpenAI gym
    - stable-baselines
    - tensorflow

.. code-block::
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

Import packages:

.. code-block:: python
    :linenos:
    
    import yfinance as yf
    from stockstats import StockDataFrame as Sdf
    
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import gym
    from stable_baselines import PPO2, DDPG, A2C, ACKTR, TD3
    from stable_baselines import DDPG
    from stable_baselines import A2C
    from stable_baselines import SAC
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.common.policies import MlpPolicy
    
    
Download Data
-------------------------------------

Yahoo Finance is a website that provides stock data, financial news, financial reports, etc. All the data provided by Yahoo Finance is free.

.. code-block:: python
    :linenos:
    
    # Download and save the data in a pandas DataFrame:
    data_df = yf.download("AAPL", start="2009-01-01", end="2020-10-23")



Preprocess Data
-------------------------------------

Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.


Implement DRL Algorithms
-------------------------------------
The implementation of the DRL algorithms are based on OpenAI Baselines and Stable Baselines. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.


Model Training
-------------------------------------

4 models: PPO A2C, DDPG, TD3

**Model 1: PPO**

.. code-block:: python
    :linenos:
    
    #tensorboard --logdir ./single_stock_tensorboard/
    env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
    model_ppo = PPO2('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
    model_ppo.learn(total_timesteps=100000,tb_log_name="run_aapl_ppo")
    #model.save('AAPL_ppo_100k')
    
    
**Model 2: DDPG**

.. code-block:: python
    :linenos:

    #tensorboard --logdir ./single_stock_tensorboard/
    env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
    model_ddpg = DDPG('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
    model_ddpg.learn(total_timesteps=100000, tb_log_name="run_aapl_ddpg")
    #model.save('AAPL_ddpg_50k')



**Model 3: A2C**

.. code-block:: python
    :linenos:

    #tensorboard --logdir ./single_stock_tensorboard/
    env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
    model_a2c = A2C('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
    model_a2c.learn(total_timesteps=100000,tb_log_name="run_aapl_a2c")
    #model.save('AAPL_a2c_50k')
    

**Model 4: TD3**

.. code-block:: python
    :linenos:

    #tensorboard --logdir ./single_stock_tensorboard/
    #DQN<DDPG<TD3
    env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
    model_td3 = TD3('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
    model_td3.learn(total_timesteps=100000,tb_log_name="run_aapl_td3")
    #model.save('AAPL_td3_50k')
    
    
**Testing data**

.. code-block:: python
    :linenos:
    
    test = data_clean[(data_clean.datadate>='2019-01-01') ]
    # the index needs to start from 0
    test=test.reset_index(drop=True)
    
**Trading**

Assume that we have $100,000 initial capital at 2019-01-01. We use the TD3 model to trade AAPL.

.. code-block:: python
    :linenos:

    model = model_td3
    env_test = DummyVecEnv([lambda: SingleStockEnv(test)])
    obs_test = env_test.reset()
    print("==============Model Prediction===========")
    for i in range(len(test.index.unique())):
        action, _states = model.predict(obs_test)
        obs_test, rewards, dones, info = env_test.step(action)
        env_test.render()
        

Backtest Our Strategy
------------------------------------

For simplicity purposes, in the article, we just calculate the Sharpe ratio and the annual return manually.

.. code-block:: python
    :linenos:

    def get_DRL_sharpe():
        df_total_value=pd.read_csv('account_value.csv',index_col=0)
        df_total_value.columns = ['account_value']
        df_total_value['daily_return']=df_total_value.pct_change(1)
        sharpe = (252**0.5)*df_total_value['daily_return'].mean()/ \
        df_total_value['daily_return'].std()
        
        annual_return = ((df_total_value['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)
        print("sharpe ratio: ", sharpe)
        return df_total_value
        
    
    def get_buy_and_hold_sharpe(test):
        test['daily_return']=test['adjcp'].pct_change(1)
        sharpe = (252**0.5)*test['daily_return'].mean()/ \
        test['daily_return'].std()
        annual_return = ((test['daily_return'].mean()+1)**252-1)*100
        print("annual return: ", annual_return)
    
        print("sharpe ratio: ", sharpe)
        #return sharpe
        