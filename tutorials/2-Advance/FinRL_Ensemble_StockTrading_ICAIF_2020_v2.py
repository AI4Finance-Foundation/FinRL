#!/usr/bin/env python
# %%
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

# %%

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
import datetime

# get_ipython().run_line_magic('matplotlib', 'inline')
import finrl.config_tickers as config_tickers
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys

sys.path.append("../FinRL-Library")

import itertools

# set cuda device to index 1
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# <a id='1.4'></a>
# ## 2.4. Create Folders

# %%

import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

#

# %%

print(DOW_30_TICKER)
# %%
#  read data
# df = pd.read_csv('../datasets/DOW30_2009-04-01-2021-01-01.csv')
# df.head()

# %%

TRAIN_START_DATE = '2009-04-01'
TRAIN_END_DATE = '2021-01-01'
TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2022-06-01'
# df = pd.read_csv('../datasets/DOW30.csv')
df = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER,
).fetch_data()

# %%

# df.to_csv(
#     f"../datasets/NAS_100_TICKER_{TRAIN_START_DATE}-{TRAIN_END_DATE}.csv", index=False
# )
df.head()

# In[56]:
# %%

df.tail()

# In[57]:
# %%

df.shape

# In[58]:
# %%

df.sort_values(["date", "tic"]).head()

# In[59]:
# %%

len(df.tic.unique())

# %%

df.tic.value_counts()

# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
# %%
# In[61]:


#  INDICATORS = ['macd',
#                'rsi_30',
#                'cci_30',
#                'dx_30']


# In[62]:
# %%

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf, 0)

# In[63]:
# %%

processed.sample(5)

# <a id='4'></a>
# # Part 5. Design Environment
# Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a **Markov Decision Process (MDP)** problem. The training process involves observing stock price change, taking an action and reward's calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds.
#
# Our trading environments, based on OpenAI Gym framework, simulate live stock markets with real market data according to the principle of time-driven simulation.
#
# The action space describes the allowed actions that the agent interacts with the environment. Normally, action a includes three actions: {-1, 0, 1}, where -1, 0, 1 represent selling, holding, and buying one share. Also, an action can be carried upon multiple shares. We use an action space {-k,…,-1, 0, 1, …, k}, where k denotes the number of shares to buy and -k denotes the number of shares to sell. For example, "Buy 10 shares of AAPL" or "Sell 10 shares of AAPL" are 10 or -10, respectively. The continuous action space needs to be normalized to [-1, 1], since the policy is defined on a Gaussian distribution, which needs to be normalized and symmetric.

# In[64]:
# %%

stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# In[65]:
# %%

env_kwargs = {
    "hmax": 100,
    "initial_amount": 25000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity":5

}
# <a id='5'></a>
# # Part 6: Implement DRL Algorithms
# * The implementation of the DRL algorithms are based on **OpenAI Baselines** and **Stable Baselines**. Stable Baselines is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.
#
# * In this notebook, we are training and validating 3 agents (A2C, PPO, DDPG) using Rolling-window Ensemble Method ([reference code](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/blob/80415db8fa7b2179df6bd7e81ce4fe8dbf913806/model/models.py#L92))

# In[66]:
# %%

rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

ensemble_agent = DRLEnsembleAgent(
    df=processed,
    train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
    val_test_period=(TEST_START_DATE, TEST_END_DATE),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)

# In[67]:
# %%

A2C_model_kwargs = {
    'n_steps': 5,
    'ent_coef': 0.005,
    'learning_rate': 0.0007
}

PPO_model_kwargs = {
    "ent_coef":0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
    "batch_size": 256
}

DDPG_model_kwargs = {
    #"action_noise":"ornstein_uhlenbeck",
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
    "batch_size": 128
}

timesteps_dict = {'a2c' : 100_000,
                  'ppo' : 100_000,
                  'ddpg' : 100_000
                  }
# %%
df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                  PPO_model_kwargs,
                                                  DDPG_model_kwargs,
                                                  timesteps_dict)
# %%


df_summary

# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# %%


unique_trade_date = processed[
    (processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)
].date.unique()

# %%


df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

df_account_value = pd.DataFrame()
for i in range(
    rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window
):
    temp = pd.read_csv("results/account_value_trade_{}_{}.csv".format("ensemble", i))
    df_account_value = df_account_value.append(temp, ignore_index=True)
sharpe = (
    (252**0.5)
    * df_account_value.account_value.pct_change(1).mean()
    / df_account_value.account_value.pct_change(1).std()
)
print("Sharpe Ratio: ", sharpe)
df_account_value = df_account_value.join(
    df_trade_date[validation_window:].reset_index(drop=True)
)

# %%


df_account_value.head()

# %%

get_ipython().run_line_magic("matplotlib", "inline")
df_account_value.account_value.plot()

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# %%

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

# %%

# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_account_value.loc[0, "date"],
    end=df_account_value.loc[len(df_account_value) - 1, "date"],
)

stats = backtest_stats(baseline_df, value_col_name="close")

# <a id='6.2'></a>
# ## 7.2 BackTestPlot

# %%
print("==============Compare to DJIA===========")
get_ipython().run_line_magic("matplotlib", "inline")
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(
    df_account_value,
    baseline_ticker="^DJI",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)

#

# %%
