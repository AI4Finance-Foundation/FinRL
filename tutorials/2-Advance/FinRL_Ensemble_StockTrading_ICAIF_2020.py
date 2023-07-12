from __future__ import annotations
<<<<<<< HEAD
# %%
from __future__ import annotations
=======


#%%

>>>>>>> dev-jdb

import warnings

warnings.filterwarnings("ignore")


# In[34]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
<<<<<<< HEAD

matplotlib.use("Agg")
=======
matplotlib.use('Agg')
>>>>>>> dev-jdb
import datetime

# get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys

sys.path.append("../FinRL-Library")

import itertools


<<<<<<< HEAD
# In[4]:


=======


# In[4]:



>>>>>>> dev-jdb
import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)


# In[5]:


tickers = config_tickers.DOW_30_TICKER
print(tickers)


# In[6]:


<<<<<<< HEAD
df = pd.read_csv("../datasets/DOW_30_TICKER.csv")
print(df.head())


=======
df = pd.read_csv('../datasets/DOW_30_TICKER.csv')
print(df.head())



>>>>>>> dev-jdb
# In[9]:


print(df.head())


# In[10]:


print(df.tail())


# In[11]:


print(df.shape)


# In[12]:


<<<<<<< HEAD
print(df.sort_values(["date", "tic"]).head())
=======
print(df.sort_values(['date', 'tic']).head())
>>>>>>> dev-jdb


# In[13]:


print(len(df.tic.unique()))


# In[14]:


print(df.tic.value_counts())


# # Part 4: Preprocess Data
# Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
# * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
# * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.

# In[15]:


<<<<<<< HEAD
tech_indicators = ["macd", "rsi_30", "cci_30", "dx_30"]
=======
tech_indicators = ['macd',
                   'rsi_30',
                   'cci_30',
                   'dx_30']
>>>>>>> dev-jdb


# In[16]:


fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=tech_indicators,
    use_turbulence=True,
<<<<<<< HEAD
    user_defined_feature=False,
)
=======
    user_defined_feature=False)
>>>>>>> dev-jdb

processed = fe.preprocess_data(df)
processed = processed.copy()
processed = processed.fillna(0)
processed = processed.replace(np.inf, 0)


# In[17]:


processed.sample(5)

# In[18]:


stock_dimension = len(processed.tic.unique())
state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[19]:


env_kwargs = {
    "hmax": 100,
    "initial_amount": 100000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
<<<<<<< HEAD
    "print_verbosity": 5,
=======
    "print_verbosity": 5

>>>>>>> dev-jdb
}

# In[20]:


rebalance_window = 63  # rebalance_window is the number of days to retrain the model
validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)
# TODO Updated start and end date with date time
<<<<<<< HEAD
train_start = "2009-04-01"
train_end = "2020-04-01"
val_test_start = "2020-04-01"
# val_test_end as current date


val_test_end = "2022-06-01"

ensemble_agent = DRLEnsembleAgent(
    df=processed,
    train_period=(train_start, train_end),
    val_test_period=(val_test_start, val_test_end),
    rebalance_window=rebalance_window,
    validation_window=validation_window,
    **env_kwargs,
)
=======
train_start = '2009-04-01'
train_end = '2020-04-01'
val_test_start = '2020-04-01'
# val_test_end as current date


val_test_end = '2022-06-01'

ensemble_agent = DRLEnsembleAgent(df=processed,
                                  train_period=(train_start, train_end),
                                  val_test_period=(val_test_start, val_test_end),
                                  rebalance_window=rebalance_window,
                                  validation_window=validation_window,
                                  **env_kwargs)
>>>>>>> dev-jdb


# In[21]:


<<<<<<< HEAD
A2C_model_kwargs = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0005}
=======
A2C_model_kwargs = {
    'n_steps': 5,
    'ent_coef': 0.01,
    'learning_rate': 0.0005
}
>>>>>>> dev-jdb

PPO_model_kwargs = {
    "ent_coef": 0.01,
    "n_steps": 2048,
    "learning_rate": 0.00025,
<<<<<<< HEAD
    "batch_size": 512,
=======
    "batch_size": 512
>>>>>>> dev-jdb
}

DDPG_model_kwargs = {
    "action_noise": "ornstein_uhlenbeck",
    "buffer_size": 10_000,
    "learning_rate": 0.0005,
<<<<<<< HEAD
    "batch_size": 512,
}

timesteps_dict = {"a2c": 1_000, "ppo": 1_000, "ddpg": 1_000}
=======
    "batch_size": 512
}

timesteps_dict = {'a2c': 1_000,
                  'ppo': 1_000,
                  'ddpg': 1_000
                  }
>>>>>>> dev-jdb


# In[22]:


<<<<<<< HEAD
df_summary = ensemble_agent.run_ensemble_strategy(
    A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict
)
=======
df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                  PPO_model_kwargs,
                                                  DDPG_model_kwargs,
                                                  timesteps_dict)
>>>>>>> dev-jdb


# In[23]:


print(df_summary)


# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# In[38]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[39]:


<<<<<<< HEAD
unique_trade_date = processed[
    (processed.date > val_test_start) & (processed.date <= val_test_end)
].date.unique()
=======
unique_trade_date = processed[(processed.date > val_test_start) & (processed.date <= val_test_end)].date.unique()
>>>>>>> dev-jdb


# In[40]:


<<<<<<< HEAD
df_trade_date = pd.DataFrame({"datadate": unique_trade_date})

df_account_value = pd.DataFrame()
for i in range(
    rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window
):
    temp = pd.read_csv(f"results/account_value_trade_ensemble_{i}.csv")
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
=======
df_trade_date = pd.DataFrame({'datadate': unique_trade_date})

df_account_value = pd.DataFrame()
for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
    temp = pd.read_csv(f'results/account_value_trade_ensemble_{i}.csv')
    df_account_value = df_account_value.append(temp, ignore_index=True)
sharpe = (252 ** 0.5) * df_account_value.account_value.pct_change(1).mean() / df_account_value.account_value.pct_change(
    1).std()
print('Sharpe Ratio: ', sharpe)
df_account_value = df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))
>>>>>>> dev-jdb


# In[41]:


df_account_value.head()


# In[42]:


# get_ipython().run_line_magic('matplotlib', 'inline')
df_account_value.account_value.plot()


# In[43]:


print("==============Get Backtest Results===========")
<<<<<<< HEAD
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
=======
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
>>>>>>> dev-jdb

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)


# In[44]:
<<<<<<< HEAD
# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_account_value.loc[0, "date"],
    end=df_account_value.loc[len(df_account_value) - 1, "date"],
)

stats = backtest_stats(baseline_df, value_col_name="close")
=======
#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_account_value.loc[0, 'date'],
    end=df_account_value.loc[len(df_account_value) - 1, 'date'])

stats = backtest_stats(baseline_df, value_col_name='close')

>>>>>>> dev-jdb


# In[49]:


print("==============Compare to DJIA===========")
# get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
<<<<<<< HEAD
backtest_plot(
    df_account_value,
    baseline_ticker="^DJI",
    baseline_start=df_account_value.loc[0, "date"],
    baseline_end=df_account_value.loc[len(df_account_value) - 1, "date"],
)

#%%
=======
backtest_plot(df_account_value,
              baseline_ticker='^DJI',
              baseline_start=df_account_value.loc[0, 'date'],
              baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])

>>>>>>> dev-jdb
