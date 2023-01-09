# %%
# !/usr/bin/env python
# coding: utf-8


# %%


import datetime
# get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, get_daily_return, get_baseline  # , get_returns
import pyfolio
from copy import deepcopy
from pprint import pprint
import pandas as pd
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")
import sys

sys.path.append("../FinRL")

import itertools

import os

if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
abspath = os.path.realpath('__file__')
print(abspath)
dname = os.path.dirname(abspath)
os.chdir('/home/james/Dropbox/Investing/RL_Agent_Examples/FinRL/tutorials/')
cwd = os.getcwd()
print("cwd", cwd)
# os.chdir("./")

# <a id='2'></a>
# # Part 3. Download Stock Data from Yahoo Finance

# -----
# class YahooDownloader:
#     Retrieving daily stock data from Yahoo Finance API
#
#     Attributes
#     ----------
#         start_date : str
#             start date of the data (modified from config.py)
#         end_date : str
#             end date of the data (modified from config.py)
#         ticker_list : list
#             a list of stock tickers (modified from config.py)
#
#     Methods
#     -------
#     fetch_data()
# %%
var = 710
variable_name = [k for k, v in locals().items() if v == 710][0]
print("Your variable name is " + variable_name)
# In[4]:
tickers = config_tickers.NAS_100_TICKER
print(f'Number of tickers: {len(tickers)}')
print(tickers)

# In[5]:


df = YahooDownloader(start_date='2009-01-01',
                     end_date='2022-09-01',
                     ticker_list=tickers).fetch_data()

# In[6]:
df.to_csv(f'./datasets/NAS_{len(tickers)}.csv', index=False)
