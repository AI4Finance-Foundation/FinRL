from finrl.config.configuration import Configuration
from finrl.config.directory_operations import create_userdata_dir
from finrl.commands import start_download_cryptodata, start_download_stockdata, start_list_markets
from pathlib import Path
from finrl.data.fetchdata import FetchData
import pandas as pd
from finrl.config import TimeRange
from datetime import datetime, timedelta
import arrow
from finrl.tools.coin_search import *

#### CREATE USER DATA DIRECTORY IN DESIGNATED PATH, IF NO NAME INDICATED DEFAULT TO user_data
####### create dir to false if only to check existence of directory

# create_userdata_dir("./user_data",create_dir=True)


# ###### Pull Configuration File (using finrl/config/configuration.py)
config = Configuration.from_files(["config.json"])

##### EXAMPLE
##### if directory path is kept none, default = user_data
# create_userdata_dir("./finrl_testing", create_dir=True)

##### args are the different options that could overide config options

# ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
#                       'user_data_dir': None, 'pairs': None, 'pairs_file': None, 
#                       'days': 160, 'timerange': None, 
#                       'download_trades': False, 'exchange': 'binance', 
#                       'timeframes': ['1d'], 'erase': False, 
#                       'dataformat_ohlcv': None, 'dataformat_trades': None}

# ######## downloads data to our local data repository as dictated by our config, or we could overide it using 'datadir'
# start_download_cryptodata(ARGS_DOWNLOAD_DATA)

# ################# fetches all our local data and outputs a df with the normal format (index:date, open, high, low, close, volume and tick symbol)
# ################ can be modified to get its own ARGS and overide config info, using config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)


# df = FetchData(config).fetch_data_crypto()

# print(df.head())

################## Either input timerange or days for period of download
# ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
#                       'user_data_dir': None, 'days': None, 'timerange': "20200101-20210101",
#                       'timeframes': ['1d'], 'erase': False}


# start_download_stockdata(ARGS_DOWNLOAD_DATA)



# df = FetchData(config).fetch_data_stock()

# print(df.head())

# ARGS_LIST_PAIRS = ["exchange": config["exchange"]["name"], "print_list", "list_pairs_print_json", "print_one_column",
#                    "print_csv", "base_currencies", "quote_currencies", "list_pairs_all"]

ARGS_LIST_PAIRS = {"exchange":config["exchange"]["name"]}

x = start_list_markets(ARGS_LIST_PAIRS, "BNB")

coins = coinSearch("BNB")

coins_to_json("./config.json", coins)

ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
                      'user_data_dir': None, 'pairs': None, 'pairs_file': None, 
                      'days': 1000, 'timerange': None, 
                      'download_trades': False, 'exchange': 'binance', 
                      'timeframes': ['1d'], 'erase': False, 
                      'dataformat_ohlcv': None, 'dataformat_trades': None}

start_download_cryptodata(ARGS_DOWNLOAD_DATA)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

# %matplotlib inline
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

from pprint import pprint

import sys
sys.path.append("../FinRL-Library")

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


config = Configuration.from_files(["config.json"])

df = FetchData(config).fetch_data_crypto()
df.shape
df.sort_values(['date','tic'],ignore_index=True).head()

fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config["TECHNICAL_INDICATORS_LIST"],
                    use_turbulence=False,
                    user_defined_feature = False)

processed = fe.preprocess_data(df)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))
processed["date"] = processed["date"].astype(str)
processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)

processed_full.sort_values(['date','tic'],ignore_index=True).head(10)

train = data_split(processed_full, '2018-05-16','2020-05-16')
trade = data_split(processed_full, '2020-05-17','2020-10-31')
print(len(train))
print(len(trade))

train.head()

trade.head()

config["TECHNICAL_INDICATORS_LIST"]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(config["TECHNICAL_INDICATORS_LIST"])*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100, 
    "initial_amount": 1000000, 
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space, 
    "stock_dim": stock_dimension, 
    "tech_indicator_list": config["TECHNICAL_INDICATORS_LIST"], 
    "action_space": stock_dimension, 
    "reward_scaling": 1e-4
    
}
e_train_gym = StockTradingEnv(df = train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

agent = DRLAgent(env = env_train)
model_a2c = agent.get_model("a2c")

trained_a2c = agent.train_model(model=model_a2c, 
                              tb_log_name='a2c',
                              total_timesteps=100000)


### Model 2: DDPG
agent = DRLAgent(env = env_train)
model_ddpg = agent.get_model("ddpg")

trained_ddpg = agent.train_model(model=model_ddpg, 
                              tb_log_name='ddpg',
                              total_timesteps=50000)

### Model 3: PPO

agent = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = agent.train_model(model=model_ppo, 
                              tb_log_name='ppo',
                              total_timesteps=100000)

### Model 4: TD3

agent = DRLAgent(env = env_train)
TD3_PARAMS = {"batch_size": 100, 
              "buffer_size": 1000000, 
              "learning_rate": 0.001}

model_td3 = agent.get_model("td3",model_kwargs = TD3_PARAMS)

trained_td3 = agent.train_model(model=model_td3, 
                              tb_log_name='td3',
                              total_timesteps=30000)

                            
  ### Model 5: SAC

agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 1000000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

trained_sac = agent.train_model(model=model_sac, 
                              tb_log_name='sac',
                              total_timesteps=80000)

data_turbulence = processed_full[(processed_full.date<'2019-01-01') & (processed_full.date>='2009-01-01')]
insample_turbulence = data_turbulence.drop_duplicates(subset=['date'])

insample_turbulence.turbulence.describe()

turbulence_threshold = np.quantile(insample_turbulence.turbulence.values,1)

turbulence_threshold


e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=model_ppo, 
    environment = e_trade_gym)

df_account_value.shape

df_account_value.head()
df_actions.head()

print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')


print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value, config, 
              baseline_start = '2019-01-01',
              baseline_end = '2021-01-01')


print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker="^DJI", start = '2019-01-01',
                                  end = '2021-01-01')

stats = backtest_stats(baseline_df, value_col_name = 'close')