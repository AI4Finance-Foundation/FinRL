# import packages

from finrl import config
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.finrl_meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
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


# read data
df=pd.read_csv("new.csv").iloc[:,1:]

# data preprocess
# print(df)
# df.loc[0,'USD (PM)']=-1
# df=df.fillna(-1)
list_ticker=['Gold','Bitcoin']
list_date=list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
combination=list(itertools.product(list_date,list_ticker))

df.columns=['date','Bitcoin','Gold']
df=df.sort_values(['date'])
processed=df.melt(id_vars=['date'],value_vars=['Gold','Bitcoin'],var_name='tic',value_name='close')
# print(df.info())
# print(processed.info())
# print(processed)
processed['Disable']=processed['close'].apply(pd.isna)
processed[processed['tic']=='Gold']=processed[processed['tic']=='Gold'].fillna(method='pad')
processed.loc[0,'close']=1324
processed_full=processed.sort_values(['date','tic'],ignore_index=True)
# processed_full=processed
# print(processed_full)
# print(processed_full.isna().any())
# processed_full.close=processed.close.astype('object')
# print(processed_full.info())
time=datetime.date(2016,9,11)

# initial = [cash, initial_stock1_share, initial_stock2_share]
initial=[1000,0,0]

# initial dataframe used to store the result of model
all_action=pd.DataFrame(columns=['date','Bitcoin','Gold'])
all_value=pd.DataFrame(columns=['date','account_value'])
all_state=pd.DataFrame(columns=['cash','Bitcoin_price','Gold_price','Bitcoin_num','Gold_num','Bitcoin_Disable','Gold_Disable'])

# trainning & trading process
while(time+datetime.timedelta(days=30)<datetime.date(2021,9,10)):
    train = data_split(processed_full, str(time),str(time+datetime.timedelta(days=31)))
    trade = data_split(processed_full, str(time+datetime.timedelta(days=30)),str(time+datetime.timedelta(days=60)))
    # print(len(processed_full))
    # print(train)
    # print(trade)
    # print(train.iloc[-1][:])

    # print(trade.loc[0]['close'])

    stock_dimension=len(train.tic.unique())
    # print("initial asset========", initial[0] + sum(initial[1:1 + stock_dimension] * trade.loc[0]['close']))
    state_space=3*stock_dimension+1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    env_kwargs = {
        "hmax": 10,
        "initial_list": initial, # Pass a initial state list to build trading env, instead of simply pass initial cash
        "buy_cost_pct": [0.01,0.02], # Different stock may need to have different cost of trading (buy or sell) in some specific problems
        "sell_cost_pct": [0.01,0.02],
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": ['Disable'], # there may be some dates that a stock is unable to trade
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "model_name":"stock exchange_SAC_coor",
        "mode":"alpha_Bitcoin=0.01, alpha_Gold=0.02"

    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    # print(type(env_train))
    agent = DRLAgent(env = env_train)
    model_PPO = agent.get_model("sac")
    trained_PPO = agent.train_model(model=model_PPO,
                                 tb_log_name='sac',
                                 total_timesteps=600)

    e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)
    # print("initial=======",initial)
    df_account_value, df_actions,df_state = DRLAgent.DRL_prediction(
        model=trained_PPO,
        environment = e_trade_gym)
    # print("first day asset:",df_account_value.iloc[0]['account_value'])
    df_actions.to_csv('action.csv')
    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    print(df_account_value)
    perf_stats_all = backtest_stats(account_value=df_account_value)
    # holding_num_share=backtest_stats()
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')
    # for Serial training, have nothing to do with FinRL lib
    initial=[df_state.iloc[-1]['cash'],df_state.iloc[-1]['Bitcoin_num'],df_state.iloc[-1]['Gold_num']]
    time=time + datetime.timedelta(days=30)
    all_action=pd.concat([all_action,df_actions],axis=0)
    all_value=pd.concat([all_value,df_account_value],axis=0)
    all_state=pd.concat([all_state,df_state],axis=0)

all_value.to_csv('all_value.csv')
all_action.to_csv('all_action.csv')
all_state.to_csv('all_state.csv')