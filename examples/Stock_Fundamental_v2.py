#!/usr/bin/env python
# coding: utf-8


# %%


import datetime
import itertools
import json
import multiprocessing
import os
import sys
import time
from copy import deepcopy

sys.path.append("../../")
cwd = os.getcwd()
print("cwd", cwd)
# sys.path.append("./FinRL")
import matplotlib
import numpy as np
import pandas as pd
import pyfolio
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.plot import backtest_stats, get_daily_return, get_baseline  # , get_returns

from Stock_env import StockTradingEnv
# get_ipython().run_line_magic('matplotlib', 'inline')
from finrl import config
from finrl import config_tickers
import time

# sys.path.append("../FinRL")

# abspath = os.path.realpath('__file__')
# print(abspath)
# dname = os.path.dirname(abspath)
# os.chdir('/home/james/Dropbox/Investing/RL_Agent_Examples/FinRL_JDB/tutorials/1-Introduction')
cwd = os.getcwd()
print("cwd", cwd)
start_date = '2019-01-01'
end_date = '2021-01-01'
# total_steps = config.TOTAL_TIME_STEPS
total_steps = 5_000
print(f'Total Steps: {total_steps}')

matplotlib.use('Agg')
# %%
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
if not os.path.exists("./images"):
    os.makedirs("./images")

# %%
os.path.realpath('__file__')
# In[4]:

tickers = config_tickers.SP_500_TICKER
# tickers = config_tickers.DOW_30_TICKER
# ticker_num = len(tickers)
# print(tickers)

# In[5]:


# df = YahooDownloader(start_date = '2009-01-01',
#                      end_date = '2021-01-01',
#                      ticker_list = config_tickers.DOW_30_TICKER).fetch_data()


# In[6]:

dtype = {'first_column': 'str', 'second_column': 'str'}
# df = pd.read_csv('../datasets/SP_500_TICKER.csv')
df = pd.read_csv('../datasets/DOW_30_TICKER.csv')
# drop rows with missing values
# df = df.dropna()
# get number of distinct tic in df
tickers = df.tic.unique()
ticker_num = len(df.tic.unique())
print(f'DF Shape: {df.shape}')
print(f'Total Steps: {total_steps}')
print(f'Using Tickers: {tickers}')
print(df.head())
# In[7]:


# df.shape
# df.to_csv('../datasets/DOW30.csv', index=False)


# In[8]:


df.head()

# In[9]:


df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# In[10]:


df.sort_values(['date', 'tic'], ignore_index=True).head()

# # Part 4: Preprocess fundamental data
# - Import finanical data downloaded from Compustat via WRDS(Wharton Research Data Service)
# - Preprocess the dataset and calculate financial ratios
# - Add those ratios to the price data preprocessed in Part 3
# - Calculate price-related ratios such as P/E and P/B

# ## 4.1 Import the financial data

# In[11]:


# Import fundamental data from my GitHub repository
url = 'https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv'

fund = pd.read_csv(url, low_memory=False)

# In[12]:


# Check the imported dataset
fund.head()

# ## 4.2 Specify items needed to calculate financial ratios
# - To learn more about the data description of the dataset, please check WRDS's website(https://wrds-www.wharton.upenn.edu/). Login will be required.

# In[13]:


# List items that are used to calculate financial ratios

items = [
    'datadate',  # Date
    'tic',  # Ticker
    'oiadpq',  # Quarterly operating income
    'revtq',  # Quartely revenue
    'niq',  # Quartely net income
    'atq',  # Total asset
    'teqq',  # Shareholder's equity
    'epspiy',  # EPS(Basic) incl. Extraordinary items
    'ceqq',  # Common Equity
    'cshoq',  # Common Shares Outstanding
    'dvpspq',  # Dividends per share
    'actq',  # Current assets
    'lctq',  # Current liabilities
    'cheq',  # Cash & Equivalent
    'rectq',  # Recievalbles
    'cogsq',  # Cost of  Goods Sold
    'invtq',  # Inventories
    'apq',  # Account payable
    'dlttq',  # Long term debt
    'dlcq',  # Debt in current liabilites
    'ltq'  # Liabilities
]

# Omit items that will not be used
fund_data = fund[items]

# In[14]:


# Rename column names for the sake of readability
fund_data = fund_data.rename(columns={
    'datadate': 'date',  # Date
    'oiadpq': 'op_inc_q',  # Quarterly operating income
    'revtq': 'rev_q',  # Quartely revenue
    'niq': 'net_inc_q',  # Quartely net income
    'atq': 'tot_assets',  # Assets
    'teqq': 'sh_equity',  # Shareholder's equity
    'epspiy': 'eps_incl_ex',  # EPS(Basic) incl. Extraordinary items
    'ceqq': 'com_eq',  # Common Equity
    'cshoq': 'sh_outstanding',  # Common Shares Outstanding
    'dvpspq': 'div_per_sh',  # Dividends per share
    'actq': 'cur_assets',  # Current assets
    'lctq': 'cur_liabilities',  # Current liabilities
    'cheq': 'cash_eq',  # Cash & Equivalent
    'rectq': 'receivables',  # Receivalbles
    'cogsq': 'cogs_q',  # Cost of  Goods Sold
    'invtq': 'inventories',  # Inventories
    'apq': 'payables',  # Account payable
    'dlttq': 'long_debt',  # Long term debt
    'dlcq': 'short_debt',  # Debt in current liabilites
    'ltq': 'tot_liabilities'  # Liabilities
})

# In[15]:


# Check the data
fund_data.head()

# ## 4.3 Calculate financial ratios
# - For items from Profit/Loss statements, we calculate LTM (Last Twelve Months) and use them to derive profitability related ratios such as Operating Maring and ROE. For items from balance sheets, we use the numbers on the day.
# - To check the definitions of the financial ratios calculated here, please refer to CFI's website: https://corporatefinanceinstitute.com/resources/knowledge/finance/financial-ratios/

# In[16]:

# Calculate financial ratios
date = pd.to_datetime(fund_data['date'], format='%Y%m%d')

tic = fund_data['tic'].to_frame('tic')

# Profitability ratios
# Operating Margin
OPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='OPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        OPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        OPM.iloc[i] = np.nan
    else:
        OPM.iloc[i] = np.sum(fund_data['op_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Net Profit Margin
NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='NPM')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        NPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        NPM.iloc[i] = np.nan
    else:
        NPM.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / np.sum(fund_data['rev_q'].iloc[i - 3:i])

# Return On Assets
ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROA')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROA[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROA.iloc[i] = np.nan
    else:
        ROA.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['tot_assets'].iloc[i]

# Return on Equity
ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='ROE')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROE[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROE.iloc[i] = np.nan
    else:
        ROE.iloc[i] = np.sum(fund_data['net_inc_q'].iloc[i - 3:i]) / fund_data['sh_equity'].iloc[i]

    # For calculating valuation ratios in the next subpart, calculate per share items in advance
# Earnings Per Share
EPS = fund_data['eps_incl_ex'].to_frame('EPS')

# Book Per Share
BPS = (fund_data['com_eq'] / fund_data['sh_outstanding']).to_frame('BPS')  # Need to check units

# Dividend Per Share
DPS = fund_data['div_per_sh'].to_frame('DPS')

# Liquidity ratios
# Current ratio
cur_ratio = (fund_data['cur_assets'] / fund_data['cur_liabilities']).to_frame('cur_ratio')

# Quick ratio
quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities']).to_frame('quick_ratio')

# Cash ratio
cash_ratio = (fund_data['cash_eq'] / fund_data['cur_liabilities']).to_frame('cash_ratio')

# Efficiency ratios
# Inventory turnover ratio
inv_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='inv_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        inv_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        inv_turnover.iloc[i] = np.nan
    else:
        inv_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['inventories'].iloc[i]

# Receivables turnover ratio
acc_rec_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_rec_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_rec_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_rec_turnover.iloc[i] = np.nan
    else:
        acc_rec_turnover.iloc[i] = np.sum(fund_data['rev_q'].iloc[i - 3:i]) / fund_data['receivables'].iloc[i]

# Payable turnover ratio
acc_pay_turnover = pd.Series(np.empty(fund_data.shape[0], dtype=object), name='acc_pay_turnover')
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_pay_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_pay_turnover.iloc[i] = np.nan
    else:
        acc_pay_turnover.iloc[i] = np.sum(fund_data['cogs_q'].iloc[i - 3:i]) / fund_data['payables'].iloc[i]

## Leverage financial ratios
# Debt ratio
debt_ratio = (fund_data['tot_liabilities'] / fund_data['tot_assets']).to_frame('debt_ratio')

# Debt to Equity ratio
debt_to_equity = (fund_data['tot_liabilities'] / fund_data['sh_equity']).to_frame('debt_to_equity')
# In[17]:


# Create a dataframe that merges all the ratios
ratios = pd.concat([date, tic, OPM, NPM, ROA, ROE, EPS, BPS, DPS,
                    cur_ratio, quick_ratio, cash_ratio, inv_turnover, acc_rec_turnover, acc_pay_turnover,
                    debt_ratio, debt_to_equity], axis=1)

# In[18]:


# Check the ratio data
ratios.head()

# In[19]:


ratios.tail()

# ## 4.4 Deal with NAs and infinite values
# - We replace N/A and infinite values with zero.

# In[20]:


# Replace NAs infinite values with zero
final_ratios = ratios.copy()
final_ratios = final_ratios.fillna(0)
final_ratios = final_ratios.replace(np.inf, 0)

# In[21]:


final_ratios.head()

# In[22]:


final_ratios.tail()

# ## 4.5 Merge stock price data and ratios into one dataframe
# - Merge the price dataframe preprocessed in Part 3 and the ratio dataframe created in this part
# - Since the prices are daily and ratios are quartely, we have NAs in the ratio columns after merging the two dataframes. We deal with this by backfilling the ratios.

# In[23]:


list_ticker = df["tic"].unique().tolist()
list_date = list(pd.date_range(df['date'].min(), df['date'].max()))
combination = list(itertools.product(list_date, list_ticker))

# Merge stock price data and ratios into one dataframe
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(df, on=["date", "tic"], how="left")
processed_full = processed_full.merge(final_ratios, how='left', on=['date', 'tic'])
processed_full = processed_full.sort_values(['tic', 'date'])

# Backfill the ratio data to make them daily
processed_full = processed_full.bfill(axis='rows')

# ## 4.6 Calculate market valuation ratios using daily stock price data

# In[24]:


# Calculate P/E, P/B and dividend yield using daily closing price
processed_full['PE'] = processed_full['close'] / processed_full['EPS']
processed_full['PB'] = processed_full['close'] / processed_full['BPS']
processed_full['Div_yield'] = processed_full['DPS'] / processed_full['close']

# Drop per share items used for the above calculation
processed_full = processed_full.drop(columns=['day', 'EPS', 'BPS', 'DPS'])
# Replace NAs infinite values with zero
processed_full = processed_full.copy()
processed_full = processed_full.fillna(0)
processed_full = processed_full.replace(np.inf, 0)

# In[25]:


# Check the final data
processed_full.sort_values(['date', 'tic'], ignore_index=True).head(10)

# <a id='4'></a>
# # Part 5. A Market Environment in OpenAI Gym-style
# The training process involves observing stock price change, taking an action and reward's calculation. By interacting with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.
#
# Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.

# ## 5.1 Data Split
# - Training data period: 2009-01-01 to 2019-01-01
# - Trade data period: 2019-01-01 to 2020-12-31

# In[26]:


train_data = data_split(processed_full, '2009-01-01', '2019-01-01')
trade_data = data_split(processed_full, '2019-01-01', '2021-01-01')
# Check the length of the two datasets
print(len(train_data))
print(len(trade_data))

# In[27]:


train_data.head()

# In[28]:


trade_data.head()

# ## 5.2 Set up the training environment

# In[29]:


# from stable_baselines3.common import logger

# In[30]:


ratio_list = ['OPM', 'NPM', 'ROA', 'ROE', 'cur_ratio', 'quick_ratio', 'cash_ratio', 'inv_turnover', 'acc_rec_turnover',
              'acc_pay_turnover', 'debt_ratio', 'debt_to_equity',
              'PE', 'PB', 'Div_yield']

stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

# In[31]:


# Parameters for the environment
env_kwargs = {
    "hmax": 100,
    "initial_amount": 10_000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ratio_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

# Establish the training environment using StockTradingEnv() class
e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)

# ## Environment for Training
#
#

# In[32]:


env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# <a id='5'></a>
# # Part 6: Train DRL Agents
# * The DRL algorithms are from **Stable Baselines 3**. Users are also encouraged to try **ElegantRL** and **Ray RLlib**.
# * FinRL library includes fine-tuned standard DRL algorithms, such as DQN, DDPG,
# Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to
# design their own DRL algorithms by adapting these DRL algorithms.

# In[33]:


# Set up the agent using DRLAgent() class using the environment created in the previous part
agent = DRLAgent(env=env_train)


# ### Agent Training: 5 algorithms (A2C, DDPG, PPO, TD3, SAC)

# ### Model 1: PPO

# In[34]:

def train_ppo(total_timesteps=50000, batch_size=128, model_name='ppo'):
    global agent, model_ppo
    print("============== Model 1: PPO: ===========")
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": batch_size,
    }
    model_ppo = agent.get_model(model_name, model_kwargs=PPO_PARAMS)
    return agent.train_model(model=model_ppo, tb_log_name='ppo', total_timesteps=total_timesteps)


# In[ ]: Training PPO


# ### Model 2: DDPG


def train_ddpg(total_timesteps=50000, batch_size=128):
    global agent, model_ddpg
    print("============== Model 2: DDPG ===========")
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": batch_size,
        "buffer_size": 50000,
        "learning_rate": 0.001,
    }
    model_ddpg = agent.get_model("ddpg")  # , model_kwargs=DDPG_PARAMS
    return agent.train_model(model=model_ddpg, tb_log_name='ddpg', total_timesteps=total_timesteps)


#

# In[ ]:

def train_a2c(total_timesteps=50000):
    global agent, model_a2c
    print("============== Model 3: A2C ===========")
    agent = DRLAgent(env=env_train)
    A2C_PARAMS = {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
    }
    model_a2c = agent.get_model("a2c")  # model_kwargs=A2C_PARAMS
    return agent.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=total_timesteps)


# print('################# Model 3: A2C #########################')
# #
# agent = DRLAgent(env=env_train)
# model_a2c = agent.get_model("a2c")
#
# trained_a2c = agent.train_model(model=model_a2c,
#                                 tb_log_name='a2c',
#                                 total_timesteps=100_000)

# ### Model 4: TD3

# In[ ]:

def train_td3(total_timesteps=50000, batch_size=128):
    global agent, model_td3
    print("============== Model 4: TD3 ===========")
    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": batch_size,
                  "buffer_size": 1000000,
                  "learning_rate": 0.001}
    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
    return agent.train_model(model=model_td3, tb_log_name='td3', total_timesteps=total_timesteps)


# print('################# Model 4: TD3 #########################')
# #
# agent = DRLAgent(env=env_train)
# TD3_PARAMS = {"batch_size": 100,
#               "buffer_size": 1000000,
#               "learning_rate": 0.001}
#
# model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
# #
# trained_td3 = agent.train_model(model=model_td3,
#                                 tb_log_name='td3',
#                                 total_timesteps=30000)
#
#
# # ### Model 5: SAC
#
# In[ ]:

def train_sac(total_timesteps=50000, sac_prams=config.SAC_PARAMS):
    global agent, model_sac
    print("============== Model 5: SAC ===========")
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
    model_sac = agent.get_model("sac", model_kwargs=sac_prams)
    return agent.train_model(model=model_sac, tb_log_name='sac', total_timesteps=total_timesteps)


# %% Training PPO
def train_seq(model_name_list=None, total_timesteps=100_000):
    """
    Train models in sequence
    :param total_timesteps:
    :param model_name_list:  ['ppo', 'ddpg', 'a2c', 'td3', 'sac']
    :return: trained model list in the  order of  ['ppo', 'ddpg', 'a2c', 'td3', 'sac']
    """
    # train the model if it is in the list
    if model_name_list is None:
        model_name_list = ['ppo', 'ddpg', 'a2c', 'td3', 'sac']

    global trained_td3, trained_sac, trained_a2c, trained_ddpg, trained_ppo
    start_time = time.time()

    if 'ppo' in model_name_list:
        trained_ppo = train_ppo(total_timesteps=total_timesteps)
    #  Training DDPG
    if 'ddpg' in model_name_list:
        trained_ddpg = train_ddpg(total_timesteps=total_timesteps)
    #  Training A2C
    if 'a2c' in model_name_list:
        trained_a2c = train_a2c(total_timesteps=total_timesteps)
    # Training TD3
    if 'td3' in model_name_list:
        trained_td3 = train_td3(total_timesteps=total_timesteps)
    #  Training SAC
    if 'sac' in model_name_list:
        trained_sac = train_sac(total_timesteps=total_timesteps)
    end_time = time.time()
    print("Total time taken to train all models: ", end_time - start_time)
    return trained_ppo, trained_ddpg, trained_a2c, trained_td3, trained_sac


# %%
# trained_ppo = train_ppo(total_timesteps=50_000)
# %%

# trained_ppo, trained_ddpg, trained_a2c, trained_td3, trained_sac = train_seq()


# # %%
# trained_ppo, trained_ddpg, trained_a2c, trained_td3, trained_sac = train_seq(model_name_list=None,
#                                                                              total_timesteps=total_timesteps)


# %%
# TODO: Turn the training into multi-processing to speed up the training process

def multi_train(model_name_list=None, total_timesteps=10_000) -> dict:
    """
    Train models in parallel
    :param model_name_list:  ['ppo', 'ddpg', 'a2c', 'td3', 'sac']
    :return: trained model in a dictionary with key as model name
    """

    if model_name_list is None:
        model_name_list = ['ppo', 'ddpg', 'a2c', 'td3', 'sac']
    global trained_td3, trained_sac, trained_a2c, trained_ddpg, trained_ppo
    print("==============Multi-Processing===========")
    # create a process pool for each model and get the results from each process
    pool = multiprocessing.Pool(processes=len(model_name_list))
    results = []
    for model_name in model_name_list:
        if model_name == 'ppo':
            results.append(pool.apply_async(train_ppo))
        elif model_name == 'ddpg':
            results.append(pool.apply_async(train_ddpg))
        elif model_name == 'a2c':
            results.append(pool.apply_async(train_a2c))
        elif model_name == 'td3':
            results.append(pool.apply_async(train_td3))
        elif model_name == 'sac':
            results.append(pool.apply_async(train_sac))
        else:
            print("Model name not found: ", model_name)
    pool.close()
    pool.join()
    # get the results from each process
    trained_models = {}
    for i, model_name in enumerate(model_name_list):
        trained_models[model_name] = results[i].get()
    return trained_models


# In[ ]:
print('################# Trade #########################')
trade_data = data_split(processed_full, '2019-01-01', '2021-01-01')
e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)


def trade(trained_model, trade_gym):
    trade_data.head()
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=trade_gym)

    return df_account_value, df_actions


# %%


# In[ ]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')


def backtesting(df_account_value, model_name):
    print("\n ppo:")
    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + model_name + "_" + now + '.csv')


# %%


# In[ ]:


# baseline stats
# print("==============Get Baseline Stats===========")
# baseline_df = get_baseline(
#     ticker="^DJI",
#     start='2019-01-01',
#     end='2021-01-01')
# print(baseline_df)
# # %%
# stats = backtest_stats(baseline_df, value_col_name='close')
# print(stats)


# <a id='6.2'></a>
# ## 7.2 BackTestPlot
# %%

def get_baseline_returns(account_value, baseline_end=config.TEST_END_DATE, baseline_start=config.TEST_START_DATE,
                         baseline_ticker="^DJI"):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    baseline_returns = baseline_returns.fillna(0.0)
    return baseline_returns


def get_returns(
        account_value,
        baseline_start=config.TRADE_START_DATE,
        baseline_end=config.TRADE_END_DATE,
        baseline_ticker="^DJI",
        value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)
    baseline_returns = None

    # baseline_returns = get_baseline_returns(baseline_end, baseline_start, baseline_ticker, df)
    # baseline_df = get_baseline(
    #     ticker=baseline_ticker, start=baseline_start, end=baseline_end
    # )
    #
    # baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    # baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    # baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    # baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    # with pyfolio.plotting.plotting_context(font_scale=1.1):
    #     pyfolio.create_full_tear_sheet(
    #         returns=test_returns, benchmark_rets=baseline_returns, set_context=False
    #     )
    return test_returns, baseline_returns


def get_daily_return(df2, value_col_name="account_value"):
    df2 = deepcopy(df2)
    df2["daily_return"] = df2[value_col_name].pct_change(1)
    df2["date"] = pd.to_datetime(df2["date"])
    df2.set_index("date", inplace=True, drop=True)
    df2.index = df2.index.tz_localize("UTC")
    return pd.Series(df2["daily_return"], index=df2.index)


# In[ ]:


print("==============Compare to DJIA===========")


# get_ipython().run_line_magic('matplotlib', 'inline')
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX

# get_returns(df_account_value_ppo,
#             baseline_ticker='^DJI',
#             baseline_start='2019-01-01',
#             baseline_end='2021-01-01')
# # %%


def create_figure(df_account_value, baseline_returns, model_name, date_start, date_end, tick_number=ticker_num,
                  t_steps=total_steps):
    test_returns = get_returns(df_account_value, baseline_start=date_start,
                               baseline_end=date_end,
                               baseline_ticker="^DJI",
                               value_col_name="account_value")
    # %%
    f = pyfolio.create_returns_tear_sheet(returns=test_returns[0], benchmark_rets=baseline_returns, return_fig=True)
    f.savefig(f'./images/{str(model_name)}' + f'_steps_{t_steps}_stocks_{tick_number}.png')

    return test_returns, baseline_returns, f


# %%

training_batch_size = 128
total_start = time.time()
# %% PPO
print("\n============== Model PPO: ===========")
start = time.time()
print("============== Training PPO ===========")
trained_ppo = train_ppo(total_timesteps=total_steps, batch_size=training_batch_size)
print("============== Trading PPO ===========")
df_account_value_ppo, df_actions_ppo = trade(trained_ppo, e_trade_gym)
print("============== Backtesting PPO ===========")
backtesting(df_account_value=df_account_value_ppo, model_name='ppo')
print("============== Plotting PPO ===========")
baseline_daily_returns = get_baseline_returns(df_account_value_ppo)
create_figure(df_account_value_ppo, baseline_daily_returns, 'ppo', config.TEST_START_DATE, config.TEST_END_DATE)
end = time.time()
training_times = {'ppo': end - start}
# %% DDPG
print("\n============== Model DDPG: ===========")
start = time.time()
print("============== Training DDPG ===========")
trained_ddpg = train_ddpg(total_timesteps=total_steps, batch_size=training_batch_size)
print("============== Trading DDPG ===========")
df_account_value_ddpg, df_actions_ddpg = trade(trained_ddpg, e_trade_gym)
print("============== Backtesting DDPG ===========")
backtesting(df_account_value=df_account_value_ddpg, model_name='ddpg')
print("============== Plotting DDPG ===========")
create_figure(df_account_value_ddpg, baseline_daily_returns, 'ddpg', start_date, end_date)
end = time.time()
training_times['ddpg'] = end - start

# %% A2C
print("\n============== Model A2C: ===========")
start = time.time()
print("============== Training A2C ===========")
trained_a2c = train_a2c(total_timesteps=total_steps)
print("============== Trading A2C ===========")
df_account_value_a2c, df_actions_a2c = trade(trained_a2c, e_trade_gym)
print("============== Backtesting A2C ===========")
backtesting(df_account_value=df_account_value_a2c, model_name='a2c')
print("============== Plotting A2C ===========")
create_figure(df_account_value_a2c, baseline_daily_returns, 'a2c', start_date, end_date)
end = time.time()
training_times['a2c'] = end - start

# %% TD3
print("\n============== Model TD3: ===========")
start = time.time()
print("============== Training TD3 ===========")
trained_td3 = train_td3(total_timesteps=total_steps)
print("============== Trading TD3 ===========")
df_account_value_td3, df_actions_td3 = trade(trained_td3, e_trade_gym)
print("============== Backtesting TD3 ===========")
backtesting(df_account_value=df_account_value_td3, model_name='td3')
print("============== Plotting TD3 ===========")
create_figure(df_account_value_td3, baseline_daily_returns, 'td3', start_date, end_date)
end = time.time()
training_times['td3'] = end - start

# %% SAC
print("\n============== Model SAC: ===========")
start = time.time()
print("============== Training SAC ===========")
trained_sac = train_sac(total_timesteps=total_steps)
print("============== Trading SAC ===========")
df_account_value_sac, df_actions_sac = trade(trained_sac, e_trade_gym)
print("============== Backtesting SAC ===========")
backtesting(df_account_value=df_account_value_sac, model_name='sac')
print("============== Plotting SAC ===========")
create_figure(df_account_value_sac, baseline_daily_returns, 'sac', start_date, end_date)
end = time.time()
training_times['sac'] = end - start

total_end = time.time()
training_times['total'] = total_end - total_start

times_json = json.dumps(training_times)
# save the times
with open('./training_times.json', 'w') as f:
    json.dump(times_json, f, indent=4)
    # f.write(times_json)


# # %%
# backtesting(df_account_value=df_account_value_a2c, model_name='a2c')
# backtesting(df_account_value=df_account_value_td3, model_name='td3')
#
# # %%
# create_figure(df_account_value_a2c, 'a2c', start_date, end_date)
# create_figure(df_account_value_td3, 'td3', start_date, end_date)


def main():
    print("==============Starting Training===========")


if __name__ == '__main__':
    main()
