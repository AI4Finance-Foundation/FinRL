# %%
# !/usr/bin/env python
# coding: utf-8
# %%
from __future__ import annotations

import datetime
from copy import deepcopy
from pprint import pprint

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import get_daily_return

# get_ipython().run_line_magic('matplotlib', 'inline')

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
abspath = os.path.realpath("__file__")
print(abspath)
dname = os.path.dirname(abspath)
os.chdir("/home/james/Dropbox/Investing/RL_Agent_Examples/FinRL/tutorials/")
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
print(f"Number of tickers: {len(tickers)}")
print(tickers)

# In[5]:

#
# df = YahooDownloader(start_date='2009-01-01',
#                      end_date='2022-09-01',
#                      ticker_list=tickers).fetch_data()


# In[6]:
# df.to_csv(f'./datasets/NAS_{len(tickers)}.csv', index=False)

# %%
df = pd.read_csv(f"./datasets/DOW30.csv")


# In[7]:


# df.shape
# df.to_csv('../datasets/DOW30.csv', index=False)


# In[8]:


df.head()

# In[9]:


df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")


# In[10]:


df.sort_values(["date", "tic"], ignore_index=True).head()


# # Part 4: Preprocess fundamental data
# - Import finanical data downloaded from Compustat via WRDS(Wharton Research Data Service)
# - Preprocess the dataset and calculate financial ratios
# - Add those ratios to the price data preprocessed in Part 3
# - Calculate price-related ratios such as P/E and P/B

# ## 4.1 Import the financial data

# In[11]:


# Import fundamental data from my GitHub repository
url = "https://raw.githubusercontent.com/mariko-sawada/FinRL_with_fundamental_data/main/dow_30_fundamental_wrds.csv"

fund = pd.read_csv(url)


# In[12]:


# Check the imported dataset
fund.head()


# ## 4.2 Specify items needed to calculate financial ratios
# - To learn more about the data description of the dataset, please check WRDS's website(https://wrds-www.wharton.upenn.edu/). Login will be required.

# In[13]:


# List items that are used to calculate financial ratios

items = [
    "datadate",  # Date
    "tic",  # Ticker
    "oiadpq",  # Quarterly operating income
    "revtq",  # Quartely revenue
    "niq",  # Quartely net income
    "atq",  # Total asset
    "teqq",  # Shareholder's equity
    "epspiy",  # EPS(Basic) incl. Extraordinary items
    "ceqq",  # Common Equity
    "cshoq",  # Common Shares Outstanding
    "dvpspq",  # Dividends per share
    "actq",  # Current assets
    "lctq",  # Current liabilities
    "cheq",  # Cash & Equivalent
    "rectq",  # Recievalbles
    "cogsq",  # Cost of  Goods Sold
    "invtq",  # Inventories
    "apq",  # Account payable
    "dlttq",  # Long term debt
    "dlcq",  # Debt in current liabilites
    "ltq",  # Liabilities
]

# Omit items that will not be used
fund_data = fund[items]


# In[14]:


# Rename column names for the sake of readability
fund_data = fund_data.rename(
    columns={
        "datadate": "date",  # Date
        "oiadpq": "op_inc_q",  # Quarterly operating income
        "revtq": "rev_q",  # Quartely revenue
        "niq": "net_inc_q",  # Quartely net income
        "atq": "tot_assets",  # Assets
        "teqq": "sh_equity",  # Shareholder's equity
        "epspiy": "eps_incl_ex",  # EPS(Basic) incl. Extraordinary items
        "ceqq": "com_eq",  # Common Equity
        "cshoq": "sh_outstanding",  # Common Shares Outstanding
        "dvpspq": "div_per_sh",  # Dividends per share
        "actq": "cur_assets",  # Current assets
        "lctq": "cur_liabilities",  # Current liabilities
        "cheq": "cash_eq",  # Cash & Equivalent
        "rectq": "receivables",  # Receivalbles
        "cogsq": "cogs_q",  # Cost of  Goods Sold
        "invtq": "inventories",  # Inventories
        "apq": "payables",  # Account payable
        "dlttq": "long_debt",  # Long term debt
        "dlcq": "short_debt",  # Debt in current liabilites
        "ltq": "tot_liabilities",  # Liabilities
    }
)


# In[15]:


# Check the data
fund_data.head()


# ## 4.3 Calculate financial ratios
# - For items from Profit/Loss statements, we calculate LTM (Last Twelve Months) and use them to derive profitability related ratios such as Operating Maring and ROE. For items from balance sheets, we use the numbers on the day.
# - To check the definitions of the financial ratios calculated here, please refer to CFI's website: https://corporatefinanceinstitute.com/resources/knowledge/finance/financial-ratios/

# In[16]:


# Calculate financial ratios
date = pd.to_datetime(fund_data["date"], format="%Y%m%d")

tic = fund_data["tic"].to_frame("tic")

# Profitability ratios
# Operating Margin
OPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="OPM")
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        OPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        OPM.iloc[i] = np.nan
    else:
        OPM.iloc[i] = np.sum(fund_data["op_inc_q"].iloc[i - 3 : i]) / np.sum(
            fund_data["rev_q"].iloc[i - 3 : i]
        )

# Net Profit Margin
NPM = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="NPM")
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        NPM[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        NPM.iloc[i] = np.nan
    else:
        NPM.iloc[i] = np.sum(fund_data["net_inc_q"].iloc[i - 3 : i]) / np.sum(
            fund_data["rev_q"].iloc[i - 3 : i]
        )

# Return On Assets
ROA = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROA")
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROA[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROA.iloc[i] = np.nan
    else:
        ROA.iloc[i] = (
            np.sum(fund_data["net_inc_q"].iloc[i - 3 : i])
            / fund_data["tot_assets"].iloc[i]
        )

# Return on Equity
ROE = pd.Series(np.empty(fund_data.shape[0], dtype=object), name="ROE")
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        ROE[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        ROE.iloc[i] = np.nan
    else:
        ROE.iloc[i] = (
            np.sum(fund_data["net_inc_q"].iloc[i - 3 : i])
            / fund_data["sh_equity"].iloc[i]
        )

# For calculating valuation ratios in the next subpart, calculate per share items in advance
# Earnings Per Share
EPS = fund_data["eps_incl_ex"].to_frame("EPS")

# Book Per Share
BPS = (fund_data["com_eq"] / fund_data["sh_outstanding"]).to_frame(
    "BPS"
)  # Need to check units

# Dividend Per Share
DPS = fund_data["div_per_sh"].to_frame("DPS")

# Liquidity ratios
# Current ratio
cur_ratio = (fund_data["cur_assets"] / fund_data["cur_liabilities"]).to_frame(
    "cur_ratio"
)

# Quick ratio
quick_ratio = (
    (fund_data["cash_eq"] + fund_data["receivables"]) / fund_data["cur_liabilities"]
).to_frame("quick_ratio")

# Cash ratio
cash_ratio = (fund_data["cash_eq"] / fund_data["cur_liabilities"]).to_frame(
    "cash_ratio"
)


# Efficiency ratios
# Inventory turnover ratio
inv_turnover = pd.Series(
    np.empty(fund_data.shape[0], dtype=object), name="inv_turnover"
)
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        inv_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        inv_turnover.iloc[i] = np.nan
    else:
        inv_turnover.iloc[i] = (
            np.sum(fund_data["cogs_q"].iloc[i - 3 : i])
            / fund_data["inventories"].iloc[i]
        )

# Receivables turnover ratio
acc_rec_turnover = pd.Series(
    np.empty(fund_data.shape[0], dtype=object), name="acc_rec_turnover"
)
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_rec_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_rec_turnover.iloc[i] = np.nan
    else:
        acc_rec_turnover.iloc[i] = (
            np.sum(fund_data["rev_q"].iloc[i - 3 : i])
            / fund_data["receivables"].iloc[i]
        )

# Payable turnover ratio
acc_pay_turnover = pd.Series(
    np.empty(fund_data.shape[0], dtype=object), name="acc_pay_turnover"
)
for i in range(0, fund_data.shape[0]):
    if i - 3 < 0:
        acc_pay_turnover[i] = np.nan
    elif fund_data.iloc[i, 1] != fund_data.iloc[i - 3, 1]:
        acc_pay_turnover.iloc[i] = np.nan
    else:
        acc_pay_turnover.iloc[i] = (
            np.sum(fund_data["cogs_q"].iloc[i - 3 : i]) / fund_data["payables"].iloc[i]
        )

## Leverage financial ratios
# Debt ratio
debt_ratio = (fund_data["tot_liabilities"] / fund_data["tot_assets"]).to_frame(
    "debt_ratio"
)

# Debt to Equity ratio
debt_to_equity = (fund_data["tot_liabilities"] / fund_data["sh_equity"]).to_frame(
    "debt_to_equity"
)


# In[17]:


# Create a dataframe that merges all the ratios
ratios = pd.concat(
    [
        date,
        tic,
        OPM,
        NPM,
        ROA,
        ROE,
        EPS,
        BPS,
        DPS,
        cur_ratio,
        quick_ratio,
        cash_ratio,
        inv_turnover,
        acc_rec_turnover,
        acc_pay_turnover,
        debt_ratio,
        debt_to_equity,
    ],
    axis=1,
)


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
list_date = list(pd.date_range(df["date"].min(), df["date"].max()))
combination = list(itertools.product(list_date, list_ticker))

# Merge stock price data and ratios into one dataframe
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    df, on=["date", "tic"], how="left"
)
processed_full = processed_full.merge(final_ratios, how="left", on=["date", "tic"])
processed_full = processed_full.sort_values(["tic", "date"])

# Backfill the ratio data to make them daily
processed_full = processed_full.bfill(axis="rows")


# ## 4.6 Calculate market valuation ratios using daily stock price data

# In[24]:


# Calculate P/E, P/B and dividend yield using daily closing price
processed_full["PE"] = processed_full["close"] / processed_full["EPS"]
processed_full["PB"] = processed_full["close"] / processed_full["BPS"]
processed_full["Div_yield"] = processed_full["DPS"] / processed_full["close"]

# Drop per share items used for the above calculation
processed_full = processed_full.drop(columns=["day", "EPS", "BPS", "DPS"])
# Replace NAs infinite values with zero
processed_full = processed_full.copy()
processed_full = processed_full.fillna(0)
processed_full = processed_full.replace(np.inf, 0)


# In[25]:


# Check the final data
processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)


# <a id='4'></a>
# # Part 5. A Market Environment in OpenAI Gym-style
# The training process involves observing stock price change, taking an action and reward's calculation. By interacting with the market environment, the agent will eventually derive a trading strategy that may maximize (expected) rewards.
#
# Our market environment, based on OpenAI Gym, simulates stock markets with historical market data.

# ## 5.1 Data Split
# - Training data period: 2009-01-01 to 2019-01-01
# - Trade data period: 2019-01-01 to 2020-12-31

# In[26]:


train_data = data_split(processed_full, "2009-01-01", "2019-01-01")
trade_data = data_split(processed_full, "2019-01-01", "2021-01-01")
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


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots=False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct)
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // self.state[index + 1]
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.initial_amount
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )

                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.close()

            # Add outputs to logger interface
            # logger.record("environment/portfolio_value", end_total_asset)
            # logger.record("environment/total_reward", tot_reward)
            # logger.record("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record("environment/total_cost", self.cost)
            # logger.record("environment/total_trades", self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if (
                self.turbulence_threshold is not None
                and self.turbulence >= self.turbulence_threshold
            ):
                actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # print("begin_total_asset:{}".format(begin_total_asset))

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # state: s -> s+1
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + [0] * self.stock_dim
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
            )
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


# In[30]:


ratio_list = [
    "OPM",
    "NPM",
    "ROA",
    "ROE",
    "cur_ratio",
    "quick_ratio",
    "cash_ratio",
    "inv_turnover",
    "acc_rec_turnover",
    "acc_pay_turnover",
    "debt_ratio",
    "debt_to_equity",
    "PE",
    "PB",
    "Div_yield",
]

stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2 * stock_dimension + len(ratio_list) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


# In[31]:


# Parameters for the environment
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": ratio_list,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
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


def train_ppo():
    global agent, model_ppo
    print("==============Model 1: PPO===========")
    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
    trained_ppo = agent.train_model(
        model=model_ppo, tb_log_name="ppo", total_timesteps=50000
    )

    return trained_ppo


# In[ ]: Training PPO


# ### Model 2: DDPG

# In[ ]:


def train_ddpg():
    global agent, model_ddpg
    print("==============Model 2: DDPG===========")
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001,
    }
    model_ddpg = agent.get_model("ddpg")  # , model_kwargs=DDPG_PARAMS
    trained_ddpg = agent.train_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
    )

    return trained_ddpg


# print('################# Model 2: DDPG #########################')
#
# agent = DRLAgent(env=env_train)
# model_ddpg = agent.get_model("ddpg")
#
# trained_ddpg = agent.train_model(model=model_ddpg,
#                                  tb_log_name='ddpg',
#                                  total_timesteps=50000)

# ### Model 3: A2C
#

# In[ ]:


def train_a2c():
    global agent, model_a2c
    print("==============Model 3: A2C===========")
    agent = DRLAgent(env=env_train)
    A2C_PARAMS = {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
    }
    model_a2c = agent.get_model("a2c")  # model_kwargs=A2C_PARAMS
    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=50000
    )

    return trained_a2c


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


def train_td3():
    global agent, model_td3
    print("==============Model 4: TD3===========")
    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
    trained_td3 = agent.train_model(
        model=model_td3, tb_log_name="td3", total_timesteps=50000
    )

    return trained_td3


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


def train_sac():
    global agent, model_sac
    print("==============Model 5: SAC===========")
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    trained_sac = agent.train_model(
        model=model_sac, tb_log_name="sac", total_timesteps=50000
    )

    return trained_sac


# %% Training PPO

trained_ppo = train_ppo()
# %% Training DDPG

trained_ddpg = train_ddpg()
# %% Training A2C

trained_a2c = train_a2c()
# %%Training TD3

trained_td3 = train_td3()
# %% Training SAC

trained_sac = train_sac()
# %%
# TODO: Turn the training into multi-processing to speed up the training process
import multiprocessing as mp


def multi_train():
    print("==============Multi-Processing===========")
    pool = mp.Pool(mp.cpu_count())
    results = [
        pool.apply_async(train_ppo),
        pool.apply_async(train_ddpg),
        pool.apply_async(train_a2c),
        pool.apply_async(train_td3),
        pool.apply_async(train_sac),
    ]
    # add results from each process to a dictionary with the key as the model name
    trained_models = {}
    for result in results:
        trained_models[result.get().model_name] = result.get()
    return trained_models
    #
    # output = [p.get() for p in results]
    # return output


# print('################# Model 5: SAC #########################')
# agent = DRLAgent(env=env_train)
# SAC_PARAMS = {
#     "batch_size": 128,
#     "buffer_size": 1000000,
#     "learning_rate": 0.0001,
#     "learning_starts": 100,
#     "ent_coef": "auto_0.1",
# }
#
# model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
#
# # In[ ]:
#
#
# trained_sac = agent.train_model(model=model_sac,
#                                 tb_log_name='sac',
#                                 total_timesteps=80000)

# ## Trading

# In[ ]:

print("################# Trade #########################")
trade_data = data_split(processed_full, "2019-01-01", "2021-01-01")
e_trade_gym = StockTradingEnv(df=trade_data, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()


# In[ ]:


trade_data.head()

# In[ ]:


df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=e_trade_gym
)

df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg, environment=e_trade_gym
)

df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, environment=e_trade_gym
)

df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3, environment=e_trade_gym
)

df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac, environment=e_trade_gym
)

# In[ ]:


# <a id='6'></a>
# # Part 7: Backtest Our Strategy
# Backtesting plays a key role in evaluating the performance of a trading strategy. Automated backtesting tool is preferred because it reduces the human error. We usually use the Quantopian pyfolio package to backtest our trading strategies. It is easy to use and consists of various individual plots that provide a comprehensive image of the performance of a trading strategy.

# <a id='6.1'></a>
# ## 7.1 BackTestStats
# pass in df_account_value, this information is stored in env class
#

# In[ ]:


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

print("\n ppo:")
perf_stats_all_ppo = backtest_stats(account_value=df_account_value_ppo)
perf_stats_all_ppo = pd.DataFrame(perf_stats_all_ppo)
perf_stats_all_ppo.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_ppo_" + now + ".csv"
)

print("\n ddpg:")
perf_stats_all_ddpg = backtest_stats(account_value=df_account_value_ddpg)
perf_stats_all_ddpg = pd.DataFrame(perf_stats_all_ddpg)
perf_stats_all_ddpg.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_ddpg_" + now + ".csv"
)

print("\n a2c:")
perf_stats_all_a2c = backtest_stats(account_value=df_account_value_a2c)
perf_stats_all_a2c = pd.DataFrame(perf_stats_all_a2c)
perf_stats_all_a2c.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_a2c_" + now + ".csv"
)

print("\n atd3:")
perf_stats_all_td3 = backtest_stats(account_value=df_account_value_td3)
perf_stats_all_td3 = pd.DataFrame(perf_stats_all_td3)
perf_stats_all_td3.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_td3_" + now + ".csv"
)

print("\n sac:")
perf_stats_all_sac = backtest_stats(account_value=df_account_value_sac)
perf_stats_all_sac = pd.DataFrame(perf_stats_all_sac)
perf_stats_all_sac.to_csv(
    "./" + config.RESULTS_DIR + "/perf_stats_all_sac_" + now + ".csv"
)

# In[ ]:


# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(ticker="^DJI", start="2019-01-01", end="2021-01-01")
print(baseline_df)
# %%
stats = backtest_stats(baseline_df, value_col_name="close")
print(stats)

# <a id='6.2'></a>
# ## 7.2 BackTestPlot
# %%


def backtest_plot(
    account_value,
    baseline_start=config.TRADE_START_DATE,
    baseline_end=config.TRADE_END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    # with pyfolio.plotting.plotting_context(font_scale=1.1):
    #     pyfolio.create_full_tear_sheet(
    #         returns=test_returns, benchmark_rets=baseline_returns, set_context=False
    #     )
    return test_returns, baseline_returns


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

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    # with pyfolio.plotting.plotting_context(font_scale=1.1):
    #     pyfolio.create_full_tear_sheet(
    #         returns=test_returns, benchmark_rets=baseline_returns, set_context=False
    #     )
    return test_returns, baseline_returns


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


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
start_date = "2019-01-01"
end_date = "2021-01-01"

test_returns, baseline_returns = get_returns(
    df_account_value_ppo,
    baseline_start=start_date,
    baseline_end=end_date,
    baseline_ticker="^DJI",
    value_col_name="account_value",
)
matplotlib.use("Agg")

f = pyfolio.create_returns_tear_sheet(
    test_returns, benchmark_rets=baseline_returns, return_fig=True
)
f.savefig("./imgs/ppo_pyfolio_results.png")

# In[ ]:

test_returns, baseline_returns = get_returns(
    df_account_value_ddpg,
    baseline_start=start_date,
    baseline_end=end_date,
    baseline_ticker="^DJI",
    value_col_name="account_value",
)

f = pyfolio.create_returns_tear_sheet(
    test_returns, benchmark_rets=baseline_returns, return_fig=True
)
f.savefig("./imgs/ddpg_pyfolio_results.png")
# In[ ]:
test_returns, baseline_returns = get_returns(
    df_account_value_a2c,
    baseline_start=start_date,
    baseline_end=end_date,
    baseline_ticker="^DJI",
    value_col_name="account_value",
)

f = pyfolio.create_returns_tear_sheet(
    test_returns, benchmark_rets=baseline_returns, return_fig=True
)
f.savefig("./imgs/a2c_pyfolio_results.png")
# In[ ]:
test_returns, baseline_returns = get_returns(
    df_account_value_sac,
    baseline_start=start_date,
    baseline_end=end_date,
    baseline_ticker="^DJI",
    value_col_name="account_value",
)

f = pyfolio.create_returns_tear_sheet(
    test_returns, benchmark_rets=baseline_returns, return_fig=True
)
f.savefig("./imgs/sac_pyfolio_results.png")
# %%
test_returns, baseline_returns = get_returns(
    df_account_value_td3,
    baseline_start=start_date,
    baseline_end=end_date,
    baseline_ticker="^DJI",
    value_col_name="account_value",
)

f = pyfolio.create_returns_tear_sheet(
    test_returns, benchmark_rets=baseline_returns, return_fig=True
)
f.savefig("./imgs/td3_pyfolio_results.png")

# In[ ]:

# get_returns(df_account_value_ddpg,
#              baseline_ticker = '^DJI',
#              baseline_start = '2019-01-01',
#              baseline_end = '2021-01-01')
#
#
# # In[ ]:
#
#
# get_returns(df_account_value_a2c,
#              baseline_ticker = '^DJI',
#              baseline_start = '2019-01-01',
#              baseline_end = '2021-01-01')
#
#
#
#
#
# # In[ ]:
#
#
# get_returns(df_account_value_td3,
#              baseline_ticker = '^DJI',
#              baseline_start = '2019-01-01',
#              baseline_end = '2021-01-01')
#
#
# # In[ ]:
#
#
# get_returns(df_account_value_sac,
#              baseline_ticker = '^DJI',
#              baseline_start = '2019-01-01',
#              baseline_end = '2021-01-01')


# In[ ]:
