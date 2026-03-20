"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to use the agents we trained to do backtest, and compare with baselines such as
Mean Variance Optimization and DJIA index.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# %% Part 1. Load data

train = pd.read_csv("train_data.csv")
trade = pd.read_csv("trade_data.csv")

train = train.set_index(train.columns[0])
train.index.names = [""]
trade = trade.set_index(trade.columns[0])
trade.index.names = [""]

# %% Part 2. Load trained agents

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

# %% Part 3. Backtesting - DRL agents

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_trade_gym = StockTradingEnv(
    df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
)

df_account_value_a2c, df_actions_a2c = (
    DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)
    if if_using_a2c
    else (None, None)
)

df_account_value_ddpg, df_actions_ddpg = (
    DRLAgent.DRL_prediction(model=trained_ddpg, environment=e_trade_gym)
    if if_using_ddpg
    else (None, None)
)

df_account_value_ppo, df_actions_ppo = (
    DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)
    if if_using_ppo
    else (None, None)
)

df_account_value_td3, df_actions_td3 = (
    DRLAgent.DRL_prediction(model=trained_td3, environment=e_trade_gym)
    if if_using_td3
    else (None, None)
)

df_account_value_sac, df_actions_sac = (
    DRLAgent.DRL_prediction(model=trained_sac, environment=e_trade_gym)
    if if_using_sac
    else (None, None)
)

# %% Part 4. Mean Variance Optimization baseline


def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")


def StockReturnsComputing(StockPrice, Rows, Columns):
    StockReturn = np.zeros([Rows - 1, Columns])
    for j in range(Columns):
        for i in range(Rows - 1):
            StockReturn[i, j] = (
                (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
            ) * 100
    return StockReturn


StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

arStockPrices = np.asarray(StockData)
[Rows, Cols] = arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

meanReturns = np.mean(arReturns, axis=0)
covReturns = np.cov(arReturns, rowvar=False)

np.set_printoptions(precision=3, suppress=True)
print("Mean returns of assets in portfolio\n", meanReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array(
    [1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))]
)

LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])

# %% Part 5. DJIA index baseline

import yfinance as yf

df_dji = yf.download("^DJI", start=TRADE_START_DATE, end=TRADE_END_DATE)
df_dji = df_dji[["Close"]].reset_index()
df_dji.columns = ["date", "close"]
df_dji["date"] = df_dji["date"].astype(str)
fst_day = df_dji["close"].iloc[0]
dji = pd.merge(
    df_dji["date"],
    df_dji["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

# %% Part 6. Compare results

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c
    else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg
    else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo
    else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3
    else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac
    else None
)

result = pd.DataFrame(
    {
        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
        "mvo": MVO_result["Mean Var"],
        "dji": dji["close"],
    }
)

print("\n=== Backtest Results ===")
print(result)

# %% Part 7. Plot

plt.rcParams["figure.figsize"] = (15, 5)
plt.figure()
result.plot()
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to backtest_result.png")
