"""
Stock NeurIPS2018 Part 3. Backtest

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introducing how to backtest trained agents against a point-in-time equal-weight
Dow basket and the DJIA index.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# %% Part 1. Load data

trade = pd.read_csv("trade_data.csv")

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
state_space = 1 + 2 * stock_dimension + (len(INDICATORS) + 1) * stock_dimension
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
    "member_col": "is_member",
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

# %% Part 4. Point-in-time equal-weight Dow basket baseline

# Rebalance over only the constituents active on each date. The prices remain
# available outside membership intervals so returns at index-entry dates are
# computed from the previous close, while the membership mask controls inclusion.
trade_prices = trade.pivot(index="date", columns="tic", values="close").sort_index()
trade_members = (
    trade.pivot(index="date", columns="tic", values="is_member")
    .reindex(index=trade_prices.index, columns=trade_prices.columns)
    .astype(bool)
)
daily_returns = trade_prices.pct_change().where(trade_members).mean(axis=1)
equal_weight_value = (1 + daily_returns.fillna(0)).cumprod() * 1_000_000
pit_equal_weight = pd.DataFrame({"portfolio_value": equal_weight_value})

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
        "point-in-time equal weight": pit_equal_weight["portfolio_value"],
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
