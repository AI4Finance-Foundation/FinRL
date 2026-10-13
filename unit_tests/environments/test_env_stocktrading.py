from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def make_env(prices, indicators=None, cost=0.001):
    """StockTradingEnv on synthetic prices of shape (days, stocks), no downloads."""
    indicators = indicators or {}
    n_days, n_stocks = prices.shape
    rows = []
    for day in range(n_days):
        for stock in range(n_stocks):
            row = {"date": f"2020-01-{day + 1:02d}", "tic": f"S{stock}"}
            row["close"] = prices[day, stock]
            for name, values in indicators.items():
                row[name] = values[day, stock]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.index = df["date"].factorize()[0]
    return StockTradingEnv(
        df=df,
        stock_dim=n_stocks,
        hmax=100,
        initial_amount=10_000,
        num_stock_shares=[0] * n_stocks,
        buy_cost_pct=[cost] * n_stocks,
        sell_cost_pct=[cost] * n_stocks,
        reward_scaling=1.0,
        state_space=1 + 2 * n_stocks + len(indicators) * n_stocks,
        action_space=n_stocks,
        tech_indicator_list=list(indicators),
    )


@pytest.fixture
def prices():
    return np.array([[50.0, 20.0], [51.0, 21.0], [52.0, 19.0], [53.0, 20.0]])


def shares(state, n_stocks=2):
    return list(state[1 + n_stocks : 1 + 2 * n_stocks])


def test_indicator_equal_to_one_does_not_block_trading(prices):
    # The first technical indicator of a stock used to be read as a
    # "cannot trade" flag, so a value of exactly 1.0 blocked every trade.
    env = make_env(prices, {"macd": np.ones_like(prices)})
    env.reset()
    state, *_ = env.step(np.array([0.5, 0.3]))
    assert shares(state) == [50, 30]


def test_trading_without_technical_indicators(prices):
    env = make_env(prices)
    env.reset()
    state, *_ = env.step(np.array([0.5, 0.3]))
    assert shares(state) == [50, 30]
    state, *_ = env.step(np.array([-0.2, -0.1]))
    assert shares(state) == [30, 20]


def test_missing_price_is_not_traded(prices):
    # A price of 0 marks missing data: no shares are bought for free and
    # held shares are not sold for nothing.
    prices[1, 1] = 0.0
    env = make_env(prices, {"macd": np.zeros_like(prices)})
    env.reset()
    state, *_ = env.step(np.array([0.5, 0.3]))
    cash_after_buy = state[0]
    state, *_ = env.step(np.array([0.0, 0.5]))
    assert shares(state) == [50, 30]
    state, *_ = env.step(np.array([0.0, -0.3]))
    assert shares(state) == [50, 0]
    assert state[0] > cash_after_buy
