from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def make_env(shares):
    stock_dim = len(shares)
    data = pd.DataFrame(
        [
            {
                "date": f"2024-01-0{day + 1}",
                "tic": f"STOCK{stock}",
                "close": 10.0 + stock + 2 * day,
                "indicator": 0.0,
            }
            for day in range(2)
            for stock in range(stock_dim)
        ],
        index=np.repeat(np.arange(2), stock_dim),
    )
    return StockTradingEnv(
        df=data,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1000,
        num_stock_shares=shares,
        buy_cost_pct=[0.0] * stock_dim,
        sell_cost_pct=[0.0] * stock_dim,
        reward_scaling=1.0,
        state_space=1 + 3 * stock_dim,
        action_space=stock_dim,
        tech_indicator_list=["indicator"],
    )


@pytest.mark.parametrize("shares", [[0], [5], [5, 3]])
def test_initial_holdings_contribute_to_state_and_reward(shares):
    env = make_env(shares)
    stock_dim = len(shares)
    holdings = slice(stock_dim + 1, 2 * stock_dim + 1)
    assert env.state[holdings] == shares

    state, _ = env.reset()
    assert state[holdings] == shares
    initial_value = 1000 + sum((10 + i) * count for i, count in enumerate(shares))
    assert env.asset_memory == [initial_value]

    state, reward, _, _, _ = env.step(np.zeros(stock_dim))
    assert state[holdings] == shares
    assert reward == 2 * sum(shares)
    assert env.asset_memory == [initial_value, initial_value + reward]


def test_reset_restores_single_stock_initial_holdings_after_selling():
    shares = [5]
    env = make_env(shares)
    state, _, _, _, _ = env.step(np.array([-1.0]))
    assert state[0] == 1050
    assert state[2] == 0

    state, _ = env.reset()
    assert state[0] == 1000
    assert state[2] == 5
    assert shares == [5]
    assert env.asset_memory == [1050]
