from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def make_env(shares, **kwargs):
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
        **kwargs,
    )


@pytest.mark.parametrize("shares", [[0], [5], [5, 3]])
@pytest.mark.parametrize("shares_type", [list, np.array], ids=["list", "array"])
def test_initial_holdings_contribute_to_state_and_reward(shares, shares_type):
    env = make_env(shares_type(shares))
    stock_dim = len(shares)
    holdings = slice(stock_dim + 1, 2 * stock_dim + 1)
    expected_state = (
        [1000] + [10.0 + i for i in range(stock_dim)] + shares + [0.0] * stock_dim
    )
    np.testing.assert_array_equal(env.state, expected_state)
    assert env.state[holdings] == shares

    state, _ = env.reset()
    np.testing.assert_array_equal(state, expected_state)
    assert state[holdings] == shares
    initial_value = 1000 + sum((10 + i) * count for i, count in enumerate(shares))
    assert env.asset_memory == [initial_value]

    state, reward, _, _, _ = env.step(np.zeros(stock_dim))
    assert state[holdings] == shares
    assert reward == 2 * sum(shares)
    assert env.asset_memory == [initial_value, initial_value + reward]


@pytest.mark.parametrize("shares", [[0], [5], [5, 3]])
@pytest.mark.parametrize("state_type", [list, np.array], ids=["list", "array"])
def test_previous_state_restores_cash_and_holdings(shares, state_type):
    stock_dim = len(shares)
    previous_values = (
        [750] + [1.0 + i for i in range(stock_dim)] + shares + [-1.0] * stock_dim
    )
    previous_state = state_type(previous_values)
    env = make_env([0] * stock_dim, initial=False, previous_state=previous_state)
    expected_state = (
        [750] + [10.0 + i for i in range(stock_dim)] + shares + [0.0] * stock_dim
    )
    np.testing.assert_array_equal(env.state, expected_state)

    state, _ = env.reset()
    np.testing.assert_array_equal(state, expected_state)
    initial_value = 750 + sum((10 + i) * count for i, count in enumerate(shares))
    assert env.asset_memory == [initial_value]

    state, reward, _, _, _ = env.step(np.zeros(stock_dim))
    expected_next_state = (
        [750] + [12.0 + i for i in range(stock_dim)] + shares + [0.0] * stock_dim
    )
    np.testing.assert_array_equal(state, expected_next_state)
    assert reward == 2 * sum(shares)
    assert env.asset_memory == [initial_value, initial_value + reward]

    env.reset()
    state, _, _, _, _ = env.step(-np.ones(stock_dim))
    assert state[0] == initial_value
    assert state[stock_dim + 1 : 2 * stock_dim + 1] == [0] * stock_dim

    state, _ = env.reset()
    np.testing.assert_array_equal(state, expected_state)
    assert env.asset_memory == [initial_value]
    np.testing.assert_array_equal(previous_state, previous_values)


@pytest.mark.parametrize("shares_type", [list, np.array], ids=["list", "array"])
def test_reset_restores_single_stock_initial_holdings_after_selling(shares_type):
    shares = shares_type([5])
    env = make_env(shares)
    state, _, _, _, _ = env.step(np.array([-1.0]))
    assert state[0] == 1050
    assert state[2] == 0

    state, _ = env.reset()
    assert state[0] == 1000
    assert state[2] == 5
    np.testing.assert_array_equal(shares, [5])
    assert env.asset_memory == [1050]
