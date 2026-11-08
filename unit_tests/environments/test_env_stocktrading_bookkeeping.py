from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

PRICES = np.array([[50.0, 20.0], [51.0, 21.0], [52.0, 19.0], [53.0, 20.0]])


def make_env(**kwargs):
    n_days, n_stocks = PRICES.shape
    rows = [
        {
            "date": f"2020-01-{day + 1:02d}",
            "tic": f"S{stock}",
            "close": PRICES[day, stock],
            "macd": 0.0,
        }
        for day in range(n_days)
        for stock in range(n_stocks)
    ]
    df = pd.DataFrame(rows)
    df.index = df["date"].factorize()[0]
    return StockTradingEnv(
        df=df,
        stock_dim=n_stocks,
        hmax=100,
        initial_amount=10_000,
        num_stock_shares=[0] * n_stocks,
        buy_cost_pct=[0.0] * n_stocks,
        sell_cost_pct=[0.0] * n_stocks,
        reward_scaling=1.0,
        state_space=1 + 3 * n_stocks,
        action_space=n_stocks,
        tech_indicator_list=["macd"],
        **kwargs,
    )


def test_reset_honours_the_seed():
    # reset(seed=...) accepted the argument and ignored it.
    env = make_env()
    env.reset(seed=123)
    first = env.np_random.random(3)
    env.reset(seed=123)
    second = env.np_random.random(3)
    env.reset(seed=456)
    third = env.np_random.random(3)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_state_memory_columns_follow_the_tickers():
    # save_state_memory hard-coded seven column names from a two-asset
    # Bitcoin/Gold competition and failed for any other universe.
    env = make_env()
    env.reset()
    env.step(np.array([0.5, 0.3]))
    env.step(np.array([0.0, 0.0]))

    df = env.save_state_memory()

    assert list(df.columns) == [
        "cash",
        "S0_close",
        "S1_close",
        "S0_shares",
        "S1_shares",
        "S0_macd",
        "S1_macd",
    ]
    assert list(df.index) == ["2020-01-01", "2020-01-02"]
    assert df["S0_shares"].tolist() == [50, 50]
    assert df["S1_shares"].tolist() == [30, 30]


def test_previous_state_default_is_not_shared_between_instances():
    a, b = make_env(), make_env()
    a.previous_state.append("marker")
    assert b.previous_state == []
