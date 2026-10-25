import random

import pandas as pd
import pytest

from finrl.meta.env_stock_trading.env_stocktrading_cashpenalty import (
    StockTradingEnvCashpenalty,
)
from finrl.meta.env_stock_trading.env_stocktrading_stoploss import (
    StockTradingEnvStopLoss,
)


@pytest.mark.parametrize(
    "env_type", [StockTradingEnvCashpenalty, StockTradingEnvStopLoss]
)
def test_random_start_uses_environment_seed_without_changing_global_rng(env_type):
    dates = pd.date_range("2024-01-01", periods=12)
    data = pd.DataFrame(
        {
            "date": dates,
            "tic": "TEST",
            "open": 1.0,
            "close": 1.0,
            "high": 1.0,
            "low": 1.0,
            "volume": 100,
        }
    )
    env = env_type(df=data, cache_indicator_data=False, random_start=True)
    global_state = random.getstate()

    env.reset(seed=123)
    first_start = env.starting_point
    env.reset(seed=123)

    assert env.starting_point == first_start
    assert random.getstate() == global_state

    env.seed(321)
    env.reset()
    next_start = env.starting_point
    env.seed(321)
    env.reset()

    assert env.starting_point == next_start
    assert random.getstate() == global_state
