from __future__ import annotations

import numpy as np

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


def test_buy_leaves_room_for_transaction_cost():
    config = {
        "price_array": np.array([[100.0], [100.0]]),
        "tech_array": np.zeros((2, 1)),
        "turbulence_array": np.zeros(2),
        "if_train": False,
    }
    env = StockTradingEnv(config, initial_capital=1000)
    env.reset()
    env.step(np.array([1.0]))
    assert env.stocks[0] == 9
    assert env.amount >= 0
