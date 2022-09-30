from __future__ import annotations

import numpy as np
import pytest

from finrl.meta.env_stock_trading.env_stocktrading_cashpenalty import (
    StockTradingEnvCashpenalty,
)
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]


@pytest.fixture(scope="session")
def indicator_list():
    return ["open", "close", "high", "low", "volume"]


@pytest.fixture(scope="session")
def data(ticker_list):
    return YahooDownloader(
        start_date="2019-01-01", end_date="2019-02-01", ticker_list=ticker_list
    ).fetch_data()


def test_zero_step(data, ticker_list):
    # Prove that zero actions results in zero stock buys, and no price changes
    init_amt = 1e6
    env = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    _ = env.reset()

    # step with all zeros
    for i in range(2):
        actions = np.zeros(len(ticker_list))
        next_state, _, _, _ = env.step(actions)
        cash = next_state[0]
        holdings = next_state[1 : 1 + len(ticker_list)]
        asset_value = env.account_information["asset_value"][-1]
        total_assets = env.account_information["total_assets"][-1]

        assert cash == init_amt
        assert init_amt == total_assets

        assert np.sum(holdings) == 0
        assert asset_value == 0

        assert env.current_step == i + 1


def test_patient(data, ticker_list):
    # Prove that we just not buying any new assets if running out of cash and the cycle is not ended
    aapl_first_close = data[data["tic"] == "AAPL"].head(1)["close"].values[0]
    init_amt = aapl_first_close
    hmax = aapl_first_close * 100
    env = StockTradingEnvCashpenalty(
        df=data,
        initial_amount=init_amt,
        hmax=hmax,
        cache_indicator_data=False,
        patient=True,
        random_start=False,
    )
    _ = env.reset()

    actions = np.array([1.0, 1.0])
    next_state, _, is_done, _ = env.step(actions)
    holdings = next_state[1 : 1 + len(ticker_list)]

    assert not is_done
    assert np.sum(holdings) == 0


@pytest.mark.xfail(reason="Not implemented")
def test_cost_penalties():
    raise NotImplementedError


@pytest.mark.xfail(reason="Not implemented")
def test_purchases():
    raise NotImplementedError


@pytest.mark.xfail(reason="Not implemented")
def test_gains():
    raise NotImplementedError


@pytest.mark.skip(reason="this test is not working correctly")
def test_validate_caching(data):
    # prove that results with or without caching don't change anything
    init_amt = 1e6
    env_uncached = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=False
    )
    env_cached = StockTradingEnvCashpenalty(
        df=data, initial_amount=init_amt, cache_indicator_data=True
    )
    _ = env_uncached.reset()
    _ = env_cached.reset()
    for i in range(10):
        actions = np.random.uniform(low=-1, high=1, size=2)
        print(f"actions: {actions}")
        un_state, un_reward, _, _ = env_uncached.step(actions)
        ca_state, ca_reward, _, _ = env_cached.step(actions)

        assert un_state == ca_state
        assert un_reward == ca_reward
