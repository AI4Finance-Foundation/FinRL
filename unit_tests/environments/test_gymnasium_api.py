from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_cryptocurrency_trading.env_btc_ccxt import BitcoinEnv
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.meta.env_stock_trading.env_stocktrading_cashpenalty import (
    StockTradingEnvCashpenalty,
)
from finrl.meta.env_stock_trading.env_stocktrading_stoploss import (
    StockTradingEnvStopLoss,
)


def _stock_frame() -> pd.DataFrame:
    rows = []
    for day in range(6):
        for asset_index, ticker in enumerate(("AAA", "BBB")):
            price = 10.0 + day + asset_index
            rows.append(
                {
                    "date": f"2024-01-{day + 1:02d}",
                    "tic": ticker,
                    "open": price,
                    "close": price,
                    "high": price + 1,
                    "low": price - 1,
                    "volume": 1_000 + day,
                }
            )
    return pd.DataFrame(rows)


def _bitcoin_env() -> BitcoinEnv:
    return BitcoinEnv(
        price_ary=np.linspace(100.0, 105.0, num=6, dtype=np.float32).reshape(-1, 1),
        tech_ary=np.arange(42, dtype=np.float32).reshape(6, 7) + 1,
        time_frequency=1,
    )


def _crypto_env() -> CryptoEnv:
    return CryptoEnv(
        config={
            "price_array": np.linspace(100.0, 107.0, num=16, dtype=np.float32).reshape(
                8, 2
            ),
            "tech_array": np.arange(24, dtype=np.float32).reshape(8, 3) + 1,
        },
        lookback=2,
    )


def _cashpenalty_env() -> StockTradingEnvCashpenalty:
    return StockTradingEnvCashpenalty(
        df=_stock_frame(),
        random_start=False,
        cache_indicator_data=False,
        print_verbosity=1_000,
    )


def _stoploss_env() -> StockTradingEnvStopLoss:
    return StockTradingEnvStopLoss(
        df=_stock_frame(),
        random_start=False,
        cache_indicator_data=False,
        print_verbosity=1_000,
    )


ENV_FACTORIES = [
    pytest.param(_bitcoin_env, id="bitcoin"),
    pytest.param(_crypto_env, id="crypto"),
    pytest.param(_cashpenalty_env, id="cashpenalty"),
    pytest.param(_stoploss_env, id="stoploss"),
]


def test_crypto_get_state_returns_only_the_numpy_observation():
    env = _crypto_env()

    state = env.get_state()

    assert isinstance(state, np.ndarray)
    assert state.shape == (env.state_dim,)
    assert state.shape == (9,)


@pytest.mark.parametrize("env_factory", ENV_FACTORIES)
def test_gymnasium_reset_and_step_contract(env_factory):
    env = env_factory()

    reset_result = env.reset(seed=7)
    assert isinstance(reset_result, tuple)
    assert len(reset_result) == 2
    observation, reset_info = reset_result
    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert reset_info == {}

    step_result = env.step(
        np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    )
    assert isinstance(step_result, tuple)
    assert len(step_result) == 5
    next_observation, reward, terminated, truncated, step_info = step_result
    assert np.asarray(next_observation).shape == env.observation_space.shape
    assert np.isscalar(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert not terminated
    assert not truncated
    assert step_info == {}


@pytest.mark.parametrize("env_factory", ENV_FACTORIES)
def test_dummy_vec_env_can_reset_and_step(env_factory):
    vec_env = DummyVecEnv([env_factory])
    try:
        observation = vec_env.reset()
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (1, *vec_env.observation_space.shape)

        action = np.zeros(
            (1, *vec_env.action_space.shape), dtype=vec_env.action_space.dtype
        )
        next_observation, rewards, dones, infos = vec_env.step(action)

        assert next_observation.shape == observation.shape
        assert rewards.shape == (1,)
        assert dones.shape == (1,)
        assert not dones[0]
        assert len(infos) == 1
        assert isinstance(infos[0], dict)
    finally:
        vec_env.close()


@pytest.mark.parametrize("env_factory", ENV_FACTORIES)
def test_dummy_vec_env_reaches_done_and_resets(env_factory):
    vec_env = DummyVecEnv([env_factory])
    try:
        observation = vec_env.reset()
        action = np.zeros(
            (1, *vec_env.action_space.shape), dtype=vec_env.action_space.dtype
        )

        for _ in range(10):
            observation, _, dones, infos = vec_env.step(action)
            if dones[0]:
                break
        else:
            pytest.fail("environment did not terminate within ten steps")

        assert "terminal_observation" in infos[0]
        assert observation.shape == (1, *vec_env.observation_space.shape)
    finally:
        vec_env.close()
