from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.meta.env_cryptocurrency_trading.env_btc_ccxt import BitcoinEnv
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv


@pytest.mark.parametrize("env_class", [BitcoinEnv, CryptoEnv])
def test_crypto_env_completes_and_resets_episode(env_class):
    prices = np.array([[100.0], [101.0], [110.0], [120.0], [130.0]])
    if env_class is BitcoinEnv:
        trading_env = BitcoinEnv(
            price_ary=prices, tech_ary=np.zeros((len(prices), 7)), time_frequency=1
        )
        steps = len(prices) - 1
    else:
        trading_env = CryptoEnv(
            {"price_array": prices, "tech_array": np.zeros_like(prices)}, lookback=2
        )
        steps = len(prices) - 2

    env = DummyVecEnv([lambda: trading_env])
    try:
        initial = env.reset()
        action = np.ones((1, 1), dtype=np.float32)
        for step in range(steps):
            obs, rewards, done, info = env.step(action)
            assert bool(done[0]) == (step == steps - 1)

        assert rewards[0] > 0
        assert trading_env.observation_space.contains(info[0]["terminal_observation"])
        np.testing.assert_array_equal(obs, initial)

        for _ in range(steps):
            _, rewards, done, _ = env.step(np.zeros_like(action))
            assert rewards[0] == 0
        assert done[0]
    finally:
        env.close()
