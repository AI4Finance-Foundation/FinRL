from __future__ import annotations

import numpy as np
import pytest
from stable_baselines3 import PPO

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


def test_load_from_file_runs_the_numpy_env_to_the_end(tmp_path):
    # DRL_prediction_load_from_file (the stable_baselines3 path of finrl.test)
    # unpacked the old gym 4-tuple from step() and a bare observation from
    # reset(), while the numpy environment speaks Gymnasium, so it raised
    # ValueError on the first step.
    n_days, n_stocks = 12, 3
    rng = np.random.default_rng(0)
    env = StockTradingEnv(
        config={
            "price_array": rng.uniform(10, 100, size=(n_days, n_stocks)),
            "tech_array": rng.normal(size=(n_days, 2 * n_stocks)),
            "turbulence_array": np.zeros(n_days),
            "if_train": False,
        }
    )
    model_path = tmp_path / "ppo"
    PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0).save(model_path)

    total_assets = DRLAgent.DRL_prediction_load_from_file("ppo", env, str(model_path))

    # the initial value plus one entry per trading day
    assert len(total_assets) == n_days
    assert total_assets[0] == pytest.approx(env.initial_total_asset)
    assert all(np.isfinite(total_assets))
