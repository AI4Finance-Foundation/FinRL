"""Regression tests for ElegantRL DRLAgent.DRL_prediction (issue #1196)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("elegantrl")

from finrl.agents.elegantrl.models import DRLAgent  # noqa: E402
from finrl.meta.env_stock_trading.env_stocktrading_np import (  # noqa: E402
    StockTradingEnv,
)


@pytest.fixture
def env_config():
    rng = np.random.default_rng(0)
    n_days, n_stocks, n_tech = 12, 3, 4
    return {
        "price_array": rng.uniform(10, 100, size=(n_days, n_stocks)),
        "tech_array": rng.normal(size=(n_days, n_stocks * n_tech)),
        "turbulence_array": rng.uniform(0, 10, size=n_days),
        "if_train": False,
    }


def test_drl_prediction_runs_on_gymnasium_env(tmp_path, env_config):
    env = StockTradingEnv(config=env_config)

    # ElegantRL saves the whole actor module to <cwd>/act.pth.
    actor = torch.nn.Sequential(
        torch.nn.Linear(env.state_dim, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, env.action_dim),
        torch.nn.Tanh(),
    )
    torch.save(actor, tmp_path / "act.pth")

    with torch.no_grad():
        assets = DRLAgent.DRL_prediction(
            model_name="ppo",
            cwd=str(tmp_path),
            net_dimension=2**7,
            environment=env,
            env_args=env_config,
        )

    # initial asset + one entry per step until the episode ends
    assert len(assets) == env.max_step + 1
    assert np.all(np.isfinite(assets))
