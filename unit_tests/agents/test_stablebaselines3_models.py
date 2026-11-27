from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from sb3_contrib import CrossQ
from sb3_contrib import TQC

from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.stablebaselines3.models import DRLEnsembleAgent
from finrl.agents.stablebaselines3.models import MODEL_KWARGS
from finrl.agents.stablebaselines3.models import MODELS


@pytest.mark.parametrize("window_size", [1, 3])
def test_ensemble_prediction_saves_final_state(tmp_path, monkeypatch, window_size):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    dates = pd.date_range("2024-01-01", periods=window_size + 2).strftime("%Y-%m-%d")
    data = pd.DataFrame(
        [
            {"date": date, "tic": ticker, "close": close}
            for day, date in enumerate(dates)
            for ticker, close in [("AAA", 10.0 + day), ("BBB", 20.0 + day)]
        ]
    )
    agent = DRLEnsembleAgent(
        df=data,
        train_period=(dates[0], dates[1]),
        val_test_period=(dates[0], dates[-1]),
        rebalance_window=window_size,
        validation_window=1,
        stock_dim=2,
        hmax=10,
        initial_amount=1000,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        reward_scaling=1.0,
        state_space=5,
        action_space=2,
        tech_indicator_list=[],
        print_verbosity=10,
    )
    model = SimpleNamespace(predict=lambda obs: (np.zeros((1, 2)), None))
    state = agent.DRL_prediction(
        model, "ensemble", [], window_size, turbulence_threshold=None, initial=True
    )

    assert state == [1000, 10.0 + window_size, 20.0 + window_size, 0, 0]
    saved = pd.read_csv(results / f"last_state_ensemble_{window_size}.csv")
    assert saved["last_state"].tolist() == state


@pytest.fixture(scope="module")
def env():
    # A continuous-action env is all get_model needs; downloading real
    # market data for this is unnecessary weight for what is a
    # construction/API smoke test, not a strategy test.
    return gym.make("Pendulum-v1")


def test_tqc_and_crossq_are_registered():
    assert MODELS["tqc"] is TQC
    assert MODELS["crossq"] is CrossQ
    assert MODEL_KWARGS["tqc"] is config.TQC_PARAMS
    assert MODEL_KWARGS["crossq"] is config.CROSSQ_PARAMS


@pytest.mark.parametrize("model_name,model_cls", [("tqc", TQC), ("crossq", CrossQ)])
def test_get_model_builds_the_right_class(env, model_name, model_cls):
    agent = DRLAgent(env=env)
    model = agent.get_model(model_name, verbose=0)
    assert isinstance(model, model_cls)


@pytest.mark.parametrize("model_name", ["tqc", "crossq"])
def test_get_model_then_learn_runs_a_few_steps(env, model_name):
    # Same drop-in `predict`/`learn` API SAC/TD3 already exercise elsewhere
    # in this file's siblings -- this only pins that TQC/CrossQ do not
    # break DRLAgent's construction path, not that they trade well.
    agent = DRLAgent(env=env)
    model = agent.get_model(model_name, verbose=0)
    DRLAgent.train_model(
        model, tb_log_name=model_name, total_timesteps=64, callbacks=None
    )
