from __future__ import annotations

import gymnasium as gym
import pytest
from sb3_contrib import CrossQ
from sb3_contrib import TQC

from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.stablebaselines3.models import MODEL_KWARGS
from finrl.agents.stablebaselines3.models import MODELS


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
