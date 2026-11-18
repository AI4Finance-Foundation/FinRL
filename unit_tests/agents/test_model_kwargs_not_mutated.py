from __future__ import annotations

import gymnasium as gym
import pytest

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.agents.stablebaselines3.models import DRLEnsembleAgent
from finrl.agents.stablebaselines3.models import MODEL_KWARGS


@pytest.fixture(scope="module")
def env():
    return gym.make("Pendulum-v1")


def _kwargs():
    return {
        "buffer_size": 100,
        "learning_starts": 10,
        "action_noise": "normal",
    }


def test_drl_agent_get_model_leaves_model_kwargs_untouched(env):
    # get_model replaced the "normal" string with a NormalActionNoise object
    # inside the caller's dict, so a second call with the same dict raised
    # KeyError when it looked the object up in NOISE.
    kwargs = _kwargs()
    agent = DRLAgent(env=env)
    agent.get_model("ddpg", model_kwargs=kwargs, verbose=0)
    assert kwargs == _kwargs()
    agent.get_model("ddpg", model_kwargs=kwargs, verbose=0)


def test_ensemble_get_model_leaves_model_kwargs_untouched(env):
    kwargs = _kwargs()
    DRLEnsembleAgent.get_model("ddpg", env, model_kwargs=kwargs, verbose=0)
    assert kwargs == _kwargs()
    DRLEnsembleAgent.get_model("ddpg", env, model_kwargs=kwargs, verbose=0)


def test_module_defaults_are_not_mutated(env):
    # With model_kwargs=None the defaults from finrl.config are used; they
    # must survive a get_model call unchanged as well.
    before = {name: dict(params) for name, params in MODEL_KWARGS.items()}
    DRLAgent(env=env).get_model("td3", verbose=0)
    assert MODEL_KWARGS == before
