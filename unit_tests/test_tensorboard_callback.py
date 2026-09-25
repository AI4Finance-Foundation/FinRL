from __future__ import annotations

import numpy as np

from finrl.agents.stablebaselines3.models import TensorboardCallback


class _FakeLogger:
    """Capture ``logger.record(key, value)`` calls made by the callback."""

    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class _FakeBuffer:
    """Minimal stand-in for an SB3 rollout/replay buffer.

    ``rewards`` has shape ``(buffer_size, n_envs)`` like the real buffers.
    ``full``/``pos`` mirror SB3's fill bookkeeping so we can exercise the
    partially-filled replay-buffer case.
    """

    def __init__(self, rewards, pos=None, full=True):
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.full = full
        self.pos = self.rewards.shape[0] if pos is None else pos


class _FakeModel:
    """An SB3-like model exposing exactly one buffer plus a logger."""

    def __init__(self, rollout_buffer=None, replay_buffer=None):
        self.logger = _FakeLogger()
        if rollout_buffer is not None:
            self.rollout_buffer = rollout_buffer
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer


def _callback_for(model):
    callback = TensorboardCallback()
    callback.model = model  # `callback.logger` is a property -> model.logger
    return callback


def test_on_rollout_end_logs_on_policy_rollout_buffer():
    # A2C/PPO expose a rollout_buffer that is full at the end of a rollout.
    model = _FakeModel(rollout_buffer=_FakeBuffer([[1.0], [2.0], [3.0]]))
    _callback_for(model)._on_rollout_end()

    records = model.logger.records
    assert records["train/reward_min"] == 1.0
    assert records["train/reward_mean"] == 2.0
    assert records["train/reward_max"] == 3.0


def test_on_rollout_end_logs_off_policy_replay_buffer():
    # Regression for #1395: DDPG/TD3/SAC expose a replay_buffer rather than a
    # rollout_buffer. The old code assumed rollout_buffer, raising a KeyError
    # and logging None values for every off-policy rollout.
    model = _FakeModel(replay_buffer=_FakeBuffer([[10.0], [20.0], [30.0]]))
    _callback_for(model)._on_rollout_end()

    records = model.logger.records
    assert records["train/reward_min"] == 10.0
    assert records["train/reward_mean"] == 20.0
    assert records["train/reward_max"] == 30.0
    assert None not in records.values()


def test_on_rollout_end_ignores_unfilled_replay_buffer_tail():
    # A replay buffer is allocated up front; only its first ``pos`` rows hold
    # real transitions until it fills, so the zero tail must be ignored.
    model = _FakeModel(
        replay_buffer=_FakeBuffer([[5.0], [7.0], [0.0], [0.0]], pos=2, full=False)
    )
    _callback_for(model)._on_rollout_end()

    records = model.logger.records
    assert records["train/reward_min"] == 5.0
    assert records["train/reward_max"] == 7.0
    assert records["train/reward_mean"] == 6.0


def test_on_rollout_end_without_buffer_is_noop():
    # Defensive: a model exposing neither buffer must not raise or log.
    model = _FakeModel()
    callback = _callback_for(model)

    assert callback._on_rollout_end() is True
    assert model.logger.records == {}
