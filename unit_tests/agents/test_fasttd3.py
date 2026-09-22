"""Tests for FastTD3 (finrl.agents.fasttd3): the C51 Bellman projection and
GPU-resident replay buffer in isolation, plus a construction/API smoke test
against Pendulum-v1 mirroring the sibling TQC/CrossQ tests in this
directory.

The projection and buffer are the two pieces of FastTD3 that have no SB3
analogue and are worth testing on their own: a wrong Bellman projection
would silently corrupt every critic update without raising, and a buffer
that drops or overwrites the wrong transitions would too.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from finrl.agents.fasttd3 import FastTD3
from finrl.agents.fasttd3.buffer import GPUReplayBuffer
from finrl.agents.fasttd3.networks import project_distribution

N_ATOMS = 11
V_MIN, V_MAX = -5.0, 5.0
SUPPORT = torch.linspace(V_MIN, V_MAX, N_ATOMS)
GAMMA = 0.9


def _project(rewards, dones, next_probs, *, v_min=V_MIN, v_max=V_MAX, support=SUPPORT):
    return project_distribution(
        rewards,
        dones,
        next_probs,
        support=support,
        gamma=GAMMA,
        v_min=v_min,
        v_max=v_max,
    )


# --------------------------------------------------------------- projection


def test_projection_conserves_mass_and_stays_non_negative():
    torch.manual_seed(0)
    for _ in range(200):
        probs = torch.softmax(torch.randn(8, N_ATOMS), dim=1)
        rewards = torch.randn(8) * 3
        dones = (torch.rand(8) > 0.7).float()
        target = _project(rewards, dones, probs)
        assert (target >= -1e-6).all()
        assert torch.allclose(target.sum(dim=1), torch.ones(8), atol=1e-4)


def test_projection_puts_all_mass_on_the_exact_landing_atom():
    """A point-mass distribution shifted by an exact grid step lands exactly
    on its target atom -- the degenerate (lo == hi) case a naive
    linear-interpolation formula silently drops (see the module docstring's
    nudge comment)."""
    next_probs = torch.zeros(4, N_ATOMS)
    next_probs[:, 5] = 1.0  # atom 5 == support value 0.0
    rewards = torch.tensor([0.0, 1.0, -1.0, 100.0])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

    target = _project(rewards, dones, next_probs)

    assert torch.allclose(target[0, 5], torch.tensor(1.0), atol=1e-5)  # Tz = 0.0
    assert torch.allclose(target[1, 6], torch.tensor(1.0), atol=1e-5)  # Tz = 1.0
    assert torch.allclose(target[2, 4], torch.tensor(1.0), atol=1e-5)  # Tz = -1.0
    # done=1 clips the reward-100 target to v_max, landing on the top atom.
    assert torch.allclose(target[3, N_ATOMS - 1], torch.tensor(1.0), atol=1e-5)


def test_projection_interpolates_linearly_between_neighboring_atoms():
    next_probs = torch.zeros(1, N_ATOMS)
    next_probs[0, 5] = 1.0
    # gamma * 0.0 == 0, so Tz = reward = 0.25; delta_z = 1.0, atoms at 0.0
    # (index 5) and 1.0 (index 6) -> b = 5.25, 75% to the lower atom.
    target = _project(torch.tensor([0.25]), torch.tensor([0.0]), next_probs)

    assert torch.allclose(target[0, 5], torch.tensor(0.75), atol=1e-5)
    assert torch.allclose(target[0, 6], torch.tensor(0.25), atol=1e-5)


def test_projection_n_atoms_two_is_the_minimum_supported_case():
    support = torch.linspace(-1.0, 1.0, 2)
    probs = torch.tensor([[0.3, 0.7]])
    target = project_distribution(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        probs,
        support=support,
        gamma=1.0,
        v_min=-1.0,
        v_max=1.0,
    )
    assert torch.allclose(target, probs, atol=1e-5)


def test_a_terminal_transition_ignores_the_next_state_entirely():
    """done=1 means the target collapses to the reward alone (no bootstrap),
    matching the un-discounted terminal-value convention every Bellman
    backup uses -- the (1 - done) gate in the projection's Tz formula."""
    next_probs = torch.zeros(1, N_ATOMS)
    next_probs[0, 0] = 1.0  # would pull the target toward v_min if it mattered
    target = _project(torch.tensor([2.0]), torch.tensor([1.0]), next_probs)
    # Tz = reward + (1 - done) * gamma * z = 2.0 exactly, regardless of z.
    expected_atom = int(round((2.0 - V_MIN) / ((V_MAX - V_MIN) / (N_ATOMS - 1))))
    assert torch.allclose(target[0, expected_atom], torch.tensor(1.0), atol=1e-5)


# -------------------------------------------------------------------- buffer


def test_buffer_add_and_sample_round_trips_a_single_transition():
    buf = GPUReplayBuffer(capacity=10, obs_dim=3, action_dim=2, device="cpu")
    obs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    actions = np.array([[0.5, -0.5]], dtype=np.float32)
    rewards = np.array([1.5], dtype=np.float32)
    next_obs = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)

    buf.add(obs, actions, rewards, next_obs, dones)
    assert len(buf) == 1

    o, a, r, no, d = buf.sample(1)
    assert torch.allclose(o, torch.tensor(obs))
    assert torch.allclose(a, torch.tensor(actions))
    assert torch.allclose(r, torch.tensor(rewards))
    assert torch.allclose(no, torch.tensor(next_obs))
    assert torch.allclose(d, torch.tensor(dones))


def test_buffer_caps_at_capacity_and_wraps_like_a_ring():
    buf = GPUReplayBuffer(capacity=5, obs_dim=1, action_dim=1, device="cpu")
    for i in range(8):  # 8 > capacity: the oldest 3 get overwritten
        buf.add(
            np.full((1, 1), i, dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.array([float(i)], dtype=np.float32),
            np.full((1, 1), i, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
        )
    assert len(buf) == 5
    o, _, _, _, _ = buf.sample(5)
    # Only observations 3..7 should ever be reachable; the ring must never
    # serve up an overwritten (stale) slot.
    assert set(o.flatten().tolist()) <= {3.0, 4.0, 5.0, 6.0, 7.0}


def test_buffer_add_accepts_a_batch_larger_than_one_env_step():
    """The unit a vectorized env actually produces (num_envs > 1)."""
    buf = GPUReplayBuffer(capacity=100, obs_dim=2, action_dim=1, device="cpu")
    n = 4
    buf.add(
        np.random.randn(n, 2).astype(np.float32),
        np.random.randn(n, 1).astype(np.float32),
        np.random.randn(n).astype(np.float32),
        np.random.randn(n, 2).astype(np.float32),
        np.zeros(n, dtype=np.float32),
    )
    assert len(buf) == n


# ----------------------------------------------------------------- end-to-end


@pytest.fixture()
def vector_env():
    # A continuous-action vector env is all FastTD3 needs -- downloading
    # real market data for this is unnecessary weight for what is a
    # construction/API smoke test, not a strategy test (same rationale as
    # the sibling TQC/CrossQ tests in this directory).
    env = gym.make_vec("Pendulum-v1", num_envs=2)
    yield env
    env.close()


def test_fasttd3_runs_a_few_steps_and_predicts_on_the_action_space(vector_env):
    model = FastTD3(
        vector_env,
        actor_net_arch=(16,),
        critic_net_arch=(16,),
        buffer_size=256,
        learning_starts=8,
        batch_size=8,
        n_atoms=11,
        seed=0,
    )
    model.learn(total_timesteps=32)

    obs, _ = vector_env.reset(seed=[1, 2])
    action, state = model.predict(obs[0])
    assert state is None
    assert vector_env.single_action_space.contains(action)


def test_fasttd3_save_and_load_round_trips_predictions(vector_env, tmp_path):
    model = FastTD3(
        vector_env,
        actor_net_arch=(16,),
        critic_net_arch=(16,),
        buffer_size=256,
        learning_starts=8,
        batch_size=8,
        n_atoms=11,
        seed=0,
    )
    model.learn(total_timesteps=32)

    path = tmp_path / "fasttd3.pt"
    model.save(path)
    loaded = FastTD3.load(path)

    obs, _ = vector_env.reset(seed=[3, 4])
    action_before, _ = model.predict(obs[0], deterministic=True)
    action_after, _ = loaded.predict(obs[0], deterministic=True)
    assert np.allclose(action_before, action_after)
