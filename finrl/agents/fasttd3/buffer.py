"""GPU-resident replay buffer for FastTD3.

The paper's own reasoning: "we store the entire buffer on the GPU to avoid
the overhead of data transfer between CPU and GPU" — meaningful for its
massively-parallel physics-sim setting where a training step is otherwise
bottlenecked on host<->device copies. Reused here for infrastructure parity:
the buffer is a fixed-size ring of pre-allocated ``torch`` tensors on
``device`` from the start, rather than a ``numpy`` buffer copied to the GPU
per batch the way SB3's default replay buffer would be used from CPU.
"""

from __future__ import annotations

import numpy as np
import torch


class GPUReplayBuffer:
    """A ring buffer of ``(obs, action, reward, next_obs, done)`` transitions.

    ``capacity`` is the number of *transitions*, not ``capacity // num_envs``
    — a FastTD3 run and a single-env SAC/TD3 run configured with the same
    buffer size hold the same amount of experience regardless of how many
    parallel copies filled it.

    ``add`` accepts one env-step's worth of transitions at once (shape
    ``(num_envs, ...)``), the natural unit a vectorized environment produces.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self._obs = torch.zeros(
            (self.capacity, obs_dim), dtype=torch.float32, device=self.device
        )
        self._actions = torch.zeros(
            (self.capacity, action_dim), dtype=torch.float32, device=self.device
        )
        self._rewards = torch.zeros(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._next_obs = torch.zeros(
            (self.capacity, obs_dim), dtype=torch.float32, device=self.device
        )
        # A "done" here means a true terminal state, never a time-limit
        # truncation — matching the bootstrap-flag convention the rest of
        # FinRL's off-policy SB3 buffers already use.
        self._dones = torch.zeros(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert one vectorized env-step: each array is ``(num_envs, ...)``."""
        n = obs.shape[0]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        end = self._ptr + n
        if end <= self.capacity:
            idx = slice(self._ptr, end)
            self._obs[idx] = obs_t
            self._actions[idx] = actions_t
            self._rewards[idx] = rewards_t
            self._next_obs[idx] = next_obs_t
            self._dones[idx] = dones_t
        else:
            # Wraps around the ring: split into a tail write and a head write.
            tail = self.capacity - self._ptr
            self._obs[self._ptr :] = obs_t[:tail]
            self._actions[self._ptr :] = actions_t[:tail]
            self._rewards[self._ptr :] = rewards_t[:tail]
            self._next_obs[self._ptr :] = next_obs_t[:tail]
            self._dones[self._ptr :] = dones_t[:tail]
            rest = n - tail
            self._obs[:rest] = obs_t[tail:]
            self._actions[:rest] = actions_t[tail:]
            self._rewards[:rest] = rewards_t[tail:]
            self._next_obs[:rest] = next_obs_t[tail:]
            self._dones[:rest] = dones_t[tail:]

        self._ptr = end % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int):
        """A uniform random minibatch, already on ``device`` — no host round trip.

        Draws from the global ``torch`` RNG (seeded once by the agent's own
        constructor via ``torch.manual_seed``) rather than a buffer-owned
        generator — a CUDA-device ``torch.Generator`` cannot be seeded from a
        CPU one, and a second, buffer-local seed would be one more source of
        nondeterminism to track.
        """
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (
            self._obs[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_obs[idx],
            self._dones[idx],
        )
