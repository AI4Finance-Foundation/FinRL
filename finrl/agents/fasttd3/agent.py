"""FastTD3 agent: TD3 + a distributional critic + CDQ, driven by a
vectorized environment and a GPU-resident replay buffer.

Not an ``sb3_contrib`` algorithm — there is no SB3 implementation to lean on
the way TQC/CrossQ do, so this is a from-scratch port of the recipe in
Seo et al. 2025 (``FastTD3: Simple, Fast, and Capable Reinforcement Learning
for Humanoid Control``), standalone the way the paper's own reference
implementation is (PyTorch, built on LeanRL rather than SB3). The
:class:`FastTD3` class exposes the same surface an SB3 algorithm does
(``.learn()``, ``.predict()``, ``.save()``, classmethod ``.load()``,
``.observation_space``, ``.action_space``) — including driving an SB3-style
``BaseCallback`` through ``.on_step()``/``.on_rollout_end()`` — so it can be
dropped into a training loop written against that interface even though this
algorithm's *internal* training loop (vectorized rollout collection,
big-batch GPU updates) looks nothing like SB3's own ``OffPolicyAlgorithm``.

**What "parallel environments" means here.** ``vector_env`` is any
``gymnasium.vector.VectorEnv``: independent environment copies in the usual
vectorized-RL sense, or — a pattern relevant to a single historical
market panel, where there is only one "level" to run in parallel —
``num_envs`` copies of the *same* training window with independently sampled
exploration noise. Either way, the only thing that needs to differ across
copies for off-policy learning is the *state-action coverage* entering the
replay buffer; TD3's exploration noise is independently sampled per copy, so
``num_envs`` copies under a shared stochastic policy visit ``num_envs``
different trajectories.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import ConvertCallback
from stable_baselines3.common.utils import configure_logger

from finrl.agents.fasttd3.buffer import GPUReplayBuffer
from finrl.agents.fasttd3.networks import Actor
from finrl.agents.fasttd3.networks import DistributionalCritic
from finrl.agents.fasttd3.networks import project_distribution
from finrl.agents.fasttd3.networks import q_values


def _unbatch_infos(infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    """Undo gymnasium vector-env info batching back to one dict per env.

    ``gymnasium.vector.SyncVectorEnv`` batches each per-env ``info`` dict
    (recursively through nested dicts) into ``{key: array_of_len_num_envs,
    "_" + key: bool_mask, ...}``. The mask keys exist to say which envs
    actually reported that key this step and are dropped; everything else is
    indexed back into the per-env list of dicts SB3's callback protocol
    generally expects in ``callback.locals["infos"]``.
    """

    def _unbatch_one(batched: dict[str, Any], i: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in batched.items():
            if key.startswith("_"):
                continue
            out[key] = _unbatch_one(value, i) if isinstance(value, dict) else value[i]
        return out

    return [_unbatch_one(infos, i) for i in range(num_envs)]


class FastTD3:
    """TD3 with twin distributional (C51) critics and CDQ target selection."""

    def __init__(
        self,
        vector_env: VectorEnv,
        *,
        actor_net_arch: tuple[int, ...] = (512, 256, 128),
        critic_net_arch: tuple[int, ...] = (1024, 512, 256),
        learning_rate: float = 3e-4,
        buffer_size: int = 100_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        gamma: float = 0.999,
        tau: float = 0.005,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.4,
        updates_per_step: int = 1,
        n_atoms: int = 101,
        v_min: float = -5.0,
        v_max: float = 5.0,
        use_compile: bool = False,
        use_amp: bool = False,
        seed: int | None = None,
        device: str | torch.device = "auto",
        verbose: int = 0,
    ) -> None:
        self.vector_env = vector_env
        self.num_envs = vector_env.num_envs
        self.observation_space: spaces.Box = vector_env.single_observation_space
        self.action_space: spaces.Box = vector_env.single_action_space
        self.logit_bound = float(self.action_space.high[0])
        obs_dim = int(np.prod(self.observation_space.shape))
        action_dim = int(np.prod(self.action_space.shape))

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.exploration_noise = float(exploration_noise)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.updates_per_step = int(updates_per_step)
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.actor_net_arch = tuple(actor_net_arch)
        self.critic_net_arch = tuple(critic_net_arch)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"

        if seed is not None:
            torch.manual_seed(int(seed))
        self._np_rng = np.random.default_rng(seed)

        self.support = torch.linspace(
            self.v_min, self.v_max, self.n_atoms, device=self.device
        )

        self.actor = Actor(
            obs_dim, action_dim, self.actor_net_arch, self.logit_bound
        ).to(self.device)
        self.actor_target = Actor(
            obs_dim, action_dim, self.actor_net_arch, self.logit_bound
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1 = DistributionalCritic(
            obs_dim, action_dim, self.critic_net_arch, n_atoms
        ).to(self.device)
        self.critic2 = DistributionalCritic(
            obs_dim, action_dim, self.critic_net_arch, n_atoms
        ).to(self.device)
        self.critic1_target = DistributionalCritic(
            obs_dim, action_dim, self.critic_net_arch, n_atoms
        ).to(self.device)
        self.critic2_target = DistributionalCritic(
            obs_dim, action_dim, self.critic_net_arch, n_atoms
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        if use_compile:
            self.actor = torch.compile(self.actor)
            self.critic1 = torch.compile(self.critic1)
            self.critic2 = torch.compile(self.critic2)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            itertools.chain(self.critic1.parameters(), self.critic2.parameters()),
            lr=learning_rate,
        )
        self.critic_scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.use_amp
        )
        self.actor_scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.use_amp
        )

        self.buffer = GPUReplayBuffer(
            buffer_size, obs_dim, action_dim, device=self.device
        )
        self.num_timesteps = 0
        self._update_count = 0
        self.logger = configure_logger(verbose=verbose)

    # ------------------------------------------------------------- training

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | None = None,
        progress_bar: bool = False,
    ) -> FastTD3:
        if callback is None:
            callback = ConvertCallback(None)
        callback.init_callback(self)
        callback.on_training_start(locals_={}, globals_={})

        obs, _ = self.vector_env.reset(seed=self._episode_seeds())
        iterator = range(0, total_timesteps, self.num_envs)
        if progress_bar:
            from tqdm import tqdm

            iterator = tqdm(iterator)

        for _ in iterator:
            actions = self._collect_actions(obs)
            next_obs, rewards, terminated, truncated, infos = self.vector_env.step(
                actions
            )
            self.buffer.add(
                obs, actions, rewards, next_obs, terminated.astype(np.float32)
            )

            callback.locals = {
                "actions": actions,
                "infos": _unbatch_infos(infos, self.num_envs),
            }
            self.num_timesteps += self.num_envs
            if not callback.on_step():
                break

            if truncated.any():
                if not truncated.all():
                    raise RuntimeError(
                        "FastTD3's vector env copies must share one episode boundary; got a "
                        "partial truncation, which means the vector env's copies have drifted "
                        "out of sync with each other."
                    )
                callback.on_rollout_end()
                obs, _ = self.vector_env.reset(seed=self._episode_seeds())
            else:
                obs = next_obs

            if len(self.buffer) >= self.learning_starts:
                for _ in range(self.updates_per_step):
                    self._update()

        callback.on_training_end()
        return self

    def _episode_seeds(self) -> list[int]:
        return [int(self._np_rng.integers(0, 2**31 - 1)) for _ in range(self.num_envs)]

    def _collect_actions(self, obs: np.ndarray) -> np.ndarray:
        if self.num_timesteps < self.learning_starts:
            shape = (self.num_envs,) + self.action_space.shape
            low, high = self.action_space.low, self.action_space.high
            return self._np_rng.uniform(low, high, size=shape).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            raw = self.actor(obs_t)
            noise = torch.randn_like(raw) * (self.exploration_noise * self.logit_bound)
            action = (raw + noise).clamp(-self.logit_bound, self.logit_bound)
        return action.cpu().numpy()

    def _update(self) -> None:
        self._update_count += 1
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.use_amp
        ):
            next_actions = self.actor_target(next_obs)
            smoothing_noise = (
                torch.randn_like(next_actions) * self.target_policy_noise
            ).clamp(-self.target_noise_clip, self.target_noise_clip) * self.logit_bound
            next_actions = (next_actions + smoothing_noise).clamp(
                -self.logit_bound, self.logit_bound
            )

            next_probs1 = self.critic1_target(next_obs, next_actions)
            next_probs2 = self.critic2_target(next_obs, next_actions)
            q1 = q_values(next_probs1, self.support)
            q2 = q_values(next_probs2, self.support)
            # CDQ (Fujimoto et al. 2018): both online critics regress toward
            # whichever target critic's distribution has the lower expected
            # value, the distributional analogue of "take the min" a
            # scalar-critic TD3/SAC uses.
            use_first = (q1 <= q2).unsqueeze(-1)
            next_probs = torch.where(use_first, next_probs1, next_probs2)
            target_dist = project_distribution(
                rewards,
                dones,
                next_probs,
                support=self.support,
                gamma=self.gamma,
                v_min=self.v_min,
                v_max=self.v_max,
            )

        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            logits1 = self.critic1.logits(obs, actions)
            logits2 = self.critic2.logits(obs, actions)
            loss1 = (
                -(target_dist * torch.log_softmax(logits1, dim=-1)).sum(dim=-1).mean()
            )
            loss2 = (
                -(target_dist * torch.log_softmax(logits2, dim=-1)).sum(dim=-1).mean()
            )
            critic_loss = loss1 + loss2

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_scaler.scale(critic_loss).backward()
        self.critic_scaler.step(self.critic_optimizer)
        self.critic_scaler.update()

        if self._update_count % self.policy_delay == 0:
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                actor_actions = self.actor(obs)
                actor_probs = self.critic1(obs, actor_actions)
                actor_loss = -q_values(actor_probs, self.support).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            self.actor_scaler.scale(actor_loss).backward()
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

    def _soft_update(self, online: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for p, tp in zip(online.parameters(), target.parameters(), strict=True):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------- inference

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        obs_arr = np.asarray(observation, dtype=np.float32)
        single = obs_arr.ndim == 1
        if single:
            obs_arr = obs_arr[None, :]
        obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_t)
            if not deterministic:
                noise = torch.randn_like(action) * (
                    self.exploration_noise * self.logit_bound
                )
                action = (action + noise).clamp(-self.logit_bound, self.logit_bound)
        action_np = action.cpu().numpy()
        return (action_np[0], None) if single else (action_np, None)

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        obs_dim = int(np.prod(self.observation_space.shape))
        action_dim = int(np.prod(self.action_space.shape))
        payload = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "num_timesteps": self.num_timesteps,
            "config": {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "logit_bound": self.logit_bound,
                "actor_net_arch": self.actor_net_arch,
                "critic_net_arch": self.critic_net_arch,
                "n_atoms": self.n_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
                "exploration_noise": self.exploration_noise,
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> FastTD3:
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg = payload["config"]
        obj = cls.__new__(cls)
        obj.device = torch.device(device)
        obj.logit_bound = float(cfg["logit_bound"])
        obj.actor_net_arch = tuple(cfg["actor_net_arch"])
        obj.critic_net_arch = tuple(cfg["critic_net_arch"])
        obj.n_atoms = int(cfg["n_atoms"])
        obj.v_min = float(cfg["v_min"])
        obj.v_max = float(cfg["v_max"])
        obj.exploration_noise = float(cfg["exploration_noise"])
        obj.support = torch.linspace(
            obj.v_min, obj.v_max, obj.n_atoms, device=obj.device
        )

        obs_dim, action_dim = int(cfg["obs_dim"]), int(cfg["action_dim"])
        obj.actor = Actor(obs_dim, action_dim, obj.actor_net_arch, obj.logit_bound).to(
            obj.device
        )
        obj.actor.load_state_dict(payload["actor"])
        obj.actor.eval()
        critic_args = (obs_dim, action_dim, obj.critic_net_arch, obj.n_atoms)
        obj.critic1 = DistributionalCritic(*critic_args).to(obj.device)
        obj.critic1.load_state_dict(payload["critic1"])
        obj.critic1.eval()
        obj.critic2 = DistributionalCritic(*critic_args).to(obj.device)
        obj.critic2.load_state_dict(payload["critic2"])
        obj.critic2.eval()

        obj.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        obj.action_space = spaces.Box(
            -obj.logit_bound, obj.logit_bound, shape=(action_dim,), dtype=np.float32
        )
        obj.num_timesteps = int(payload.get("num_timesteps", 0))
        # Inference-only round trip: no vector env, no buffer, no optimizers.
        # Resuming training from a loaded checkpoint is not supported —
        # re-deriving those from `path` alone would need information (the
        # vector env, the optimizer state) this format deliberately doesn't
        # carry.
        obj.vector_env = None
        obj.buffer = None
        obj.num_envs = None
        return obj
