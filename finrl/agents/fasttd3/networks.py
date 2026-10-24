"""Actor and distributional twin critics for FastTD3.

Not built on Stable-Baselines3 like the other ``finrl/agents/*`` backends:
FastTD3 (Seo et al. 2025) has no SB3 implementation to lean on the way
TQC/CrossQ (both from ``sb3-contrib``) do, so this is a from-scratch port of
the recipe in the paper's own reference implementation
(``younggyoseo/fast-td3``, standalone PyTorch built on LeanRL rather than
SB3). This module carries the two purely-parametric pieces; the training
loop, replay buffer and vectorized-environment handling live in
:mod:`finrl.agents.fasttd3.agent` and :mod:`finrl.agents.fasttd3.buffer`.

**Actor.** A deterministic policy over the environment's *logit* action space
(``[-logit_bound, +logit_bound]^n``, not ``[-1, 1]``): a plain MLP produces a
pre-activation, squashed by ``tanh`` and rescaled by ``logit_bound`` — the
standard TD3/DDPG deterministic-policy convention, adapted to this
environment's action bound instead of assuming one.

**Distributional critic (Bellemare et al. 2017, C51).** Each of the twin
critics estimates the return distribution as a categorical distribution over
``n_atoms`` fixed support points spaced between ``v_min`` and ``v_max``,
rather than a single expected value — the mechanism the paper credits with
most of FastTD3's stability gain, and the reason "twin critics" alone (SAC,
TQC) is not automatically "distributional". ``q_values`` collapses a critic's
categorical output back to a scalar (``sum(p_i * z_i)``) wherever a plain
Q-value is needed (e.g. the actor's policy-gradient objective); the training
loop keeps the full distribution wherever the loss is computed, since
collapsing early is exactly what a *non*-distributional critic does.
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """A plain feed-forward MLP: ``sizes[0] -> ... -> sizes[-1]``.

    Deliberately no layer normalization or residual paths: the paper reports
    both "tend to slow down training without significant gains" for this
    algorithm specifically (unlike CrossQ, where batch renormalization is
    the entire point) — kept out rather than added and left unused.
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), activation()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic policy: obs -> logits in ``[-logit_bound, +logit_bound]^n``."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden: tuple[int, ...], logit_bound: float
    ) -> None:
        super().__init__()
        self.logit_bound = float(logit_bound)
        self.net = _mlp([obs_dim, *hidden, action_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.logit_bound


class DistributionalCritic(nn.Module):
    """One C51-style critic: obs+action -> a categorical distribution over
    ``n_atoms`` fixed support points ``z in [v_min, v_max]``.

    ``forward`` returns the atom *probabilities* (``softmax`` over the raw
    logits), never a collapsed scalar — the training loop needs the full
    distribution to compute the cross-entropy loss against a projected
    target distribution (:func:`project_distribution`).
    """

    def __init__(
        self, obs_dim: int, action_dim: int, hidden: tuple[int, ...], n_atoms: int
    ) -> None:
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.net = _mlp([obs_dim + action_dim, *hidden, self.n_atoms])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([obs, action], dim=-1))
        return torch.softmax(logits, dim=-1)

    def logits(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Raw (pre-softmax) atom logits — what the cross-entropy loss reads."""
        return self.net(torch.cat([obs, action], dim=-1))


def q_values(probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Collapse a categorical distribution (batch, n_atoms) to E[Q] (batch,)."""
    return (probs * support).sum(dim=-1)


def project_distribution(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_probs: torch.Tensor,
    *,
    support: torch.Tensor,
    gamma: float,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """The C51 Bellman projection (Bellemare et al. 2017, Algorithm 1).

    Shifts and rescales the next-state atoms by the one-step return
    ``r + gamma * (1 - done) * z``, clips back onto ``[v_min, v_max]``, and
    distributes each shifted atom's probability mass onto its two nearest
    neighbors on the *fixed* support — the projection a categorical critic
    needs because ``r + gamma * z`` does not in general land back on the
    support it started from.

    ``rewards``/``dones``: shape ``(batch,)``. ``next_probs``: shape
    ``(batch, n_atoms)``, the *target* critic's distribution at the sampled
    next state/action (already CDQ-selected — see ``agent.py``). Returns a
    ``(batch, n_atoms)`` target distribution, each row summing to 1.
    """
    batch, n_atoms = next_probs.shape
    delta_z = (v_max - v_min) / (n_atoms - 1)

    # r + gamma * (1 - done) * z, clipped to the support's range.
    tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma * support.unsqueeze(
        0
    )
    tz = tz.clamp(v_min, v_max)

    b = (
        tz - v_min
    ) / delta_z  # continuous position on the fixed grid, in [0, n_atoms - 1]
    lo = b.floor().long()
    hi = b.ceil().long()
    # b landing exactly on a grid point (lo == hi) would otherwise drop that
    # atom's mass entirely: both (hi - b) and (b - lo) are zero. Nudge the
    # pair one grid step apart, toward whichever neighbor exists; b itself is
    # unchanged, so the linear-interpolation weights below still put 100% of
    # the mass on the original (untouched) index and 0% on the nudged one —
    # correct, not a special case, just the same formula with lo != hi.
    eq = lo == hi
    nudge_lo = eq & (lo > 0)
    # Mutually exclusive with nudge_lo: nudging *both* sides for a middle
    # atom would open a 2-wide gap (hi - lo == 2) and double its mass
    # instead of preserving it, so hi only moves where lo did not.
    nudge_hi = eq & ~nudge_lo & (hi < n_atoms - 1)
    lo = torch.where(nudge_lo, lo - 1, lo)
    hi = torch.where(nudge_hi, hi + 1, hi)

    target = torch.zeros_like(next_probs)
    lo_weight = (hi.float() - b) * next_probs
    hi_weight = (b - lo.float()) * next_probs
    target.scatter_add_(1, lo.clamp(0, n_atoms - 1), lo_weight)
    target.scatter_add_(1, hi.clamp(0, n_atoms - 1), hi_weight)
    return target
