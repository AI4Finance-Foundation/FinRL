"""
Crypto Trading Part 2 — Train

Loads the preprocessed numpy arrays produced by Part 1, wraps CryptoEnv in a
Gymnasium-compatible interface, and trains two DRL agents (PPO and SAC) using
Stable-Baselines3.

Agents trained:
  • PPO  — saved to trained_models/crypto_ppo
  • SAC  — saved to trained_models/crypto_sac

Requires: crypto_data_arrays.npz  (from Part 1)
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import os

import gymnasium as gym
from gymnasium import spaces

from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv

TRAINED_MODEL_DIR = "trained_models"
RESULTS_DIR       = "results"


def check_and_make_directories(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_FILE       = "crypto_data_arrays.npz"
INITIAL_CAPITAL = 1_000_000.0   # USD
BUY_COST_PCT    = 0.001         # 0.1 % taker fee (Binance standard)
SELL_COST_PCT   = 0.001
LOOKBACK        = 1

# Total environment timesteps for training
PPO_TIMESTEPS = 200_000
SAC_TIMESTEPS = 200_000

PPO_PARAMS = {
    "n_steps"      : 2048,
    "batch_size"   : 64,
    "ent_coef"     : 0.01,
    "learning_rate": 2.5e-4,
}
SAC_PARAMS = {
    "batch_size"    : 256,
    "buffer_size"   : 100_000,
    "learning_rate" : 1e-4,
    "learning_starts": 1_000,
    "ent_coef"      : "auto_0.1",
}

# ── Gymnasium wrapper ──────────────────────────────────────────────────────────

class CryptoEnvWrapper(gym.Env):
    """Thin Gymnasium wrapper around FinRL's CryptoEnv.

    CryptoEnv uses the legacy 4-tuple step() return, which is not compatible
    with SB3 2.x / Gymnasium.  This wrapper translates the API.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_array: np.ndarray,
        tech_array: np.ndarray,
        initial_capital: float = 1_000_000.0,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        lookback: int = 1,
    ) -> None:
        super().__init__()
        config = {"price_array": price_array, "tech_array": tech_array}
        self._env = CryptoEnv(
            config=config,
            lookback=lookback,
            initial_capital=initial_capital,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_dim,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, _info = self._env.step(action)
        return obs, float(reward), bool(done), False, {}

    def render(self):
        pass

    def close(self):
        self._env.close()

    # Convenience properties used by the backtest script
    @property
    def total_asset(self) -> float:
        return float(self._env.total_asset)

    @property
    def initial_total_asset(self) -> float:
        return float(self._env.initial_total_asset)

    @property
    def episode_return(self) -> float:
        return float(self._env.episode_return)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_env(price_array, tech_array, **kwargs):
    """Factory that returns a zero-argument lambda, as required by DummyVecEnv."""
    def _init():
        return CryptoEnvWrapper(price_array, tech_array, **kwargs)
    return _init


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    check_and_make_directories([TRAINED_MODEL_DIR, RESULTS_DIR])

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Loading training arrays …")
    print("=" * 60)

    data = np.load(DATA_FILE)
    price_array = data["train_price_array"]
    tech_array  = data["train_tech_array"]

    print(f"  price_array : {price_array.shape}")
    print(f"  tech_array  : {tech_array.shape}")

    env_kwargs = dict(
        price_array    = price_array,
        tech_array     = tech_array,
        initial_capital= INITIAL_CAPITAL,
        buy_cost_pct   = BUY_COST_PCT,
        sell_cost_pct  = SELL_COST_PCT,
        lookback       = LOOKBACK,
    )

    # ── 2. Build vectorised environment ───────────────────────────────────────
    print("\nBuilding training environment …")
    env_train = DummyVecEnv([make_env(**env_kwargs)])

    # Quick sanity-check
    dummy = CryptoEnvWrapper(**env_kwargs)
    print(f"  observation_space : {dummy.observation_space}")
    print(f"  action_space      : {dummy.action_space}")

    # ── 3. Train PPO ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2 — Training PPO …")
    print("=" * 60)

    model_ppo = PPO(
        policy="MlpPolicy",
        env=env_train,
        verbose=1,
        **PPO_PARAMS,
    )
    model_ppo.learn(total_timesteps=PPO_TIMESTEPS, tb_log_name="ppo_crypto")
    model_ppo.save(f"{TRAINED_MODEL_DIR}/crypto_ppo")
    print(f"\nPPO model saved → {TRAINED_MODEL_DIR}/crypto_ppo")

    # ── 4. Train SAC ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 3 — Training SAC …")
    print("=" * 60)

    # SAC needs its own fresh env (buffer is off-policy, so it rebuilds anyway)
    env_train_sac = DummyVecEnv([make_env(**env_kwargs)])

    model_sac = SAC(
        policy="MlpPolicy",
        env=env_train_sac,
        verbose=1,
        **SAC_PARAMS,
    )
    model_sac.learn(total_timesteps=SAC_TIMESTEPS, tb_log_name="sac_crypto")
    model_sac.save(f"{TRAINED_MODEL_DIR}/crypto_sac")
    print(f"\nSAC model saved → {TRAINED_MODEL_DIR}/crypto_sac")

    print("\nDone — run FinRL_CryptoTrading_3_backtest.py next.")


if __name__ == "__main__":
    main()
