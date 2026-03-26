"""
Crypto Trading Part 3 — Backtest

Loads trained PPO and SAC models, runs them on the held-out test set, and
compares performance against a simple buy-and-hold (equal-weight) baseline.

Metrics reported per agent:
  • Total return (%)
  • Annualised Sharpe ratio  (1-minute granularity → √525 600 scaling)
  • Maximum drawdown (%)
  • Final portfolio value ($)

Output:
  crypto_backtest_result.png  — portfolio value over time
  crypto_backtest_metrics.csv — summary table
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC

from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv

TRAINED_MODEL_DIR = "trained_models"

# Re-use the wrapper defined in Part 2
# (import it if running as a module; duplicate inline for standalone use)
import gymnasium as gym
from gymnasium import spaces


# ── Gymnasium wrapper (same as Part 2) ────────────────────────────────────────


class CryptoEnvWrapper(gym.Env):
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
            low=-np.inf, high=np.inf, shape=(self._env.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._env.action_dim,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._env.reset(), {}

    def step(self, action):
        obs, reward, done, _info = self._env.step(action)
        return obs, float(reward), bool(done), False, {}

    @property
    def total_asset(self) -> float:
        return float(self._env.total_asset)

    @property
    def initial_total_asset(self) -> float:
        return float(self._env.initial_total_asset)


# ── Configuration ──────────────────────────────────────────────────────────────

DATA_FILE = "crypto_data_arrays.npz"
INITIAL_CAPITAL = 1_000_000.0
BUY_COST_PCT = 0.001
SELL_COST_PCT = 0.001
LOOKBACK = 1
PAIR_LIST = ["BTC/USDT", "ETH/USDT"]

MINUTES_PER_YEAR = 525_600  # 365 × 24 × 60


# ── Prediction helper ──────────────────────────────────────────────────────────


def run_backtest(
    model,
    price_array: np.ndarray,
    tech_array: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Step a trained model through the test environment.

    Returns
    -------
    total_assets : np.ndarray  shape (T+1,)  — portfolio value at each step
    actions      : np.ndarray  shape (T, n_cryptos)
    """
    env = CryptoEnvWrapper(
        price_array=price_array,
        tech_array=tech_array,
        initial_capital=INITIAL_CAPITAL,
        buy_cost_pct=BUY_COST_PCT,
        sell_cost_pct=SELL_COST_PCT,
        lookback=LOOKBACK,
    )

    obs, _ = env.reset()
    total_assets = [env.initial_total_asset]
    actions_log = []

    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_assets.append(env.total_asset)
        actions_log.append(action.copy())
        step += 1
        if step % 10_000 == 0:
            print(f"  [{label}] step {step:,} — portfolio ${env.total_asset:,.0f}")

    print(f"  [{label}] finished — {step:,} steps, final ${env.total_asset:,.0f}")
    return np.array(total_assets), np.array(actions_log)


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_metrics(total_assets: np.ndarray, label: str) -> dict:
    """Compute backtest statistics from a portfolio-value series."""
    returns = np.diff(total_assets) / total_assets[:-1]  # per-minute returns

    total_return = (total_assets[-1] / total_assets[0] - 1) * 100

    # Annualised Sharpe (risk-free rate assumed 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(MINUTES_PER_YEAR)
        else:
            sharpe = 0.0

    # Maximum drawdown
    peak = np.maximum.accumulate(total_assets)
    drawdown = (total_assets - peak) / peak
    max_dd = drawdown.min() * 100

    return {
        "Agent": label,
        "Total Return (%)": round(total_return, 2),
        "Sharpe (ann.)": round(sharpe, 4),
        "Max Drawdown (%)": round(max_dd, 2),
        "Final Value ($)": round(total_assets[-1], 2),
    }


# ── Buy-and-hold baseline ──────────────────────────────────────────────────────


def buy_and_hold(price_array: np.ndarray, initial_capital: float) -> np.ndarray:
    """Equal-weight buy-and-hold on all assets from t=0."""
    n_assets = price_array.shape[1]
    capital_each = initial_capital / n_assets
    # Allocate at t=0 prices; ignore transaction cost for simplicity
    shares = capital_each / price_array[0]  # (n_assets,)
    portfolio_vals = (price_array * shares).sum(axis=1)  # (T,)
    return portfolio_vals


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    # ── 1. Load test data ─────────────────────────────────────────────────────
    print("=" * 60)
    print("Part 1 — Loading test arrays …")
    print("=" * 60)

    data = np.load(DATA_FILE, allow_pickle=True)
    price_array = data["test_price_array"]
    tech_array = data["test_tech_array"]
    date_ary = data["test_date_ary"]

    print(f"  price_array : {price_array.shape}")
    print(f"  tech_array  : {tech_array.shape}")
    T = price_array.shape[0]

    # ── 2. Load trained models ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 2 — Loading models …")
    print("=" * 60)

    model_ppo = PPO.load(f"{TRAINED_MODEL_DIR}/crypto_ppo")
    model_sac = SAC.load(f"{TRAINED_MODEL_DIR}/crypto_sac")
    print("  PPO and SAC loaded.")

    # ── 3. Run backtests ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 3 — Running backtests …")
    print("=" * 60)

    assets_ppo, _ = run_backtest(model_ppo, price_array, tech_array, "PPO")
    assets_sac, _ = run_backtest(model_sac, price_array, tech_array, "SAC")

    # ── 4. Buy-and-hold baseline ──────────────────────────────────────────────
    bah_vals = buy_and_hold(price_array, INITIAL_CAPITAL)

    # ── 5. Compute metrics ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 4 — Metrics")
    print("=" * 60)

    rows = [
        compute_metrics(assets_ppo, "PPO"),
        compute_metrics(assets_sac, "SAC"),
        compute_metrics(bah_vals, "Buy & Hold"),
    ]
    df_metrics = pd.DataFrame(rows).set_index("Agent")
    print("\n", df_metrics.to_string())
    df_metrics.to_csv("crypto_backtest_metrics.csv")
    print("\nMetrics saved → crypto_backtest_metrics.csv")

    # ── 6. Plot ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Part 5 — Plotting …")
    print("=" * 60)

    # Align index: use minute-sampled timestamps if available, else integer index
    n_plot = min(len(assets_ppo), len(assets_sac), len(bah_vals))
    x = np.arange(n_plot)

    # Normalise to initial capital for easy comparison
    def norm(arr):
        return arr[:n_plot] / INITIAL_CAPITAL * 100

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        "BTC/USDT + ETH/USDT  |  1-minute DRL Backtest\n"
        f"Initial capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=13,
    )

    # — Panel 1: portfolio value (%) —
    ax1 = axes[0]
    ax1.plot(x, norm(assets_ppo), label="PPO", linewidth=1.0, color="steelblue")
    ax1.plot(x, norm(assets_sac), label="SAC", linewidth=1.0, color="darkorange")
    ax1.plot(
        x,
        norm(bah_vals),
        label="Buy & Hold",
        linewidth=1.0,
        color="green",
        linestyle="--",
    )
    ax1.axhline(100, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_ylabel("Portfolio Value (% of initial capital)")
    ax1.set_title("Portfolio Value Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # — Panel 2: drawdown (%) —
    def drawdown_pct(arr):
        a = arr[:n_plot]
        pk = np.maximum.accumulate(a)
        return (a - pk) / pk * 100

    ax2 = axes[1]
    ax2.fill_between(
        x, drawdown_pct(assets_ppo), 0, alpha=0.4, color="steelblue", label="PPO DD"
    )
    ax2.fill_between(
        x, drawdown_pct(assets_sac), 0, alpha=0.4, color="darkorange", label="SAC DD"
    )
    ax2.fill_between(
        x, drawdown_pct(bah_vals), 0, alpha=0.3, color="green", label="Buy & Hold DD"
    )
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Minutes into backtest")
    ax2.set_title("Drawdown")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("crypto_backtest_result.png", dpi=150, bbox_inches="tight")
    print("Chart saved → crypto_backtest_result.png")

    print("\nAll done!")


if __name__ == "__main__":
    main()
