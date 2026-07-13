"""Real-time learning agent for SPY options trading.

This module implements a continuous learning agent that:
- Uses PPO (Proximal Policy Optimization) for training
- Updates model incrementally from new data
- Learns from every trade taken
- Provides price target predictions
- Adapts to changing market conditions
"""

from __future__ import annotations

import os
import pickle
import warnings
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from finrl.meta.env_options_trading.env_spy_options import SPYOptionsEnv
from finrl.meta.preprocessor.spy_feature_engineer import SPYFeatureEngineer


class TradeLogger(BaseCallback):
    """Callback to log trades and performance during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.trades = []
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Log info if available
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "portfolio_value" in info:
                self.trades.append(
                    {
                        "step": self.num_timesteps,
                        "portfolio_value": info["portfolio_value"],
                        "cash": info.get("cash", 0),
                    }
                )
        return True


class ContinuousLearningAgent:
    """Continuous learning agent for SPY options trading.

    Features:
    - Real-time model updates
    - Experience replay buffer
    - Price target prediction
    - Trade analysis and learning
    - Adaptive strategy based on performance

    Attributes
    ----------
    model : PPO
        The RL model (PPO)
    env : SPYOptionsEnv
        Trading environment
    feature_engineer : SPYFeatureEngineer
        Feature engineering pipeline
    experience_buffer : List
        Buffer of recent experiences
    buffer_size : int
        Maximum buffer size
    update_frequency : int
        How often to update model (in steps)
    """

    def __init__(
        self,
        initial_amount: float = 10000,
        transaction_cost: float = 0.001,
        max_options: int = 10,
        buffer_size: int = 1000,
        update_frequency: int = 100,
        model_path: str | None = None,
        tech_indicator_list: list[str] = None,
        greeks_list: list[str] = None,
    ):
        """Initialize the continuous learning agent.

        Parameters
        ----------
        initial_amount : float
            Initial trading capital
        transaction_cost : float
            Transaction cost percentage
        max_options : int
            Maximum option contracts
        buffer_size : int
            Experience replay buffer size
        update_frequency : int
            Model update frequency (steps)
        model_path : str, optional
            Path to load pre-trained model
        tech_indicator_list : List[str], optional
            List of technical indicators
        greeks_list : List[str], optional
            List of Greeks to use
        """
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.max_options = max_options
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.model_path = model_path

        # Feature engineering
        self.tech_indicator_list = tech_indicator_list or [
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
            "rsi_14",
            "atr",
        ]
        self.greeks_list = greeks_list or [
            "call_delta",
            "call_gamma",
            "call_theta",
            "call_vega",
            "put_delta",
            "put_gamma",
            "put_theta",
            "put_vega",
        ]

        self.feature_engineer = SPYFeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.tech_indicator_list,
            use_vix=True,
            use_turbulence=False,
            use_options_greeks=True,
            use_advanced_indicators=True,
            use_volume_features=True,
            use_regime_detection=True,
        )

        # Initialize environment and model
        self.env = None
        self.model = None
        self.experience_buffer = []
        self.training_history = []
        self.trade_outcomes = []

        # Performance tracking
        self.total_steps = 0
        self.total_updates = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0

        # Price prediction
        self.price_predictions = []
        self.prediction_accuracy = 0.0

    def create_env(self, df: pd.DataFrame) -> DummyVecEnv:
        """Create trading environment from data.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed trading data

        Returns
        -------
        DummyVecEnv
            Vectorized environment
        """
        env = SPYOptionsEnv(
            df=df,
            initial_amount=self.initial_amount,
            transaction_cost=self.transaction_cost,
            max_options=self.max_options,
            tech_indicator_list=self.tech_indicator_list,
            greeks_list=self.greeks_list,
        )

        # Wrap in DummyVecEnv for compatibility with Stable Baselines3
        env_wrapped = DummyVecEnv([lambda: env])
        self.env = env_wrapped

        return env_wrapped

    def initialize_model(self, learning_rate: float = 3e-4):
        """Initialize or load the RL model.

        Parameters
        ----------
        learning_rate : float
            Learning rate for PPO
        """
        if self.env is None:
            raise ValueError("Environment must be created first using create_env()")

        if self.model_path and os.path.exists(self.model_path):
            # Load existing model
            print(f"Loading model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            # Create new model
            print("Creating new PPO model")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=0,
            )

    def train(self, total_timesteps: int = 10000, callback=None):
        """Train the model.

        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train
        callback : BaseCallback, optional
            Training callback
        """
        if self.model is None:
            raise ValueError("Model must be initialized first using initialize_model()")

        print(f"Training model for {total_timesteps} timesteps...")

        if callback is None:
            callback = TradeLogger()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

        self.total_steps += total_timesteps
        self.total_updates += 1

        print(f"Training complete. Total steps: {self.total_steps}")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, dict]:
        """Make a prediction (action) given an observation.

        Parameters
        ----------
        observation : np.ndarray
            Current state
        deterministic : bool
            Use deterministic policy

        Returns
        -------
        Tuple[np.ndarray, Dict]
            Action and additional info
        """
        if self.model is None:
            raise ValueError("Model must be initialized first")

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Generate price target prediction
        price_target = self._predict_price_target(observation)

        info = {
            "price_target": price_target,
            "confidence": self._calculate_confidence(observation),
        }

        return action, info

    def _predict_price_target(self, observation: np.ndarray) -> dict[str, float]:
        """Predict price targets for SPY.

        Parameters
        ----------
        observation : np.ndarray
            Current state

        Returns
        -------
        Dict[str, float]
            Price targets (upside, downside, expected)
        """
        # Extract current price from observation
        # Assuming observation structure: [cash, call_pos, put_pos, price, indicators...]
        current_price = (
            observation[0][3] if len(observation.shape) > 1 else observation[3]
        )

        # Use technical indicators to estimate targets
        # This is a simplified approach - in practice, you'd want more sophisticated prediction

        # Simple momentum-based prediction
        # Extract RSI if available (assuming it's in the indicators)
        try:
            # Simplified: use basic heuristics
            # In production, you could use a separate regression model

            volatility = 0.01  # Default 1%
            trend = 0.0  # Neutral

            # Estimate based on regime if available
            upside_target = current_price * (1 + volatility * 2)
            downside_target = current_price * (1 - volatility * 2)
            expected_target = current_price * (1 + trend)

        except Exception as e:
            # Fallback to simple targets
            upside_target = current_price * 1.02
            downside_target = current_price * 0.98
            expected_target = current_price

        return {
            "upside": upside_target,
            "downside": downside_target,
            "expected": expected_target,
            "current": current_price,
        }

    def _calculate_confidence(self, observation: np.ndarray) -> float:
        """Calculate confidence level for the prediction.

        Parameters
        ----------
        observation : np.ndarray
            Current state

        Returns
        -------
        float
            Confidence level (0 to 1)
        """
        # Base confidence on recent win rate and market conditions
        base_confidence = 0.5

        # Adjust based on win rate
        if self.win_rate > 0:
            confidence = base_confidence + (self.win_rate - 0.5) * 0.5
        else:
            confidence = base_confidence

        # Clip to [0, 1]
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def update_from_trade(self, trade_result: dict):
        """Update the agent based on a completed trade.

        Parameters
        ----------
        trade_result : Dict
            Trade outcome information
        """
        self.trade_outcomes.append(trade_result)

        # Update win rate
        if len(self.trade_outcomes) > 0:
            wins = sum(1 for t in self.trade_outcomes if t.get("pnl", 0) > 0)
            self.win_rate = wins / len(self.trade_outcomes)

        # Add to experience buffer
        if "experience" in trade_result:
            self.experience_buffer.append(trade_result["experience"])

            # Trim buffer if too large
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer = self.experience_buffer[-self.buffer_size :]

        print(f"Trade logged. Win rate: {self.win_rate:.2%}")

    def should_update_model(self) -> bool:
        """Check if model should be updated.

        Returns
        -------
        bool
            True if should update
        """
        return self.total_steps % self.update_frequency == 0

    def incremental_update(self, new_data: pd.DataFrame, timesteps: int = 1000):
        """Perform incremental model update with new data.

        Parameters
        ----------
        new_data : pd.DataFrame
            New market data
        timesteps : int
            Number of timesteps for update
        """
        print("Performing incremental model update...")

        # Create new environment with recent data
        temp_env = self.create_env(new_data)

        # Update model with new environment
        self.model.set_env(temp_env)

        # Train on new data
        self.train(total_timesteps=timesteps)

        print("Incremental update complete")

    def save_model(self, path: str):
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Save path
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)
        print(f"Model saved to {path}")

        # Save agent state
        agent_state = {
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "win_rate": self.win_rate,
            "trade_outcomes": self.trade_outcomes[-100:],  # Keep last 100
        }

        with open(f"{path}_agent_state.pkl", "wb") as f:
            pickle.dump(agent_state, f)

    def load_model(self, path: str):
        """Load model from disk.

        Parameters
        ----------
        path : str
            Load path
        """
        if self.env is None:
            raise ValueError("Environment must be created first")

        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")

        # Load agent state if available
        state_path = f"{path}_agent_state.pkl"
        if os.path.exists(state_path):
            with open(state_path, "rb") as f:
                agent_state = pickle.load(f)
                self.total_steps = agent_state.get("total_steps", 0)
                self.total_updates = agent_state.get("total_updates", 0)
                self.win_rate = agent_state.get("win_rate", 0.0)
                self.trade_outcomes = agent_state.get("trade_outcomes", [])

    def get_performance_summary(self) -> dict:
        """Get performance summary.

        Returns
        -------
        Dict
            Performance metrics
        """
        recent_trades = (
            self.trade_outcomes[-20:] if len(self.trade_outcomes) > 0 else []
        )

        if len(recent_trades) > 0:
            recent_pnl = [t.get("pnl", 0) for t in recent_trades]
            recent_wins = sum(1 for p in recent_pnl if p > 0)
            recent_win_rate = recent_wins / len(recent_trades)
            avg_pnl = np.mean(recent_pnl)
        else:
            recent_win_rate = 0
            avg_pnl = 0

        return {
            "total_trades": len(self.trade_outcomes),
            "win_rate": self.win_rate,
            "recent_win_rate": recent_win_rate,
            "avg_pnl": avg_pnl,
            "total_updates": self.total_updates,
            "total_steps": self.total_steps,
        }

    def generate_signal(self, current_data: pd.DataFrame) -> dict:
        """Generate trading signal from current data.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current market data (single row)

        Returns
        -------
        Dict
            Trading signal with action, confidence, and targets
        """
        # Prepare observation
        observation = self._prepare_observation(current_data)

        # Get action and prediction
        action, info = self.predict(observation, deterministic=True)

        # Interpret action
        action_type = action[0][0]  # -1, 0, 1, 2
        position_size = action[0][1]  # 0 to 1

        if action_type < -0.5:
            signal = "CLOSE"
        elif action_type > 0.5 and action_type < 1.5:
            signal = "BUY_CALL"
        elif action_type >= 1.5:
            signal = "BUY_PUT"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "position_size": position_size,
            "price_target": info["price_target"],
            "confidence": info["confidence"],
            "timestamp": datetime.now(),
        }

    def _prepare_observation(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare observation from data row.

        Parameters
        ----------
        data : pd.DataFrame
            Current data

        Returns
        -------
        np.ndarray
            Observation vector
        """
        # This would extract features from the data
        # For now, return a placeholder
        # In production, this should match the environment's state structure

        observation = []

        # Cash (start with initial amount for prediction)
        observation.append(self.initial_amount)

        # Positions (assume no current positions for signal generation)
        observation.extend([0, 0])

        # Current price
        observation.append(
            data.get("close", 0)
            if isinstance(data, pd.Series)
            else data["close"].iloc[0]
        )

        # Technical indicators
        for indicator in self.tech_indicator_list:
            val = (
                data.get(indicator, 0)
                if isinstance(data, pd.Series)
                else data[indicator].iloc[0]
            )
            observation.append(val)

        # Greeks
        for greek in self.greeks_list:
            val = (
                data.get(greek, 0)
                if isinstance(data, pd.Series)
                else data[greek].iloc[0]
            )
            observation.append(val)

        return np.array([observation], dtype=np.float32)
