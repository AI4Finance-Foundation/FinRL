from __future__ import annotations

import logging
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.data_processors.processor_ccxt import CCXTEngineer
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv


class BTCRLAnalysisBot:
    def __init__(self, initial_capital=100000):
        self.ccxt_eng = CCXTEngineer()
        self.initial_capital = initial_capital
        self.model = None

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, start_date, end_date):
        """Fetch and prepare data for training"""
        self.logger.info("Fetching training data...")
        data = self.ccxt_eng.data_fetch(
            start=start_date, end=end_date, pair_list=["BTC/USDT"], period="1h"
        )

        # Calculate technical indicators
        data["rsi"] = self.calculate_rsi(data["close"])
        data["macd"], data["signal"] = self.calculate_macd(data["close"])
        data["bb_upper"], data["bb_middle"], data["bb_lower"] = (
            self.calculate_bollinger_bands(data["close"])
        )

        return data

    def train_model(self, train_data, total_timesteps=100000):
        """Train the RL model"""
        self.logger.info("Preparing training environment...")

        # Configure training environment
        train_env_config = {
            "price_array": train_data[["close"]].values,
            "tech_array": train_data[
                ["rsi", "macd", "signal", "bb_upper", "bb_middle", "bb_lower"]
            ].values,
            "if_train": True,
        }

        env = CryptoEnv(
            config=train_env_config,
            initial_capital=self.initial_capital,
            buy_cost_pct=0.001,
            sell_cost_pct=0.001,
        )

        # Initialize agent
        agent = DRLAgent(env=env)

        self.logger.info("Training model...")
        PPO_PARAMS = {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 128,
        }

        model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
        self.model = agent.train_model(
            model=model_ppo, tb_log_name="ppo_crypto", total_timesteps=total_timesteps
        )

        self.logger.info("Training completed!")

    def generate_rl_prediction(self, current_data):
        """Generate trading signal using the trained RL model"""
        if self.model is None:
            raise ValueError("Model not trained! Please train the model first.")

        # Prepare state for prediction
        state = self.prepare_state(current_data)

        # Get action from model
        action, _ = self.model.predict(state)

        # Convert action to trading signal
        signal = self.interpret_action(action)
        return signal

    def run_analysis(self, use_rl=True):
        """Run continuous real-time analysis"""
        self.logger.info("Starting BTC/USD Analysis Bot...")

        while True:
            try:
                # Fetch current market data
                current_data = self.ccxt_eng.data_fetch(
                    start=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                    pair_list=["BTC/USDT"],
                    period="1h",
                )

                if use_rl and self.model:
                    # Use RL model for prediction
                    signal = self.generate_rl_prediction(current_data)
                else:
                    # Use traditional technical analysis
                    signal = self.generate_traditional_signals(current_data)

                self.logger.info(f"\n=== BTC/USD Analysis Report ===")
                self.logger.info(
                    f"Current Price: ${current_data['close'].iloc[-1]:,.2f}"
                )
                self.logger.info(f"Recommendation: {signal}")
                self.logger.info("=" * 30 + "\n")

                time.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)

    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower


def main():
    # Initialize bot
    bot = BTCRLAnalysisBot(initial_capital=100000)

    # Train the model (optional)
    training_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    training_end = datetime.now().strftime("%Y-%m-%d")

    train_data = bot.prepare_data(training_start, training_end)
    bot.train_model(train_data)

    # Run real-time analysis
    bot.run_analysis(use_rl=True)


if __name__ == "__main__":
    main()
