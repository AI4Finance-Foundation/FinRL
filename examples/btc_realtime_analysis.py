from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Tuple

import ccxt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class BTCAnalysisBot:
    def __init__(self, exchange_id="binance"):
        """Initialize the BTC analysis bot"""
        self.exchange = getattr(ccxt, exchange_id)()
        self.symbol = "BTC/USDT"
        self.timeframes = {
            "short_term": "5m",  # 5 minutes
            "medium_term": "1h",  # 1 hour
            "long_term": "1d",  # 1 day
        }

        # Configure logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def fetch_ohlcv_data(self, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for the specified timeframe"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Moving averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_upper"] = df["bb_middle"] + 2 * df["close"].rolling(window=20).std()
        df["bb_lower"] = df["bb_middle"] - 2 * df["close"].rolling(window=20).std()

        return df

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Generate trading signals based on technical indicators"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        signals = {
            "rsi": {
                "value": latest["rsi"],
                "signal": (
                    "SELL"
                    if latest["rsi"] > 70
                    else "BUY" if latest["rsi"] < 30 else "HOLD"
                ),
                "strength": abs(50 - latest["rsi"]) / 50,
            },
            "macd": {
                "value": latest["macd"],
                "signal": "BUY" if latest["macd"] > latest["signal"] else "SELL",
                "strength": abs(latest["macd"] - latest["signal"])
                / abs(latest["macd"]),
            },
            "bollinger": {
                "value": latest["close"],
                "signal": (
                    "SELL"
                    if latest["close"] > latest["bb_upper"]
                    else "BUY" if latest["close"] < latest["bb_lower"] else "HOLD"
                ),
                "strength": min(
                    abs(latest["close"] - latest["bb_middle"])
                    / (latest["bb_upper"] - latest["bb_middle"]),
                    1,
                ),
            },
        }

        return signals

    def analyze_market_sentiment(self, timeframes_data: dict[str, pd.DataFrame]) -> str:
        """Analyze overall market sentiment across different timeframes"""
        sentiment_scores = []

        for timeframe, df in timeframes_data.items():
            signals = self.generate_signals(df)

            # Convert signals to numeric scores (-1 for SELL, 0 for HOLD, 1 for BUY)
            score = 0
            for indicator, data in signals.items():
                if data["signal"] == "BUY":
                    score += 1 * data["strength"]
                elif data["signal"] == "SELL":
                    score -= 1 * data["strength"]

            sentiment_scores.append(score)

        # Weight the timeframes (short-term less, long-term more)
        weights = [0.2, 0.3, 0.5]  # 5m, 1h, 1d
        weighted_sentiment = np.average(sentiment_scores, weights=weights)

        if weighted_sentiment > 0.5:
            return "STRONG BUY"
        elif weighted_sentiment > 0.2:
            return "BUY"
        elif weighted_sentiment < -0.5:
            return "STRONG SELL"
        elif weighted_sentiment < -0.2:
            return "SELL"
        else:
            return "HOLD"

    def run_analysis(self):
        """Run continuous real-time analysis"""
        self.logger.info("Starting BTC/USD Analysis Bot...")

        while True:
            try:
                # Fetch data for all timeframes
                timeframes_data = {}
                for name, timeframe in self.timeframes.items():
                    df = self.fetch_ohlcv_data(timeframe)
                    if df is not None:
                        timeframes_data[name] = self.calculate_indicators(df)

                if not timeframes_data:
                    self.logger.error("Failed to fetch data for all timeframes")
                    time.sleep(60)
                    continue

                # Current price
                current_price = timeframes_data["short_term"].iloc[-1]["close"]

                # Generate analysis for each timeframe
                self.logger.info("\n=== BTC/USD Analysis Report ===")
                self.logger.info(f"Current Price: ${current_price:,.2f}")

                for timeframe_name, df in timeframes_data.items():
                    signals = self.generate_signals(df)
                    self.logger.info(f"\n{timeframe_name.upper()} Analysis:")
                    for indicator, data in signals.items():
                        self.logger.info(
                            f"{indicator.upper()}: {data['signal']} (Strength: {data['strength']:.2f})"
                        )

                # Overall sentiment
                sentiment = self.analyze_market_sentiment(timeframes_data)
                self.logger.info(f"\nOVERALL RECOMMENDATION: {sentiment}")
                self.logger.info("=" * 30 + "\n")

                # Wait before next analysis
                time.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)


def main():
    bot = BTCAnalysisBot()
    bot.run_analysis()


if __name__ == "__main__":
    main()
