import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from stable_baselines3 import PPO
from gym import spaces
import torch

class BTCAnalysisEnv:
    def __init__(self):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )

    def process_data(self, df):
        """Convert market data to state"""
        # Normalize the features
        state = np.array([
            df['close'].pct_change().iloc[-1],
            df['volume'].pct_change().iloc[-1],
            (df['close'] - df['close'].rolling(20).mean()).iloc[-1] / df['close'].iloc[-1],
            (df['rsi'] - 50) / 50,
            df['macd'].iloc[-1] / df['close'].iloc[-1],
            df['signal'].iloc[-1] / df['close'].iloc[-1],
            (df['close'] - df['bb_middle']) / (df['bb_upper'] - df['bb_lower']).iloc[-1],
            df['close'].pct_change(5).iloc[-1],
            df['close'].pct_change(10).iloc[-1],
            df['close'].pct_change(20).iloc[-1],
        ], dtype=np.float32)
        return state

class BTCRealtimeAnalyzer:
    def __init__(self, model_path=None):
        """Initialize the analyzer with optional pre-trained model"""
        self.exchange = ccxt.binance()
        self.symbol = 'BTC/USDT'
        self.env = BTCAnalysisEnv()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model if provided
        self.model = None
        if model_path:
            try:
                self.model = PPO.load(model_path)
                self.logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")

    def fetch_data(self, timeframe='1h', limit=100):
        """Fetch market data and calculate indicators"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate indicators
            df['rsi'] = self.calculate_rsi(df['close'])
            df['macd'], df['signal'] = self.calculate_macd(df['close'])
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None

    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    def analyze_technical_indicators(self, df):
        """Generate trading signals based on technical indicators"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = {
            'rsi': {
                'value': latest['rsi'],
                'signal': 'SELL' if latest['rsi'] > 70 else 'BUY' if latest['rsi'] < 30 else 'HOLD',
                'strength': abs(50 - latest['rsi']) / 50
            },
            'macd': {
                'value': latest['macd'],
                'signal': 'BUY' if latest['macd'] > latest['signal'] else 'SELL',
                'strength': abs(latest['macd'] - latest['signal']) / abs(latest['macd'])
            },
            'bollinger': {
                'value': latest['close'],
                'signal': 'SELL' if latest['close'] > latest['bb_upper'] else 'BUY' if latest['close'] < latest['bb_lower'] else 'HOLD',
                'strength': min(abs(latest['close'] - latest['bb_middle']) / (latest['bb_upper'] - latest['bb_middle']), 1)
            }
        }
        return signals

    def get_rl_prediction(self, df):
        """Get trading signal from the RL model"""
        if self.model is None:
            return None
        
        try:
            state = self.env.process_data(df)
            action, _ = self.model.predict(state)
            
            # Convert action to signal
            signal_strength = abs(float(action[0]))
            signal = {
                'value': float(action[0]),
                'signal': 'BUY' if action[0] > 0.2 else 'SELL' if action[0] < -0.2 else 'HOLD',
                'strength': signal_strength
            }
            return signal
        except Exception as e:
            self.logger.error(f"Error in RL prediction: {e}")
            return None

    def run_analysis(self):
        """Run continuous real-time analysis"""
        self.logger.info("Starting BTC/USD Real-time Analysis...")
        
        while True:
            try:
                # Fetch current market data
                df = self.fetch_data()
                if df is None:
                    time.sleep(60)
                    continue
                
                current_price = df['close'].iloc[-1]
                current_time = df['timestamp'].iloc[-1]
                
                # Technical Analysis
                tech_signals = self.analyze_technical_indicators(df)
                
                # RL Analysis
                rl_signal = self.get_rl_prediction(df) if self.model else None
                
                # Print Analysis
                self.logger.info("\n=== BTC/USD Analysis Report ===")
                self.logger.info(f"Time: {current_time}")
                self.logger.info(f"Current Price: ${current_price:,.2f}")
                
                self.logger.info("\nTechnical Indicators:")
                for indicator, data in tech_signals.items():
                    self.logger.info(f"{indicator.upper()}: {data['signal']} (Strength: {data['strength']:.2f})")
                
                if rl_signal:
                    self.logger.info("\nRL Model Prediction:")
                    self.logger.info(f"Signal: {rl_signal['signal']} (Strength: {rl_signal['strength']:.2f})")
                
                # Combined Analysis
                tech_sentiment = sum([1 if s['signal'] == 'BUY' else -1 if s['signal'] == 'SELL' else 0 
                                   for s in tech_signals.values()]) / len(tech_signals)
                
                if rl_signal:
                    final_sentiment = (tech_sentiment + rl_signal['value']) / 2
                else:
                    final_sentiment = tech_sentiment
                
                overall_signal = (
                    'STRONG BUY' if final_sentiment > 0.5 else
                    'BUY' if final_sentiment > 0.2 else
                    'STRONG SELL' if final_sentiment < -0.5 else
                    'SELL' if final_sentiment < -0.2 else
                    'HOLD'
                )
                
                self.logger.info(f"\nOVERALL RECOMMENDATION: {overall_signal}")
                self.logger.info("=" * 30 + "\n")
                
                # Wait before next analysis
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(60)

def main():
    # Initialize analyzer (optionally with a pre-trained model)
    analyzer = BTCRealtimeAnalyzer(model_path=None)  # Add model path if available
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
