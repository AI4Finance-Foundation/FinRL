"""Main SPY Options Trading Tool with Real-time Learning.

This is the main entry point for the SPY trading system that:
- Updates every minute with new data
- Uses options Greeks for strike selection
- Learns from every trade
- Optimizes across multiple timeframes using CAGR
- Provides price targets and trading signals
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finrl.meta.preprocessor.options_processor import OptionsProcessor
from finrl.meta.preprocessor.spy_feature_engineer import SPYFeatureEngineer
from spy_trading_tool.learning_agent import ContinuousLearningAgent
from spy_trading_tool.timeframe_optimizer import TimeframeOptimizer


class SPYTradingTool:
    """Main SPY options trading tool with continuous learning.

    Features:
    - Minute-by-minute updates
    - Options Greeks analysis
    - Multi-indicator signals
    - CAGR-based timeframe optimization
    - Real-time learning from trades
    - Price target predictions

    Attributes
    ----------
    ticker : str
        Stock ticker (SPY)
    initial_capital : float
        Starting capital
    options_processor : OptionsProcessor
        Options data and Greeks processor
    feature_engineer : SPYFeatureEngineer
        Feature engineering pipeline
    agent : ContinuousLearningAgent
        RL agent for trading decisions
    timeframe_optimizer : TimeframeOptimizer
        Multi-timeframe optimizer
    """

    def __init__(
        self,
        ticker: str = 'SPY',
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        max_options: int = 10,
        timeframes: List[str] = None,
        risk_free_rate: float = 0.05,
        model_path: Optional[str] = None,
        auto_save: bool = True,
        save_dir: str = './models',
    ):
        """Initialize the SPY trading tool.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol
        initial_capital : float
            Initial trading capital
        transaction_cost : float
            Transaction cost percentage
        max_options : int
            Maximum option contracts
        timeframes : List[str], optional
            Timeframes to analyze
        risk_free_rate : float
            Risk-free interest rate
        model_path : str, optional
            Path to load pre-trained model
        auto_save : bool
            Auto-save model after updates
        save_dir : str
            Directory for saving models
        """
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_options = max_options
        self.risk_free_rate = risk_free_rate
        self.auto_save = auto_save
        self.save_dir = save_dir

        # Create save directory if needed
        os.makedirs(save_dir, exist_ok=True)

        # Initialize components
        print("Initializing SPY Trading Tool...")

        # Options processor
        self.options_processor = OptionsProcessor(
            ticker=ticker,
            risk_free_rate=risk_free_rate
        )

        # Feature engineer
        self.feature_engineer = SPYFeatureEngineer(
            use_technical_indicator=True,
            use_vix=True,
            use_turbulence=False,
            use_options_greeks=True,
            use_advanced_indicators=True,
            use_volume_features=True,
            use_regime_detection=True,
            risk_free_rate=risk_free_rate,
        )

        # Learning agent
        self.agent = ContinuousLearningAgent(
            initial_amount=initial_capital,
            transaction_cost=transaction_cost,
            max_options=max_options,
            model_path=model_path,
        )

        # Timeframe optimizer
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h']
        self.timeframe_optimizer = TimeframeOptimizer(
            timeframes=self.timeframes,
            risk_free_rate=risk_free_rate,
        )

        # State
        self.current_data = None
        self.current_signal = None
        self.current_price_target = None
        self.last_update_time = None
        self.update_count = 0

        # Trading state
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {
            'calls': 0,
            'puts': 0,
        }
        self.trade_log = []

        # Performance tracking
        self.daily_returns = []
        self.performance_history = []

        print("SPY Trading Tool initialized successfully!")

    def fetch_real_time_data(self, timeframe: str = '1m', lookback: int = 100) -> pd.DataFrame:
        """Fetch real-time market data.

        Parameters
        ----------
        timeframe : str
            Data timeframe
        lookback : int
            Number of periods to fetch

        Returns
        -------
        pd.DataFrame
            Market data
        """
        print(f"Fetching real-time data ({timeframe})...")

        # Fetch stock data
        stock_data = self.options_processor.get_real_time_data(
            timeframe=timeframe,
            period='5d' if timeframe in ['1m', '5m'] else '1mo'
        )

        if stock_data.empty:
            print("Warning: No stock data fetched")
            return pd.DataFrame()

        # Add ticker column
        stock_data['tic'] = self.ticker

        # Ensure required columns
        if 'date' not in stock_data.columns and 'timestamp' in stock_data.columns:
            stock_data['date'] = stock_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Take last N rows
        stock_data = stock_data.tail(lookback).copy()

        print(f"Fetched {len(stock_data)} data points")
        return stock_data

    def engineer_features(self, data: pd.DataFrame, include_options: bool = True) -> pd.DataFrame:
        """Engineer features from raw data.

        Parameters
        ----------
        data : pd.DataFrame
            Raw market data
        include_options : bool
            Include options Greeks

        Returns
        -------
        pd.DataFrame
            Data with engineered features
        """
        print("Engineering features...")

        if data.empty:
            return data

        # Preprocess with feature engineering
        processed_data = self.feature_engineer.preprocess_spy_data(
            data,
            include_options=include_options
        )

        print(f"Features engineered: {len(processed_data.columns)} columns")
        return processed_data

    def get_current_signal(self) -> Dict:
        """Get current trading signal.

        Returns
        -------
        Dict
            Trading signal with action, confidence, targets
        """
        if self.current_data is None or self.current_data.empty:
            return {
                'signal': 'HOLD',
                'confidence': 0,
                'price_target': None,
                'timestamp': datetime.now(),
            }

        # Get latest data point
        latest_data = self.current_data.iloc[-1]

        # Generate signal from agent
        signal = self.agent.generate_signal(latest_data)

        self.current_signal = signal
        return signal

    def get_price_target(self) -> Dict:
        """Get current price target prediction.

        Returns
        -------
        Dict
            Price targets (upside, downside, expected)
        """
        signal = self.get_current_signal()
        return signal.get('price_target', {})

    def update(self, timeframe: str = '1m') -> Dict:
        """Perform a complete update cycle.

        Parameters
        ----------
        timeframe : str
            Timeframe to use

        Returns
        -------
        Dict
            Update results
        """
        print(f"\n{'='*60}")
        print(f"UPDATE {self.update_count + 1} - {datetime.now()}")
        print(f"{'='*60}")

        try:
            # 1. Fetch latest data
            raw_data = self.fetch_real_time_data(timeframe=timeframe, lookback=100)

            if raw_data.empty:
                print("No data available, skipping update")
                return {'status': 'no_data'}

            # 2. Engineer features
            processed_data = self.engineer_features(raw_data, include_options=True)

            if processed_data.empty:
                print("Feature engineering failed, skipping update")
                return {'status': 'feature_error'}

            self.current_data = processed_data

            # 3. Get trading signal
            signal = self.get_current_signal()

            print(f"\nSignal: {signal['signal']}")
            print(f"Confidence: {signal['confidence']:.2%}")

            if signal.get('price_target'):
                targets = signal['price_target']
                print(f"Current Price: ${targets.get('current', 0):.2f}")
                print(f"Upside Target: ${targets.get('upside', 0):.2f}")
                print(f"Downside Target: ${targets.get('downside', 0):.2f}")

            # 4. Check if model should be updated
            if self.agent.should_update_model() and len(processed_data) > 50:
                print("\nPerforming incremental model update...")
                self.agent.incremental_update(processed_data, timesteps=500)

                if self.auto_save:
                    model_path = os.path.join(self.save_dir, f'spy_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
                    self.agent.save_model(model_path)

            # 5. Update metrics
            self.update_count += 1
            self.last_update_time = datetime.now()

            # Performance summary
            perf = self.agent.get_performance_summary()
            print(f"\nPerformance:")
            print(f"  Total Trades: {perf['total_trades']}")
            print(f"  Win Rate: {perf['win_rate']:.2%}")
            print(f"  Model Updates: {perf['total_updates']}")

            return {
                'status': 'success',
                'signal': signal,
                'performance': perf,
                'timestamp': self.last_update_time,
            }

        except Exception as e:
            print(f"Error during update: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'error': str(e)}

    def run_continuous(self, update_interval: int = 60, max_updates: Optional[int] = None):
        """Run continuous trading loop with periodic updates.

        Parameters
        ----------
        update_interval : int
            Update interval in seconds (default: 60 = 1 minute)
        max_updates : int, optional
            Maximum number of updates (None = run indefinitely)
        """
        print(f"\n{'='*60}")
        print("STARTING CONTINUOUS SPY TRADING TOOL")
        print(f"{'='*60}")
        print(f"Update Interval: {update_interval} seconds")
        print(f"Ticker: {self.ticker}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Max Updates: {max_updates or 'Unlimited'}")
        print(f"{'='*60}\n")

        update_num = 0

        try:
            while True:
                # Perform update
                result = self.update(timeframe='1m')

                # Check if should stop
                if max_updates and update_num >= max_updates:
                    print(f"\nReached maximum updates ({max_updates}). Stopping.")
                    break

                update_num += 1

                # Wait for next update
                print(f"\nWaiting {update_interval} seconds for next update...")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\n\nStopping trading tool (user interrupted)...")

        finally:
            # Save final model
            if self.auto_save:
                final_model_path = os.path.join(self.save_dir, 'spy_model_final.zip')
                self.agent.save_model(final_model_path)
                print(f"\nFinal model saved to {final_model_path}")

            # Print final summary
            self.print_summary()

    def train_initial_model(self, training_days: int = 30, timesteps: int = 10000):
        """Train initial model on historical data.

        Parameters
        ----------
        training_days : int
            Number of days of historical data
        timesteps : int
            Training timesteps
        """
        print(f"\n{'='*60}")
        print("TRAINING INITIAL MODEL")
        print(f"{'='*60}")

        # Fetch historical data
        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=training_days)).strftime('%Y-%m-%d')

        print(f"Fetching historical data: {start_date} to {end_date}")

        downloader = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=[self.ticker]
        )

        data = downloader.fetch_data()

        if data.empty:
            print("No historical data available")
            return

        print(f"Fetched {len(data)} data points")

        # Engineer features (without live options data for historical)
        processed_data = self.engineer_features(data, include_options=False)

        # Create environment
        self.agent.create_env(processed_data)

        # Initialize model
        self.agent.initialize_model()

        # Train
        print(f"\nTraining for {timesteps} timesteps...")
        self.agent.train(total_timesteps=timesteps)

        # Save model
        if self.auto_save:
            model_path = os.path.join(self.save_dir, 'spy_model_initial.zip')
            self.agent.save_model(model_path)
            print(f"\nInitial model saved to {model_path}")

        print("\nInitial training complete!")

    def optimize_timeframes(self) -> pd.DataFrame:
        """Optimize across timeframes and display results.

        Returns
        -------
        pd.DataFrame
            Optimization results
        """
        print(f"\n{'='*60}")
        print("TIMEFRAME OPTIMIZATION")
        print(f"{'='*60}")

        # Fetch data for each timeframe
        timeframe_data = {}

        for tf in self.timeframes:
            print(f"\nFetching data for {tf}...")
            data = self.fetch_real_time_data(timeframe=tf, lookback=100)

            if not data.empty:
                # Simulate portfolio values (simplified)
                # In production, you'd backtest properly
                prices = data['close'].values
                portfolio_values = (prices / prices[0]) * self.initial_capital
                timeframe_data[tf] = portfolio_values

        # Optimize
        best_tf, metrics = self.timeframe_optimizer.optimize(timeframe_data)

        print(f"\n{self.timeframe_optimizer.generate_report()}")

        return self.timeframe_optimizer.get_optimization_summary()

    def print_summary(self):
        """Print comprehensive summary of trading session."""
        print(f"\n{'='*60}")
        print("TRADING SESSION SUMMARY")
        print(f"{'='*60}")

        print(f"\nSession Info:")
        print(f"  Total Updates: {self.update_count}")
        print(f"  Last Update: {self.last_update_time}")

        perf = self.agent.get_performance_summary()
        print(f"\nTrading Performance:")
        print(f"  Total Trades: {perf['total_trades']}")
        print(f"  Overall Win Rate: {perf['win_rate']:.2%}")
        print(f"  Recent Win Rate: {perf['recent_win_rate']:.2%}")
        print(f"  Avg PnL: ${perf['avg_pnl']:.2f}")

        print(f"\nModel Info:")
        print(f"  Total Training Steps: {perf['total_steps']}")
        print(f"  Total Updates: {perf['total_updates']}")

        if self.current_signal:
            print(f"\nLast Signal:")
            print(f"  Action: {self.current_signal['signal']}")
            print(f"  Confidence: {self.current_signal['confidence']:.2%}")

        print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    # Configuration
    TICKER = 'SPY'
    INITIAL_CAPITAL = 10000
    UPDATE_INTERVAL = 60  # seconds (1 minute)
    MAX_UPDATES = None  # Run indefinitely (or set a number for testing)

    # Initialize tool
    tool = SPYTradingTool(
        ticker=TICKER,
        initial_capital=INITIAL_CAPITAL,
        auto_save=True,
        save_dir='./spy_models',
    )

    # Train initial model (optional - comment out if loading existing model)
    print("\nTrain initial model? (y/n): ", end='')
    if input().lower().strip() == 'y':
        tool.train_initial_model(training_days=30, timesteps=10000)

    # Optimize timeframes
    print("\nRun timeframe optimization? (y/n): ", end='')
    if input().lower().strip() == 'y':
        tool.optimize_timeframes()

    # Run continuous trading
    print("\nStarting continuous trading...")
    tool.run_continuous(
        update_interval=UPDATE_INTERVAL,
        max_updates=MAX_UPDATES
    )


if __name__ == '__main__':
    main()
