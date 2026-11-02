# SPY Options Trading Tool with Real-time Learning

A comprehensive, AI-powered trading system for SPY options that combines options Greeks analysis, multi-timeframe optimization, and continuous reinforcement learning to generate intelligent trading signals and price targets.

## 🚀 Features

### Core Capabilities
- ✅ **Real-time Learning**: Continuously learns from every trade using Proximal Policy Optimization (PPO)
- ✅ **Options Greeks Analysis**: Calculate and use Delta, Gamma, Theta, Vega for optimal strike selection
- ✅ **Multi-Timeframe Optimization**: Analyzes 1m, 5m, 15m, 1h, 1d timeframes using CAGR
- ✅ **Price Target Prediction**: Provides upside, downside, and expected price targets
- ✅ **Minute-by-Minute Updates**: Self-updating system that refreshes every minute
- ✅ **Advanced Indicators**: 20+ technical indicators including RSI, MACD, Bollinger Bands, ATR, ADX
- ✅ **Risk Management**: Comprehensive risk metrics including Sharpe, Sortino, Calmar ratios
- ✅ **Performance Tracking**: Real-time equity curves, drawdown analysis, and trade logging
- ✅ **Visualization**: Beautiful charts for equity, drawdown, and returns distribution

### Technical Features
- **Reinforcement Learning**: Uses Stable Baselines3 PPO for decision making
- **Feature Engineering**: 50+ engineered features including volume, regime detection, and Greeks
- **Experience Replay**: Maintains buffer of recent experiences for continuous improvement
- **Adaptive Strategy**: Automatically adjusts based on market conditions
- **Model Persistence**: Auto-save and load trained models
- **Comprehensive Logging**: Trade-by-trade logging with JSON export

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Components](#components)
7. [Performance Metrics](#performance-metrics)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 - 3.11
- pip package manager
- Git

### Step 1: Clone Repository
```bash
cd /home/user/FinRL
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Dependencies
```bash
pip install stable-baselines3
pip install gymnasium
pip install yfinance
pip install matplotlib
pip install scipy
pip install pandas numpy
pip install stockstats
```

### Step 3: Verify Installation
```bash
cd spy_trading_tool
python config.py  # Should print configuration
```

---

## ⚡ Quick Start

### Basic Usage

```python
from spy_trading_tool import SPYTradingTool

# Initialize the tool
tool = SPYTradingTool(
    ticker='SPY',
    initial_capital=10000,
    auto_save=True
)

# Train initial model (first time only)
tool.train_initial_model(training_days=30, timesteps=10000)

# Run continuous trading
tool.run_continuous(
    update_interval=60,  # Update every minute
    max_updates=None     # Run indefinitely
)
```

### Command Line Usage

```bash
# Run the main trading tool
python spy_trader.py

# The tool will prompt you to:
# 1. Train initial model (y/n)
# 2. Run timeframe optimization (y/n)
# 3. Start continuous trading
```

### Single Update Example

```python
from spy_trading_tool import SPYTradingTool

tool = SPYTradingTool()

# Perform a single update
result = tool.update(timeframe='1m')

# Get current signal
signal = tool.get_current_signal()
print(f"Signal: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")

# Get price targets
targets = tool.get_price_target()
print(f"Current: ${targets['current']:.2f}")
print(f"Upside: ${targets['upside']:.2f}")
print(f"Downside: ${targets['downside']:.2f}")
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SPY Trading Tool                          │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────┐
│   Options     │  │   Feature    │  │  Learning    │
│  Processor    │  │  Engineer    │  │   Agent      │
│               │  │              │  │   (PPO)      │
│ - Greeks      │  │ - 50+ Feat.  │  │ - Cont. Lrn  │
│ - IV Calc     │  │ - Multi-TF   │  │ - Experience │
│ - Strike Sel  │  │ - Regime Det │  │ - Prediction │
└───────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                 ┌──────────────────┐
                 │   Timeframe      │
                 │   Optimizer      │
                 │   (CAGR-based)   │
                 └──────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Performance     │
                 │   Tracker        │
                 │  - Metrics       │
                 │  - Visualization │
                 └──────────────────┘
```

### Component Hierarchy

1. **Data Layer**
   - `OptionsProcessor`: Fetches options data and calculates Greeks
   - `YahooDownloader`: Downloads historical stock data

2. **Feature Engineering Layer**
   - `SPYFeatureEngineer`: Engineers 50+ features from raw data
   - Technical indicators, volume analysis, regime detection

3. **Environment Layer**
   - `SPYOptionsEnv`: Gymnasium environment for RL training
   - Simulates options trading with Greeks

4. **Agent Layer**
   - `ContinuousLearningAgent`: PPO-based RL agent
   - Continuous learning from trades

5. **Optimization Layer**
   - `TimeframeOptimizer`: Multi-timeframe CAGR optimization
   - Adaptive timeframe selection

6. **Tracking Layer**
   - `PerformanceTracker`: Metrics, logging, visualization

---

## ⚙️ Configuration

Edit `spy_trading_tool/config.py` to customize settings:

### Key Parameters

```python
# Trading Settings
TICKER = 'SPY'
INITIAL_CAPITAL = 10000
TRANSACTION_COST = 0.001
MAX_OPTIONS = 10

# Timeframes
TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d']
DEFAULT_TIMEFRAME = '1m'
UPDATE_INTERVAL = 60  # seconds

# Learning
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1000
UPDATE_FREQUENCY = 100

# Risk Management
MAX_DRAWDOWN_THRESHOLD = 0.20
MAX_POSITION_SIZE = 0.10
DAILY_LOSS_LIMIT = 0.05

# Features
USE_VIX = True
USE_ADVANCED_INDICATORS = True
USE_VOLUME_FEATURES = True
USE_REGIME_DETECTION = True
```

---

## 📚 Usage Examples

### Example 1: Basic Trading Session

```python
from spy_trading_tool import SPYTradingTool

# Initialize
tool = SPYTradingTool(
    initial_capital=10000,
    max_options=5
)

# Load pre-trained model (if exists)
tool.agent.load_model('./spy_models/spy_model_final.zip')

# Run for 100 updates (100 minutes)
tool.run_continuous(
    update_interval=60,
    max_updates=100
)
```

### Example 2: Timeframe Optimization

```python
from spy_trading_tool import SPYTradingTool

tool = SPYTradingTool()

# Optimize across timeframes
results = tool.optimize_timeframes()

print(results)
# Output:
#        cagr  sharpe  max_drawdown  calmar  win_rate  total_return  score
# 5m    0.18    2.34        -0.08    2.25      0.62         0.05    0.092
# 1m    0.15    2.01        -0.10    1.50      0.58         0.04    0.078
# ...
```

### Example 3: Custom Feature Engineering

```python
from finrl.meta.preprocessor.spy_feature_engineer import SPYFeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# Download data
downloader = YahooDownloader(
    start_date='2024-01-01',
    end_date='2024-12-31',
    ticker_list=['SPY']
)
data = downloader.fetch_data()

# Engineer features
engineer = SPYFeatureEngineer(
    use_technical_indicator=True,
    use_options_greeks=False,  # Skip for historical data
    use_advanced_indicators=True
)

processed = engineer.preprocess_spy_data(data, include_options=False)
print(f"Features: {list(processed.columns)}")
```

### Example 4: Performance Analysis

```python
from spy_trading_tool.performance_tracker import PerformanceTracker

tracker = PerformanceTracker(initial_capital=10000)

# Simulate some trades
tracker.log_trade({'type': 'BUY_CALL', 'pnl': 150})
tracker.log_trade({'type': 'CLOSE_CALL', 'pnl': -50})
tracker.update_equity(10100)

# Calculate metrics
metrics = tracker.calculate_metrics()
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")

# Generate report
report = tracker.generate_report()
print(report)

# Plot equity curve
tracker.plot_equity_curve(save_path='./equity.png')
```

### Example 5: Manual Signal Generation

```python
from spy_trading_tool import SPYTradingTool
import pandas as pd

tool = SPYTradingTool()

# Get latest data
data = tool.fetch_real_time_data(timeframe='1m')
processed = tool.engineer_features(data)

# Get signal
signal = tool.get_current_signal()

print(f"Action: {signal['signal']}")
print(f"Confidence: {signal['confidence']:.2%}")
print(f"Price Target: {signal['price_target']}")

# Interpret
if signal['signal'] == 'BUY_CALL':
    print(f"➡️ BUY CALL with {signal['position_size']:.1%} of max position")
elif signal['signal'] == 'BUY_PUT':
    print(f"➡️ BUY PUT with {signal['position_size']:.1%} of max position")
elif signal['signal'] == 'CLOSE':
    print("➡️ CLOSE all positions")
else:
    print("➡️ HOLD current positions")
```

---

## 🧩 Components

### 1. Options Processor (`options_processor.py`)

Handles options data and Greeks calculation.

**Key Methods:**
- `fetch_options_chain()`: Get options chain for SPY
- `calculate_greeks()`: Calculate Delta, Gamma, Theta, Vega, Rho
- `select_optimal_strike()`: Choose best strike based on Greeks
- `get_current_price()`: Get real-time SPY price

**Example:**
```python
from finrl.meta.preprocessor.options_processor import OptionsProcessor

processor = OptionsProcessor(ticker='SPY')
current_price = processor.get_current_price()
options_chain = processor.fetch_options_chain()
options_with_greeks = processor.calculate_greeks(options_chain, current_price)

best_call = processor.select_optimal_strike(
    options_with_greeks,
    strategy='balanced',
    option_type='call'
)
print(f"Best Call Strike: ${best_call['strike']:.2f}")
print(f"Delta: {best_call['delta']:.3f}")
```

### 2. Feature Engineer (`spy_feature_engineer.py`)

Engineers features from raw market data.

**Features Generated:**
- Technical: MACD, RSI, Bollinger, ADX, ATR, etc.
- Volume: OBV, MFI, Volume Ratios
- Regime: Trend, Volatility, Market State
- Greeks: Delta, Gamma, Theta, Vega (if available)

**Example:**
```python
from finrl.meta.preprocessor.spy_feature_engineer import SPYFeatureEngineer

engineer = SPYFeatureEngineer()
features = engineer.get_feature_names()
print(f"Total features: {len(features)}")
```

### 3. Trading Environment (`env_spy_options.py`)

Gymnasium-compatible environment for RL training.

**State Space:**
- Cash, positions, current price, indicators, Greeks

**Action Space:**
- Action type: CLOSE (-1), HOLD (0), BUY_CALL (1), BUY_PUT (2)
- Position size: 0 to 1 (fraction of max)

**Reward:**
- Portfolio value change (scaled)
- Penalties for excessive trading

### 4. Learning Agent (`learning_agent.py`)

Continuous learning RL agent using PPO.

**Features:**
- Incremental model updates
- Experience replay buffer
- Price target prediction
- Trade outcome learning

**Example:**
```python
from spy_trading_tool.learning_agent import ContinuousLearningAgent

agent = ContinuousLearningAgent(initial_amount=10000)
agent.create_env(processed_data)
agent.initialize_model()
agent.train(total_timesteps=10000)

# Save model
agent.save_model('./my_model.zip')
```

### 5. Timeframe Optimizer (`timeframe_optimizer.py`)

Optimizes strategy across multiple timeframes.

**Metrics:**
- CAGR
- Sharpe Ratio
- Calmar Ratio
- Max Drawdown
- Win Rate

**Example:**
```python
from spy_trading_tool.timeframe_optimizer import TimeframeOptimizer

optimizer = TimeframeOptimizer(timeframes=['1m', '5m', '1h'])

# Mock portfolio data
timeframe_data = {
    '1m': np.array([10000, 10100, 10150, 10200]),
    '5m': np.array([10000, 10200, 10300, 10400]),
    '1h': np.array([10000, 10150, 10250, 10350]),
}

best_tf, metrics = optimizer.optimize(timeframe_data)
print(f"Best timeframe: {best_tf}")
print(f"CAGR: {metrics['cagr']:.2%}")
```

### 6. Performance Tracker (`performance_tracker.py`)

Tracks and visualizes trading performance.

**Capabilities:**
- Equity curve tracking
- Metrics calculation (Sharpe, Sortino, Calmar)
- Trade logging
- Visualization (equity, drawdown, returns)
- Report generation

---

## 📊 Performance Metrics

### Metrics Calculated

| Metric | Description | Good Value |
|--------|-------------|------------|
| Total Return | Overall profit/loss % | > 10% |
| CAGR | Compound annual growth rate | > 15% |
| Win Rate | % of profitable trades | > 55% |
| Sharpe Ratio | Risk-adjusted returns | > 1.5 |
| Sortino Ratio | Downside risk-adjusted returns | > 2.0 |
| Calmar Ratio | Return / max drawdown | > 3.0 |
| Max Drawdown | Largest peak-to-trough decline | < -15% |
| Profit Factor | Gross profit / gross loss | > 1.5 |

### Example Output

```
SPY OPTIONS TRADING - PERFORMANCE REPORT
======================================================================

PORTFOLIO SUMMARY
----------------------------------------------------------------------
  Initial Capital:        $10,000.00
  Current Equity:         $11,250.00
  Total Return:           +12.50%
  Profit/Loss:            +$1,250.00

TRADING STATISTICS
----------------------------------------------------------------------
  Total Trades:           45
  Win Rate:               62.22%
  Average Win:            $85.50
  Average Loss:           $42.30
  Profit Factor:          1.85

RISK METRICS
----------------------------------------------------------------------
  Sharpe Ratio:           2.15
  Sortino Ratio:          3.22
  Calmar Ratio:           4.17
  Maximum Drawdown:       -8.50%
  Current Drawdown:       -2.10%
```

---

## 🔧 Advanced Usage

### Custom Strategy Development

```python
from spy_trading_tool import SPYTradingTool

class CustomSPYTool(SPYTradingTool):
    def get_current_signal(self):
        # Override to implement custom logic
        signal = super().get_current_signal()

        # Add custom filters
        if self.current_data['rsi_14'].iloc[-1] > 70:
            signal['signal'] = 'CLOSE'  # Force close on overbought

        return signal

# Use custom tool
tool = CustomSPYTool()
tool.run_continuous()
```

### Multi-Asset Extension

```python
# Extend to trade multiple tickers
tickers = ['SPY', 'QQQ', 'IWM']

tools = {
    ticker: SPYTradingTool(ticker=ticker, initial_capital=3333)
    for ticker in tickers
}

# Run all in parallel (requires threading)
```

### Integration with Brokers

```python
# Example integration with Alpaca
from alpaca_trade_api import REST

api = REST(api_key, secret_key, base_url)

tool = SPYTradingTool()
signal = tool.get_current_signal()

if signal['signal'] == 'BUY_CALL':
    # Place actual option order
    api.submit_order(
        symbol='SPY',
        qty=1,
        side='buy',
        type='limit',
        # ... option parameters
    )
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "No data available"**
- Solution: Check internet connection, verify market is open

**Issue: "Model training takes too long"**
- Solution: Reduce `timesteps` parameter or use smaller dataset

**Issue: "ImportError: No module named..."**
- Solution: Install missing dependencies: `pip install -r requirements.txt`

**Issue: "Options data not available"**
- Solution: Use historical mode without options Greeks for backtesting

**Issue: "Memory error during training"**
- Solution: Reduce `buffer_size` and `lookback_period`

### Debug Mode

Enable debug mode in `config.py`:
```python
DEBUG_MODE = True
VERBOSE = 2
```

---

## 📝 File Structure

```
spy_trading_tool/
│
├── __init__.py                    # Package initialization
├── config.py                      # Configuration settings
├── README.md                      # This file
│
├── spy_trader.py                  # Main trading tool
├── learning_agent.py              # RL agent with continuous learning
├── timeframe_optimizer.py         # Multi-timeframe CAGR optimizer
├── performance_tracker.py         # Performance metrics & visualization
│
└── (models and results created at runtime)
    ├── spy_models/                # Saved RL models
    └── spy_results/               # Trading results, logs, plots
```

---

## 🚀 Next Steps

1. **Backtest**: Test on historical data before live trading
2. **Paper Trade**: Run in paper trading mode to validate
3. **Optimize**: Fine-tune hyperparameters in `config.py`
4. **Monitor**: Use performance tracker to analyze results
5. **Scale**: Increase capital and options contracts gradually

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Trading options involves substantial risk of loss. Past performance does not guarantee future results. Always:

- Start with paper trading
- Never risk more than you can afford to lose
- Understand options risks thoroughly
- Consult a financial advisor
- Test extensively before live trading

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📧 Support

For questions or issues:
- Create an issue on GitHub
- Check existing documentation
- Review configuration settings

---

## 🙏 Acknowledgments

- Built on FinRL framework
- Uses Stable Baselines3 for RL
- Options data from yfinance
- Inspired by modern quantitative trading research

---

**Happy Trading! 📈**
