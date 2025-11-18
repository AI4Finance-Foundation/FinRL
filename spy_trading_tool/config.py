"""Configuration file for SPY Trading Tool.

This file contains all configurable parameters for the trading system.
"""

from typing import List

# =====================================================================
# GENERAL SETTINGS
# =====================================================================

# Stock ticker
TICKER = 'SPY'

# Initial trading capital
INITIAL_CAPITAL = 10000

# Transaction cost (as decimal, e.g., 0.001 = 0.1%)
TRANSACTION_COST = 0.001

# Maximum number of option contracts to hold
MAX_OPTIONS = 10

# Risk-free rate (annual, as decimal)
RISK_FREE_RATE = 0.05

# =====================================================================
# TIMEFRAME SETTINGS
# =====================================================================

# Timeframes to analyze and optimize
TIMEFRAMES: List[str] = ['1m', '5m', '15m', '1h', '1d']

# Default timeframe for trading
DEFAULT_TIMEFRAME = '1m'

# Lookback period for each timeframe (number of periods)
LOOKBACK_PERIOD = 100

# Reoptimization frequency (in minutes)
REOPTIMIZE_FREQ = 60

# =====================================================================
# TECHNICAL INDICATORS
# =====================================================================

# List of technical indicators to use
TECH_INDICATORS: List[str] = [
    'macd',           # Moving Average Convergence Divergence
    'rsi_30',         # Relative Strength Index (30 period)
    'rsi_14',         # Relative Strength Index (14 period)
    'cci_30',         # Commodity Channel Index
    'dx_30',          # Directional Index
    'close_30_sma',   # Simple Moving Average (30 period)
    'close_60_sma',   # Simple Moving Average (60 period)
    'atr',            # Average True Range
    'adx',            # Average Directional Index
    'boll_ub',        # Bollinger Upper Band
    'boll_lb',        # Bollinger Lower Band
]

# Options Greeks to use
GREEKS_LIST: List[str] = [
    'call_delta',
    'call_gamma',
    'call_theta',
    'call_vega',
    'put_delta',
    'put_gamma',
    'put_theta',
    'put_vega',
]

# Use VIX indicator
USE_VIX = True

# Use turbulence index
USE_TURBULENCE = False

# Use advanced indicators
USE_ADVANCED_INDICATORS = True

# Use volume features
USE_VOLUME_FEATURES = True

# Use regime detection
USE_REGIME_DETECTION = True

# =====================================================================
# LEARNING AGENT SETTINGS
# =====================================================================

# Learning rate for PPO
LEARNING_RATE = 3e-4

# Experience buffer size
BUFFER_SIZE = 1000

# Model update frequency (in steps)
UPDATE_FREQUENCY = 100

# Training timesteps for initial model
INITIAL_TRAINING_TIMESTEPS = 10000

# Training timesteps for incremental updates
INCREMENTAL_UPDATE_TIMESTEPS = 500

# Number of days for initial training
INITIAL_TRAINING_DAYS = 30

# PPO specific parameters
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64
PPO_N_EPOCHS = 10
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_RANGE = 0.2
PPO_ENT_COEF = 0.01

# =====================================================================
# REAL-TIME TRADING SETTINGS
# =====================================================================

# Update interval in seconds (60 = 1 minute)
UPDATE_INTERVAL = 60

# Maximum number of updates (None = unlimited)
MAX_UPDATES = None

# Auto-save model after updates
AUTO_SAVE = True

# Model save directory
MODEL_SAVE_DIR = './spy_models'

# Results save directory
RESULTS_DIR = './spy_results'

# Trade log file
TRADE_LOG_FILE = './spy_results/trades.log'

# =====================================================================
# VISUALIZATION SETTINGS
# =====================================================================

# Enable plotting
ENABLE_PLOTS = True

# Plot save directory
PLOT_SAVE_DIR = './spy_results/plots'

# Plot DPI
PLOT_DPI = 300

# =====================================================================
# OPTIONS SETTINGS
# =====================================================================

# Use real-time options data
USE_OPTIONS_DATA = True

# Options expiration preference (None = nearest)
OPTIONS_EXPIRATION = None

# Strike selection strategy: 'aggressive', 'balanced', 'conservative'
STRIKE_SELECTION_STRATEGY = 'balanced'

# =====================================================================
# RISK MANAGEMENT
# =====================================================================

# Maximum drawdown threshold (as decimal, e.g., 0.20 = 20%)
MAX_DRAWDOWN_THRESHOLD = 0.20

# Stop trading if drawdown exceeds threshold
STOP_ON_MAX_DRAWDOWN = True

# Position size limits (as percentage of capital)
MAX_POSITION_SIZE = 0.10  # 10% of capital per trade

# Daily loss limit (as percentage of capital)
DAILY_LOSS_LIMIT = 0.05  # 5% of capital

# =====================================================================
# NOTIFICATION SETTINGS (OPTIONAL)
# =====================================================================

# Enable email notifications
ENABLE_EMAIL_NOTIFICATIONS = False

# Email settings (if enabled)
EMAIL_FROM = ''
EMAIL_TO = ''
EMAIL_SMTP_SERVER = 'smtp.gmail.com'
EMAIL_SMTP_PORT = 587
EMAIL_PASSWORD = ''

# Notification triggers
NOTIFY_ON_TRADE = True
NOTIFY_ON_ERROR = True
NOTIFY_ON_DAILY_SUMMARY = True

# =====================================================================
# ADVANCED SETTINGS
# =====================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Verbose output level (0 = minimal, 1 = normal, 2 = detailed)
VERBOSE = 1

# Enable debug mode
DEBUG_MODE = False

# Data cache directory
CACHE_DIR = './cache'

# Cache expiration (in minutes)
CACHE_EXPIRATION = 5

# =====================================================================
# BACKTESTING SETTINGS
# =====================================================================

# Backtest start date
BACKTEST_START_DATE = '2023-01-01'

# Backtest end date (None = today)
BACKTEST_END_DATE = None

# Backtest split ratio (train/test)
TRAIN_TEST_SPLIT = 0.8

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_config() -> dict:
    """Get all configuration as a dictionary.

    Returns
    -------
    dict
        Configuration dictionary
    """
    return {
        # General
        'ticker': TICKER,
        'initial_capital': INITIAL_CAPITAL,
        'transaction_cost': TRANSACTION_COST,
        'max_options': MAX_OPTIONS,
        'risk_free_rate': RISK_FREE_RATE,

        # Timeframes
        'timeframes': TIMEFRAMES,
        'default_timeframe': DEFAULT_TIMEFRAME,
        'lookback_period': LOOKBACK_PERIOD,
        'reoptimize_freq': REOPTIMIZE_FREQ,

        # Indicators
        'tech_indicators': TECH_INDICATORS,
        'greeks_list': GREEKS_LIST,
        'use_vix': USE_VIX,
        'use_turbulence': USE_TURBULENCE,
        'use_advanced_indicators': USE_ADVANCED_INDICATORS,
        'use_volume_features': USE_VOLUME_FEATURES,
        'use_regime_detection': USE_REGIME_DETECTION,

        # Learning
        'learning_rate': LEARNING_RATE,
        'buffer_size': BUFFER_SIZE,
        'update_frequency': UPDATE_FREQUENCY,
        'initial_training_timesteps': INITIAL_TRAINING_TIMESTEPS,
        'incremental_update_timesteps': INCREMENTAL_UPDATE_TIMESTEPS,

        # Trading
        'update_interval': UPDATE_INTERVAL,
        'max_updates': MAX_UPDATES,
        'auto_save': AUTO_SAVE,
        'model_save_dir': MODEL_SAVE_DIR,
        'results_dir': RESULTS_DIR,

        # Options
        'use_options_data': USE_OPTIONS_DATA,
        'strike_selection_strategy': STRIKE_SELECTION_STRATEGY,

        # Risk
        'max_drawdown_threshold': MAX_DRAWDOWN_THRESHOLD,
        'max_position_size': MAX_POSITION_SIZE,
        'daily_loss_limit': DAILY_LOSS_LIMIT,
    }


def print_config():
    """Print current configuration."""
    config = get_config()

    print("\n" + "="*70)
    print("SPY TRADING TOOL - CONFIGURATION")
    print("="*70)

    for key, value in config.items():
        print(f"{key:.<40} {value}")

    print("="*70 + "\n")


if __name__ == '__main__':
    print_config()
