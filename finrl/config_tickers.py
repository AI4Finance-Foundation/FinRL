from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta  # For technical indicators
import torch
import yfinance as yf
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv  # For wrapping the env

from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_daily_return

# FinRL imports
# Stable Baselines3 imports
# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
