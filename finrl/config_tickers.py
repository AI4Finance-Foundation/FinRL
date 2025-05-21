import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta # For technical indicators
import os
from finrl.config_tickers import DOW_30_TICKER
# FinRL imports
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.plot import backtest_stats, backtest_plot, get_daily_return

# Stable Baselines3 imports
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv # For wrapping the env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Set random seeds for reproducibility
import torch
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)



