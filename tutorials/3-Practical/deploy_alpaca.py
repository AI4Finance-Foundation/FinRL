import numpy as np
import pandas as pd

from finrl.train import train
from finrl.test import test
from finrl.config_tickers import DOW_30_TICKER
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from alpaca_trade import AlpacaPaperTrading
from common import *

ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim
candle_time_interval = '15Min'  # '1Min'

env = StockTradingEnv

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":512, "target_step":5000, "eval_gap":30,
        "eval_times":1} 
#if you want to use larger datasets (change to longer period), and it raises error, 
#please try to increase "target_step". It should be larger than the episode steps. 

paper_trading_erl = AlpacaPaperTrading(ticker_list = DOW_30_TICKER, 
                                       time_interval = candle_time_interval, 
                                       drl_lib = 'elegantrl', 
                                       agent = 'ppo', 
                                       cwd = './papertrading_erl_retrain', 
                                       net_dim = 512, 
                                       state_dim = state_dim, 
                                       action_dim= action_dim, 
                                       API_KEY = API_KEY, 
                                       API_SECRET = API_SECRET, 
                                       API_BASE_URL = API_BASE_URL, 
                                       tech_indicator_list = INDICATORS, 
                                       turbulence_thresh=30, 
                                       max_stock=1e2)
paper_trading_erl.run()