import streamlit as st
import datetime  as dt
# img = 'choice_distribution.png'

st.header('FinRL GUI', anchor=None)
st.subheader('Train', anchor=None)
StockPool = st.selectbox("Select a stock pool", ['DOW_30_TICKER', 'CHINESE_STOCK_TICKER', 'NAS_100_TICKER', 'SP_500_TICKER'])
NumStock = st.slider("Select the number of stocks to trade", 1, 50, 30)
Algo = st.selectbox("Select an algorithm", ['PPO', 'TD3', 'SAC'])
TrainStartDate = st.date_input("Select a start date for training", value=dt.date(2022, 6, 11))
TrainEndDate = st.date_input("Select an end date for training", value=dt.date(2022, 8, 11))
TrainTradeInterval = st.radio("Select a trade interval", ['5Min', '1Min', '15Min', '30Min', '60Min'], key=1)
Log = st.checkbox("Track log", value=True)


st.subheader('BackTest', anchor=None)
TestStartDate = st.date_input("Select a start date for backtest", value=dt.date(2019, 1, 1))
TestEndDate = st.date_input("Select an end date for backtest", value=dt.date(2023, 1, 1))
TestTradeInterval = st.radio("Select a trade interval", ['5Min', '1Min', '15Min', '30Min', '60Min'], key=2)
Baseline = st.selectbox("Select a baseline", ['^DJI',])
print(TestStartDate, TestEndDate, TestTradeInterval)
# st.image(img, width=1000)


# test
from finrl.train import train
from finrl.test import test
from finrl.config_tickers import *
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from common import *
from train_alpaca import MODEL_IDX

ticker_list = eval(StockPool)
action_dim = len(ticker_list)

env = StockTradingEnv

account_value = test(start_date = TestStartDate, 
      end_date = TestEndDate,
      ticker_list = eval(StockPool)[:int(NumStock)], 
      data_source = 'alpaca',
      time_interval= TestTradeInterval, 
      technical_indicator_list= INDICATORS,
      drl_lib='elegantrl', 
      env=env,
      model_name=Algo.lower(), 
      API_KEY = API_KEY, 
      API_SECRET = API_SECRET, 
      API_BASE_URL = API_BASE_URL,
#       erl_params=ERL_PARAMS,
      cwd=f'./papertrading_erl/{MODEL_IDX}', #current_working_dir
      if_plot=True, # to return a dataframe for backtest_plot
      break_step=1e7)

#baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
        ticker=Baseline, 
        start = TestStartDate,
        end = TestEndDate)

stats = backtest_stats(baseline_df, value_col_name = 'close')

print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(account_value, 
             baseline_ticker = Baseline, 
             baseline_start = TestStartDate,
             baseline_end = TestEndDate)
