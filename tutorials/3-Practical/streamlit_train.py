import streamlit as st
import datetime  as dt
from finrl.train import train
from finrl.test import test
from finrl.config_tickers import *
from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.env_stock_trading.env_stock_papertrading import AlpacaPaperTrading
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, get_daily_return, get_baseline
from train_and_test import backtest_plot, get_baseline
from common import *
from train_alpaca import MODEL_IDX

# img = 'choice_distribution.png'

st.header('FinRL GUI', anchor=None)
st.subheader('Train', anchor=None)
StockPool = st.selectbox("Select a stock pool", ['DOW_30_TICKER', 'CHINESE_STOCK_TICKER', 'NAS_100_TICKER', 'SP_500_TICKER'])
NumStock = st.slider("Select the number of stocks to trade", 1, 50, 30)
Algo = st.selectbox("Select an algorithm", ['PPO', 'TD3', 'SAC'])
AlgoLib = st.selectbox("Select an library", ['elegantrl', 'rllib', 'stable_baselines3'])
TrainStartDate = st.date_input("Select a start date for training", value=dt.date(2022, 6, 11)).strftime("%Y-%-m-%-d")  # datetime.data() to string
TrainEndDate = st.date_input("Select an end date for training", value=dt.date(2022, 8, 11)).strftime("%Y-%-m-%-d")
TrainTradeInterval = st.radio("Select a trade interval", ['5Min', '1Min', '15Min', '30Min', '60Min'], key=1)
Log = st.checkbox("Track log", value=True)

ticker_list = eval(StockPool)[:int(NumStock)]
env = StockTradingEnv
save_path = f'./papertrading_erl/{MODEL_IDX}/'

if st.button('Train'):
      ERL_PARAMS = {"learning_rate": 3e-6, "batch_size": 2048, "gamma": 0.985,
              "seed": 312, "net_dimension": 512, "target_step": 5000, "eval_gap": 30,
              "eval_times": 1}
      st.text('Training Hyperparameters:')
      Hyperparams = st.json(ERL_PARAMS)
      st.text(f'Confs and Models saved to: {save_path}')

      train(start_date=TrainStartDate,
      end_date=TrainEndDate,
      ticker_list=ticker_list,
      data_source='alpaca',
      time_interval=TrainTradeInterval,
      technical_indicator_list=INDICATORS,
      drl_lib=AlgoLib,
      env=env,
      model_name=Algo.lower(),
      API_KEY=API_KEY,
      API_SECRET=API_SECRET,
      API_BASE_URL=API_BASE_URL,
      erl_params=ERL_PARAMS,
      cwd=save_path,  # current_working_dir
      wandb=False,
      break_step=1e7)


st.subheader('BackTest', anchor=None)
TestStartDate = st.date_input("Select a start date for backtest", value=dt.date(2022, 6, 1)).strftime("%Y-%-m-%-d") 
TestEndDate = st.date_input("Select an end date for backtest", value=dt.date(2022, 9, 1)).strftime("%Y-%-m-%-d") 
TestTradeInterval = st.radio("Select a trade interval", ['5Min', '1Min', '15Min', '30Min', '60Min'], key=2)
Baseline = st.selectbox("Select a baseline", ['^DJI',])
# print(TestStartDate, TestEndDate, TestTradeInterval)
# st.image(img, width=1000)

if st.button('BackTest'):  # test

      account_value = test(start_date = TestStartDate, 
            end_date = TestEndDate,
            ticker_list = ticker_list, 
            data_source = 'alpaca',
            time_interval= TestTradeInterval, 
            technical_indicator_list= INDICATORS,
            drl_lib=AlgoLib, 
            env=env,
            model_name=Algo.lower(), 
            API_KEY = API_KEY, 
            API_SECRET = API_SECRET, 
            API_BASE_URL = API_BASE_URL,
      #       erl_params=ERL_PARAMS,
            cwd=save_path, #current_working_dir
            if_plot=True, # to return a dataframe for backtest_plot
            break_step=1e7)

      #baseline stats
      print("==============Get Baseline Stats===========")
      baseline_df = get_baseline(
            ticker = Baseline, 
            start = TestStartDate,
            end = TestEndDate)

      stats = backtest_stats(baseline_df, value_col_name = 'close')

      print("==============Compare to DJIA===========")
      # %matplotlib inline
      # S&P 500: ^GSPC
      # Dow Jones Index: ^DJI
      # NASDAQ 100: ^NDX
      figs = backtest_plot(account_value, baseline_df)
      from PIL import Image

      figs.savefig(save_path + '/backtest.png')
      image = Image.open(save_path + '/backtest.png')
      st.subheader('BackTest Results', anchor=None)
      st.image(image, caption='Results')


else:
      pass