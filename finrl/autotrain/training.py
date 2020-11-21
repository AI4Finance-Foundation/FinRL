import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats



def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(start_date = config.START_DATE,
                     end_date = config.END_DATE,
                     ticker_list = config.DOW_30_TICKER).fetch_data()
    print("==============Start Feature Engineering===========")
    df = FeatureEngineer(df,feature_number=5,
                        use_technical_indicator=True,
                        use_turbulence=True).preprocess_data()


    train = data_split(df, config.START_DATE,config.START_TRADE_DATE)
    trade = data_split(df,config.START_TRADE_DATE,config.END_DATE)
    env_setup = EnvSetup(stock_dim = len(train.tic.unique()))
    env_train = env_setup.create_env_training(data = train,
                                          env_class = StockEnvTrain)
    agent = DRLAgent(env = env_train)
    print("==============Model Training===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    a2c_params_tuning = {'n_steps':5, 
                  'ent_coef':0.005, 
                  'learning_rate':0.0007,
                  'verbose':0,
                  'timesteps':100000}
    model_a2c = agent.train_A2C(model_name = "A2C_{}".format(now), model_params = a2c_params_tuning)

    print("==============Start Trading===========")
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                         env_class = StockEnvTrade,
                                         turbulence_threshold=250) 

    df_account_value = DRLAgent.DRL_prediction(model=model_a2c,
                            test_data = trade,
                            test_env = env_trade,
                            test_obs = obs_trade)
    df_account_value.to_csv("./"+config.RESULTS_DIR+"/"+now+'.csv')

    print("==============Get backtest results===========")
    perf_stats_all = BackTestStats(df_account_value)
    print(perf_stats_all)