import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
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
    df = FeatureEngineer(df,
                        use_technical_indicator=True,
                        use_turbulence=True).preprocess_data()


    # Training & Trade data split
    train = data_split(df, config.START_DATE,config.START_TRADE_DATE)
    trade = data_split(df, config.START_TRADE_DATE,config.END_DATE)

    # data normalization
    #feaures_list = list(train.columns)
    #feaures_list.remove('date')
    #feaures_list.remove('tic')
    #feaures_list.remove('close')
    #print(feaures_list)
    #data_normaliser = preprocessing.StandardScaler()
    #train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
    #trade[feaures_list] = data_normaliser.fit_transform(trade[feaures_list])

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension 

    env_setup = EnvSetup(stock_dim = stock_dimension,
                         state_space = state_space,
                         hmax = 100,
                         initial_amount = 1000000,
                         transaction_cost_pct = 0.001)

    env_train = env_setup.create_env_training(data = train,
                                          env_class = StockEnvTrain)
    agent = DRLAgent(env = env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    sac_params_tuning={
                      'batch_size': 64,
                     'buffer_size': 100000,
                      'ent_coef':'auto_0.1',
                     'learning_rate': 0.0001,
                     'learning_starts':200,
                     'timesteps': 50000,
                     'verbose': 0}

    model = agent.train_SAC(model_name = "SAC_{}".format(now), model_params = sac_params_tuning)

    print("==============Start Trading===========")
    env_trade, obs_trade = env_setup.create_env_trading(data = trade,
                                         env_class = StockEnvTrade,
                                         turbulence_threshold=250) 

    df_account_value,df_actions = DRLAgent.DRL_prediction(model=model,
                                                          test_data = trade,
                                                          test_env = env_trade,
                                                          test_obs = obs_trade)
    df_account_value.to_csv("./"+config.RESULTS_DIR+"/df_account_value_"+now+'.csv')
    df_actions.to_csv("./"+config.RESULTS_DIR+"/df_actions_"+now+'.csv')

    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./"+config.RESULTS_DIR+"/perf_stats_all_"+now+'.csv')