from __future__ import annotations

import copy
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import sys
from finrl.meta.data_processors.func import date2str, str2date
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.meta.data_processor import DataProcessor

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, plot_return
import itertools

from finrl import config
from finrl import config_tickers
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

def main():
    TRAIN_START_DATE = '2021-01-01'
    TRAIN_END_DATE = '2022-10-01'
    TRADE_START_DATE = '2022-10-02'
    TRADE_END_DATE = '2023-02-01'
    if_using_a2c = True
    if_using_ddpg = True
    if_using_ppo = True
    if_using_td3 = True
    if_using_sac = True


    sys.path.append('../FinRL')
    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])

    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRADE_END_DATE,
                         ticker_list=config_tickers.DOW_30_TICKER).fetch_data()
    df.sort_values(['date', 'tic'], ignore_index=True).head()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False)

    processed = fe.preprocess_data(df)
    list_ticker = processed['tic'].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(combination, columns=['date', 'tic']).merge(processed, on=['date', 'tic'], how='left')
    init_train_trade_data = init_train_trade_data[init_train_trade_data['date'].isin(processed['date'])]
    init_train_trade_data = init_train_trade_data.sort_values(['date', 'tic'])

    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_trade_data.sort_values(['date', 'tic'], ignore_index=True).head(10)

    init_train_data = data_split(init_train_trade_data, TRAIN_START_DATE, TRAIN_END_DATE)
    init_trade_data = data_split(init_train_trade_data, TRADE_START_DATE, TRADE_END_DATE)

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f'Stock Dimension: {stock_dimension}, State Space: {state_space}')
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        'hmax': 100,
        'initial_amount': 1000000,
        'num_stock_shares': num_stock_shares,
        'buy_cost_pct': buy_cost_list,
        'sell_cost_pct': sell_cost_list,
        'state_space': state_space,
        'stock_dim': stock_dimension,
        'tech_indicator_list': INDICATORS,
        'action_space': stock_dimension,
        'reward_scaling': 1e-4
    }

    e_train_gym = StockTradingEnv(df=init_train_data, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))



    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model('a2c')

    if if_using_a2c:
        # set up logger
        tmp_path = RESULTS_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)

    trained_a2c = agent.train_model(model=model_a2c,
                                    tb_log_name='a2c',
                                    total_timesteps=50000) if if_using_a2c else None

    agent = DRLAgent(env=env_train)
    model_ddpg = agent.get_model('ddpg')

    if if_using_ddpg:
        # set up logger
        tmp_path = RESULTS_DIR + '/ddpg'
        new_logger_ddpg = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
        # Set new logger
        model_ddpg.set_logger(new_logger_ddpg)
    trained_ddpg = agent.train_model(model=model_ddpg,
                                     tb_log_name='ddpg',
                                     total_timesteps=50000) if if_using_ddpg else None

    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        'n_steps': 2048,
        'ent_coef': 0.01,
        'learning_rate': 0.00025,
        'batch_size': 128,
    }
    model_ppo = agent.get_model('ppo', model_kwargs=PPO_PARAMS)
    if if_using_ppo:
        # set up logger
        tmp_path = RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
        # Set new logger
        model_ppo.set_logger(new_logger_ppo)
    trained_ppo = agent.train_model(model=model_ppo,
                                    tb_log_name='ppo',
                                    total_timesteps=50000) if if_using_ppo else None

    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {'batch_size': 100,
                  'buffer_size': 1000000,
                  'learning_rate': 0.001}
    model_td3 = agent.get_model('td3', model_kwargs=TD3_PARAMS)
    if if_using_td3:
        # set up logger
        tmp_path = RESULTS_DIR + '/td3'
        new_logger_td3 = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
        # Set new logger
        model_td3.set_logger(new_logger_td3)
    trained_td3 = agent.train_model(model=model_td3,
                                    tb_log_name='td3',
                                    total_timesteps=50000) if if_using_td3 else None

    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        'batch_size': 128,
        'buffer_size': 100000,
        'learning_rate': 0.0001,
        'learning_starts': 100,
        'ent_coef': 'auto_0.1',
    }
    model_sac = agent.get_model('sac', model_kwargs=SAC_PARAMS)
    if if_using_sac:
        # set up logger
        tmp_path = RESULTS_DIR + '/sac'
        new_logger_sac = configure(tmp_path, ['stdout', 'csv', 'tensorboard'])
        # Set new logger
        model_sac.set_logger(new_logger_sac)
    trained_sac = agent.train_model(model=model_sac,
                                    tb_log_name='sac',
                                    total_timesteps=50000) if if_using_sac else None

    e_trade_gym = StockTradingEnv(df=init_trade_data, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    trained_model = trained_a2c
    result_a2c, actions_a2c = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym) if if_using_a2c else None, None

    trained_model = trained_ddpg
    result_ddpg, actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym) if if_using_ddpg else None, None

    trained_model = trained_ppo
    result_ppo, actions_ppo = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym) if if_using_ppo else None, None

    trained_model = trained_td3
    result_td3, actions_td3 = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym) if if_using_td3 else None, None

    trained_model = trained_sac
    result_sac, actions_sac = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym) if if_using_sac else None, None

    ### used in python version, instead of notebook version
    if isinstance(result_a2c, tuple):
        actions_a2c = result_a2c[1]
        result_a2c = result_a2c[0]
    if isinstance(result_ddpg, tuple):
        actions_ddpg = result_ddpg[1]
        result_ddpg = result_ddpg[0]
    if isinstance(result_td3, tuple):
        actions_td3 = result_td3[1]
        result_td3 = result_td3[0]
    if isinstance(result_ppo, tuple):
        actions_ppo = result_ppo[1]
        result_ppo = result_ppo[0]
    if isinstance(result_sac, tuple):
        actions_sac = result_sac[1]
        result_sac = result_sac[0]

    # store actions
    store_actions = True
    if store_actions:
        actions_a2c.to_csv('actions_a2c.csv') if if_using_a2c else None
        actions_ddpg.to_csv('actions_ddpg.csv') if if_using_ddpg else None
        actions_td3.to_csv('actions_td3.csv') if if_using_td3 else None
        actions_ppo.to_csv('actions_ppo.csv') if if_using_ppo else None
        actions_sac.to_csv('actions_sac.csv') if if_using_sac else None


    # baseline stats
    print('==============Get Baseline Stats===========')
    df_dji_ = get_baseline(
        ticker='^DJI',
        start=TRADE_START_DATE,
        end=TRADE_END_DATE)
    stats = backtest_stats(df_dji_, value_col_name='close')
    print('stats of dji: ', stats)
    df_dji = pd.DataFrame()
    df_dji['date'] = df_dji_['date']
    df_dji['account_value'] = df_dji_['close'] / df_dji_['close'].tolist()[0] * env_kwargs['initial_amount']
    df_dji.to_csv('df_dji.csv')

    tmp_df_dji = copy.deepcopy(df_dji)
    tmp_df_dji.rename(columns={'account_value': 'DJI'}, inplace=True)
    result = tmp_df_dji

    if if_using_a2c:
        result_a2c.rename(columns={'account_value': 'A2C'}, inplace=True)
        result = pd.merge(result, result_a2c, how='left')
    if if_using_ddpg:
        result_ddpg.rename(columns={'account_value': 'DDPG'}, inplace=True)
        result = pd.merge(result, result_ddpg, how='left')
    if if_using_td3:
        result_td3.rename(columns={'account_value': 'TD3'}, inplace=True)
        result = pd.merge(result, result_td3, how='left')
    if if_using_ppo:
        result_ppo.rename(columns={'account_value': 'PPO'}, inplace=True)
        result = pd.merge(result, result_ppo, how='left')
    if if_using_sac:
        result_sac.rename(columns={'account_value': 'SAC'}, inplace=True)
        result = pd.merge(result, result_sac, how='left')

    # remove the rows with nan
    result = result.dropna(axis=0, how='any')
    # make sure that the first row is env_kwargs['initial_amount']
    for col in result.columns:
        if col != 'date' and result[col].tolist()[0] != env_kwargs['initial_amount']:
            result[col] = result[col] / result[col].tolist()[0] * env_kwargs['initial_amount']

    result = result.reset_index(drop=True)
    print('result: ', result)
    result.to_csv('result.csv')
    # plt.rcParams['figure.figsize'] = (15, 5)
    # plt.figure()
    # result.plot()
    # plot fig
    plot_return(result=result, \
                column_as_x='date', \
                if_need_calc_return=True, \
                savefig_filename='stock.png', \
                xlabel='Date', \
                ylabel='Return',
                if_transfer_date=True,
                num_days_xticks = 20,
                )



if __name__ == '__main__':
    main()
