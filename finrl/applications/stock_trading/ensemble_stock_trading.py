def main():
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    import datetime


    from finrl.config_tickers import DOW_30_TICKER
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

    from pprint import pprint

    import sys
    sys.path.append("../FinRL-Library")

    import itertools

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

    check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])
    print(DOW_30_TICKER)
    TRAIN_START_DATE = '2009-04-01'
    TRAIN_END_DATE = '2021-01-01'
    TEST_START_DATE = '2021-01-01'
    TEST_END_DATE = '2022-06-01'

    df = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TEST_END_DATE,
                         ticker_list=DOW_30_TICKER).fetch_data()

    df.sort_values(['date', 'tic']).head()

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_turbulence=True,
                         user_defined_feature=False)

    processed = fe.preprocess_data(df)
    processed = processed.copy()
    processed = processed.fillna(0)
    processed = processed.replace(np.inf, 0)

    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 5

    }

    rebalance_window = 63  # rebalance_window is the number of days to retrain the model
    validation_window = 63  # validation_window is the number of days to do validation and trading (e.g. if validation_window=63, then both validation and trading period will be 63 days)

    ensemble_agent = DRLEnsembleAgent(df=processed,
                                      train_period=(TRAIN_START_DATE, TRAIN_END_DATE),
                                      val_test_period=(TEST_START_DATE, TEST_END_DATE),
                                      rebalance_window=rebalance_window,
                                      validation_window=validation_window,
                                      **env_kwargs)

    A2C_model_kwargs = {
        'n_steps': 5,
        'ent_coef': 0.005,
        'learning_rate': 0.0007
    }

    PPO_model_kwargs = {
        "ent_coef": 0.01,
        "n_steps": 2048,
        "learning_rate": 0.00025,
        "batch_size": 128
    }

    DDPG_model_kwargs = {
        # "action_noise":"ornstein_uhlenbeck",
        "buffer_size": 10_000,
        "learning_rate": 0.0005,
        "batch_size": 64
    }

    timesteps_dict = {'a2c': 10_000,
                      'ppo': 10_000,
                      'ddpg': 10_000
                      }
    df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                      PPO_model_kwargs,
                                                      DDPG_model_kwargs,
                                                      timesteps_dict)

    unique_trade_date = processed[(processed.date > TEST_START_DATE) & (processed.date <= TEST_END_DATE)].date.unique()

    df_trade_date = pd.DataFrame({'datadate': unique_trade_date})

    df_account_value = pd.DataFrame()
    for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
        temp = pd.read_csv('results/account_value_trade_{}_{}.csv'.format('ensemble', i))
        df_account_value = df_account_value.append(temp, ignore_index=True)
    sharpe = (252 ** 0.5) * df_account_value.account_value.pct_change(
        1).mean() / df_account_value.account_value.pct_change(1).std()
    print('Sharpe Ratio: ', sharpe)
    df_account_value = df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

    df_account_value.account_value.plot()

    print("==============Get Backtest Results===========")
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

    perf_stats_all = backtest_stats(account_value=df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)

    # baseline stats
    print("==============Get Baseline Stats===========")
    baseline_df = get_baseline(
        ticker="^DJI",
        start=df_account_value.loc[0, 'date'],
        end=df_account_value.loc[len(df_account_value) - 1, 'date'])

    stats = backtest_stats(baseline_df, value_col_name='close')

    print("==============Compare to DJIA===========")

    # S&P 500: ^GSPC
    # Dow Jones Index: ^DJI
    # NASDAQ 100: ^NDX
    backtest_plot(df_account_value,
                  baseline_ticker='^DJI',
                  baseline_start=df_account_value.loc[0, 'date'],
                  baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])


if __name__ == '__main__':
    main()

