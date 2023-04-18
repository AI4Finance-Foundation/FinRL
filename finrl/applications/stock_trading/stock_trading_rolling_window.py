from __future__ import annotations

import copy
import datetime
import itertools
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure

from finrl import config
from finrl import config_tickers
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import DATA_SAVE_DIR
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config_tickers import DOW_30_TICKER
from finrl.main import check_and_make_directories
from finrl.meta.data_processor import DataProcessor
from finrl.meta.data_processors.func import calc_train_trade_data
from finrl.meta.data_processors.func import calc_train_trade_starts_ends_if_rolling
from finrl.meta.data_processors.func import date2str
from finrl.meta.data_processors.func import str2date
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.plot import backtest_plot
from finrl.plot import backtest_stats
from finrl.plot import get_baseline
from finrl.plot import get_daily_return
from finrl.plot import plot_return

# matplotlib.use('Agg')


def stock_trading_rolling_window(
    train_start_date: str,
    train_end_date: str,
    trade_start_date: str,
    trade_end_date: str,
    rolling_window_length: int,
    if_store_actions: bool = True,
    if_store_result: bool = True,
    if_using_a2c: bool = True,
    if_using_ddpg: bool = True,
    if_using_ppo: bool = True,
    if_using_sac: bool = True,
    if_using_td3: bool = True,
):
    # sys.path.append("../FinRL")
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )
    date_col = "date"
    tic_col = "tic"
    df = YahooDownloader(
        start_date=train_start_date, end_date=trade_end_date, ticker_list=DOW_30_TICKER
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed[tic_col].unique().tolist()
    list_date = list(
        pd.date_range(processed[date_col].min(), processed[date_col].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    init_train_trade_data = pd.DataFrame(
        combination, columns=[date_col, tic_col]
    ).merge(processed, on=[date_col, tic_col], how="left")
    init_train_trade_data = init_train_trade_data[
        init_train_trade_data[date_col].isin(processed[date_col])
    ]
    init_train_trade_data = init_train_trade_data.sort_values([date_col, tic_col])

    init_train_trade_data = init_train_trade_data.fillna(0)

    init_train_data = data_split(
        init_train_trade_data, train_start_date, train_end_date
    )
    init_trade_data = data_split(
        init_train_trade_data, trade_start_date, trade_end_date
    )

    stock_dimension = len(init_train_data.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    initial_amount = 1000000
    env_kwargs = {
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    # split the init_train_data and init_trade_data to subsets
    init_train_dates = init_train_data[date_col].unique()
    init_trade_dates = init_trade_data[date_col].unique()

    (
        train_starts,
        train_ends,
        trade_starts,
        trade_ends,
    ) = calc_train_trade_starts_ends_if_rolling(
        init_train_dates, init_trade_dates, rolling_window_length
    )

    result = pd.DataFrame()
    actions_a2c = pd.DataFrame(columns=DOW_30_TICKER)
    actions_ddpg = pd.DataFrame(columns=DOW_30_TICKER)
    actions_ppo = pd.DataFrame(columns=DOW_30_TICKER)
    actions_sac = pd.DataFrame(columns=DOW_30_TICKER)
    actions_td3 = pd.DataFrame(columns=DOW_30_TICKER)

    for i in range(len(train_starts)):
        print("i: ", i)
        train_data, trade_data = calc_train_trade_data(
            i,
            train_starts,
            train_ends,
            trade_starts,
            trade_ends,
            init_train_data,
            init_trade_data,
            date_col,
        )
        e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        # train

        if if_using_a2c:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["A2C"].iloc[-1]
            e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env=env_train)
            model_a2c = agent.get_model("a2c")
            # set up logger
            tmp_path = RESULTS_DIR + "/a2c"
            new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_a2c.set_logger(new_logger_a2c)
            trained_a2c = agent.train_model(
                model=model_a2c, tb_log_name="a2c", total_timesteps=50000
            )

        if if_using_ddpg:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["DDPG"].iloc[-1]
            e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env=env_train)
            model_ddpg = agent.get_model("ddpg")
            # set up logger
            tmp_path = RESULTS_DIR + "/ddpg"
            new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_ddpg.set_logger(new_logger_ddpg)
            trained_ddpg = agent.train_model(
                model=model_ddpg, tb_log_name="ddpg", total_timesteps=40000
            )

        if if_using_ppo:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["PPO"].iloc[-1]
            e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env=env_train)
            PPO_PARAMS = {
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0001,
                "batch_size": 64,
            }
            model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
            # set up logger
            tmp_path = RESULTS_DIR + "/ppo"
            new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_ppo.set_logger(new_logger_ppo)
            trained_ppo = agent.train_model(
                model=model_ppo, tb_log_name="ppo", total_timesteps=50000
            )

        if if_using_sac:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["SAC"].iloc[-1]
            e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env=env_train)
            SAC_PARAMS = {
                "batch_size": 64,
                "buffer_size": 100000,
                "learning_rate": 0.00015,
                "learning_starts": 100,
                "ent_coef": "auto_0.1",
            }
            model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
            # set up logger
            tmp_path = RESULTS_DIR + "/sac"
            new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_sac.set_logger(new_logger_sac)
            trained_sac = agent.train_model(
                model=model_sac, tb_log_name="sac", total_timesteps=50000
            )

        if if_using_td3:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["TD3"].iloc[-1]
            e_train_gym = StockTradingEnv(df=train_data, **env_kwargs)
            env_train, _ = e_train_gym.get_sb_env()
            agent = DRLAgent(env=env_train)
            TD3_PARAMS = {
                "batch_size": 64,
                "buffer_size": 100000,
                "learning_rate": 0.0008,
            }
            model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
            # set up logger
            tmp_path = RESULTS_DIR + "/td3"
            new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
            # Set new logger
            model_td3.set_logger(new_logger_td3)
            trained_td3 = agent.train_model(
                model=model_td3, tb_log_name="td3", total_timesteps=50000
            )

        # trade
        # this e_trade_gym is initialized, then it will be used if i == 0
        e_trade_gym = StockTradingEnv(
            df=trade_data,
            turbulence_threshold=70,
            risk_indicator_col="vix",
            **env_kwargs,
        )

        if if_using_a2c:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["A2C"].iloc[-1]
                e_trade_gym = StockTradingEnv(
                    df=trade_data,
                    turbulence_threshold=70,
                    risk_indicator_col="vix",
                    **env_kwargs,
                )
            result_a2c, actions_i_a2c = DRLAgent.DRL_prediction(
                model=trained_a2c, environment=e_trade_gym
            )

        if if_using_ddpg:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["DDPG"].iloc[-1]
                e_trade_gym = StockTradingEnv(
                    df=trade_data,
                    turbulence_threshold=70,
                    risk_indicator_col="vix",
                    **env_kwargs,
                )
            result_ddpg, actions_i_ddpg = DRLAgent.DRL_prediction(
                model=trained_ddpg, environment=e_trade_gym
            )

        if if_using_ppo:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["PPO"].iloc[-1]
                e_trade_gym = StockTradingEnv(
                    df=trade_data,
                    turbulence_threshold=70,
                    risk_indicator_col="vix",
                    **env_kwargs,
                )
            result_ppo, actions_i_ppo = DRLAgent.DRL_prediction(
                model=trained_ppo, environment=e_trade_gym
            )

        if if_using_sac:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["SAC"].iloc[-1]
                e_trade_gym = StockTradingEnv(
                    df=trade_data,
                    turbulence_threshold=70,
                    risk_indicator_col="vix",
                    **env_kwargs,
                )
            result_sac, actions_i_sac = DRLAgent.DRL_prediction(
                model=trained_sac, environment=e_trade_gym
            )

        if if_using_td3:
            if len(result) >= 1:
                env_kwargs["initial_amount"] = result["TD3"].iloc[-1]
                e_trade_gym = StockTradingEnv(
                    df=trade_data,
                    turbulence_threshold=70,
                    risk_indicator_col="vix",
                    **env_kwargs,
                )
            result_td3, actions_i_td3 = DRLAgent.DRL_prediction(
                model=trained_td3, environment=e_trade_gym
            )

        # in python version, we should check isinstance, but in notebook version, it is not necessary
        if if_using_a2c and isinstance(result_a2c, tuple):
            actions_i_a2c = result_a2c[1]
            result_a2c = result_a2c[0]
        if if_using_ddpg and isinstance(result_ddpg, tuple):
            actions_i_ddpg = result_ddpg[1]
            result_ddpg = result_ddpg[0]
        if if_using_ppo and isinstance(result_ppo, tuple):
            actions_i_ppo = result_ppo[1]
            result_ppo = result_ppo[0]
        if if_using_sac and isinstance(result_sac, tuple):
            actions_i_sac = result_sac[1]
            result_sac = result_sac[0]
        if if_using_td3 and isinstance(result_td3, tuple):
            actions_i_td3 = result_td3[1]
            result_td3 = result_td3[0]

        # merge actions
        actions_a2c = pd.concat([actions_a2c, actions_i_a2c]) if if_using_a2c else None
        actions_ddpg = (
            pd.concat([actions_ddpg, actions_i_ddpg]) if if_using_ddpg else None
        )
        actions_ppo = pd.concat([actions_ppo, actions_i_ppo]) if if_using_ppo else None
        actions_sac = pd.concat([actions_sac, actions_i_sac]) if if_using_sac else None
        actions_td3 = pd.concat([actions_td3, actions_i_td3]) if if_using_td3 else None

        # dji_i
        trade_start = trade_starts[i]
        trade_end = trade_ends[i]
        dji_i_ = get_baseline(ticker="^DJI", start=trade_start, end=trade_end)
        dji_i = pd.DataFrame()
        dji_i[date_col] = dji_i_[date_col]
        dji_i["DJI"] = dji_i_["close"]
        # dji_i.rename(columns={'account_value': 'DJI'}, inplace=True)

        # select the rows between trade_start and trade_end (not included), since some values may not in this region
        dji_i = dji_i.loc[
            (dji_i[date_col] >= trade_start) & (dji_i[date_col] < trade_end)
        ]

        # init result_i by dji_i
        result_i = dji_i

        # rename column name of result_a2c, result_ddpg, etc., and then put them to result_i
        if if_using_a2c:
            result_a2c.rename(columns={"account_value": "A2C"}, inplace=True)
            result_i = pd.merge(result_i, result_a2c, how="left")
        if if_using_ddpg:
            result_ddpg.rename(columns={"account_value": "DDPG"}, inplace=True)
            result_i = pd.merge(result_i, result_ddpg, how="left")
        if if_using_ppo:
            result_ppo.rename(columns={"account_value": "PPO"}, inplace=True)
            result_i = pd.merge(result_i, result_ppo, how="left")
        if if_using_sac:
            result_sac.rename(columns={"account_value": "SAC"}, inplace=True)
            result_i = pd.merge(result_i, result_sac, how="left")
        if if_using_td3:
            result_td3.rename(columns={"account_value": "TD3"}, inplace=True)
            result_i = pd.merge(result_i, result_td3, how="left")

        # remove the rows with nan
        result_i = result_i.dropna(axis=0, how="any")

        # merge result_i to result
        result = pd.concat([result, result_i], axis=0)

    # store actions
    if if_store_actions:
        actions_a2c.to_csv("actions_a2c.csv") if if_using_a2c else None
        actions_ddpg.to_csv("actions_ddpg.csv") if if_using_ddpg else None
        actions_ppo.to_csv("actions_ppo.csv") if if_using_ppo else None
        actions_sac.to_csv("actions_sac.csv") if if_using_sac else None
        actions_td3.to_csv("actions_td3.csv") if if_using_td3 else None

    # calc the column name of strategies, including DJI
    col_strategies = []
    for col in result.columns:
        if col != date_col and col != "" and "Unnamed" not in col:
            col_strategies.append(col)

    # make sure that the first row of DJI is initial_amount
    col = "DJI"
    result[col] = result[col] / result[col].iloc[0] * initial_amount
    result = result.reset_index(drop=True)

    # stats
    for col in col_strategies:
        stats = backtest_stats(result, value_col_name=col)
        print("\nstats of " + col + ": \n", stats)

    # print and save result
    print("result: ", result)
    if if_store_result:
        result.to_csv("result.csv")

    # plot fig
    plot_return(
        result=result,
        column_as_x=date_col,
        if_need_calc_return=True,
        savefig_filename="stock_trading_rolling_window.png",
        xlabel="Date",
        ylabel="Return",
        if_transfer_date=True,
        num_days_xticks=20,
    )


if __name__ == "__main__":
    train_start_date = "2009-01-01"
    train_end_date = "2022-07-01"
    trade_start_date = "2022-07-01"
    trade_end_date = "2022-11-01"
    rolling_window_length = 22  # num of trading days in a rolling window
    if_store_actions = True
    if_store_result = True
    if_using_a2c = True
    if_using_ddpg = True
    if_using_ppo = True
    if_using_sac = True
    if_using_td3 = True
    stock_trading_rolling_window(
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        trade_start_date=trade_start_date,
        trade_end_date=trade_end_date,
        rolling_window_length=rolling_window_length,
        if_store_actions=if_store_actions,
        if_store_result=if_store_result,
        if_using_a2c=if_using_a2c,
        if_using_ddpg=if_using_ddpg,
        if_using_ppo=if_using_ppo,
        if_using_sac=if_using_sac,
        if_using_td3=if_using_td3,
    )
