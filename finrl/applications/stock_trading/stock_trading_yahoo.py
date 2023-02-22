from __future__ import annotations


def main():
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    # matplotlib.use('Agg')
    import datetime

    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    from finrl.agents.stablebaselines3.models import DRLAgent
    from stable_baselines3.common.logger import configure
    from finrl.meta.data_processor import DataProcessor

    from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
    from pprint import pprint

    import sys

    sys.path.append("../FinRL")

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

    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )

    TRAIN_START_DATE = "2009-01-01"
    TRAIN_END_DATE = "2020-07-01"
    TRADE_START_DATE = "2020-07-01"
    TRADE_END_DATE = "2021-10-31"

    df = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=config_tickers.DOW_30_TICKER,
    ).fetch_data()
    print(config_tickers.DOW_30_TICKER)

    df.sort_values(["date", "tic"], ignore_index=True).head()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])

    processed_full = processed_full.fillna(0)

    processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

    mvo_df = processed_full.sort_values(["date", "tic"], ignore_index=True)[
        ["date", "tic", "close"]
    ]

    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(len(train))
    print(len(trade))

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))

    agent = DRLAgent(env=env_train)

    if_using_a2c = True
    if_using_ddpg = True
    if_using_ppo = True
    if_using_td3 = True
    if_using_sac = True

    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")

    if if_using_a2c:
        # set up logger
        tmp_path = RESULTS_DIR + "/a2c"
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)

    trained_a2c = (
        agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=50000)
        if if_using_a2c
        else None
    )

    agent = DRLAgent(env=env_train)
    model_ddpg = agent.get_model("ddpg")

    if if_using_ddpg:
        # set up logger
        tmp_path = RESULTS_DIR + "/ddpg"
        new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ddpg.set_logger(new_logger_ddpg)

    trained_ddpg = (
        agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000)
        if if_using_ddpg
        else None
    )

    agent = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 128,
    }
    model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    if if_using_ppo:
        # set up logger
        tmp_path = RESULTS_DIR + "/ppo"
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ppo.set_logger(new_logger_ppo)

    trained_ppo = (
        agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=50000)
        if if_using_ppo
        else None
    )

    agent = DRLAgent(env=env_train)
    TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}

    model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)

    if if_using_td3:
        # set up logger
        tmp_path = RESULTS_DIR + "/td3"
        new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_td3.set_logger(new_logger_td3)

    trained_td3 = (
        agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=50000)
        if if_using_td3
        else None
    )

    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }

    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)

    if if_using_sac:
        # set up logger
        tmp_path = RESULTS_DIR + "/sac"
        new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_sac.set_logger(new_logger_sac)

    trained_sac = (
        agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=50000)
        if if_using_sac
        else None
    )

    data_risk_indicator = processed_full[
        (processed_full.date < TRAIN_END_DATE)
        & (processed_full.date >= TRAIN_START_DATE)
    ]
    insample_risk_indicator = data_risk_indicator.drop_duplicates(subset=["date"])

    insample_risk_indicator.vix.describe()

    insample_risk_indicator.vix.quantile(0.996)

    insample_risk_indicator.turbulence.describe()

    insample_risk_indicator.turbulence.quantile(0.996)

    e_trade_gym = StockTradingEnv(
        df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs
    )
    # env_trade, obs_trade = e_trade_gym.get_sb_env()

    trained_moedl = trained_a2c
    df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
        model=trained_moedl, environment=e_trade_gym
    )

    trained_moedl = trained_ddpg
    df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
        model=trained_moedl, environment=e_trade_gym
    )

    trained_moedl = trained_ppo
    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_moedl, environment=e_trade_gym
    )

    trained_moedl = trained_td3
    df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
        model=trained_moedl, environment=e_trade_gym
    )

    trained_moedl = trained_sac
    df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
        model=trained_moedl, environment=e_trade_gym
    )

    fst = mvo_df
    fst = fst.iloc[0 * 29 : 0 * 29 + 29, :]
    tic = fst["tic"].tolist()

    mvo = pd.DataFrame()

    for k in range(len(tic)):
        mvo[tic[k]] = 0

    for i in range(mvo_df.shape[0] // 29):
        n = mvo_df
        n = n.iloc[i * 29 : i * 29 + 29, :]
        date = n["date"][i * 29]
        mvo.loc[date] = n["close"].tolist()

    from scipy import optimize
    from scipy.optimize import linprog

    # function obtains maximal return portfolio using linear programming

    def MaximizeReturns(MeanReturns, PortfolioSize):
        # dependencies

        c = np.multiply(-1, MeanReturns)
        A = np.ones([PortfolioSize, 1]).T
        b = [1]
        res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="simplex")

        return res

    def MinimizeRisk(CovarReturns, PortfolioSize):
        def f(x, CovarReturns):
            func = np.matmul(np.matmul(x, CovarReturns), x.T)
            return func

        def constraintEq(x):
            A = np.ones(x.shape)
            b = 1
            constraintVal = np.matmul(A, x.T) - b
            return constraintVal

        xinit = np.repeat(0.1, PortfolioSize)
        cons = {"type": "eq", "fun": constraintEq}
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        opt = optimize.minimize(
            f,
            x0=xinit,
            args=(CovarReturns),
            bounds=bnds,
            constraints=cons,
            tol=10**-3,
        )

        return opt

    def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
        def f(x, CovarReturns):
            func = np.matmul(np.matmul(x, CovarReturns), x.T)
            return func

        def constraintEq(x):
            AEq = np.ones(x.shape)
            bEq = 1
            EqconstraintVal = np.matmul(AEq, x.T) - bEq
            return EqconstraintVal

        def constraintIneq(x, MeanReturns, R):
            AIneq = np.array(MeanReturns)
            bIneq = R
            IneqconstraintVal = np.matmul(AIneq, x.T) - bIneq
            return IneqconstraintVal

        xinit = np.repeat(0.1, PortfolioSize)
        cons = (
            {"type": "eq", "fun": constraintEq},
            {"type": "ineq", "fun": constraintIneq, "args": (MeanReturns, R)},
        )
        lb = 0
        ub = 1
        bnds = tuple([(lb, ub) for x in xinit])

        opt = optimize.minimize(
            f,
            args=(CovarReturns),
            method="trust-constr",
            x0=xinit,
            bounds=bnds,
            constraints=cons,
            tol=10**-3,
        )

        return opt

    def StockReturnsComputing(StockPrice, Rows, Columns):
        import numpy as np

        StockReturn = np.zeros([Rows - 1, Columns])
        for j in range(Columns):  # j: Assets
            for i in range(Rows - 1):  # i: Daily Prices
                StockReturn[i, j] = (
                    (StockPrice[i + 1, j] - StockPrice[i, j]) / StockPrice[i, j]
                ) * 100

        return StockReturn

    # Obtain optimal portfolio sets that maximize return and minimize risk

    # Dependencies
    import numpy as np
    import pandas as pd

    # input k-portfolio 1 dataset comprising 15 stocks
    # StockFileName = './DJIA_Apr112014_Apr112019_kpf1.csv'

    Rows = 1259  # excluding header
    Columns = 15  # excluding date
    portfolioSize = 29  # set portfolio size

    # read stock prices in a dataframe
    # df = pd.read_csv(StockFileName,  nrows= Rows)

    # extract asset labels
    # assetLabels = df.columns[1:Columns+1].tolist()
    # print(assetLabels)

    # extract asset prices
    # StockData = df.iloc[0:, 1:]
    StockData = mvo.head(mvo.shape[0] - 336)
    TradeData = mvo.tail(336)
    # df.head()
    TradeData.to_numpy()

    # compute asset returns
    arStockPrices = np.asarray(StockData)
    [Rows, Cols] = arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    # compute mean returns and variance covariance matrix of returns
    meanReturns = np.mean(arReturns, axis=0)
    covReturns = np.cov(arReturns, rowvar=False)

    # set precision for printing results
    np.set_printoptions(precision=3, suppress=True)

    # display mean returns and variance-covariance matrix of returns
    print("Mean returns of assets in k-portfolio 1\n", meanReturns)
    print("Variance-Covariance matrix of returns\n", covReturns)

    from pypfopt.efficient_frontier import EfficientFrontier

    ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
    raw_weights_mean = ef_mean.max_sharpe()
    cleaned_weights_mean = ef_mean.clean_weights()
    mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
    mvo_weights

    LastPrice = np.array([1 / p for p in StockData.tail(1).to_numpy()[0]])
    Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
    Initial_Portfolio

    Portfolio_Assets = TradeData @ Initial_Portfolio
    MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
    MVO_result

    df_result_a2c = df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    df_result_ddpg = df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    df_result_td3 = df_account_value_td3.set_index(df_account_value_td3.columns[0])
    df_result_ppo = df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    df_result_sac = df_account_value_sac.set_index(df_account_value_sac.columns[0])

    result = pd.merge(df_result_a2c, df_result_ddpg, left_index=True, right_index=True)
    result = pd.merge(result, df_result_td3, left_index=True, right_index=True)
    result = pd.merge(result, df_result_ppo, left_index=True, right_index=True)
    result = pd.merge(result, df_result_sac, left_index=True, right_index=True)
    result = pd.merge(result, MVO_result, left_index=True, right_index=True)
    result.columns = ["a2c", "ddpg", "td3", "ppo", "sac", "mean var"]

    plt.rcParams["figure.figsize"] = (15, 5)
    plt.figure()
    result.plot()


if __name__ == "__main__":
    main()
