import ray

from finrl.apps.config import (
    DOW_30_TICKER,
    TECHNICAL_INDICATORS_LIST,
    TEST_END_DATE,
    TEST_START_DATE,
    RLlib_PARAMS,
)
from finrl.finrl_meta.env_stock_trading.env_stocktrading import StockTradingEnv


def test(
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        drl_lib,
        env,
        model_name,
        if_vix=True,
        **kwargs
):
    # import DRL agents
    from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
    from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib
    from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl

    # import data processor
    from finrl.finrl_meta.data_processor import DataProcessor

    # fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2 ** 7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )

        return episode_total_assets

    elif drl_lib == "rllib":
        # load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )

        return episode_total_assets

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
    )

    # demo for rllib
    ray.shutdown()  # always shutdown previous session if any
    account_value_rllib = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="rllib",
        env=env,
        model_name="ppo",
        cwd="./test_ppo/checkpoint_000030/checkpoint-30",
        rllib_params=RLlib_PARAMS,
    )

    # demo for stable baselines3
    account_value_sb3 = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="stable_baselines3",
        env=env,
        model_name="sac",
        cwd="./test_sac.zip",
    )
