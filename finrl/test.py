from __future__ import annotations
import pandas as pd
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.train import download_data
from collections import defaultdict

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
    if_plot=False,
    **kwargs,
):
    env_config = download_data(    
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        if_vix,
        if_train=False,
        **kwargs)
    price_array = env_config["price_array"]
    tech_array = env_config["tech_array"]
    turbulence_array = env_config["turbulence_array"]

    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
    elif drl_lib == "rllib":
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )
    elif drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

    if if_plot:
        # print(env_config['date_array'].unique()[-1000:])
        unique_dates = env_config['date_array'].unique() # get unique dates (year-month-day hour-minute-second), 
        list_myd = []
        list_hms = []
        for d in unique_dates:
            [myd, ms] = str(d).split(' ')  # separate year-month-day hour-minute-second
            list_myd.append(myd)
            list_hms.append(ms)
        # print(list_myd, list_ms)

        # get last minute data as the data for a day
        def list_duplicates(seq):
            tally = defaultdict(list)
            for i,item in enumerate(seq):
                tally[item].append(i)
            return ((key,locs) for key,locs in tally.items() 
                                    if len(locs)>1)

        dates = []  # year-month-day only date tag
        values = [] # asset value at the end of each day
        for dup in sorted(list_duplicates(list_myd)):
            dates.append(dup[0])
            values.append(episode_total_assets[dup[1][-1]])
        # print(dates, values)

        df_account_value = pd.DataFrame(
            {"date": dates, "account_value": values}
            )
        return df_account_value
    else:
        return episode_total_assets



if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # # demo for stable baselines3
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac.zip",
    # )
