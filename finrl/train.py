from __future__ import annotations

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.wandb import init_wandb

# construct environment

def download_data(    
        start_date,
        end_date,
        ticker_list,
        data_source,
        time_interval,
        technical_indicator_list,
        if_vix=True,
        if_train=False,
        **kwargs
        ):

    data_path = 'data/'
    data_file_name = f'{data_source}_{start_date}_{end_date}.pkl'
    file_name = data_path + data_file_name
    dp = DataProcessor(data_source, **kwargs)
    import pandas as pd
    import os
    if os.path.isfile(file_name):
        # load if exist
        try:
            data = pd.read_pickle(file_name)
            dp.set_meta_data(start_date, end_date, time_interval, technical_indicator_list)
            print(f"Load data from {file_name}")
        except:
            print(f"Failed to load data from {file_name}")
    else:
        # download data
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = dp.add_vix(data)

        # save data
        os.makedirs(data_path, exist_ok=True)
        data.to_pickle(file_name) 

    print('The data looks like: \n', data.head(40))  # display the data
    print(f'Full data shape: {data.shape}')
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": if_train,
        "date_array": pd.to_datetime(data['timestamp']+pd.Timedelta(hours=4))  # ny to utc by default
    }
    return env_config

def train(
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
        if_train=True,
        **kwargs)
    price_array = env_config["price_array"]
    tech_array = env_config["tech_array"]
    turbulence_array = env_config["turbulence_array"]

    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    args={
            'wandb_project': 'FinRL',
            'wandb_group': '',
            'wandb_entity': 'quantumiracle',            
            'wandb_name': str(cwd),
    }
    init_wandb(args)

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
        import numpy as np
        # turbulence_array = np.expand_dims(turbulence_array, axis=-1)
        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model, model_config = agent_rllib.get_model(model_name)
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":

    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #
    # # demo for stable-baselines3
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac",
    #     agent_params=SAC_PARAMS,
    #     total_timesteps=1e4,
    # )
