import ray

from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.finrl_meta.data_processor import DataProcessor


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
        **kwargs
):
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
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
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

        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training finished!")
        trained_model.save(cwd)
        print("Trained model saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    from finrl.apps.config import DOW_30_TICKER
    from finrl.apps.config import TECHNICAL_INDICATORS_LIST
    from finrl.apps.config import TRAIN_START_DATE
    from finrl.apps.config import TRAIN_END_DATE
    from finrl.apps.config import ERL_PARAMS
    from finrl.apps.config import RLlib_PARAMS
    from finrl.apps.config import SAC_PARAMS

    # construct environment
    from finrl.finrl_meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

    env = StockTradingEnv

    # demo for elegantrl
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        erl_params=ERL_PARAMS,
        break_step=1e5,
    )

    # demo for rllib
    ray.shutdown()  # always shutdown previous session if any
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="rllib",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        rllib_params=RLlib_PARAMS,
        total_episodes=30,
    )

    # demo for stable-baselines3
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=TECHNICAL_INDICATORS_LIST,
        drl_lib="stable_baselines3",
        env=env,
        model_name="sac",
        cwd="./test_sac",
        agent_params=SAC_PARAMS,
        total_timesteps=1e4,
    )
