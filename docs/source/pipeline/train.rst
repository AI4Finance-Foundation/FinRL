:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Train an agent
=======================

/finrl/train.py
-------------------

.. code-block:: python
    :linenos:

    import ray
    from finrl.drl_agents.elegantrl.models import DRLAgent as DRLAgent_erl
    from finrl.drl_agents.rllib.models import DRLAgent as DRLAgent_rllib
    from finrl.drl_agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
    from finrl.finrl_meta.data_processor import DataProcessor


    def train(start_date, end_date, ticker_list, data_source, time_interval,
              technical_indicator_list, drl_lib, env, model_name, if_vix=True,
              **kwargs):
        # fetch data
        DP = DataProcessor(data_source, **kwargs) #initialize NeoFinRL Data Processor (DP)
        data = DP.download_data(ticker_list, start_date, end_date, time_interval) #download data
        data = DP.clean_data(data) #clean data (check raw data, fill NaN data)
        data = DP.add_technical_indicator(data, technical_indicator_list) #such as RSI, MACD, BOLL, CCI
        if if_vix: #whether to use the VIX index to control risk
            data = DP.add_vix(data) #add VIX index
        price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix) #transfrom pd.DataFrame into Numpy.array
        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array, #VIX or turbulence (risk array)
            "if_train": True,
        }
        env_instance = env(config=env_config) #build environment using prepared datasets

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
