# common library
import pandas as pd
import numpy as np
import time
import gym

#from finrl.apps import config

# RL models from RLlib ray
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.ddpg import ddpg
from ray.rllib.agents.ddpg import td3
from ray.rllib.agents.a3c import a2c
from ray.rllib.agents.sac import sac

MODELS = {"a2c": a2c, "ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo}

#MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
        price_array: numpy array
            OHLC data
        tech_array: numpy array
            techical data
        turbulence_array: numpy array
            turbulence/risk data
    Methods
    -------
        PPOTrainer()
            the implementation for PPO algorithm
        A2CTrainer()
            the implementation for A2C algorithm
        DDPGTrainer()
            the implementation for DDPG algorithm
        TD3Trainer()
            the implementation for TD3 algorithm
        SACTrainer()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """
    # @staticmethod
    # def DRL_prediction(model, environment):
    #     test_env, test_obs = environment.get_sb_env()
    #     """make a prediction"""
    #     account_memory = []
    #     actions_memory = []
    #     test_env.reset()
    #     for i in range(len(environment.df.index.unique())):
    #         action, _states = model.predict(test_obs)
    #         #account_memory = test_env.env_method(method_name="save_asset_memory")
    #         #actions_memory = test_env.env_method(method_name="save_action_memory")
    #         test_obs, rewards, dones, info = test_env.step(action)
    #         if i == (len(environment.df.index.unique()) - 2):
    #           account_memory = test_env.env_method(method_name="save_asset_memory")
    #           actions_memory = test_env.env_method(method_name="save_action_memory")
    #         if dones[0]:
    #             print("hit end!")
    #             break
    #     return account_memory[0], actions_memory[0]


    def __init__(self, 
                 env, 
                 price_array,
                 tech_array,
                 turbulence_array):
        self.env = env
        self.price_array=price_array
        self.tech_array=tech_array
        self.turbulence_array=turbulence_array

    def get_model(
        self,
        model_name,
        #policy="MlpPolicy",
        #policy_kwargs=None,
        #model_kwargs=None,
    ):
        if model_name not in MODELS:
           raise NotImplementedError("NotImplementedError")

        #if model_kwargs is None:
        #    model_kwargs = MODEL_KWARGS[model_name]

        model = MODELS[model_name]
        #get algorithm default configration based on algorithm in RLlib
        if model_name == 'a2c':
            model_config = model.A2C_DEFAULT_CONFIG.copy()
        elif model_name == 'td3':
            model_config = model.TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = model.DEFAULT_CONFIG.copy()
        #pass env, log_level, price_array, tech_array, and turbulence_array to config
        model_config['env'] = self.env
        model_config["log_level"] = "WARN"
        model_config['env_config'] = {'price_array':self.price_array,
                                      'tech_array':self.tech_array,
                                      'turbulence_array':self.turbulence_array,
                                      'if_train':True}

        return model, model_config

    def train_model(self, model, model_name, model_config,total_episodes=100):
        if model_name not in MODELS:
           raise NotImplementedError("NotImplementedError")
        ray.init() # Other Ray APIs will not work until `ray.init()` is called.
        if model_name == 'ppo':
            trainer = model.PPOTrainer(env=self.env, config=model_config)
        elif model_name == 'a2c':
            trainer = model.A2CTrainer(env=self.env, config=model_config)
        elif model_name == 'ddpg':
            trainer = model.DDPGTrainer(env=self.env, config=model_config)           
        elif model_name == 'td3':
            trainer = model.TD3Trainer(env=self.env, config=model_config)
        elif model_name == 'sac':
            trainer = model.SACTrainer(env=self.env, config=model_config)

        for i in range(total_episodes):
            trainer.train()

        ray.shutdown()

        #save the trained model
        cwd = './test_'+str(model_name)
        trainer.save(cwd)

        return trainer


