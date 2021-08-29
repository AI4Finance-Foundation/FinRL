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
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset 
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """
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

    @staticmethod
    def DRL_prediction(model, 
                        model_name,
                        env, 
                        price_array, 
                        tech_array, 
                        turbulence_array, 
                        agent_path='./test_ppo/checkpoint_000100/checkpoint-100'):
        if model_name not in MODELS:
           raise NotImplementedError("NotImplementedError")
           
        if model_name == 'a2c':
            model_config = model.A2C_DEFAULT_CONFIG.copy()
        elif model_name == 'td3':
            model_config = model.TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = model.DEFAULT_CONFIG.copy()
        model_config['env'] = env
        model_config["log_level"] = "WARN"
        model_config['env_config'] = {'price_array':price_array,
                                        'tech_array':tech_array,
                                        'turbulence_array':turbulence_array,
                                        'if_train':False}
        env_config = {'price_array':price_array,
                'tech_array':tech_array,
                'turbulence_array':turbulence_array,
                'if_train':False}
        env_instance = env(config=env_config)

        #ray.init() # Other Ray APIs will not work until `ray.init()` is called.
        if model_name == 'ppo':
            trainer = model.PPOTrainer(env=env, config=model_config)
        elif model_name == 'a2c':
            trainer = model.A2CTrainer(env=env, config=model_config)
        elif model_name == 'ddpg':
            trainer = model.DDPGTrainer(env=env, config=model_config)           
        elif model_name == 'td3':
            trainer = model.TD3Trainer(env=env, config=model_config)
        elif model_name == 'sac':
            trainer = model.SACTrainer(env=env, config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except:
            raise ValueError('Fail to load agent!')

        #test on the testing env
        state = env_instance.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(env_instance.initial_total_asset)
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)

            total_asset = env_instance.amount + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        ray.shutdown()
        print('episode return: ' + str(episode_return))
        print('Test Finished!')   
        return episode_total_assets
