# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
#from stable_baselines import SAC
#from stable_baselines import TD3

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.config import config


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
    train_PPO()
        the implementation for PPO algorithm
    train_A2C()
        the implementation for A2C algorithm
    train_DDPG()
        the implementation for DDPG algorithm
    train_TD3()
        the implementation for TD3 algorithm      
    train_SAC()
        the implementation for SAC algorithm 
    DRL_prediction() 
        make a prediction in a test dataset and get results
    """
    def __init__(self, env):
        self.env = env

    def train_A2C(self, model_name, policy = "MlpPolicy", model_params = config.A2C_PARAMS):
        """A2C model"""
        from stable_baselines3 import A2C
        from stable_baselines3.a2c import MlpPolicy

        env_train = self.env
        start = time.time()
        model = A2C(policy, env_train, 
                    n_steps = model_params['n_steps'],
                    ent_coef = model_params['ent_coef'],
                    learning_rate = model_params['learning_rate'],
                    verbose = model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "A2C_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (A2C): ', (end-start)/60,' minutes')
        return model


    def train_DDPG(self, model_name, policy = "MlpPolicy", model_params = config.DDPG_PARAMS):
        """DDPG model"""
        from stable_baselines3 import DDPG
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))


        start = time.time()
        model = DDPG(policy, 
                    env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate=model_params['learning_rate'],
                    action_noise=action_noise,
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "DDPG_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end-start)/60,' minutes')
        return model


    def train_TD3(self, model_name, policy = "MlpPolicy", model_params = config.TD3_PARAMS):
        """TD3 model"""
        from stable_baselines3 import TD3
        from stable_baselines3.td3.policies import MlpPolicy
        from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

        env_train = self.env

        n_actions = env_train.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

        start = time.time()
        model = TD3(policy, env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate = model_params['learning_rate'],
                    action_noise = action_noise,
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "TD3_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (DDPG): ', (end-start)/60,' minutes')
        return model

    def train_SAC(self, model_name, policy = "MlpPolicy", model_params = config.SAC_PARAMS):
        """TD3 model"""
        from stable_baselines3 import SAC
        from stable_baselines3.sac import MlpPolicy

        env_train = self.env

        start = time.time()
        model = SAC(policy, env_train,
                    batch_size=model_params['batch_size'],
                    buffer_size=model_params['buffer_size'],
                    learning_rate = model_params['learning_rate'],
                    learning_starts=model_params['learning_starts'],
                    ent_coef=model_params['ent_coef'],
                    verbose=model_params['verbose'],
                    tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                    )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "SAC_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (SAC): ', (end-start)/60,' minutes')
        return model


    def train_PPO(self, model_name, policy = "MlpPolicy", model_params = config.PPO_PARAMS):
        """PPO model"""
        from stable_baselines3 import PPO
        from stable_baselines3.ppo import MlpPolicy

        env_train = self.env

        start = time.time()
        model = PPO(policy, env_train,
                     n_steps = model_params['n_steps'],
                     ent_coef = model_params['ent_coef'],
                     learning_rate = model_params['learning_rate'],
                     batch_size = model_params['batch_size'],
                     verbose = model_params['verbose'],
                     tensorboard_log = f"{config.TENSORBOARD_LOG_DIR}/{model_name}"
                     )
        model.learn(total_timesteps=model_params['timesteps'], tb_log_name = "PPO_run")
        end = time.time()

        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
        print('Training time (PPO): ', (end-start)/60,' minutes')
        return model

    @staticmethod
    def DRL_prediction(model, test_data, test_env, test_obs):
        """make a prediction"""
        start = time.time()
        account_memory = []
        actions_memory = []
        for i in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(test_data.index.unique()) - 2):
                account_memory = test_env.env_method(method_name = 'save_asset_memory')
                actions_memory = test_env.env_method(method_name = 'save_action_memory')
        end = time.time()
        return account_memory[0], actions_memory[0]