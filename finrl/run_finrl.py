# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import SAC
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv

# customized env
from env.StockTradingRLEnvSingleStock import StockEnv

#import json
#import datetime as dt

from stockstats import StockDataFrame as Sdf
from sklearn import preprocessing
#import pickle

## data preprocessing part




# EnvName = 'PongNoFrameskip-v4'
# EnvType = 'atari'

# EnvName = 'CartPole-v0'
EnvName = 'Pendulum-v0'
EnvType = 'classic_control'

# EnvName = 'BipedalWalker-v2'
# EnvType = 'box2d'

# EnvName = 'Ant-v2'
# EnvType = 'mujoco'

# EnvName = 'FetchPush-v1'
# EnvType = 'robotics' 

# EnvName = 'FishSwim-v0'
# EnvType = 'dm_control'

# EnvName = 'ReachTarget'
# EnvType = 'rlbench'
# env = build_env(EnvName, EnvType, state_type='vision')

AlgName = 'SAC'
env = build_env(EnvName, EnvType)
alg_params, learn_params = call_default_params(env, EnvType, AlgName)
alg = eval(AlgName+'(**alg_params)')
alg.learn(env=env, mode='train', render=False, **learn_params)
alg.learn(env=env, mode='test', render=True, **learn_params)

# AlgName = 'DPPO'
# number_workers = 2  # need to specify number of parallel workers in parallel algorithms like A3C and DPPO
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, AlgName)
# alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

# AlgName = 'PPO'
# env = build_env(EnvName, EnvType)
# alg_params, learn_params = call_default_params(env, EnvType, AlgName)
# alg_params['method'] = 'clip'    # specify 'clip' or 'penalty' method for different version of PPO and DPPO
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

# AlgName = 'A3C'
# number_workers = 2  # need to specify number of parallel workers
# env = build_env(EnvName, EnvType, nenv=number_workers)
# alg_params, learn_params = call_default_params(env, EnvType, 'A3C')
# alg = eval(AlgName+'(**alg_params)')
# alg.learn(env=env,  mode='train', render=False, **learn_params)
# alg.learn(env=env,  mode='test', render=True, **learn_params)

env.close()
