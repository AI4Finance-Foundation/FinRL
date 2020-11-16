import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from finrl.config import config
from stable_baselines.common.vec_env import DummyVecEnv
#import pickle

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 1

# transaction fee: 2/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.002
TURBULENCE_THRESHOLD = 120


class EnvSetup:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        df
        feature_number : str
            start date of the data (modified from config.py)
        use_technical_indicator : str
            end date of the data (modified from config.py)
        use_turbulence : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """
    def __init__(self, 
        df,
        hmax = 100,
        initial_amount = 1000000,
        transaction_cost_pct = 0.001,
        turbulence_threshold=150,
        reward_scaling = 1e-4):

        self.df = df
        self.stock_dim = len(self.df.tic.unique())
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = config.TECHNICAL_INDICATORS_LIST
        self.turbulence_threshold=turbulence_threshold
        # account balance + close price + shares + technical indicators
        self.state_space = 1 + 2*self.stock_dim + len(self.tech_indicator_list)*self.stock_dim
        self.action_space = self.stock_dim


    def create_env_training(self, env_class):
        env_train = DummyVecEnv([lambda: env_class(df = self.df,
                                                    stock_dim = self.stock_dim,
                                                    hmax = self.hmax,
                                                    initial_amount = self.initial_amount,
                                                    transaction_cost_pct = self.transaction_cost_pct,
                                                    reward_scaling = self.reward_scaling,
                                                    state_space = self.state_space,
                                                    action_space = self.action_space,
                                                    tech_indicator_list = self.tech_indicator_list)])
        return env_train


    def create_env_validation(self, env_class):
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

    def create_env_trading(self, env_class):
        env_trade = DummyVecEnv([lambda: env_class(df = self.df,
                                            stock_dim = self.stock_dim,
                                            hmax = self.hmax,
                                            initial_amount = self.initial_amount,
                                            transaction_cost_pct = self.transaction_cost_pct,
                                            reward_scaling = self.reward_scaling,
                                            state_space = self.state_space,
                                            action_space = self.action_space,
                                            tech_indicator_list = self.tech_indicator_list,
                                            turbulence_threshold=self.turbulence_threshold)])
        obs_trade = env_trade.reset()


        return env_trade, obs_trade
