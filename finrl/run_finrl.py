# common library
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

# customized env
from env.EnvSingleStock import StockEnv
# preprocessor
from preprocessing.preprocessors import *
# config
from config.config import *
# model
from model.models import *


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    df = preprocess_data()
    df = add_turbulence(df)

    # divide train and test
    train = data_split(df, start=20090000, end=20170000)
    test = data_split(df, start=20170000, end=20200512)

    ## set up train & test environment
    # training env
    env_train = DummyVecEnv([lambda: StockEnv(train)])
    # testing env
    env_test = DummyVecEnv([lambda: StockEnv(test)])
    obs_test = env_test.reset()

    ## model training
    print("==============Model Training===========")
    model = train_A2C(env_train, model_name = "A2C_80k_spy", timesteps=80000)

    print("==============Model Prediction===========")
    for i in range(len(test.index.unique())):
        action, _states = model.predict(obs_test)
        obs_test, rewards, dones, info = env_test.step(action)
        env_test.render()

    #_logger.info(f"saving model version: {_version}")
    #save_pipeline(pipeline_to_persist=pipeline.price_pipe)


if __name__ == "__main__":
    run_model()