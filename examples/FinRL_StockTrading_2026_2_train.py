"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

# %% Part 1. Prepare directories

check_and_make_directories([TRAINED_MODEL_DIR])

# %% Part 2. Build environment

train = pd.read_csv("train_data.csv")
train = train.set_index(train.columns[0])
train.index.names = [""]

stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

# %% Part 3. Train DRL Agents

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

# --- Agent 1: A2C ---
agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")
if if_using_a2c:
    tmp_path = RESULTS_DIR + "/a2c"
    new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_a2c.set_logger(new_logger_a2c)

trained_a2c = (
    agent.train_model(model=model_a2c, tb_log_name="a2c", total_timesteps=20000)
    if if_using_a2c
    else None
)
if if_using_a2c:
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c")

# --- Agent 2: DDPG ---
agent = DRLAgent(env=env_train)
model_ddpg = agent.get_model("ddpg")
if if_using_ddpg:
    tmp_path = RESULTS_DIR + "/ddpg"
    new_logger_ddpg = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ddpg.set_logger(new_logger_ddpg)

trained_ddpg = (
    agent.train_model(model=model_ddpg, tb_log_name="ddpg", total_timesteps=20000)
    if if_using_ddpg
    else None
)
if if_using_ddpg:
    trained_ddpg.save(TRAINED_MODEL_DIR + "/agent_ddpg")

# --- Agent 3: PPO ---
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
if if_using_ppo:
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

trained_ppo = (
    agent.train_model(model=model_ppo, tb_log_name="ppo", total_timesteps=20000)
    if if_using_ppo
    else None
)
if if_using_ppo:
    trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo")

# --- Agent 4: TD3 ---
agent = DRLAgent(env=env_train)
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.001,
}
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
if if_using_td3:
    tmp_path = RESULTS_DIR + "/td3"
    new_logger_td3 = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_td3.set_logger(new_logger_td3)

trained_td3 = (
    agent.train_model(model=model_td3, tb_log_name="td3", total_timesteps=20000)
    if if_using_td3
    else None
)
if if_using_td3:
    trained_td3.save(TRAINED_MODEL_DIR + "/agent_td3")

# --- Agent 5: SAC ---
agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
if if_using_sac:
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

trained_sac = (
    agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=20000)
    if if_using_sac
    else None
)
if if_using_sac:
    trained_sac.save(TRAINED_MODEL_DIR + "/agent_sac")

print("All agents trained and saved to", TRAINED_MODEL_DIR)
