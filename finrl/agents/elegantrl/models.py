"""
DRL models from ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
"""

from __future__ import annotations

import torch
from elegantrl.agents import *
from elegantrl.train.config import Config
from elegantrl.train.run import train_agent

MODELS = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
    "ppo": AgentPPO,
    "a2c": AgentA2C,
}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo"]
# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
#
# NOISE = {
#     "normal": NormalActionNoise,
#     "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
# }


class DRLAgent:
    """Implementations of DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
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

    def __init__(self, env, price_array, tech_array, turbulence_array):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array

    def get_model(self, model_name, model_kwargs):
        self.env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "turbulence_array": self.turbulence_array,
            "if_train": True,
        }
        self.model_kwargs = model_kwargs
        self.gamma = model_kwargs.get("gamma", 0.985)

        env = self.env
        env.env_num = 1
        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        stock_dim = self.price_array.shape[1]
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        self.action_dim = stock_dim
        self.env_args = {
            "env_name": "StockEnv",
            "config": self.env_config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "if_discrete": False,
            "max_step": self.price_array.shape[0] - 1,
        }

        model = Config(agent_class=agent, env_class=env, env_args=self.env_args)
        model.if_off_policy = model_name in OFF_POLICY_MODELS
        if model_kwargs is not None:
            try:
                model.break_step = int(
                    2e5
                )  # break training if 'total_step > break_step'
                model.net_dims = (
                    128,
                    64,
                )  # the middle layer dimension of MultiLayer Perceptron
                model.gamma = self.gamma  # discount factor of future rewards
                model.horizon_len = model.max_step
                model.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small
                model.learning_rate = model_kwargs.get("learning_rate", 1e-4)
                model.state_value_tau = 0.1  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
                model.eval_times = model_kwargs.get("eval_times", 2**5)
                model.eval_per_step = int(2e4)
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = total_timesteps
        train_agent(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, env_args):
        import torch

        gpu_id = 0  # >=0 means GPU ID, -1 means CPU
        agent_class = MODELS[model_name]
        stock_dim = env_args["price_array"].shape[1]
        state_dim = 1 + 2 + 3 * stock_dim + env_args["tech_array"].shape[1]
        action_dim = stock_dim
        env_args = {
            "env_num": 1,
            "env_name": "StockEnv",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "if_discrete": False,
            "max_step": env_args["price_array"].shape[0] - 1,
            "config": env_args,
        }

        actor_path = f"{cwd}/act.pth"
        net_dim = [2**7]

        """init"""
        env = environment
        env_class = env
        args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
        args.cwd = cwd
        act = agent_class(
            net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args
        ).act
        parameters_dict = {}
        act = torch.load(actor_path)
        for name, param in act.named_parameters():
            parameters_dict[name] = torch.tensor(param.detach().cpu().numpy())

        act.load_state_dict(parameters_dict)

        if_discrete = env.if_discrete
        device = next(act.parameters()).device
        state = env.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [env.initial_total_asset]
        max_step = env.max_step
        for steps in range(max_step):
            s_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)
            a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
            action = (
                a_tensor.detach().cpu().numpy()[0]
            )  # not need detach(), because using torch.no_grad() outside
            state, reward, done, _ = env.step(action)
            total_asset = env.amount + (env.price_ary[env.day] * env.stocks).sum()
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env.initial_total_asset
            episode_returns.append(episode_return)
            if done:
                break
        print("Test Finished!")
        print("episode_retuen", episode_return)
        return episode_total_assets
