"""
DRL models to solve the portfolio optimization task with reinforcement learning.
This agent was developed to work with environments like PortfolioOptimizationEnv.
"""
from __future__ import annotations

from .algorithms import PolicyGradient

MODELS = {"pg": PolicyGradient}


class DRLAgent:
    """Implementation for DRL algorithms for portfolio optimization.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        env: Gym environment class.
    """

    def __init__(self, env):
        """Agent initialization.

        Args:
            env: Gym environment to be used in training.
        """
        self.env = env

    def get_model(
        self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None
    ):
        """Setups DRL model.

        Args:
            model_name: Name of the model according to MODELS list.
            device: Device used to instantiate neural networks.
            model_kwargs: Arguments to be passed to model class.
            policy_kwargs: Arguments to be passed to policy class.

        Note:
            model_kwargs and policy_kwargs are dictionaries. The keys must be strings
            with the same names as the class arguments. Example for model_kwargs::

            { "lr": 0.01, "policy": EIIE }

        Returns:
            An instance of the model.
        """
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        # add device settings
        model_kwargs["device"] = device
        policy_kwargs["device"] = device

        # add policy_kwargs inside model_kwargs
        model_kwargs["policy_kwargs"] = policy_kwargs

        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=100):
        """Trains portfolio optimization model.

        Args:
            model: Instance of the model.
            episoded: Number of episodes.

        Returns:
            An instance of the trained model.
        """
        model.train(episodes)
        return model

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        policy=None,
        online_training_period=10,
        learning_rate=None,
        optimizer=None,
    ):
        """Tests a model in a testing environment.

        Args:
            model: Instance of the model.
            test_env: Gym environment to be used in testing.
            policy: Policy architecture to be used. If None, it will use the training
            architecture.
            online_training_period: Period in which an online training will occur. To
                disable online learning, use a very big value.
            batch_size: Batch size to train neural network. If None, it will use the
                training batch size.
            lr: Policy neural network learning rate. If None, it will use the training
                learning rate
            optimizer: Optimizer of neural network. If None, it will use the training
                optimizer
        """
        model.test(test_env, policy, online_training_period, learning_rate, optimizer)
