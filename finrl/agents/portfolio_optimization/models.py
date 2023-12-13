"""
DRL models to solve the portfolio optimization task with reinforcement learning.
This agent was developed to work with environments like PortfolioOptimizationEnv.
"""

from .algorithms import PolicyGradient

MODELS = {"pg": PolicyGradient}


class DRLAgent:
    """
    Implementation for DRL algorithms for portfolio optimization.

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

    def __init__(self, env):
        self.env = env

    def get_model(self, model_name, model_kwargs=None, policy_kwargs=None):
        if model_name not in MODELS:
            raise NotImplementedError("The model requested was not implemented.")

        model = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if policy_kwargs is not None:
            model_kwargs["policy_kwargs"] = policy_kwargs
        return model(self.env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=100):
        model.train(episodes)

    @staticmethod
    def DRL_validation(
        model,
        test_env,
        policy=None,
        online_training_period=10,
        learning_rate=None,
        optimizer=None,
    ):
        model.test(test_env, policy, online_training_period, learning_rate, optimizer)
