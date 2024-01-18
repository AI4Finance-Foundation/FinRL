from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import EIIE
from .utils import PVM
from .utils import ReplayBuffer
from .utils import RLDataset


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy is updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent
        train_policy: Policy used in training.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        optimizer=AdamW,
        device="cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
          env: Training Environment.
          policy: Policy architecture to be used.
          policy_kwargs: Arguments to be used in the policy network.
          validation_env: Validation environment.
          batch_size: Batch size to train neural network.
          lr: policy Neural network learning rate.
          optimizer: Optimizer of neural network.
          device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.device = device
        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer)

    def _setup_train(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before training.

        Args:
          env: environment.
          policy: Policy architecture to be used.
          batch_size: Batch size to train neural network.
          lr: Policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.train_env = env

        # neural networks
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def train(self, episodes=100):
        """Training sequence.

        Args:
            episodes: Number of episodes to simulate.
        """
        for i in tqdm(range(1, episodes + 1)):
            obs = self.train_env.reset()  # observation
            self.train_pvm.reset()  # reset portfolio vector memory
            done = False

            while not done:
                # define last_action and action and update portfolio vector memory
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = self.train_policy(obs_batch, last_action_batch)
                self.train_pvm.add(action)

                # run simulation step
                next_obs, reward, done, info = self.train_env.step(action)

                # add experience to replay buffer
                exp = (obs, last_action, info["price_variation"], info["trf_mu"])
                self.train_buffer.append(exp)

                # update policy networks
                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent()

                obs = next_obs

            # gradient ascent with episode remaining buffer data
            self._gradient_ascent()

            # validation step
            if self.validation_env:
                self.test(self.validation_env)

    def _setup_test(self, env, policy, batch_size, lr, optimizer):
        """Initializes algorithm before testing.

        Args:
          env: Environment.
          policy: Policy architecture to be used.
          batch_size: batch size to train neural network.
          lr: policy neural network learning rate.
          optimizer: Optimizer of neural network.
        """
        # environment
        self.test_env = env

        # process None arguments
        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        # neural networks
        # define policy
        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        # replay buffer and portfolio vector memory
        self.test_buffer = ReplayBuffer(capacity=batch_size)
        self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

        # dataset and dataloader
        dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test(
        self, env, policy=None, online_training_period=10, lr=None, optimizer=None
    ):
        """Tests the policy with online learning.

        Args:
          env: Environment to be used in testing.
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

        Note:
            To disable online learning, set learning rate to 0 or a very big online
            training period.
        """
        self._setup_test(env, policy, online_training_period, lr, optimizer)

        obs = self.test_env.reset()  # observation
        self.test_pvm.reset()  # reset portfolio vector memory
        done = False
        steps = 0

        while not done:
            steps += 1
            # define last_action and action and update portfolio vector memory
            last_action = self.test_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            action = self.test_policy(obs_batch, last_action_batch)
            self.test_pvm.add(action)

            # run simulation step
            next_obs, reward, done, info = self.test_env.step(action)

            # add experience to replay buffer
            exp = (obs, last_action, info["price_variation"], info["trf_mu"])
            self.test_buffer.append(exp)

            # update policy networks
            if steps % online_training_period == 0:
                self._gradient_ascent(test=True)

            obs = next_obs

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
        """
        # get batch data from dataloader
        obs, last_actions, price_variations, trf_mu = (
            next(iter(self.test_dataloader))
            if test
            else next(iter(self.train_dataloader))
        )
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)

        # define policy loss (negative for gradient ascent)
        mu = (
            self.test_policy.mu(obs, last_actions)
            if test
            else self.train_policy.mu(obs, last_actions)
        )
        policy_loss = -torch.mean(
            torch.log(torch.sum(mu * price_variations * trf_mu, dim=1))
        )

        # update policy network
        if test:
            self.test_policy.zero_grad()
            policy_loss.backward()
            self.test_optimizer.step()
        else:
            self.train_policy.zero_grad()
            policy_loss.backward()
            self.train_optimizer.step()
