# Portfolio Optimization Agents

This directory contains architectures and algorithms commonly used in portfolio optimization agents.

To instantiate the model, it's necessary to have an instance of [PortfolioOptimizationEnv](/finrl/meta/env_portfolio_optimization/). In the example below, we use the `DRLAgent` class to instantiate a policy gradient ("pg") model. With the dictionary `model_kwargs`, we can set the `PolicyGradient` class parameters and, whith the dictionary `policy_kwargs`, it's possible to change the parameters of the chosen architecture.

```python
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

# set PolicyGradient algorithm arguments
model_kwargs = {
    "lr": 0.01,
    "policy": EIIE,
}

# set EIIE architecture arguments
policy_kwargs = {
    "k_size": 4
}

model = DRLAgent(train_env).get_model("pg", model_kwargs, policy_kwargs)
```

In the example below, the model is trained in 5 episodes (we define an episode as a complete period of the used environment).

```python
DRLAgent.train_model(model, episodes=5)
```

It's important that the architecture and the environment have the same `time_window` defined. By default, both of them use 50 timesteps as `time_window`. For more details about what is a time window, check this [article](https://doi.org/10.5753/bwaif.2023.231144).

### Policy Gradient Algorithm

The class `PolicyGradient` implements the Policy Gradient algorithm used in *Jiang et al* paper. This algorithm is inspired by DDPG (deep deterministic policy gradient), but there are a couple of differences:
- DDPG is an actor-critic algorithm, so it has an actor and a critic neural network. The algorithm below, however, doesn't have a critic neural network and uses the portfolio value as value function: the policy will be updated to maximize the portfolio value.
- DDPG usually makes use of a noise parameter in the action during training to create an exploratory behavior. PG algorithm, on the other hand, has a full-exploit approach.
- DDPG randomly samples experiences from its replay buffer. The implemented policy gradient, however, samples a sequential batch of experiences in time, to make it possible to calculate the variation of the portfolio value in the batch and use it as value function.

The algorithm was implemented as follows:
1. Initializes policy network and replay buffer;
2. For each episode, do the following:
    1. For each period of `batch_size` timesteps, do the following:
        1. For each timestep, define an action to be performed, simulate the timestep and save the experiences in the replay buffer.
        2. After `batch_size` timesteps are simulated, sample the replay buffer.
        4. Calculate the value function: $V = \sum\limits_{t=1}^{batch\_size} ln(\mu_{t}(W_{t} \cdot P_{t}))$, where $W_{t}$ is the action performed at timestep t, $P_{t}$ is the price variation vector at timestep t and $\mu_{t}$ is the transaction remainder factor at timestep t. Check *Jiang et al* paper for more details.
        5. Perform gradient ascent in the policy network.
    2. If, in the and of episode, there is sequence of remaining experiences in the replay buffer, perform steps 1 to 5 with the remaining experiences.

### References

If you are using one of them in your research, you can use the following references.

#### EIIE Architecture and Policy Gradient algorithm

[A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem](https://doi.org/10.48550/arXiv.1706.10059)
```
@misc{jiang2017deep,
      title={A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem},
      author={Zhengyao Jiang and Dixing Xu and Jinjun Liang},
      year={2017},
      eprint={1706.10059},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```

#### EI3 Architecture

[A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management](https://doi.org/10.1145/3357384.3357961)
```
@inproceedings{shi2018multiscale,
               author = {Shi, Si and Li, Jianjun and Li, Guohui and Pan, Peng},
               title = {A Multi-Scale Temporal Feature Aggregation Convolutional Neural Network for Portfolio Management},
               year = {2019},
               isbn = {9781450369763},
               publisher = {Association for Computing Machinery},
               address = {New York, NY, USA},
               url = {https://doi.org/10.1145/3357384.3357961},
               doi = {10.1145/3357384.3357961},
               booktitle = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
               pages = {1613–1622},
               numpages = {10},
               keywords = {portfolio management, reinforcement learning, inception network, convolution neural network},
               location = {Beijing, China},
               series = {CIKM '19} }
```
