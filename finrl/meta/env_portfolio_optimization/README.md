# PortfolioOptimizationEnv (POE)

This environment simulates the effects of the market in a portfolio that is periodically rebalanced through a reinforcement learning agent. At every timestep $t$, the agent is responsible for determining a portfolio vector $W_{t}$ which contains the percentage of money invested in each stock. The environment, then, utilizes data provided by the user to simulate the new portfolio value at time-step $t+1$.

For more details on the formulation of this problem, check the following paper:

[POE: A General Portfolio Optimization Environment for FinRL](https://doi.org/10.5753/bwaif.2023.231144)
```
@inproceedings{bwaif,
 author = {Caio Costa and Anna Costa},
 title = {POE: A General Portfolio Optimization Environment for FinRL},
 booktitle = {Anais do II Brazilian Workshop on Artificial Intelligence in Finance},
 location = {João Pessoa/PB},
 year = {2023},
 keywords = {},
 issn = {0000-0000},
 pages = {132--143},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/bwaif.2023.231144},
 url = {https://sol.sbc.org.br/index.php/bwaif/article/view/24959}
}
```

## Inputs
This environment simulates the interactions between an agent and the financial market
based on data provided by a dataframe. The dataframe contains the time series of
features defined by the user (such as closing, high and low prices) and must have
a time and a tic column with a list of datetimes and ticker symbols respectively.
An example of dataframe is shown below:
``````
    date        high            low             close           tic
0   2020-12-23  0.157414        0.127420        0.136394        ADA-USD
1   2020-12-23  34.381519       30.074295       31.097898       BNB-USD
2   2020-12-23  24024.490234    22802.646484    23241.345703    BTC-USD
3   2020-12-23  0.004735        0.003640        0.003768        DOGE-USD
4   2020-12-23  637.122803      560.364258      583.714600      ETH-USD
... ...         ...             ...             ...             ...
``````

## Actions

At each time step, the environment expects an action that is a one-dimensional Box of shape (n+1,), where $n$ is the number of stocks in the portfolio. This action is called *portfolio vector* and contains, for the remaining cash and for each stock, the percentage of allocated money.

For example: given a portfolio of three stocks, a valid portfolio vector would b $W_{t} = [0.25, 0.4, 0.2, 0.15]$. In this example, 25% of the money is not invested (remaining cash), 40% is invested in stock 1, 20% in stock 2 and 15% in sotck 3.

**Note:** It's important that the sum of the values in the portfolio vator is equal (or very close) to 1. If it's not, POE will apply a softmax normalization.

## Observations

POE can return two types of observations during simulation: a Dict or a Box.

- The box is a three-dimensional array of shape $(f, n, t)$, where $f$ s the number of features, $n$ is the number of stocks in the portfolio and $t$ is the time series timw window. This observation basically only contains the current state of the agent.

- The dict representation, on the other hand, is a dictionary containing the state and the last portfolio vector, like below:

```json
{
"state": "three-dimensional Box (f, n, t representing the time series",
"last_action": "one-dimensional Box (n+1,) representing the portfolio weights"
}
```

## Rewards
Given the simulation of timestep $t$, the reward is given by the following formula: $r_{t} = ln(V_{t}/V_{t-1})$, where $V_{t}$ is the value of the portfolio at time $t$. By using this formulation, the reward is negative whenever the portfolio value decreases due to a rebalancing and is positive otherwise.

## Example
A jupyter notebook using this environment can be found [here](/examples/FinRL_PortfolioOptimizationEnv_Demo.ipynb).
