:github_url: https://github.com/AI4Finance-Foundation/FinRL

Environment Layer
=================

As the most popular open source library that shows how to build well designed environments for different reinforcement learning (RL) tasks, OpenAI gym is the standard of after coming libraries that building RL environments. FinRL-Meta strictly follows OpenAI gym's way to produce market environments by using the well cleaned data from data layer.

Users can build their environments based on FinRL-Meta environments easily, share their results and compare the strategies' performance. We will add more environments for convenience in the future.


Incorporating trading constraints to model market frictions:
To better simulate real-world markets, we incorporate common market frictions (e.g., transaction costs and investor risk aversion) and portfolio restrictions (e.g., non-negative balance).

Market liquidity: Although market prices will be affected by trading orders, especially when orders are frequently executed at the close price, we assume that the market price will not be affected by the DRL trading agent. 
Flexible account settings: Users can choose whether to allow buying on margin or short-selling.  % We support different options for buying on margin and short-selling.
Transaction cost: We incorporate the transaction cost to reflect market friction, e.g., $0.1\%$ of each buy or sell trade.
Risk-control for market crash: In FinRL, a financial turbulence index is used to control risk during market crash situations. However, calculating the turbulence index is time-consuming. It may take minutes, which is not suitable for paper trading and live trading. We replace the financial turbulence index with the volatility index (VIX) that can be accessed immediately. %When VIX exceeds a given threshold, the agent will sell all the assets to avoid a massive loss in the market crash. 

Multiprocessing training via vector environment: We utilize GPUs for multiprocessing training, namely, the vector environment technique of Isaac Gym , which significantly accelerates the training process.  In each CUDA core, a trading agent interacts with a market environment to produce transitions in the form of (state, action, reward, next state). Then, all the transitions are stored in a replay buffer to update a learner. By adopting this technique, we successfully achieve the multiprocessing simulation of hundreds of market environments to improve the performance of DRL trading agents on large datasets.
