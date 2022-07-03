:github_url: https://github.com/AI4Finance-Foundation/FinRL

Environment Layer
=================

FinRL-Meta follows the OpenAI gym-style to create market environments using the cleaned data from the data layer. It provides hundreds of environments with a common interface. Users can build their environments based on FinRL-Meta environments easily, share their results and compare the strategies’ performance. We will add more environments for convenience in the future.

Incorporating trading constraints to model market frictions
--------------------------------------------------------------------

To better simulate real-world markets, we incorporate common market frictions (e.g., transaction costs and investor risk aversion) and portfolio restrictions (e.g., non-negative balance).

- **Flexible account settings**: Users can choose whether to allow buying on margin or short-selling.
- **Transaction cost**: We incorporate the transaction cost to reflect market friction, e.g., 0.1% of each buy or sell trade.
- **Risk-control for market crash**: In FinRL, a financial turbulence index is used to control risk during market crash situations. However, calculating the turbulence index is time-consuming. It may take minutes, which is not suitable for paper trading and live trading. We replace the financial turbulence index with the volatility index (VIX) that can be accessed immediately.

Multiprocessing training via vector environment
---------------------------------------------------

We utilize GPUs for multiprocessing training, namely, the vector environment technique of Isaac Gym, which significantly accelerates the training process. In each CUDA core, a trading agent interacts with a market environment to produce transitions in the form of {state, action, reward, next state}. Then, all the transitions are stored in a replay buffer to update a learner. By adopting this technique, we successfully achieve the multiprocessing simulation of hundreds of market environments to improve the performance of DRL trading agents on large datasets.
