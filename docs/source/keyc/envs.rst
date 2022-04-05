:github_url: https://github.com/AI4Finance-Foundation/FinRL

==========================
Stock Market Environments
==========================

Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a Markov Decision Process (MDP) problem. FinRL-Meta first preprocesses the market data, and then builds an environment. The environemnt observes the change of stock price, and the agent takes an action and receives the reward from the environment, and finally the agent adjusts its strategy accordingly. By interacting with the environment, the smart agent will derive a trading strategy to maximize the long-term accumulated reward (Q-value). 
