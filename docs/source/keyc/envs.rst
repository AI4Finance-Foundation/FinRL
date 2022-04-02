:github_url: https://github.com/AI4Finance-Foundation/FinRL

==========================
Stock Market Environments
==========================

Considering the stochastic and interactive nature of the automated stock trading tasks, a financial task is modeled as a Markov Decision Process (MDP) problem. FinRL-Meta first preprocesses the market data, and then builds an environment. The environemnt observes stock price change, and the agent takes an action and reward’s calculation to have the agent adjusting its strategy accordingly. By interacting with the environment, the trading agent will derive a trading strategy with the maximized rewards as time proceeds. 
