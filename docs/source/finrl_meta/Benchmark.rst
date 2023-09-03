:github_url: https://github.com/AI4Finance-Foundation/FinRL

=============================
Benchmark
=============================

Performance Metrics
====================

FinRL-Meta provides the following unified metrics to measure the trading performance:

- **Cumulative return:** :math:`R = \frac{V - V_0}{V_0}`, where V is final portfolio value, and :math:`V_0` is original capital.
- **Annualized return:** :math:`r = (1+R)^\frac{365}{t}-1`, where t is the number of trading days.
- **Annualized volatility:** :math:`{\sigma}_a = \sqrt{\frac{\sum_{i=1}^{n}{(r_i-\bar{r})^2}}{n-1}}`, where :math:`r_i` is the annualized return in year i, :math:`\bar{r}` is the average annualized return, and n is the number of years.
- **Sharpe ratio:** :math:`S = \frac{r - r_f}{{\sigma}_a}`, where :math:`r_f` is the risk-free rate.
- **Max. drawdown** The maximal percentage loss in portfolio value.

The following baseline trading strategies are provided for comparisons:

• **Passive trading strategy**, a well-known long-term strategy. The investors just buy and hold selected stocks or indexes without further activities.
• ****Mean-variance and min-variance strategy**, both strategies look for a balance between risks and profits. It selects a diversified portfolio to achieve higher profits at lower risk.
• **Equally weighted strategy**, a portfolio allocation strategy that gives equal weights to different assets, avoiding allocating overly high weights on particular stocks.

Tutorials in Jupyter Notebooks
===========================================

For educational purposes, we provide Jupyter notebooks as tutorials to help newcomers get familiar with the whole pipeline. Notebooks can be found `here <https://github.com/AI4Finance-Foundation/FinRL-Tutorials>`_

• Stock trading: We apply popular DRL algorithms to trade multiple stocks.
• Portfolio allocation: We use DRL agents to optimize asset allocation in a set of stocks.
• Cryptocurrency trading: We reproduce the experiment on 10 popular cryptocurrencies.
• Multi-agent RL for liquidation strategy analysis: We reproduce the experiment in [7]. The multi-agent optimizes the shortfalls in the liquidation task, which is to sell given shares of one stock sequentially within a given period, considering the costs arising from the market impact and the risk aversion.
• Ensemble strategy for stock trading: We reproduce the experiment in that employed an ensemble strategy of several DRL algorithms on the stock trading task.
• Paper trading demo: We provide a demo for paper trading. Users could combine their own strategies or trained agents in paper trading.
• China A-share demo: We provide a demo based on the China A-share market data.
• Hyperparameter tuning: We provide several demos for hyperparameter tuning using Optuna or Ray Tune, since hyperparameter tuning is critical for better performance.
