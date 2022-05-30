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

Experiment Settings
====================
