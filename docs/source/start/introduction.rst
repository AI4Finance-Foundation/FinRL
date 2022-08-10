:github_url: https://github.com/AI4Finance-Foundation/FinRL

=======================
Introduction
=======================

.. contents:: Table of Contents
    :depth: 2

**Design Principles**

- Plug-and-Play (PnP): modularity; handle different markets (say T0 vs. T+1).
- Completeness and universal: multiple markets; various data sources (APIs, Excel, etc); user-friendly variables.
- Avoid hard-coded parameters.
- Closing the sim-real gap using the “training-testing-trading” pipeline: simulation for training and connecting real-time APIs for testing/trading.
- Efficient data sampling: accelerate the data sampling process is the key to DRL training! From the ElegantRL project. We know that multi-processing is powerful to reduce the training time (scheduling between CPU + GPU).
- Transparency: a virtual env that is invisible to the upper layer.
- Flexibility and extensibility: inheritance might be helpful here.



**Contributions**


    - FinRL is an open source framework for financial reinforcement learning. Trading environments incorporating market frictions are provided.
    - Trading tasks accompanied by hands-on tutorials are available in a beginner-friendly and reproducible fashion. Customization is feasible.
    - FinRL has good scalability, with fine-tuned state-of-the-art DRL algorithms. Adjusting the implementations to the rapid changing stock market is well supported.
    - Typical use cases are selected to establish benchmarks for the quantitative finance community. Standard backtesting and evaluation metrics are also provided for easy and effective performance evaluation.

With FinRL library, the implementation of powerful DRL trading strategies becomes more accessible, efficient and delightful.
