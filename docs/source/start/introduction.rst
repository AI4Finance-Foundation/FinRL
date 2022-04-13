:github_url: https://github.com/AI4Finance-Foundation/FinRL

=======================
Introduction
=======================

.. contents:: Table of Contents
    :depth: 3

Design Principles
=======================

- Plug-and-Play (PnP): Modularity; Handle different markets (say T0 vs. T+1)
- Completeness and universal: Multiple markets; Various data sources (APIs, Excel, etc); User-friendly variables.
- Avoid hard-coded parameters
- Closing the sim-real gap using the “training-testing-trading” pipeline: simulation for training and connecting real-time APIs for testing/trading.
- Efficient data sampling: accelerate the data sampling process is the key to DRL training! From the ElegantRL project. We know that multi-processing is powerful to reduce the training time (scheduling between CPU + GPU).
- ransparency: a virtual env that is invisible to the upper layer
- Flexibility and extensibility: Inheritance might be helpful here



Features
=======================

The features are summarized as follows: 

Unique three-layer architecture
------------------------------------

**Three-layer architecture**: The three layers of FinRL library are **stock market environment (FinRL-Meta)**, **DRL trading agent**, and **stock trading applications**. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior working-well decisions or to make new actions hoping to get greater rewards. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer.

.. image:: ../image/finrl_framework.png
    :width: 80%
    :align: center


FinRL-Meta: Market Simulator
------------------------------------

For data processing and building environment for DRL in finance, AI4Finance has maintained another project: `FinRL-Meta <https://github.com/AI4Finance-Foundation/FinRL-Meta>`_.

In the *Three-Layer Architecture* section, there will be detailed explanation about how FinRL-Meta works.


ElegantRL: DRL library
------------------------------------

One sentence summary of reinforcement learning (RL): in RL, an agent learns by continuously interacting with an unknown environment, in a trial-and-error manner, making sequential decisions under uncertainty and achieving a balance between exploration (new territory) and exploitation (using knowledge learned from experiences).

Deep reinforcement learning (DRL) has great potential to solve real-world problems that are challenging to humans, such as self-driving cars, gaming, natural language processing (NLP), and financial trading. Starting from the success of AlphaGo, various DRL algorithms and applications are emerging in a disruptive manner. The ElegantRL library enables researchers and practitioners to pipeline the disruptive “design, development and deployment” of DRL technology.

The library to be presented is featured with “elegant” in the following aspects:

    - Lightweight: core codes have less than 1,000 lines, e.g., helloworld.
    - Efficient: the performance is comparable with Ray RLlib.
    - Stable: more stable than Stable Baseline 3.

ElegantRL supports state-of-the-art DRL algorithms, including discrete and continuous ones, and provides user-friendly tutorials in Jupyter notebooks. The ElegantRL implements DRL algorithms under the Actor-Critic framework, where an Agent (a.k.a, a DRL algorithm) consists of an Actor network and a Critic network. Due to the completeness and simplicity of code structure, users are able to easily customize their own agents.


Implemented Algorithms

.. image:: ../image/alg_compare.png


Contributions of FinRL
=======================

    - FinRL is an open source library specifically designed and implemented for quantitative finance. Trading environments incorporating market frictions are used and provided. 
    - Trading tasks accompanied by hands-on tutorials with built-in DRL agents are available in a beginner-friendly and reproducible fashion using Jupyter notebook. Customization of trading time steps is feasible.
    - FinRL has good scalability, with fine-tuned state-of-the-art DRL algorithms. Adjusting the implementations to the rapid changing stock market is well supported. 
    - Typical use cases are selected to establish a benchmark for the quantitative finance community. Standard backtesting and evaluation metrics are also provided for easy and effective performance evaluation. 

With FinRL Library, the implementation of powerful DRL driven trading strategies is more accessible, efficient and delightful.

