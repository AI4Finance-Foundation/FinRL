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

Three-layer architecture
------------------------------------

**Three-layer architecture**: The three layers of FinRL library are **stock market environment (FinRL-Meta)**, **DRL trading agent**, and **stock trading applications**. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior working-well decisions or to make new actions hoping to get greater rewards. 

.. image:: ../image/finrl_framework.png
    :width: 80%
    :align: center


**FinRL-Meta: Market Simulator**

For data processing and building environment for DRL in finance, AI4Finance has maintained another project: `FinRL-Meta <https://github.com/AI4Finance-Foundation/FinRL-Meta>`_.

In the *Three-Layer Architecture* section, there will be detailed explanation about how FinRL-Meta works.


**ElegantRL: DRL library**


FinRL contains fine-tuned standard DRL algorithms in ElegantRL, Stable Baseline 3, and RLlib. ElegantRL is a scalable and elastic DRL library that maintained by AI4Finance, with faster and more stable performance than Stable Baseline 3 and RLlib. In the *Three-Layer Architecture* section, there will be detailed explanation about how ElegantRL accomplish its role in FinRL perfectly. If interested, please refer to ElegantRL's `GitHub page <https://github.com/AI4Finance-Foundation/ElegantRL>`_ or `documentation <https://elegantrl.readthedocs.io>`_.

With those three powerful DRL libraries, FinRL provides the following algorithms for users:

.. image:: ../image/alg_compare.png


Contributions
=======================

    - FinRL is an open source library specifically designed and implemented for quantitative finance. Trading environments incorporating market frictions are used and provided. 
    - Trading tasks accompanied by hands-on tutorials with built-in DRL agents are available in a beginner-friendly and reproducible fashion using Jupyter notebook. Customization of trading time steps is feasible.
    - FinRL has good scalability, with fine-tuned state-of-the-art DRL algorithms. Adjusting the implementations to the rapid changing stock market is well supported. 
    - Typical use cases are selected to establish a benchmark for the quantitative finance community. Standard backtesting and evaluation metrics are also provided for easy and effective performance evaluation. 

With FinRL Library, the implementation of powerful DRL driven trading strategies is more accessible, efficient and delightful.

