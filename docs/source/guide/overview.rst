:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

Overview
=======================

Design Principles
----------------------

    - **Full-stack framework**: To provide a full-stack DRL framework with finance-oriented optimizations, including market data APIs, data preprocessing, DRL algorithms, and automated backtesting. Users can transparently make use of such a development pipeline. 

    - **Customization**: To maintain modularity and extensibility in development by including state-of-the-art DRL algorithms and supporting design of new algorithms. The DRL algorithms can be used to construct trading strategies by simple configurations.

    - **Reproducibility and hands-on-tutoring**: To provide tutorials such as step-by-step Jupyter notebooks and user guide to help users walk through the pipeline and reproduce the use cases.


Architecture of the FinRL Library
------------------------------------

    - **Three-layer architecture**: The three layers of FinRL library are stock market environment, DRL trading agent, and stock trading applications. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior working-well decisions or to make new actions hoping to get greater rewards. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer.

    - **Modularity**: Each layer includes several modules and each module defines a separate function. One can select certain modules from any layer to implement his/her stock trading task. Furthermore, updating existing modules is possible.

    - **Simplicity, Applicability and Extendibility**: Specifically designed for automated stock trading, FinRL presents DRL algorithms as modules. In this way, FinRL is made accessible yet not demanding. FinRL provides three trading tasks as use cases that can be easily reproduced. Each layer includes reserved interfaces that allow users to develop new modules.

    - **Better Market Environment Modeling**: We build a trading simulator that replicates live stock market and provides backtesting support that incorporates important market frictions such as transaction cost, market liquidity and the investor’s degree of risk-aversion. All of those are crucial among key determinants of net returns.

.. image:: ../image/FinRL-Architecture.png


Implemented Algorithms
------------------------------------

.. image:: ../image/alg_compare.png
