:github_url: https://github.com/AI4Finance-Foundation/FinRL

===========================
Three-layer Architecture
===========================

FinRL library consists of three layers: **stock market environment (FinRL-Meta)**, **DRL trading agent** and **stock trading applications**. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior workingwell decisions or to make new actions hoping to get greater Q-value. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer.

.. image:: ../image/finrl_framework.png
   :width: 80%
   :align: center

**Modularity**: Each layer includes several modules and each module defines a separate function. One can select certain modules from a layer to implement his/her stock trading task. Furthermore, updating existing modules is possible.

**Simplicity, Applicability and Extendibility**: Specifically designed for automated stock trading, FinRL presents DRL algorithms as modules. In this way, FinRL is made accessible yet not demanding. FinRL provides three trading tasks as use cases that can be easily reproduced. Each layer includes reserved interfaces that allow users to develop new modules.

**Better Market Environment Modeling**: We build a trading simulator that replicates live stock markets and provides backtesting support that incorporates important market frictions such as transaction cost, market liquidity and the investor’s degree of risk-aversion. All of those are crucial among key determinants of net returns.

Please refer to the following pages for more specific explanation:

.. toctree::
   :maxdepth: 1

   three_layer/environments
   three_layer/agents
   three_layer/applications
