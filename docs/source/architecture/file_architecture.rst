:github_url: https://github.com/AI4Finance-Foundation/FinRL

=================
File Architecture
=================

FinRL's file architecture strictly follow the :ref:`three-layer architecture`.

.. code:: bash

    FinRL
    ├── finrl (the main folder)
    │   ├── applications
    │   	├── cryptocurrency_trading
    │   	├── high_frequency_trading
    │   	├── portfolio_allocation
    │   	└── stock_trading
    │   ├── agents
    │   	├── elegantrl
    │   	├── rllib
    │   	└── stablebaseline3
    │   ├── finrl_meta
    │   	├── data_processors
    │   	├── env_cryptocurrency_trading
    │   	├── env_portfolio_allocation
    │   	├── env_stock_trading
    │   	├── preprocessor
    │   	├── data_processor.py
    │   	└── finrl_meta_config.py
    │   ├── config.py
    │   ├── config_tickers.py
    │   ├── main.py
    │   ├── train.py
    │   ├── test.py
    │   ├── trade.py
    └───└── plot.py