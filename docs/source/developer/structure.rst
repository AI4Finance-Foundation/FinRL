:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

File Structure
============================

.. code:: bash
    
    FinRL
    ├── finrl (main folder)
    │   ├── apps
    │   	├── cryptocurrency_trading
    │   	├── high_frequency_trading
    │   	├── portfolio_allocation
    │   	├── stock_trading
    │   	└── config.py
    │   ├── drl_agents
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
    │   ├── main.py
    │   ├── train.py
    │   ├── test.py
    │   ├── trade.py
    │   └── plot.py
    ├── tutorial (tutorial notebooks and educational files)
    ├── unit_testing (make sure verified codes working on env & data)
    │   ├── test_env
    │   	└── test_env_cashpenalty.py
    │   └── test_marketdata
    │   	└── test_yahoodownload.py
    ├── setup.cfg
    ├── setup.py
    ├── requirements.txt
    └── README.md
