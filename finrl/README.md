This folder has three subfolders: 
+ apps: trading tasks, 
+ drl_agents: DRL algorithms, from ElegantRL, RLlib, or Stable Baselines 3 (SB3). Users can plug in any DRL lib and play.
+ finrl_meta: market environments, we merge the stable ones from the active [FinRL-Meta repo](https://github.com/AI4Finance-Foundation/FinRL-Meta).

Then, we employ a train-test-trade pipeline by three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (this folder)
│   ├── apps
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
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
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── train.py
│   ├── test.py
│   ├── trade.py
│   └── plot.py
```
