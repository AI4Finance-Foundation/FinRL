This folder has three subfolders:
+ applications: trading tasks,
+ agents: DRL algorithms, from ElegantRL, RLlib, or Stable Baselines 3 (SB3). Users can plug in any DRL lib and play.
+ meta: market environments, we merge the stable ones from the active [FinRL-Meta repo](https://github.com/AI4Finance-Foundation/FinRL-Meta).

Then, we employ a train-test-trade pipeline by three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (this folder)
│   ├── applications
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	└── stock_trading
│   ├── agents
│   	├── elegantrl
│   	├── rllib
│   	└── stablebaseline3
│   ├── meta
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
