This main folder is organized into three subfolders: 
+ apps: tens of trading tasks, 
+ drl_agents: tens of DRL algorithms, from ElegantRL, RLlib, or Stable Baselines 3 (SB3). Users can plug in any DRL lib and play.
+ finrl_meta: hundreds of market environments, we marge the stable ones from the active [FinRL-Meta repo](https://github.com/AI4Finance-Foundation/FinRL-Meta).

Then, we employ a train-test-trade pipeline, via three files:
+ train.py, 
+ test.py
+ trade.py.
