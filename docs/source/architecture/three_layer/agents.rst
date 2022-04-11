:github_url: https://github.com/AI4Finance-Foundation/FinRL

2. DRL Agents
============================

DRL agents are built based on fine-tuned standard DRL algorithms depending on three famous DRL library: ElegantRL, Stable Baseline 3, and RLlib. 

Supported algorithms includes: DQN, DDPG, Multi-Agent DDPG, PPO, SAC, A2C and TD3. We also allow users to design their own DRL algorithms by adapting these DRL algorithms, e.g., Adaptive DDPG, or employing ensemble methods. The comparison of DRL algorithms is shown in the table bellow:

.. image:: ../../image/alg_compare.png
   :align: center

Users are able to choose the DRL agent they want to use before training. DRL agents might have different performance in various tasks.
