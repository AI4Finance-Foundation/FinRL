.. Finrl Library documentation master file, created by
   sphinx-quickstart on Wed Nov 18 08:14:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/AI4Finance-Foundation/FinRL

Welcome to FinRL Library!
=====================================================================================================

.. meta::
   :description: FinRL Library is an open source framework that facilitates beginners to expose themselves to quantitative finance and to develop their own stock trading strategies using deep reinforcement learning, it collects the most practical reinforcement learning algorithms, frameworks and applications(DQN, DDPG, PPO, SAC, A2C, TD3, etc.).
   :keywords: finance ai, openai, artificial intelligence in finance, machine learning, deep reinforcement learning, DRL, RL, machine learning neural networks, deep q network, multi agent reinforcement learning

.. image:: image/logo_transparent_background.png
   :target:  https://github.com/AI4Finance-Foundation/FinRL

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

To efficiently automate trading, **AI4Finance** community provides this demonstrative and educational resource. FinRL is an open source deep reinforcement learning (DRL) framework for researchers and practitioners. 

.. _FinRL: https://github.com/AI4Finance-Foundation/FinRL

Reinforcement learning (RL) trains an agent to solve tasks by trial and error, while DRL uses deep neural networks as function approximators. DRL balances exploration (of uncharted territory) and exploitation (of current knowledge), and has been recognized as a competitive edge for automated trading. DRL framework is powerful in solving dynamic decision making problems by learning through interactions with an unknown environment, thus provides two major advantages: portfolio scalability and market model independence. In quantitative finance, automated trading is essentially making dynamic decisions, namely **to decide where to trade, at what price, and what quantity**, over a highly stochastic and complex stock market. Taking many complex financial factors into account, DRL trading agents build a multi-factor model and provide algorithmic trading strategies, which are difficult for human traders.

`FinRL Library`_ provides a framework that supports various markets, SOTA DRL algorithms, benchmarks of many quant finance tasks, live trading, etc.

.. _FinRL Library: https://github.com/AI4Finance-Foundation/FinRL

Join or discuss FinRL: `AI4Finance mailing list <https://groups.google.com/u/1/g/ai4finance>`_.

Feel free to leave us feedback: report bugs using `Github issues`_ or discuss FinRL development in the Slack Channel.

.. _Github issues: https://github.com/AI4Finance-LLC/FinRL-Library/issues

.. image:: image/join_slack.png
   :target: https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-jyaottie-hHqU6TdvuhMHHAMXaLw_~w
   :width: 400
   :align: center


.. toctree::
   :maxdepth: 1
   :hidden:
   
   Home <self>


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   
   start/introduction
   start/first_glance
   start/three_layer
   start/installation
   start/quick_start


.. toctree:: 
   :maxdepth: 1
   :caption: FinRL-Meta

   finrl_meta/background
   finrl_meta/overview
   finrl_meta/Data_layer
   finrl_meta/Environment_layer
   finrl_meta/Benchmark


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorial/Guide
   tutorial/1-Introduction
   tutorial/2-Advance
   tutorial/3-Practical
   tutorial/4-Optimization
   tutorial/5-Others
   
   
.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/file_architecture
   developer_guide/development_setup


.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/publication
   reference/reference.md


.. toctree::
   :maxdepth: 2
   :caption: FAQ

   faq
