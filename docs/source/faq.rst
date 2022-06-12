:github_url: https://github.com/AI4Finance-LLC/FinRL-Library

=============================
FAQ
=============================

:Version: 0.3
:Date: 05-29-2022
:Contributors: Roberto Fray da Silva, Xiao-Yang Liu, Ziyi Xia, Ming Zhu


This document contains the most frequently asked questions related to the FinRL Library, which are based on questions posted on the slack channels and Github_ issues.

.. _Github: https://github.com/AI4Finance-Foundation/FinRL


Outline
==================

    - :ref:`Section-1`

    - :ref:`Section-2`

    - :ref:`Section-3`

    - :ref:`Section-4`

    - :ref:`Section-5`


.. _Section-1:

1-Inputs and datasets
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">Can I use FinRL for crypto? </font>`

	*not yet. We're developing this functionality*

    - :raw-html:`<font color="#A52A2A">Can I use FinRL for live trading?  </font>`

	*not yet. We're developing this functionality*

    - :raw-html:`<font color="#A52A2A">Can I use FinRL for forex? </font>`

	*not yet. We're developing this functionality*

    - :raw-html:`<font color="#A52A2A">Can I use FinRL for futures? </font>`

	*not yet*

    -  :raw-html:`<font color="#A52A2A">What is the best data source for free daily data?</font>`

	*Yahoo Finance (through the yfinance library)*

    - :raw-html:`<font color="#A52A2A">What is the best data source for minute data? </font>`

	*Yahoo Finance (only up to last 7 days), through the yfinance library. It is the only option besides scraping (or paying for a service provider)*

    - :raw-html:`<font color="#A52A2A">Does FinRL support trading with leverage? </font>`

	*no, as this is more of an execution strategy related to risk control. You can use it as part of your system, adding the risk control part as a separate component*

    - :raw-html:`<font color="#A52A2A">Can a sentiment feature be added to improve the model's performance? </font>`

	*yes, you can add it. Remember to check on the code that this additional feature is being fed to the model (state)*

    - :raw-html:`<font color="#A52A2A">Is there a good free source for market sentiment to use as a feature?  </font>`

	*no, you'll have to use a paid service or library/code to scrape news and obtain the sentiment from them (normally, using deep learning and NLP)*

.. _Section-2:

2-Code and implementation
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">Does FinRL supports GPU training?  </font>`

	*yes, it does*

    - :raw-html:`<font color="#A52A2A">The code works for daily data but gives bad results on intraday frequency.</font>`

	*yes, because the current parameters are defined for daily data. You'll have to tune the model for intraday trading*

    - :raw-html:`<font color="#A52A2A">Are there different reward functions available? </font>`

	*not yet, but we're working on providing different reward functions and an easy way to set your own reward function*

    - :raw-html:`<font color="#A52A2A">Can I use a pre-trained model?  </font>`

	*yes, but none is available at the moment. Sometimes in the literature you'll find this referred to as transfer learning*

    - :raw-html:`<font color="#A52A2A">What is the most important hyperparameter to tune on the models?  </font>`

	*each model has its own hyperparameters, but the most important is the total_timesteps (think of it as epochs in a neural network: even if all the other hyperparameters are optimal, with few epochs the model will have a bad performance). The other important hyperparameters, in general, are: learning_rate, batch_size, ent_coef, buffer_size, policy, and reward scaling*

    - :raw-html:`<font color="#A52A2A">What are some libraries I could use to better tune the models? </font>`

	*there are several, such as: ray rllib and optuna. You'll have to implement them by yourself on the code, as this is not supported yet*

    - :raw-html:`<font color="#A52A2A">What DRL models can i use with FinRL?  </font>`

	*all the DRL models on Stable Baselines 3. We tested the following models with success: A2C, A3C, DDPG, PPO, SAC, TD3, TRPO. You can also create your own model, using the OpenAI Gym structure*

    - :raw-html:`<font color="#A52A2A">The model is presenting strange results OR is not training.   </font>`

	*Please update to latest version (https://github.com/AI4Finance-LLC/FinRL-Library), check if the hyperparameters used were not outside a normal range (ex: learning rate too high), and run the code again. If you still have problems, please check Section 2 (What to do when you experience problems)*

    - :raw-html: `<font color="#A52A2A">What to do when you experience problems? </font>`

    *1. Check if it is not already answered on this FAQ 2. Check if it is posted on the GitHub repo* `issues <https://github.com/AI4Finance-LLC/FinRL-Library/issues>`_. If not, welcome to submit an issue on GitHub 3. Use the correct channel on the AI4Finance slack or Wechat group.*

    - :raw-html: `<font color="#A52A2A">Does anyone know if there is a trading environment for a single stock? There is one in the docs, but the collab link seems to be broken. </font>`

        *We did not update the single stock for long time. The performance for single stock is not very good, since the state space is too small so that the agent extract little information from the environment. Please use the multi stock environment, and after training only use the single stock to trade.*


.. _Section-3:

3-Model evaluation
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">The model did not beat buy and hold (BH) with my data. Is the model or code wrong?  </font>`

	*not exactly. Depending on the period, the asset, the model chosen, and the hyperparameters used, BH may be very difficult to beat (it's almost never beaten on stocks/periods with low volatility and steady growth). Nevertheless, update the library and its dependencies (the github repo has the most recent version), and check the example notebook for the specific environment type (single, multi, portfolio optimization) to see if the code is running correctly*

    - :raw-html:`<font color="#A52A2A">How does backtesting works in the library?  </font>`

	*we use the Pyfolio backtest library from Quantopian ( https://github.com/quantopian/pyfolio ), especially the simple tear sheet and its charts. In general, the most important metrics are: annual returns, cumulative returns, annual volatility, sharpe ratio, calmar ratio, stability, and max drawdown*

    - :raw-html:`<font color="#A52A2A">Which metrics should I use for evaluting the model?  </font>`

	*there are several metrics, but we recommend the following, as they are the most used in the market: annual returns, cumulative returns, annual volatility, sharpe ratio, calmar ratio, stability, and max drawdown*

    - :raw-html:`<font color="#A52A2A">Which models should I use as a baseline for comparison?  </font>`

	*we recommend using buy and hold (BH), as it's a strategy that can be followed on any market and tends to provide good results in the long run. You can also compare with other DRL models and trading strategies such as the minimum variance portfolio*

.. _Section-4:

4-Miscellaneous
========================================================================

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">I'm interested, but I know nothing. How should I start? </font>`

    *1. read the documentation from the very beginning 2. go through * `tutorials <https://github.com/AI4Finance-Foundation/FinRL/tree/master/tutorials>`_ *3. read our papers*

    - :raw-html:`<font color="#A52A2A">What is the development roadmap for the library?  </font>`

	*this is available on our Github repo* https://github.com/AI4Finance-LLC/FinRL-Library

    - :raw-html:`<font color="#A52A2A">How can I contribute to the development?  </font>`

	*participate on the slack channels, check the current issues and the roadmap, and help any way you can (sharing the library with others, testing the library of different markets/models/strategies, contributing with code development, etc)*

    - :raw-html:`<font color="#A52A2A">What are some good references before I start using the library?  </font>`

	*please read* :ref:`Section-1`

    - :raw-html:`<font color="#A52A2A">What are some good RL references for people from finance? What are some good finance references for people from ML? </font>`

	*please read* :ref:`Section-4`

    - :raw-html:`<font color="#A52A2A">What new SOTA models will be incorporated on FinRL?  </font>`

	*please check our development roadmap at our Github repo: https://github.com/AI4Finance-LLC/FinRL-Library*
	
    - :raw-html:`<font color="#A52A2A">What's the main difference between FinRL and FinRL-Meta?  </font>`

	*FinRL is the project of financial RL for education and demonstration purpose, while FinRL-Meta is with aim for financial big data and metaverse in data driven financial RL.*

.. _Section-5:
    
5-Common issues/bugs
====================================
- Package trading_calendars reports errors in Windows system:\
    Trading_calendars is not maintained now. It may report errors in Windows system (python>=3.7). These are two possible solutions: 1).Use python=3.6 environment. 2).Replace trading_calendars with exchange_caldenars.
