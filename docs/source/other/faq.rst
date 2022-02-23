:github_url: https://github.com/AI4Finance-LLC/FinRL-Library


FAQ
=============================

:Version: 0.1
:Date: 02-06-2021
:Contributors: Roberto Fray da Silva, Xiao-Yang Liu



Description
----------------

This document contains the most frequently asked questions related to the FinRL Library, based on questions posted on the slack channels and Github_ issues.

.. _Github: https://github.com/AI4Finance-LLC/FinRL-Library


Outline
----------------

    - :ref:`Section-1`

    - :ref:`Section-2`

    - :ref:`Section-3`

	      - :ref:`Section-3-1`

	      - :ref:`Section-3-2`

	      - :ref:`Section-3-3`

	      - :ref:`Section-3-4`

    - :ref:`Section-4`

		- :ref:`Section-4-1`

		- :ref:`Section-4-2`
		
		- :ref:`Section-4-3`

    - :ref:`Section-5`


.. _Section-1:

Section 1  Where to start?
--------------------------------

    - Read the paper that describes the FinRL library: Liu, X.Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B. and Wang, C.D., 2020. FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance. Deep RL Workshop, NeurIPS 2020. https://arxiv.org/abs/2011.09607

    - Read the post related to the type of environment you want to work on (multi stock trading, portfolio optimization) https://github.com/AI4Finance-LLC/FinRL-Library, Section "News"

    - Install the library following the instructions at the official Github repo: https://github.com/AI4Finance-LLC/FinRL-Library

    - Run the Jupyter notebooks related to the type of environment you want to work on notebooks folder of the library https://github.com/AI4Finance-LLC/FinRL-Library/tree/master/notebooks

    - Enter on the AI4Finance slack: https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-kq0c9het-FCSU6Y986OnSw6Wb5EkEYw


.. _Section-2:

Section 2 What to do when you experience problems?
----------------------------------------------------------------

    - If any questions arise, please follow this sequence of activities (it allows us to focus on the main issues that need to be solved, instead of repeatedly answering the same questions):

        - Check if it is not already answered on this FAQ

        - Check if it is not posted on the Github repo issues:https://github.com/AI4Finance-LLC/FinRL-Library/issues

        - Use the correct slack channel on the AI4Finance slack.


.. _Section-3:

Section 3 Most frequently asked questions related to the FinRL Library
---------------------------------------------------------------------------

.. _Section-3-1:

Subsection 3.1  Inputs and datasets
-----------------------------------------------------------------

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

.. _Section-3-2:

Subsection 3.2 Code and implementation
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - :raw-html:`<font color="#A52A2A">Does FinRL supports GPU training?  </font>`

	*yes, it does*

    - :raw-html:`<font color="#A52A2A">The code works for daily data but gives bad results on intraday frequency.</font>`

	*yes, because the current parameters are defined for daily data. You'll have to tune the model for intraday trading*

    - :raw-html:`<font color="#A52A2A">Are there different reward functions available? </font>`

	*not yet, but we're working on providing different reward functions and an easy way to code your own reward function*

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

.. _Section-3-3:

Subsection 3.3 Model evaluation
-----------------------------------------------------------------

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

.. _Section-3-4:

Subsection 3.4 Miscellaneous
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

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

.. _Section-4:

Section 4 References for diving deep into Deep Reinforcement Learning (DRL)
------------------------------------------------------------------------------

.. _Section-4-1:

Subsection 4.1 General resources
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - OpenAI Spinning UP DRL, educational resource
        https://spinningup.openai.com/en/latest/

    - Awesome-ai-in-finance
        https://github.com/georgezouq/awesome-ai-in-finance

    - penAI Gym
        https://github.com/openai/gym

    - Stable Baselines 3
        contains the implementations of all models used by FinRL
        https://github.com/DLR-RM/stable-baselines3

    - Ray RLlib
        https://docs.ray.io/en/master/rllib.html

    - Policy gradient algorithms
        https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

    - Fischer, T.G., 2018. Reinforcement learning in financial markets-a survey (No. 12/2018). FAU Discussion Papers in Economics. (:raw-html:`<font color="#A52A2A">a survey on the use of RL for finance </font>`)

    - Li, Y., 2018. Deep reinforcement learning. arXiv preprint arXiv:1810.06339. (:raw-html:`<font color="#A52A2A">an in-depth review of DRL and its main models and components</font>`)

    - Charpentier, A., Elie, R. and Remlinger, C., 2020. Reinforcement learning in economics and finance. arXiv preprint arXiv:2003.10014. (:raw-html:`<font color="#A52A2A">an in-depth review of uses of RL and DRL in finance</font>`)

    - Kolm, P.N. and Ritter, G., 2020. Modern perspectives on reinforcement learning in finance. Modern Perspectives on Reinforcement Learning in Finance (September 6, 2019). The Journal of Machine Learning in Finance, 1(1) (:raw-html:`<font color="#A52A2A">an in-depth review of uses of RL and DRL in finance</font>`)

    - Practical Deep Reinforcement Learning Approach for Stock Trading, paper and codes, Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.


.. _Section-4-2:

Subsection 4.2 Papers related to the implemented DRL models
-----------------------------------------------------------------

	.. role:: raw-html(raw)
	   :format: html

    - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D. and Riedmiller, M., 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 	(:raw-html:`<font color="#A52A2A">the first paper that proposed (with success) the use of DL in RL</font>`)

    - Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G. and Petersen, S., 2015. Human-level control through deep reinforcement learning. Nature, 518(7540), pp.529-533 (:raw-html:`<font color="#A52A2A">an excellent review paper of important concepts on DRL</font>`)

    - Lillicrap, T.P., Hunt, J.J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D. and Wierstra, D., 2015. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (:raw-html:`<font color="#A52A2A">paper that proposed the DDPG model</font>`)

    - Fujimoto, S., Hoof, H. and Meger, D., 2018, July. Addressing function approximation error in actor-critic methods. In International Conference on Machine Learning (pp. 1587-1596). PMLR (:raw-html:`<font color="#A52A2A">paper that proposed the TD3 model</font>`)

    - Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (:raw-html:`<font color="#A52A2A">paper that proposed the PPO model</font>`)

    - Mnih, V., Badia, A.P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D. and Kavukcuoglu, K., 2016, June. Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937). PMLR (:raw-html:`<font color="#A52A2A">paper that proposed the A3C model</font>`)

    - https://openai.com/blog/baselines-acktr-a2c/ (:raw-html:`<font color="#A52A2A">description of the implementation of the A2C model</font>`)

    - Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P., 2015, June. Trust region policy optimization. In International conference on machine learning (pp. 1889-1897). PMLR (:raw-html:`<font color="#A52A2A">description of the implementation of the TRPO model</font>`)
    
  
.. _Section-4-3:

Subsection 4.3 Challenges of DataOps and MLOps
-----------------------------------------------------------------

 
    - Paleyes, A., Urma, R.G. and Lawrence, N.D., 2020. Challenges in deploying machine learning: a survey of case studies. arXiv preprint arXiv:2011.09926.

.. _Section-5:
    
Section 5  Common issues/bugs
--------------------------------
- Package trading_calendars reports errors in Windows system:\
    Trading_calendars is not maintained now. It may report erros in Windows system (python>=3.7). These are two possible solutions: 1.Use python=3.6 environment 2.Replace trading_calendars with exchange_caldenars.
