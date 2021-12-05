# FinRL: Deep Reinforcement Learning for Quantitative Finance [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]
 
[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-A-Deep-Reinforcement-Learning-Library-for-Quantitative-Finance%20&url=hhttps://github.com/AI4Finance-LLC/FinRL-Library&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-LLC%2FFinRL-Library
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-LLC/FinRL-Library
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-LLC%2FFinRL-Library

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="45%"/>
</div>

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

**Our Mission**: efficiently automate trading. We continuously develop and share codes for finance. 

**Our Vision**: AI community has accumulated an open-source code ocean over the past decade. We believe applying these intellectual and engineering properties to finance will initiate a paradigm shift from the conventional trading routine to an automated machine learning approach, even **RLOps in finance**.

[**FinRL**](https://finrl.readthedocs.io/en/latest/index.html) is the first open-source framework to demonstrate the great potential of applying deep reinforcement learning in quantitative finance. We help practitioners establish the development pipeline of trading strategies using **deep reinforcement learning (DRL)**. **A DRL agent learns by continuously interacting with an environment in a trial-and-error manner, making sequential decisions under uncertainty, and achieving a balance between exploration and exploitation**. 

**News: We will release codes for both paper trading and live trading. Please let us know your coding needs.**

Join to discuss FinRL: [AI4Finance mailing list](https://groups.google.com/u/1/g/ai4finance), AI4Finance Slack channel:


<a href="https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-v670l1jm-dzTgIT9fHZIjjrqprrY0kg" target="\_blank">
	<div align="center">
		<img src=figs/join_slack.png width="35%"/>
	</div>
</a>
<b>Follow us on WeChat:</b>
	<div align="center">
		<img src=http://www.tensorlet.org/wp-content/uploads/2021/01/qrcode_for_gh_feece88824ab_258.jpg width="15%" />
	</div>

The ecosystem of FinRL:

**FinRL 1.0**: entry-level for beginners, with a demonstrative and educational purpose.

**FinRL 2.0**: intermediate-level for full-stack developers and professionals, [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL).  

**FinRL 3.0**: advanced-level for investment banks and hedge funds, a cloud-native solution [FinRL-podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer). 

**FinRL 0.0**: hundreds of training/testing/trading environments in [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Metaverse).

FinRL provides a unified framework for various markets, SOTA DRL algorithms, finance tasks (portfolio allocation, cryptocurrency trading, high-frequency trading), live trading support, etc. 

## Outline

- [Tutorials](#Tutorials)
- [News](#News)
- [Overview](#Overview)
- [Status Update](#Status-Update)
- [Installation](#Installation)
- [Contributions](#Contributions)
- [Publications](#Publications)
- [Citing FinRL](#Citing-FinRL)
- [Welcome Contributions](#To-Contribute)
- [LICENSE](#LICENSE)

## Tutorials

+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Multiple Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Portfolio Allocation](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Single Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac)
+ [Towardsdatascience] [ElegantRL: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b)
+ [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791)
+ [MLearning.ai] [FinRL for Quantitative Finance: plug-and-play DRL algorithms](https://medium.com/mlearning-ai/finrl-for-quantitative-finance-plug-and-play-drl-algorithms-11cf494d28b1)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f)
+ [Analytics Vidhya] [Hyperparameter tuning using optuna for FinRL](https://medium.com/analytics-vidhya/hyperparameter-tuning-using-optuna-for-finrl-8a49506d2741)

## News
+ [央广网] [2021 IDEA大会开启AI思想盛宴 沈向洋理事长发布六大前沿产品](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [量化投资与机器学习] [基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)
+ [运筹OR帷幄] [领读计划NO.10 | 基于深度增强学习的量化交易机器人：从AlphaGo到FinRL的演变过程](https://zhuanlan.zhihu.com/p/353557417)
+ [深度强化实验室] [【重磅推荐】哥大开源“FinRL”: 一个用于量化金融自动交易的深度强化学习库](https://blog.csdn.net/deeprl/article/details/114828024)
+ [商业新知] [金融科技讲座回顾|AI4Finance: 从AlphaGo到FinRL](https://www.shangyexinzhi.com/article/4170766.html)
+ [Analyticsindiamag.com] [How To Automate Stock Market Using FinRL (Deep Reinforcement Learning Library)?](https://analyticsindiamag.com/stock-market-prediction-using-finrl/)
+ [Kaggle] [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199313)
+ [矩池云Matpool] [在矩池云上如何运行FinRL股票交易策略框架](http://www.python88.com/topic/111918)
+ [财智无界] [金融学会常务理事陈学彬: 深度强化学习在金融资产管理中的应用](https://www.sohu.com/a/486837028_120929319)
+ [Neurohive] [FinRL: глубокое обучение с подкреплением для трейдинга](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)
+ [ICHI.PRO] [양적 금융을위한 FinRL: 단일 주식 거래를위한 튜토리얼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)


## Overview

A video about [FinRL library](http://www.youtube.com/watch?v=ZSGJjtM-5jA). The [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g) for quantative finance.


<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

Supported Data Sources: 
|Data Source |Type |Range and Frequency |Request Limits|Raw Data|
|  ----  |  ----  |  ----  |  ----  |  ----  | 
|Yahoo! Finance| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | 
|CCXT| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| 
|WRDS.TAQ| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|
|Alpaca| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV|
|RiceQuant| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| 
|JoinQuant| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV|
|QuantConnect| US Securities| 1998-now, 1s| NA| OHLCV|
	
## DRL Algorithms 

[ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) implements Deep Q Learning (DQN), Double DQN, DDPG, A2C, SAC, PPO, TD3, GAE, MADDPG, etc. using PyTorch. 

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0 
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>

## Installation

+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## Contributions

- FinRL is the first open-source framework to demonstrate the great potential of applying DRL algorithms in quantitative finance. We build an ecosystem around the FinRL framework, which seeds the rapidly growing AI4Finance community. 
- The application layer provides interfaces for users to customize FinRL to their own trading tasks. Automated backtesting tool and performance metrics are provided to help quantitative traders iterate trading strategies at a high turnover rate. Profitable trading strategies are reproducible and hands-on tutorials are provided in a beginner-friendly fashion. Adjusting the trained models to the rapidly changing markets is also possible. 
- The agent layer provides state-of-the-art DRL algorithms that are adapted to finance with fine-tuned hyperparameters. Users can add new DRL algorithms. 
- The environment layer includes not only a collection of historical data APIs, but also live trading APIs. They are reconfigured into standard OpenAI gym-style environments. Moreover, it incorporates market frictions and allows users to customize the trading time granularity. 

## Publications

We published [papers in FinTech](http://tensorlet.org/projects/ai-in-finance/) at [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ) and now arrive at this project:

+ FinRL-Meta: Data-driven deep reinforcement learning in quantitative finance, Data-Centric AI Workshop, NeurIPS 2021.
+ Explainable deep reinforcement learning for portfolio management: An empirical approach. [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3958005) ACM International Conference on AI in Finance, ICAIF 2021.
+ FinRL-Podracer: High performance and scalable deep reinforcement learning for quantitative finance. ACM International Conference on AI in Finance, ICAIF 2021.
+ [FinRL](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949): Deep reinforcement learning framework to automate trading in quantitative finance, ACM International Conference on AI in Finance, ICAIF 2021.
+ [FinRL](https://arxiv.org/abs/2011.09607): A deep reinforcement learning library for automated stock trading in quantitative finance, Deep RL Workshop, NeurIPS 2020.
+ Deep reinforcement learning for automated stock trading: An ensemble strategy, [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) and [codes](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020), ACM International Conference on AI in Finance, ICAIF 2020.
+ Multi-agent reinforcement learning for liquidation strategy analysis, [paper](https://arxiv.org/abs/1906.11046) and [codes](https://github.com/WenhangBao/Multi-Agent-RL-for-Liquidation). Workshop on Applications and Infrastructure for Multi-Agent Learning, ICML 2019.
+ Practical deep reinforcement learning approach for stock trading, [paper](https://arxiv.org/abs/1811.07522) and [codes](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018), Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.

## Citing FinRL
```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
```

```
@article{liu2021finrl,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Gao, Jiechao and Wang, Christina Dan},
    title   = {{FinRL}: Deep reinforcement learning framework to automate trading in quantitative finance},
    journal = {ACM International Conference on AI in Finance (ICAIF)},
    year    = {2021}
}

```

## To Contribute

Welcome to join **AI4Finance Foundation** community!

Please check [Contributing Guidances](https://github.com/AI4Finance-LLC/FinRL/blob/master/contributing.md).

### Contributors

Thanks to our contributors! 
<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>


## LICENSE

MIT License

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

