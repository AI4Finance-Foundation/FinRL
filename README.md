# FinRL: Deep Reinforcement Learning for Quantitative Finance [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]
 
[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-Foundation/FinRL
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="45%"/> 
</div>


**Our Mission**: to efficiently automate trading. We continuously develop and share codes for finance. 

**Our Vision**: AI community has accumulated an open-source code ocean over the past decade. Applying these intellectual and engineering properties to finance will initiate a paradigm shift from the conventional trading routine to an automated machine learning approach, even **RLOps in finance**. 

**FinRL** ([website](https://finrl.readthedocs.io/en/latest/index.html)) is **the first open-source project** to explore the great potential of deep reinforcement learning in finance. We help practitioners pipeline a trading strategy using **deep reinforcement learning (DRL)**. 

The FinRL ecosystem is a unified framework, including various markets, state-of-the-art algorithms, financial tasks (portfolio management, cryptocurrency trading, high-frequency trading), live trading, etc. 

| Roadmap  | Level | Users | Example | Desription | 
|----|----|----|----|----|
| 0.0 (Preparation) | preparation | practitioners of financial big data | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Metaverse)| a universe of market environments|
| 1.0 (Proof-of-Concept)| entry-level | beginners | [this repo](https://github.com/AI4Finance-Foundation/FinRL) | demonstration, education |
| 2.0 (Professional) | intermediate-level  | full-stack developers, professionals | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | financially optimized DRL algorithms |
| 3.0 (Production) | advance-level | investment banks, hedge funds | [Podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | cloud-native solution |


## Outline

- [Overview](#Overview)
- [File Structure](#File-Structure)
- [Supported Data Sources](#Supported-Data-Sources)
- [Installation](#Installation)
- [Status Update](#Status-Update)
- [Contributions](#Contributions)
- [Tutorials](#Tutorials)
- [News](#News)
- [Citing FinRL](#Citing-FinRL)
- [Welcome Contributions](#To-Contribute)
- [Sponsorship](#Sponsorship)
- [LICENSE](#LICENSE)

## Overview

FinRL has three layers: applications, drl agents, and market environments.

For a trading task (on the top), an agent (in the middle) interacts with an environment (at the bottom), making sequential decisions.


<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>


Run [FinRL_StockTrading_NeurIPS_2018.ipynb](https://github.com/AI4Finance-Foundation/FinRL/blob/master/FinRL_StockTrading_NeurIPS_2018.ipynb) step by step for a quick start.

A video about [FinRL library](http://www.youtube.com/watch?v=ZSGJjtM-5jA) at the [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g).

## File Structure

Correspondingly, the main folder **finrl** has three subfolders **apps, drl_agents, finrl_meta**. 

We employ a **train-test-trade** pipeline by three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (main folder)
│   ├── apps
│   	├── cryptocurrency_trading
│   	├── high_frequency_trading
│   	├── portfolio_allocation
│   	├── stock_trading
│   	└── 
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
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│   
├── tutorial (tutorial notebooks and educational files)
├── unit_testing (make sure verified codes working on env & data)
│   ├── test_env
│   	└── test_env_cashpenalty.py
│   └── test_marketdata
│   	└── test_yahoodownload.py
├── setup.py
├── requirements.txt
└── README.md
```

## Supported Data Sources 

|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Alpaca](https://alpaca.markets/docs/introduction/)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/home/home)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
|[tusharepro](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS.TAQ](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[Yahoo! Finance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|

OHLCV: open, high, low, and close prices; volume.   adj_close: adjusted close price

Technical indicators: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'.  Users also can add new features. 


## Installation

+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

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

## Contributions

- FinRL is the first open-source framework to demonstrate the great potential of applying DRL algorithms in quantitative finance. We build an ecosystem around the FinRL framework, which seeds the rapidly growing AI4Finance community. 
- The application layer provides interfaces for users to customize FinRL to their own trading tasks. Automated backtesting tool and performance metrics are provided to help quantitative traders iterate trading strategies at a high turnover rate. Profitable trading strategies are reproducible and hands-on tutorials are provided in a beginner-friendly fashion. Adjusting the trained models to the rapidly changing markets is also possible. 
- The agent layer provides state-of-the-art DRL algorithms that are adapted to finance with fine-tuned hyperparameters. Users can add new DRL algorithms. 
- The environment layer includes not only a collection of historical data APIs, but also live trading APIs. They are reconfigured into standard OpenAI gym-style environments. Moreover, it incorporates market frictions and allows users to customize the trading time granularity. 


## Tutorials

+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Multiple Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Portfolio Allocation](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Single Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac)
+ [Towardsdatascience] [ElegantRL-Podracer: A Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning](https://elegantrl.medium.com/elegantrl-podracer-scalable-and-elastic-library-for-cloud-native-deep-reinforcement-learning-bafda6f7fbe0)
+ [Towardsdatascience] [ElegantRL: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b)
+ [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791)
+ [MLearning.ai] [Hyperparameter Optimization using Ray tune for FinRL models](https://medium.com/mlearning-ai/hyperparameter-optimization-using-ray-tune-for-finrl-models-42df2937d53d)
+ [MLearning.ai] [An Empirical Approach to Explain Deep Reinforcement Learning in Portfolio Management Task](https://medium.com/mlearning-ai/an-empirical-approach-to-explain-deep-reinforcement-learning-in-portfolio-management-task-e65a42225d9d)
+ [MLearning.ai] [FinRL for Quantitative Finance: plug-and-play DRL algorithms](https://medium.com/mlearning-ai/finrl-for-quantitative-finance-plug-and-play-drl-algorithms-11cf494d28b1)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f)
+ [DataDrivenInvestor] [FinRL-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)
+ [DataDrivenInvestor] [A Data Scientist’s Approach for Algorithmic Trading using Deep Reinforcement Learning: An End-to-end Tutorial for Paper Trading](https://medium.datadriveninvestor.com/a-data-scientists-approach-for-algorithmic-trading-using-deep-reinforcement-learning-an-be8da40b2230)
+ [Analytics Vidhya] [Weights and Biases-ify FinRL with Stable Baselines3 models](https://medium.com/analytics-vidhya/weights-and-biases-ify-stable-baselines-models-in-finrl-f11b67f2a6a7)
+ [Analytics Vidhya] [Hyperparameter tuning using optuna for FinRL](https://medium.com/analytics-vidhya/hyperparameter-tuning-using-optuna-for-finrl-8a49506d2741)
+ [Analytics Vidhya] [A hitchhikers guide to FinRL: A Deep Reinforcement Learning Framework for Quantitative Finance](https://medium.com/analytics-vidhya/a-hitchhikers-guide-to-finrl-a-deep-reinforcement-learning-framework-for-quantitative-finance-e624c508f763)
+ [Analyticsindiamag.com] [How To Automate Stock Market Using FinRL (Deep Reinforcement Learning Library)?](https://analyticsindiamag.com/stock-market-prediction-using-finrl/)


## News
+ [央广网] [2021 IDEA大会于福田圆满落幕：群英荟萃论道AI 多项目发布亮点纷呈](http://tech.cnr.cn/techph/20211123/t20211123_525669092.shtml)
+ [央广网] [2021 IDEA大会开启AI思想盛宴 沈向洋理事长发布六大前沿产品](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [IDEA新闻] [2021 IDEA大会发布产品FinRL-Meta——基于数据驱动的强化学习金融风险模拟系统](https://idea.edu.cn/news/20211213143128.html)
+ [知乎] [FinRL-Meta基于数据驱动的强化学习金融元宇宙](https://zhuanlan.zhihu.com/p/437804814)
+ [量化投资与机器学习] [基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)
+ [运筹OR帷幄] [领读计划NO.10 | 基于深度增强学习的量化交易机器人：从AlphaGo到FinRL的演变过程](https://zhuanlan.zhihu.com/p/353557417)
+ [深度强化实验室] [【重磅推荐】哥大开源“FinRL”: 一个用于量化金融自动交易的深度强化学习库](https://blog.csdn.net/deeprl/article/details/114828024)
+ [商业新知] [金融科技讲座回顾|AI4Finance: 从AlphaGo到FinRL](https://www.shangyexinzhi.com/article/4170766.html)
+ [Kaggle] [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199313)
+ [矩池云Matpool] [在矩池云上如何运行FinRL股票交易策略框架](http://www.python88.com/topic/111918)
+ [财智无界] [金融学会常务理事陈学彬: 深度强化学习在金融资产管理中的应用](https://www.sohu.com/a/486837028_120929319)
+ [Neurohive] [FinRL: глубокое обучение с подкреплением для трейдинга](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)
+ [ICHI.PRO] [양적 금융을위한 FinRL: 단일 주식 거래를위한 튜토리얼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)


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

We published [FinTech papers](http://tensorlet.org/projects/ai-in-finance/), check [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ), resulting in this project. Closely related papers are given in the [list](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/FinRL_papers.md). 


## Join and Contribute

Welcome to the **AI4Finance Foundation** community!

Join to discuss FinRL: [AI4Finance mailing list](https://groups.google.com/u/1/g/ai4finance), AI4Finance Slack channel:


<a href="https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-v670l1jm-dzTgIT9fHZIjjrqprrY0kg" target="\_blank">
	<div align="center">
		<img src=figs/join_slack.png width="35%"/>
	</div>
</a>
<b>Follow us on WeChat:</b>
	<div align="center">
		<img src=http://www.tensorlet.org/wp-content/uploads/2021/01/qrcode_for_gh_feece88824ab_258.jpg width="25%" />
	</div>
</b>

Please check [Contributing Guidances](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/Contributing.md).

### Contributors

Thanks! 

<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>

### Sponsorship

Welcome gift money to support AI4Finance, a non-profit academic community. Use the links in the right, or scan the following vemo QR code:

Detailed sponsorship records can be found at [Issue #425](https://github.com/AI4Finance-Foundation/FinRL/issues/425)

<a target="\_blank">
	<div align="center">
		<img src=figs/Xiao-Yang-Liu_AI4Finance_vemo.png width="35%"/>
	</div>
</a>


## LICENSE

MIT License

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**

