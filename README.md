# FinRL: Financial Reinforcement Learning [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]

[1.1]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.org/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.org/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-Financial-Deep-Reinforcement-Learning%20&url=https://github.com/AI4Finance-Foundation/FinRL&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-Foundation/FinRL
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-Foundation%2FFinRL

<div align="center">
<img align="center" src=figs/logo_transparent_background.png width="55%"/>
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)
[![Documentation Status](https://readthedocs.org/projects/finrl/badge/?version=latest)](https://finrl.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/AI4Finance-Foundation/finrl.svg?color=brightgreen)



**Financial reinforcement learning (FinRL)** ([Document website](https://finrl.readthedocs.io/en/latest/index.html)) is **the first open-source framework** for financial reinforcement learning. FinRL has evolved into an **ecosystem**

| Dev Roadmap  | Stage | Users | Project | Desription |
|----|----|----|----|----|
| 0.0 (Preparation) | entrance | practitioners | [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)| gym-style market environments |
| 1.0 (Proof-of-Concept)| full-stack | developers | [this repo](https://github.com/AI4Finance-Foundation/FinRL) | automatic pipeline |
| 2.0 (Professional) | profession | experts | [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) | algorithms |
| 3.0 (Production) | service | hedge funds | [Podracer](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | cloud-native deployment |


## Outline

- [Overview](#Overview)
- [File Structure](#File-Structure)
- [Supported Data Sources](#Supported-Data-Sources)
- [Installation](#Installation)
- [Status Update](#Status-Update)
- [Tutorials](#Tutorials)
- [Publications](#Publications)
- [News](#News)
- [Citing FinRL](#Citing-FinRL)
- [Welcome Contributions](#To-Contribute)
- [Sponsorship](#Sponsorship)
- [LICENSE](#LICENSE)

## Overview

FinRL has three layers: market environments, agents, and applications.  For a trading task (on the top), an agent (in the middle) interacts with a market environment (at the bottom), making sequential decisions.

<div align="center">
<img align="center" src=figs/finrl_framework.png>
</div>

A quick start: Stock_NeurIPS2018.ipynb. Videos [FinRL](http://www.youtube.com/watch?v=ZSGJjtM-5jA) at [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g).


## File Structure

The main folder **finrl** has three subfolders **applications, agents, meta**. We employ a **train-test-trade** pipeline with three files: train.py, test.py, and trade.py.

```
FinRL
├── finrl (main folder)
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
│       ├── meta_config_tickers.py
│   	└── meta_config.py
│   ├── config.py
│   ├── config_tickers.py
│   ├── main.py
│   ├── plot.py
│   ├── train.py
│   ├── test.py
│   └── trade.py
│
├── examples
├── unit_tests (unit tests to verify codes on env & data)
│   ├── environments
│   	└── test_env_cashpenalty.py
│   └── downloaders
│   	├── test_yahoodownload.py
│   	└── test_alpaca_downloader.py
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
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|


<!-- |Data Source |Type |Max Frequency |Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|    AkShare |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Alpaca |  US Stocks, ETFs |  1 min |  OHLCV |  Prices, indicators |
|    Alpha Vantage | Stock, ETF, forex, crypto, technical indicators | 1 min |  OHLCV  & Prices, indicators |
|    Baostock |  CN Securities |  5 min |  OHLCV |  Prices, indicators |
|    Binance |  Cryptocurrency |  1 s |  OHLCV |  Prices, indicators |
|    CCXT |  Cryptocurrency |  1 min  |  OHLCV |  Prices, indicators |
|    currencyapi |  Exchange rate | 1 day |  Exchange rate | Exchange rate, indicators |
|    currencylayer |  Exchange rate | 1 day  |  Exchange rate | Exchange rate, indicators |
|    EOD Historical Data | US stocks, and ETFs |  1 day  |  OHLCV  | Prices, indicators |
|    Exchangerates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    findatapy |  CN Securities | 1 day  |  OHLCV |  Prices, indicators |
|    Financial Modeling prep | US stocks, currencies, crypto |  1 min |  OHLCV  | Prices, indicators |
|    finnhub | US Stocks, currencies, crypto |   1 day |  OHLCV  | Prices, indicators |
|    Fixer |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    IEXCloud |  NMS US securities | 1 day  | OHLCV |  Prices, indicators |
|    JoinQuant |  CN Securities |  1 min  |  OHLCV |  Prices, indicators |
|    Marketstack | 50+ countries |  1 day  |  OHLCV | Prices, indicators |
|    Open Exchange Rates |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    pandas\_datareader |  US Securities |  1 day |  OHLCV | Prices, indicators |
|    pandas-finance |  US Securities |  1 day  |  OHLCV  & Prices, indicators |
|    Polygon |  US Securities |  1 day  |  OHLCV  | Prices, indicators |
|    Quandl | 250+ sources |  1 day  |  OHLCV  | Prices, indicators |
|    QuantConnect |  US Securities |  1 s |  OHLCV |  Prices, indicators |
|    RiceQuant |  CN Securities |  1 ms  |  OHLCV |  Prices, indicators |
|    Tiingo | Stocks, crypto |  1 day  |  OHLCV  | Prices, indicators |
|    Tushare |  CN Securities | 1 min  |  OHLCV |  Prices, indicators |
|    WRDS |  US Securities |  1 ms  |  Intraday Trades | Prices, indicators |
|    XE |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    Xignite |  Exchange rate |  1 day  |  Exchange rate | Exchange rate, indicators |
|    YahooFinance |  US Securities | 1 min  |  OHLCV  |  Prices, indicators |
|    ystockquote |  US Securities |  1 day  |  OHLCV | Prices, indicators | -->



OHLCV: open, high, low, and close prices; volume. adjusted_close: adjusted close price

Technical indicators: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add new features.


## Installation
+ [Install description for all operating systems (MAC OS, Ubuntu, Windows 10)](./docs/source/start/installation.rst)
+ [FinRL for Quantitative Finance: Install and Setup Tutorial for Beginners](https://ai4finance.medium.com/finrl-for-quantitative-finance-install-and-setup-tutorial-for-beginners-1db80ad39159)

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>

* 2022-06-25
	0.3.5: Formal release of FinRL, neo_finrl is chenged to FinRL-Meta with related files in directory: *meta*.
* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>


## Tutorials

+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)

A complete list at [blogs](https://github.com/AI4Finance-Foundation/Blogs)


## Publications

|Title |Conference |Link|Citations|Year|
|  ----  |  ----  |  ----  |  ----  |  ----  |
|**FinRL-Meta**: FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning| NeurIPS 2022| [paper](https://arxiv.org/abs/2211.03107) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta) | 1 | 2022 |
|**FinRL**: Deep reinforcement learning framework to automate trading in quantitative finance| ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3955949) | 19 | 2021 |
|**FinRL-Podracer**: High performance and scalable deep reinforcement learning for quantitative finance | ACM International Conference on AI in Finance (ICAIF) | [paper](https://arxiv.org/abs/2111.05188) [code](https://github.com/AI4Finance-Foundation/FinRL_Podracer) | 8 | 2021 |
|Explainable deep reinforcement learning for portfolio management: An empirical approach| ACM International Conference on AI in Finance (ICAIF) | [paper](https://arxiv.org/abs/2111.03995) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_PortfolioAllocation_Explainable_DRL/FinRL_PortfolioAllocation_Explainable_DRL.py](https://github.com/AI4Finance-Foundation/FinRL-Tutorials/tree/master/2-Advance))| 3 | 2021 |
|**FinRL**: A deep reinforcement learning library for automated stock trading in quantitative finance| NeurIPS 2020 Deep RL Workshop  | [paper](https://arxiv.org/abs/2011.09607) | 46 | 2020 |
|Deep reinforcement learning for automated stock trading: An ensemble strategy| ACM International Conference on AI in Finance (ICAIF) | [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) [code](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/tutorials/2-Advance/FinRL_Ensemble_StockTrading_ICAIF_2020/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) | 94 | 2020 |
|Practical deep reinforcement learning approach for stock trading | NeurIPS 2018 Workshop on Challenges and Opportunities for AI in Financial Services| [paper](https://arxiv.org/abs/1811.07522) [code](https://github.com/AI4Finance-Foundation/DQN-DDPG_Stock_Trading](https://github.com/AI4Finance-Foundation/FinRL/tree/master/examples))| 127 | 2018 |


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
@article{liu2022finrl_meta,
  title={FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning},
  author={Liu, Xiao-Yang and Xia, Ziyi and Rui, Jingyang and Gao, Jiechao and Yang, Hongyang and Zhu, Ming and Wang, Christina Dan and Wang, Zhaoran and Guo, Jian},
  journal={NeurIPS},
  year={2022}
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

```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
```

```
@article{liu2018practical,
  title={Practical deep reinforcement learning approach for stock trading},
  author={Liu, Xiao-Yang and Xiong, Zhuoran and Zhong, Shan and Yang, Hongyang and Walid, Anwar},
  journal={NeurIPS Workshop on Deep Reinforcement Learning},
  year={2018}
}
```

We published [FinRL papers](http://tensorlet.org/projects/ai-in-finance/) that are listed at [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ). Previous papers are given in the [list](https://github.com/AI4Finance-Foundation/FinRL/blob/master/tutorials/FinRL_papers.md).


## Join and Contribute

Welcome to **AI4Finance** community!

Discuss FinRL via [AI4Finance mailing list](https://groups.google.com/u/1/g/ai4finance) and AI4Finance Slack channel:


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

Thank you!

<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>

### Sponsorship

Welcome gift money to support AI4Finance, a non-profit community.

Network: USDT-TRC20

<a target="\_blank">
	<div align="center">
		<img src=figs/okx.jpeg width="35%"/>
	</div>
</a>



## LICENSE

MIT License

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
