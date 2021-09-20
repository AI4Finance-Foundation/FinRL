# FinRL: A Deep Reinforcement Learning Framework for Quantitative Finance [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]
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

**News: we plan to share our codes for both paper trading and live trading. Please actively share your interests with our community.**

**FinRL** is an open-source framework to help practitioners establish the development pipeline of trading strategies based on **deep reinforcement learning (DRL)**. **An agent learns by continuously interacting with an environment in a trial-and-error manner, making sequential decisions under uncertainty, and achieving a balance between exploration and exploitation**. The open-source community **AI4Finance** (efficiently automating trading) provides resources about deep reinforcement learning in quantitative finance. It accelerates the paradigm shift from the conventional machine learning approach to **RLOps in finance**.


Join or discuss FinRL: [AI4Finance mailing list](https://groups.google.com/u/1/g/ai4finance), AI4Finance Slack channel:


<a href="https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-v670l1jm-dzTgIT9fHZIjjrqprrY0kg" target="\_blank">
	<div align="center">
		<img src=figs/join_slack.png width="35%"/>
	</div>
</a>

Roadmaps of FinRL:

**FinRL 1.0**: entry-level for beginners, with a demonstrative and educational purpose.

**FinRL 2.0**: intermediate-level for full-stack developers and professionals. Check out [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL).  

**FinRL 3.0**: advanced-level for investment banks and hedge funds. Check out our cloud-native solution [GPU-podracer](https://github.com/AI4Finance-Foundation/GPU_Podracer). 

**FinRL 0.0**: tens of training/testing/trading environments in [NeoFinRL](https://github.com/AI4Finance-Foundation/NeoFinRL).

FinRL provides a unified DRL framework for various markets, SOTA DRL algorithms, benchmark finance tasks (portfolio allocation, cryptocurrency trading, high-frequency trading), live trading, etc. 

## Outline

- [Prior Arts](#Prior-Arts)
- [News](#Tutorials-and-News)
- [Overview](#Overview)
- [Status Update](#Status-Update)
- [Installation](#Installation)
	- [Docker Installation](#Docker-Installation)
	- [Prerequisites](#Prerequisites)
	- [Dependencies](#Dependencies)
- [Contributions](#Contributions)
	- [Citing FinRL](#Citing-FinRL)
	- [Welcome Contributions](#Call-for-Contributions)
- [LICENSE](#LICENSE)

## Prior Arts

We published [papers in FinTech](http://tensorlet.org/projects/ai-in-finance/) at [Google Scholar](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XsdPXocAAAAJ) and now arrive at this project:

+ [FinRL](https://arxiv.org/abs/2011.09607): A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, Deep RL Workshop, NeurIPS 2020.
+ Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) and [codes](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020), ACM International Conference on AI in Finance, ICAIF 2020.
+ Multi-agent Reinforcement Learning for Liquidation Strategy Analysis, [paper](https://arxiv.org/abs/1906.11046) and [codes](https://github.com/WenhangBao/Multi-Agent-RL-for-Liquidation). Workshop on Applications and Infrastructure for Multi-Agent Learning, ICML 2019.
+ Practical Deep Reinforcement Learning Approach for Stock Trading, [paper](https://arxiv.org/abs/1811.07522) and [codes](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018), Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.

## Tutorials and News
+ [Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Multiple Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Portfolio Allocation](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd)
+ [Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Single Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac)
+ [Towardsdatascience] [ElegantRL: A Lightweight and Stable Deep Reinforcement Learning Library](https://towardsdatascience.com/elegantrl-a-lightweight-and-stable-deep-reinforcement-learning-library-95cef5f3460b)
+ [Towardsdatascience] [ElegantRL: Mastering PPO Algorithms](https://medium.com/@elegantrl/elegantrl-mastering-the-ppo-algorithm-part-i-9f36bc47b791)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part I)](https://elegantrl.medium.com/elegantrl-demo-stock-trading-using-ddpg-part-i-e77d7dc9d208)
+ [MLearning.ai] [ElegantRL Demo: Stock Trading Using DDPG (Part II)](https://medium.com/mlearning-ai/elegantrl-demo-stock-trading-using-ddpg-part-ii-d3d97e01999f)
+ [Analyticsindiamag.com] [How To Automate Stock Market Using FinRL (Deep Reinforcement Learning Library)?](https://analyticsindiamag.com/stock-market-prediction-using-finrl/)
+ [Kaggle] [Jane Street Market Prediction](https://www.kaggle.com/c/jane-street-market-prediction/discussion/199313)
+ [量化投资与机器学习] [基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)
+ [运筹OR帷幄] [领读计划NO.10 | 基于深度增强学习的量化交易机器人：从AlphaGo到FinRL的演变过程](https://zhuanlan.zhihu.com/p/353557417)
+ [深度强化实验室] [【重磅推荐】哥大开源“FinRL”: 一个用于量化金融自动交易的深度强化学习库](https://blog.csdn.net/deeprl/article/details/114828024)
+ [商业新知] [金融科技讲座回顾|AI4Finance: 从AlphaGo到FinRL](https://www.shangyexinzhi.com/article/4170766.html)
+ [矩池云Matpool] [在矩池云上如何运行FinRL股票交易策略框架](http://www.python88.com/topic/111918)
+ [财智无界] [金融学会常务理事陈学彬: 深度强化学习在金融资产管理中的应用](https://www.sohu.com/a/486837028_120929319)
+ [Neurohive] [FinRL: глубокое обучение с подкреплением для трейдинга](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)
+ [ICHI.PRO] [양적 금융을위한 FinRL: 단일 주식 거래를위한 튜토리얼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)


## Overview

A video about FinRL library. The [AI4Finance Youtube Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g) for quantative finance.

[<div align="center"><img src="http://img.youtube.com/vi/ZSGJjtM-5jA/0.jpg" width="70%"></div>](http://www.youtube.com/watch?v=ZSGJjtM-5jA)

<div align="center">
<img align="center" src=figs/Poster_FinRL.jpg width="600">
</div>
	
## DRL Algorithms 

We implemented Deep Q Learning (DQN), Double DQN, DDPG, A2C, SAC, PPO, TD3, GAE, MADDPG, etc. using PyTorch and OpenAI Gym. 

## Status Update
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>
	
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0 
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
* 2021-08-25
	0.3.1: pytorch version with a three-layer architecture, apps (financial tasks), drl_agents (drl algorithms), neo_finrl (gym env)
</div>
</details>

## Installation

### Installation (Recommend using cloud service - Google Colab or AWS EC2)

Clone the repository:
```shell
git clone https://github.com/AI4Finance-Foundation/FinRL.git
```

Install the unstable development version of FinRL using **pip**:
```shell
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
```


### Prerequisites
For [OpenAI Baselines](https://github.com/openai/baselines), you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows:

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#prerequisites).
    
#### Create and Activate Python Virtual-Environment (optional but highly recommended)
cd into this repository:
```bash
cd FinRL
```
Under the root directory /FinRL, create a Python virtual-environment:
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages. 

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv **venv** under the root directory /FinRL
```bash
virtualenv -p python3 venv
```
To activate a virtualenv:
```
source venv/bin/activate
```
To activate a virtualenv on windows:
```
venv\Scripts\activate
```
### Dependencies

The script has been tested running under **Python >= 3.6.0**, with the following packages installed:

```shell
pip install -r requirements.txt
```

#### Stable-Baselines3 using Pytorch

#### About [Stable-Baselines 3](https://github.com/DLR-RM/stable-baselines3)

Stable-Baselines3 is a set of improved implementations of reinforcement learning algorithms in PyTorch. It is the next major version of Stable Baselines. If you have questions regarding Stable-baselines package, please refer to [Stable-baselines3 installation guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html). Install the Stable Baselines package using pip:

```
pip install stable-baselines3[extra]
```
A migration guide from SB2 to SB3 can be found in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html).

#### Stable-Baselines using Tensorflow 2.0
Still [Under Development](https://github.com/Stable-Baselines-Team/stable-baselines-tf2)

## Docker Installation

### Option 1: Run a container via scripts

```bash
# grant access to execute scripting (read it, it's harmless)
$ sudo chmod -R 777 docker/bin

# build the container!
$ ./docker/bin/build_container.sh

# start notebook on port 8887!
$ ./docker/bin/start_notebook.sh

# proceed to party!
```

### Option 2: Run a container manually

Build the container:
```bash
$ docker build -f docker/Dockerfile -t finrl docker/
```

Start the container:
```bash
$ docker run -it --rm -v ${PWD}:/home -p 8888:8888 finrl
```
Note: The default container run starts jupyter lab in the root directory, allowing you to run scripts, notebooks, etc.

### Run 
```shell
python main.py --mode=train
```
### Backtesting

Use Quantopian's [pyfolio package](https://github.com/quantopian/pyfolio) to do the backtesting.


### Data
The stock data we use is pulled from Yahoo Finance API (the following time windows show the data split in the paper; users can customized to new time windows).
<div align="center">
<img src=figs/example_data.PNG width="600">
</div>

## Contributions

- FinRL is an open source library specifically designed and implemented for quant finance. Trading environments incorporating market frictions are used and provided.
- Trading tasks accompanied by hands-on tutorials with built-in DRL agents are available in a beginner-friendly and reproducible fashion using Jupyter notebook. Customization of trading time steps is feasible.
- FinRL has good scalability, with a broad range of fine-tuned state-of-the-art DRL algorithms. Adjusting the implementations to the rapid changing stock market is well supported.
- Typical use cases are selected and used to establish a benchmark for the quantitative finance community. Standard backtesting and evaluation metrics are also provided for easy and effective performance evaluation. 

## Citing FinRL
```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    journal = {Deep RL Workshop, NeurIPS 2020},
    title   = {FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance},
    url     = {https://arxiv.org/pdf/2011.09607.pdf},
    year    = {2020}
}
```

## Call for Contributions

Will maintain FinRL with the "AI4Finance" community and welcome your contributions!

Please check the [Contributing Guidances](https://github.com/AI4Finance-LLC/FinRL/blob/master/contributing.md).

### Contributors

Thanks to all the people who contribute. 
<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>

## Support various markets
   Support more markets for users to test their stategies.
## SOTA DRL algorithms 
   Maintain a pool of SOTA DRL algorithms.
## Benchmarks for typical trading tasks
   To help quants have better evaluations, we will maintain benchmarks for many trading tasks, upon which you can improve for your own tasks.
## Support live trading
   Supporting live trading can close the simulation-reality gap, which allows quants to switch to the real market when they are confident with the results.
   
## LICENSE

MIT License
