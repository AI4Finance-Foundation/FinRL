# FinRL: A Deep Reinforcement Learning Library for Quantitative Finance [![twitter][1.1]][1] [![facebook][1.2]][2] [![google+][1.3]][3] [![linkedin][1.4]][4]
[1.1]: http://www.tensorlet.com/wp-content/uploads/2021/01/button_twitter_22x22.png
[1.2]: http://www.tensorlet.com/wp-content/uploads/2021/01/facebook-button_22x22.png
[1.3]: http://www.tensorlet.com/wp-content/uploads/2021/01/button_google_22.xx_.png
[1.4]: http://www.tensorlet.com/wp-content/uploads/2021/01/button_linkedin_22x22.png

[1]: https://twitter.com/intent/tweet?text=FinRL-A-Deep-Reinforcement-Learning-Library-for-Quantitative-Finance%20&url=hhttps://github.com/AI4Finance-LLC/FinRL-Library&hashtags=DRL&hashtags=AI
[2]: https://www.facebook.com/sharer.php?u=http%3A%2F%2Fgithub.com%2FAI4Finance-LLC%2FFinRL-Library
[3]: https://plus.google.com/share?url=https://github.com/AI4Finance-LLC/FinRL-Library
[4]: https://www.linkedin.com/sharing/share-offsite/?url=http%3A%2F%2Fgithub.com%2FAI4Finance-LLC%2FFinRL-Library

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)
[![Downloads](https://pepy.tech/badge/finrl/week)](https://pepy.tech/project/finrl)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl.svg)](https://pypi.org/project/finrl/)

FinRL is an open source library that provides practitioners a unified framework for pipeline strategy development. **In reinforcement learning (or Deep RL), an agent learns by continuously interacting with an environment, in a trial-and-error manner, making sequential decisions under uncertainty and achieving a balance between exploration and exploitation.** The open source community **AI4Finance** (to efficiently automate trading) provides educational resources about deep reinforcement learning (DRL) in quantitative finance. 

**To contribute?**  Please check the call for contributions at the end of this page.

Feel free to report bugs using Github issues, join our mailing list: [AI4Finance](https://groups.google.com/u/1/g/ai4finance), and discuss FinRL in the slack channel:

<br/>

<a href="https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-myfwpei8-omxzFpznO~hMvg_FMnYRXg" target="\_blank">
	<div align="center">
		<img src=figs/join_slack.png width="40%"/>
	</div>
</a>

<br/>

Roadmaps of FinRL:

**FinRL 1.0**: entry-level toturials for beginners, with a demonstrative and educational purpose.

**FinRL 2.0**: intermediate-level framework for full-stack developers and professionals. As a Starter, check out [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)  

FinRL provides a unified machine learning framework for various markets, SOTA DRL algorithms, benchmark finance tasks (portfolio allocation, cryptocurrency trading, high-frequency trading), live trading, etc. 

## Table of Contents

- [Prior Arts](#Prior-Arts)
- [News](#News)
- [Overview](#Overview)
- [Status](#Status)
- [Installation](#Installation)
	- [Docker Installation](#Docker-Installation)
	- [Prerequisites](#Prerequisites)
	- [Dependencies](#Dependencies)
- [Contributions](#Contributions)
	- [Citing FinRL](#Citing-FinRL)
	- [Call for Contributions](#Call-for-Contributions)
- [LICENSE](#LICENSE)

# Prior Arts:

We published [papers in FinTech](http://www.tensorlet.com/projects/ai-in-finance/) and now arrive at this project:

4). [FinRL](https://arxiv.org/abs/2011.09607): A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, Deep RL Workshop, NeurIPS 2020.

3). Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) and [codes](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020), ACM International Conference on AI in Finance, ICAIF 2020.

2). Multi-agent Reinforcement Learning for Liquidation Strategy Analysis, [paper](https://arxiv.org/abs/1906.11046) and [codes](https://github.com/WenhangBao/Multi-Agent-RL-for-Liquidation). Workshop on Applications and Infrastructure for Multi-Agent Learning, ICML 2019.

1). Practical Deep Reinforcement Learning Approach for Stock Trading, [paper](https://arxiv.org/abs/1811.07522) and [codes](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018), Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.

# News
[Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Single Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac)

[Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Multiple Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530)

[Towardsdatascience] [FinRL for Quantitative Finance: Tutorial for Portfolio Allocation](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd)

[Towardsdatascience] [Deep Reinforcement Learning for Automated Stock Trading](https://towardsdatascience.com/deep-reinforcement-learning-for-automated-stock-trading-f1dad0126a02)

[Analyticsindiamag.com] [How To Automate The Stock Market Using FinRL (Deep Reinforcement Learning Library)?](https://analyticsindiamag.com/stock-market-prediction-using-finrl/)

[量化投资与机器学习] [基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)

[Neurohive] [FinRL: глубокое обучение с подкреплением для трейдинга](https://neurohive.io/ru/gotovye-prilozhenija/finrl-glubokoe-obuchenie-s-podkrepleniem-dlya-trejdinga/)

[ICHI.PRO] [양적 금융을위한 FinRL: 단일 주식 거래를위한 튜토리얼](https://ichi.pro/ko/yangjeog-geum-yung-eul-wihan-finrl-dan-il-jusig-geolaeleul-wihan-tyutolieol-61395882412716)


# Overview

A YouTube video about FinRL library.  [YouTube] [AI4Finance Channel](https://www.youtube.com/channel/UCrVri6k3KPBa3NhapVV4K5g) for quant finance.

[<img src="http://img.youtube.com/vi/ZSGJjtM-5jA/0.jpg" width="70%">](http://www.youtube.com/watch?v=ZSGJjtM-5jA)


<img src=figs/Poster_FinRL.jpg width="800">

# DRL Algorithms 

We implemented Deep Q Learning (DQN), Double DQN, DDPG, A2C, SAC, PPO, TD3, GAE, MADDPG, MuZero, etc. using PyTorch and OpenAI Gym. 

# Status
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>
	
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0 
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>

# Installation:

## Docker Installation

### Option 1: Use the bin

```bash
# grant access to execute scripting (read it, it's harmless)
$ sudo chmod -R 777 docker/bin

# build the container!
$ ./docker/bin/build_container.sh

# start notebook on port 8887!
$ ./docker/bin/start_notebook.sh

# proceed to party!
```

### Option 2: Do it manually

Build the container:
```bash
$ docker build -f docker/Dockerfile -t finrl docker/
```

Start the container:
```bash
$ docker run -it --rm -v ${PWD}:/home -p 8888:8888 finrl
```
Note: The default container run starts jupyter lab in the root directory, allowing you to run scripts, notebooks, etc.

## Bare-metal installation (More difficult)

Clone this repository:
```shell
git clone https://github.com/AI4Finance-LLC/FinRL-Library.git
```

Install the unstable development version of FinRL:
```shell
pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
```


## Prerequisites
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
    
#### Create and Activate Python Virtual-Environment (Optional but highly recommended)
cd into this repository:
```bash
cd FinRL-Library
```
Under folder /FinRL-Library, create a Python virtual-environment:
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages. 

**Virtualenvs can also avoid packages conflicts.**

Create a virtualenv **venv** under folder /FinRL-Library
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
## Dependencies

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


### Run 
```shell
python main.py --mode=train
```
### Backtesting

Use Quantopian's [pyfolio package](https://github.com/quantopian/pyfolio) to do the backtesting.


### Data
The stock data we use is pulled from Yahoo Finance API.

(The following time line is used in the paper; users can update to new time windows.)

<img src=figs/example_data.PNG width="600">


# Contributions

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

### Contributors

Thanks to all the people who contribute. 
<a href="https://github.com/AI4Finance-LLC/FinRL-Library/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Finance-LLC/FinRL-Library" />
</a>

## Support various markets
   Would like to support more asset markets, so that the users can test their stategies.
## SOTA DRL algorithms 
   Will continue to maintian a pool of DRL algorithms that can be treated as SOTA implementations.
## Benchmarks for typical trading tasks
   To help quants have better evaluations, will maintain benchmarks for many trading tasks, upon which you can improve for your own tasks.
## Support live trading
   Supporting live trading can close the simulation-reality gap, it will enable quants to switch to the real market when they are confident with their strategies.
   
# LICENSE

MIT
