# FinRL: A Deep Reinforcement Learning Library for Quantitative Finance

[![Downloads](https://pepy.tech/badge/finrl)](https://pepy.tech/project/finrl)


FinRL is the open source library for practitioners. To efficiently automate trading, **AI4Finance** provides this educational resource and makes it easier to learn about deep reinforcement learning (DRL) in quantitative finance. 

Reinforcement learning (RL) trains an agent how to solve tasks by trial and error, while DRL combines RL with deep learning. DRL balances exploration (of uncharted territory) and exploitation (of current knowledge), and has been recognized as an advantageous approach for automated trading. DRL framework is powerful in solving dynamic decision making problems by learning through interaction with an unknown environment, thus provides two major advantages: portfolio scala-bility and market model independence. In quantitative finance, automated trading is essentially making dynamic decisions, namely **to decide where to trade, at what price, and what quantity**, over a highlystochastic and complex stock market. Taking many complex financialfactors into account, DRL trading agents build a multi-factor model and provide algorithmic trading strategies, which are difficult for human traders
 
FinRL provides a framework that supports various markets, SOTA DRL algorithms, benchmarks of many quant finance tasks, live trading, etc.  

To contribute?  Please check the call for contributions at the end of this page.

Feel free to leave us feedback: report bugs using Github issues or discuss FinRL development in the slack channel.



<br/>

<a href="https://join.slack.com/t/ai4financeworkspace/shared_invite/zt-jyaottie-hHqU6TdvuhMHHAMXaLw_~w" target="\_blank">
	<div align="center">
		<img src=figs/join_slack.png width="40%"/>
	</div>
</a>

<br/>


## Prior Arts:

We published the following papers and now arrive at this project:

4). [FinRL](https://arxiv.org/abs/2011.09607): A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, Deep RL Workshop, NeurIPS 2020.

3). Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy, [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) and [codes](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020), ACM International Conference on AI in Finance, ICAIF 2020.

2). Multi-agent Reinforcement Learning for Liquidation Strategy Analysis, [paper](https://arxiv.org/abs/1906.11046) and [codes](https://github.com/WenhangBao/Multi-Agent-RL-for-Liquidation). Workshop on Applications and Infrastructure for Multi-Agent Learning, ICML 2019.

1). Practical Deep Reinforcement Learning Approach for Stock Trading, [paper](https://arxiv.org/abs/1811.07522) and [codes](https://github.com/AI4Finance-LLC/Deep-Reinforcement-Learning-for-Stock-Trading-DDPG-Algorithm-NIPS-2018), Workshop on Challenges and Opportunities for AI in Financial Services, NeurIPS 2018.

## Status
<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>
	
* 2020-12-14
  	Upgraded to **Pytorch** with stable-baselines3; Remove tensorflow 1.0 at this moment, under development to support tensorflow 2.0 
* 2020-11-27
  	0.1: Beta version with tensorflow 1.5
</div>
</details>


## Overview
As deep reinforcement learning (DRL) has been recognized as an effective approach in quantitative finance, getting hands-on experiences is attractive to beginners. However, to train a practical DRL trading agent that decides where to trade, at what price, and what quantity involves error-prone and arduous development and debugging. 

We introduce a DRL library FinRL that facilitates beginners to expose themselves to quantitative finance and to develop their own stock trading strategies. Along with easily-reproducible tutorials, FinRL library allows users to streamline their own developments and to compare with existing schemes easily.  Within FinRL, virtual environments are configured with stock market datasets, trading agents are trained with neural networks, and extensive backtesting is analyzed via trading performance. Moreover, it incorporates important trading constraints such as transaction cost, market liquidity and the investor’s degree of risk-aversion. 

FinRL is featured with completeness, hands-on tutorial and reproducibility that favors beginners: (i) at multiple levels of time granularity, FinRL simulates trading environments across various stock markets, including NASDAQ-100, DJIA, S&P 500, HSI, SSE 50, and CSI 300; (ii) organized in a layered architecture with modular structure, FinRL provides fine-tuned state-of-the-art DRL algorithms (DQN, DDPG, PPO, SAC, A2C, TD3, etc.), commonly-used reward functions and standard evaluation baselines to alleviate the debugging work-loads and promote the reproducibility, and (iii) being highly extendable, FinRL reserves a complete set of user-import interfaces. 

Furthermore, we incorporated three application demonstrations, namely single stock trading, multiple stock trading, and portfolio allocation. 



## Guiding Principles
- **Completeness.** Our library shall cover components of the DRL framework completely, which is a fundamental requirement;
- **Hands-on tutorials.** We aim for a library that is friendly to beginners. Tutorials with detailed walk-through will help users to explore the functionalities of our library;
- **Reproducibility.** Our library shall guarantee reproducibility to ensure the transparency and also provide users with confidence in what they have done.


## Architecture of the FinRL Library
- **Three-layer architecture:** The three layers of FinRL library are stock market environment, DRL trading agent, and stock trading applications. The agent layer interacts with the environment layer in an exploration-exploitation manner, whether to repeat prior working-well decisions or to make new actions hoping to get greater rewards. The lower layer provides APIs for the upper layer, making the lower layer transparent to the upper layer.
- **Modularity:** Each layer includes several modules and each module defines a separate function. One can select certain modules from any layer to implement his/her stock trading task. Furthermore, updating existing modules is possible.
- **Simplicity, Applicability and Extendibility:** Specifically designed for automated stock trading, FinRL presents DRL algorithms as modules. In this way, FinRL is made accessible yet not demanding. FinRL provides three trading tasks as use cases that can be easily reproduced. Each layer includes reserved interfaces that allow users to develop new modules.
- **Better Market Environment Modeling:** We build a trading simulator that replicates live stock market and provides backtesting support that incorporates important market frictions such as transaction cost, market liquidity and the investor’s degree of risk-aversion. All of those are crucial among key determinants of net returns.

<img src=figs/FinRL-Architecture.png width="800">

## Implemented Algorithms
<img src=figs/alg_compare.PNG width="800">

## Medium Blogs
[FinRL for Quantitative Finance: Tutorial for Single Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-single-stock-trading-37d6d7c30aac)

[FinRL for Quantitative Finance: Tutorial for Multiple Stock Trading](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-multiple-stock-trading-7b00763b7530)

[FinRL for Quantitative Finance: Tutorial for Portfolio Allocation](https://towardsdatascience.com/finrl-for-quantitative-finance-tutorial-for-portfolio-allocation-9b417660c7cd)

## Related Reports
【量化投资与机器学习】[基于深度强化学习的股票交易策略框架（代码+文档)](https://www.mdeditor.tw/pl/p5Gg)

## Installation:

Clone this repository
```shell
git clone https://github.com/AI4Finance-LLC/FinRL-Library.git
```

Install the unstable development version of FinRL:
```shell
pip install git+https://github.com/AI4Finance-LLC/FinRL-Library.git
```

## Docker Installation

Build the container:
```bash
$ docker build -f docker/Dockerfile -t finrl docker/
```

Start the container
Note: The default container run starts jupyter lab in the root directory, allowing you to run scripts, notebooks, etc.
```bash
$ docker run -it --rm -v ${PWD}:/home -p 8888:8888 finrl
```

### Prerequisites
For [OpenAI Baselines](https://github.com/openai/baselines), you'll need system packages CMake, OpenMPI and zlib. Those can be installed as follows

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
    
### Create and Activate Python Virtual-Environment (Optional but highly recommended)
cd into this repository
```bash
cd FinRL-Library
```
Under folder /FinRL-Library, create a Python virtual-environment
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

The script has been tested running under **Python >= 3.6.0**, with the folowing packages installed:

```shell
pip install -r requirements.txt
```

## Stable-Baselines3 using Pytorch

### About [Stable-Baselines 3](https://github.com/DLR-RM/stable-baselines3)

Stable-Baselines3 is a set of improved implementations of reinforcement learning algorithms in PyTorch. It is the next major version of Stable Baselines. If you have questions regarding Stable-baselines package, please refer to [Stable-baselines3 installation guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html). Install the Stable Baselines package using pip:

```
pip install stable-baselines3[extra]

```
A migration guide from SB2 to SB3 can be found in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html).

## Stable-Baselines using Tensorflow 2.0
Still [Under Development](https://github.com/Stable-Baselines-Team/stable-baselines-tf2)


## Run 
```shell
python main.py --mode=train
```
## Backtesting

Use Quantopian's [pyfolio package](https://github.com/quantopian/pyfolio) to do the backtesting.


## Status

<details><summary><b>Version History</b> <i>[click to expand]</i></summary>
<div>
* 0.0.1
    Simple version
</div>
</details>

## Data
The stock data we use is pulled from Yahoo Finance API

(The following time line is used in the paper; users can update to new time windows.)

<img src=figs/example_data.PNG width="600">

## Performance
<img src=figs/performance.PNG width="600">

## Contributions
- FinRL is an open source library specifically designed and implemented for quantitative finance. Trading environments incorporating market frictions are used and provided.
- Trading tasks accompanied by hands-on tutorials with built-in DRL agents are available in a beginner-friendly and reproducible fashion using Jupyter notebook. Customization of trading time steps is feasible.
- FinRL has good scalability, with a broad range of fine-tuned state-of-the-art DRL algorithms. Adjusting the implementations to the rapid changing stock market is well supported.
- Typical use cases are selected and used to establish a benchmark for the quantitative finance community. Standard backtesting and evaluation metrics are also provided for easy and effective performance evaluation. 
## Citing FinRL
```
@article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    journal = {Deep RL Workshop, NeurIPS 2020},
    title   = {{FinRL: A Deep Reinforcement Learning Library forAutomated Stock Trading in Quantitative Finance}},
    url     = {},
    year    = {2020}
}
```

# Call for Contributions

We will maintain the open source FinRL library for the "AI + finance" community and welcome you to join as contributors!

## Support various markets
   We would like to support more asset markets, so that the users can test their stategies.
## SOTA DRL algorithms 
   We will continue to maintian a pool of DRL algorithms that can be treated as SOTA implementations.
## Benchmarks for typical trading tasks
   To help quants have better evaluations, here we maintain benchmarks for many trading tasks, upon which you can improve for your own tasks.
## Support live trading
   Supporting live trading can close the simulation-reality gap, it will enable quant to switch to the real market when they are confident with their strategies.

