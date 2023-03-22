We show a workflow of applying RL in algorithmic trading, which is a reproduction and improvement of the process in the [NeurIPS 2018 paper](https://arxiv.org/abs/1811.07522).

# Usage

## Step I. Data

First, run the notebook: *Stock_NeurIPS2018_1_Data.ipynb*.

It downloads and preprocesses stocks' OHLCV data.

It generates two csv files: *train.csv*, *trade.csv*. You can check the provided two sample files.

## Step II. Train a Trading Agent

Second, run the notebook: *Stock_NeurIPS2018_2_Train.ipynb*.

It shows how to process the data into an OpenAI gym-style envrionment, and then train a DRL agent.

It will generate a trained RL model .zip file. Here, we also provided a training A2C model in .zip file.

## Step III. Backtest

Finally, run the notebook: *Stock_NeurIPS2018_3_Backtest.ipynb*.

It backtests the trained agent and compares with two baselines: Mean-Variance Optimization and the market DJIA index, respectively.

At the end, it will plot a figure of the portfolio value during the backtest process.
