:github_url: https://github.com/AI4Finance-Foundation/FinRL

=================
Section 1. Data
=================

Part 1. Install Packages
==================================
..  code-block:: python
    ## install required packages
    !pip install swig
    !pip install wrds
    !pip install pyportfolioopt
    ## install finrl library
    !pip install git+https://github.com/AI4Finance-Foundation/FinRL.git

..  code-block:: python
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

import itertools

Part 2. Fetch data
==================================

`yfinance <https://github.com/ranaroussi/yfinance>`_ is an open-source library that provides APIs fetching historical data form Yahoo Finance. In FinRL, we have a class called YahooDownloader that use yfinance to fetch data from Yahoo Finance.

**OHLCV**: Data downloaded are in the form of OHLCV, corresponding to **open, high, low, close, volume,** respectively. OHLCV is important because they contain most of numerical information of a stock in time series. From OHLCV, traders can get further judgement and prediction like the momentum, people's interest, market trends, etc.

Data for a single ticker
----------------------------------------

**using yfinance**
..  code-block:: python
    aapl_df_yf = yf.download(tickers = "aapl", start='2020-01-01', end='2020-01-31')

**using FinRL**

In FinRL's YahooDownloader, we modified the data frame to the form that convenient for further data processing process. We use adjusted close price instead of close price, and add a column representing the day of a week (0-4 corresponding to Monday-Friday).

..  code-block:: python
    aapl_df_finrl = YahooDownloader(start_date = '2020-01-01',
                                    end_date = '2020-01-31',
                                    ticker_list = ['aapl']).fetch_data()

Data for the chosen ticker
----------------------------------------
..  code-block:: python
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2020-07-01'
    TRADE_START_DATE = '2020-07-01'
    TRADE_END_DATE = '2021-10-29'
..  code-block:: python
    df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
                             end_date = TRADE_END_DATE,
                             ticker_list = config_tickers.DOW_30_TICKER).fetch_data()

Part 3. Preprocess Data
==================================

We need to check for missing data and do feature engineering to convert the data point into a state.

- **Adding technical indicators**. In practical trading, various information needs to be taken into account, such as historical prices, current holding shares, technical indicators, etc. Here, we demonstrate two trend-following technical indicators: MACD and RSI.
- **Adding turbulence index**. Risk-aversion reflects whether an investor prefers to protect the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the turbulence index that measures extreme fluctuation of asset price.

Hear let's take MACD as an example. Moving average convergence/divergence (MACD) is one of the most commonly used indicator showing bull and bear market. Its calculation is based on EMA (Exponential Moving Average indicator, measuring trend direction over a period of time.)
