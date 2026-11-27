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

**using FXMacroData for daily FX spot data and macro events**

FXMacroData provides daily FX spot rates for currency pairs such as EUR/USD.
The downloader maps each daily rate to FinRL's OHLCV shape, with open, high,
low, and close set to the FX spot rate and volume set to 0. FXMacroData also
provides official macro announcements, release-calendar rows, and forecast
groups that can be joined into a trading state.

..  code-block:: python
    from finrl.meta.preprocessor.fxmacrodatadownloader import FXMacroDataDownloader
    from finrl.meta.data_processors.processor_fxmacrodata import FXMacroDataProcessor

    eurusd_df_finrl = FXMacroDataDownloader(start_date = '2020-01-01',
                                            end_date = '2020-01-31',
                                            ticker_list = ['EURUSD']).fetch_data()

    processor = FXMacroDataProcessor()
    macro_df = processor.download_macro_data(currency = 'usd',
                                             indicator_list = ['inflation',
                                                               'policy_rate'],
                                             start_date = '2020-01-01',
                                             end_date = '2020-01-31')
    eurusd_with_macro = processor.add_macro_features(eurusd_df_finrl, macro_df,
                                                     date_column = 'date')

**using Aperiodic for crypto and RWA perpetual futures**

Aperiodic provides OHLCV bars (1m to 1d, UTC) for perpetual futures on
Binance, OKX and Hyperliquid, including Hyperliquid HIP-3 markets for
tokenized stocks, indices and commodities, and contracts that have since been
delisted, so a universe can be built without survivorship bias. Install the
SDK with ``pip install "aperiodic[pandas]"`` (Python 3.11+).

A ticker names the perpetual as ``BASE-QUOTE`` on the venue set by
``exchange``:

=================  ===============  ==========================================
Ticker             ``exchange``     Contract
=================  ===============  ==========================================
``BTC-USDT``       ``binance``      BTC perpetual settled in USDT
``BTC-USDT``       ``okx``          the same pair on OKX
``BTC-USDC``       ``hyperliquid``  Hyperliquid's BTC perpetual
``XYZ:TSLA-USDC``  ``hyperliquid``  HIP-3 Tesla perpetual listed by ``XYZ``
``BTC-USD:BTC``    ``okx``          inverse perpetual settled in BTC
=================  ===============  ==========================================

Append ``@binance``, ``@okx`` or ``@hyperliquid`` to a ticker to mix venues
in one list; the ``tic`` column keeps the ticker as written. Full Aperiodic
symbols such as ``perpetual-BTC-USDT:USDT`` are accepted as well, and
``list_tickers()`` returns every ticker on a venue, delisted ones included.

..  code-block:: python

    from finrl.config import INDICATORS
    from finrl.meta.data_processor import DataProcessor

    dp = DataProcessor('aperiodic', exchange = 'binance',
                       derivative_features = ['funding_rate', 'open_interest'])
    df = dp.download_data(ticker_list = ['BTC-USDT', 'ETH-USDT',
                                         'XYZ:TSLA-USDC@hyperliquid'],
                          start_date = '2026-07-01',
                          end_date = '2026-09-30',
                          time_interval = '1h')
    df = dp.clean_data(df)
    df = dp.add_technical_indicator(df, INDICATORS + ['funding_rate',
                                                      'open_interest'])
    df = dp.add_turbulence(df)
    price_array, tech_array, turbulence_array = dp.df_to_array(df, if_vix = False)

``derivative_features`` takes ``funding_rate``, ``open_interest``,
``open_interest_notional`` and ``basis_bps``. They are joined on the bar's open
time and hold the last value observed during that bar. Listing them with the
indicators carries them into ``tech_array``. ``clean_data`` aligns all tickers
on one clock: a missing bar repeats the previous close with zero volume, so a
delisted contract keeps its last close and one listed after ``start_date``
starts at its first close. Perpetuals have no VIX, so use ``add_turbulence``
with ``if_vix = False`` for the risk signal.
``AperiodicDownloader`` in ``finrl.meta.preprocessor.aperiodicdownloader``
returns the same data with a ``date`` column, like ``YahooDownloader``.

Without a key, ``preview = True`` reads a small public sample (the exact
parameters are listed at https://aperiodic.io/catalog#preview). A free account
key, set as ``APERIODIC_API_KEY`` or passed as ``api_key`` (sign up at
https://aperiodic.io), covers a fixed sample of markets and dates; paid plans
cover every perpetual and year, see https://aperiodic.io/pricing. A request
outside your access raises an error with the API's message rather than
returning an empty frame.

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
