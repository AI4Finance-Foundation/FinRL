from __future__ import annotations

import datetime
import itertools

import numpy as np
import pandas as pd
import yfinance as yf

from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


TRAIN_START_DATE = "2000-01-01"
TRAIN_END_DATE = "2024-01-01"
TRADE_START_DATE = "2023-01-01"
TRADE_END_DATE = "2024-01-01"


df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER
    + config_tickers.NAS_100_TICKER
    + config_tickers.SP_500_TICKER,
).fetch_data()


df_raw.head()


fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
)

processed = fe.preprocess_data(df_raw)


list_ticker = processed["tic"].unique().tolist()
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])

processed_full = processed_full.fillna(0)


processed_full.head()

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(len(train))
print(len(trade))

train.to_csv("train_data.csv")
trade.to_csv("trade_data.csv")
