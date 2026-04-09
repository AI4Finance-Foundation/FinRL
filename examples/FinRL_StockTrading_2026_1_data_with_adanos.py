"""
Stock Trading 2026 Part 1 (optional Adanos sentiment features).

This example keeps the classic FinRL Yahoo + technical-indicator pipeline and
optionally augments it with lagged market sentiment features when ADANOS_API_KEY
is available.
"""

from __future__ import annotations

import itertools
import os

import pandas as pd

from finrl.config import INDICATORS
from finrl.meta.preprocessor.adanos_sentiment import ADANOS_SENTIMENT_FEATURES
from finrl.meta.preprocessor.adanos_sentiment import add_adanos_market_sentiment
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

TRAIN_START_DATE = "2025-12-01"
TRAIN_END_DATE = "2026-02-15"
TRADE_START_DATE = "2026-02-15"
TRADE_END_DATE = "2026-03-20"
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]

api_key = os.getenv("ADANOS_API_KEY")

df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=TICKERS,
).fetch_data()

feature_engineer = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=False,
    user_defined_feature=False,
)

processed = feature_engineer.preprocess_data(df_raw)

list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"]).fillna(0)

processed_with_sentiment = add_adanos_market_sentiment(
    processed_full,
    api_key=api_key,
    days=90,
)

if api_key:
    feature_columns = INDICATORS + ADANOS_SENTIMENT_FEATURES
    print("Added Adanos sentiment features:")
    print(ADANOS_SENTIMENT_FEATURES)
else:
    feature_columns = INDICATORS
    print("ADANOS_API_KEY is not set. Proceeding without optional sentiment features.")

train = data_split(processed_with_sentiment, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_with_sentiment, TRADE_START_DATE, TRADE_END_DATE)

train.to_csv("train_data_with_adanos.csv")
trade.to_csv("trade_data_with_adanos.csv")

print("Feature columns to feed into the environment:")
print(feature_columns)
print("Saved train_data_with_adanos.csv and trade_data_with_adanos.csv")
