"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.
"""

from __future__ import annotations

import itertools

import pandas as pd
import yfinance as yf

from finrl.config import INDICATORS
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.dow import DOW_30_HISTORICAL_UNIVERSE
from finrl.meta.preprocessor.dow import add_dow_membership
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# %% Part 1. Fetch data - Single ticker

# Using yfinance directly
aapl_df_yf = yf.download(tickers="aapl", start="2020-01-01", end="2020-01-31")
print("=== yfinance download ===")
print(aapl_df_yf.head())

# Using FinRL's YahooDownloader
aapl_df_finrl = YahooDownloader(
    start_date="2020-01-01",
    end_date="2020-01-31",
    ticker_list=["aapl"],
).fetch_data()
print("\n=== FinRL YahooDownloader ===")
print(aapl_df_finrl.head())

# %% Part 2. Fetch data - DOW 30 tickers

print("\n=== Historical Dow ticker universe ===")
print(DOW_30_HISTORICAL_UNIVERSE)

df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=list(DOW_30_HISTORICAL_UNIVERSE),
).fetch_data()
print("\n=== Raw data ===")
print(df_raw.head())

# %% Part 3. Preprocess data

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=INDICATORS,
    use_vix=True,
    use_turbulence=True,
    user_defined_feature=False,
    keep_incomplete_tickers=True,
)

processed = fe.preprocess_data(df_raw)

list_ticker = list(DOW_30_HISTORICAL_UNIVERSE)
list_date = list(
    pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
)
combination = list(itertools.product(list_date, list_ticker))

processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
    processed, on=["date", "tic"], how="left"
)
processed_full = processed_full[processed_full["date"].isin(processed["date"])]
processed_full = processed_full.sort_values(["date", "tic"])
processed_full = add_dow_membership(processed_full)
active_rows = processed_full["is_member"]
if processed_full.loc[active_rows, "close"].isna().any():
    missing = processed_full.loc[
        active_rows & processed_full["close"].isna(), ["date", "tic"]
    ]
    raise ValueError(
        "Yahoo Finance data is missing for active Dow constituents; "
        f"inspect these rows before training:\n{missing.head(10)}"
    )
feature_columns = [
    column
    for column in processed_full.columns
    if column not in {"date", "tic", "is_member"}
]
processed_full[feature_columns] = processed_full.groupby("tic", sort=False)[
    feature_columns
].transform(lambda values: values.ffill().bfill())
processed_full[feature_columns] = processed_full[feature_columns].fillna(0)
processed_full["is_member"] = processed_full["is_member"].astype(int)

print("\n=== Processed data ===")
print(processed_full.head())

# %% Part 4. Split and save data

train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(f"\nTrain data length: {len(train)}")
print(f"Trade data length: {len(trade)}")

train.to_csv("train_data.csv")
trade.to_csv("trade_data.csv")
print("Data saved to train_data.csv and trade_data.csv")
