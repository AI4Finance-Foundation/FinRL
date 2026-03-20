"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.
"""

import itertools

import pandas as pd
import yfinance as yf

from finrl import config_tickers
from finrl.config import INDICATORS, TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
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

print("\n=== DOW 30 Tickers ===")
print(config_tickers.DOW_30_TICKER)

df_raw = YahooDownloader(
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER,
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
