from __future__ import annotations

import gc

import pandas as pd

from finrl.config_tickers import TAI_0050_TICKER
from finrl.meta.data_processors.processor_sinopac import SinopacProcessor
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.shioajidownloader import SinopacDownloader


TRAIN_START_DATE = "2023-04-13"
TRAIN_END_DATE = "2024-04-13"
TRADE_START_DATE = "2024-04-13"
TRADE_END_DATE = "2024-07-31"


def process_ticker_data(ticker):
    print(f"Processing data for ticker: {ticker}")
    df_raw = SinopacDownloader(
        start_date=TRAIN_START_DATE, end_date=TRADE_END_DATE, ticker_list=[ticker]
    ).fetch_data()

    df_raw.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "amount": "Amount",
        },
        inplace=True,
    )

    processor = SinopacProcessor(
        API_KEY="3Tn2BbtCzbaU1KSy8yyqLa4m7LEJJyhkRCDrK2nknbcu",
        API_SECRET="Epakqh1Nt4inC3hsqowE2XjwQicPNzswkuLjtzj2WKpR",
    )

    cleaned_df = processor.clean_data(df_raw)
    df_with_indicators = processor.add_technical_indicator(cleaned_df)
    df_with_vix = processor.add_vix(df_with_indicators)
    df_with_turbulence = processor.add_turbulence(df_with_vix, time_period=252)

    # Save processed data for each ticker to a separate file
    df_with_turbulence.to_csv(f"data_{ticker}.csv")

    # Explicitly delete unused objects and collect garbage
    del df_raw, cleaned_df, df_with_indicators, df_with_vix, df_with_turbulence
    gc.collect()


df_final = pd.DataFrame()
for ticker in TAI_0050_TICKER:
    process_ticker_data(ticker)
    # Load processed data from file and concatenate
    df_ticker = pd.read_csv(f"data_{ticker}.csv")
    df_final = pd.concat([df_final, df_ticker], ignore_index=True)
    del df_ticker  # free up memory
    gc.collect()

train = data_split(df_final, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(df_final, TRADE_START_DATE, TRADE_END_DATE)
train.to_csv("train_data.csv")
trade.to_csv("trade_data.csv")
