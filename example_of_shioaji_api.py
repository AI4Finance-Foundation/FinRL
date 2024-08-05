from __future__ import annotations

import datetime
import gc
import itertools

import numpy as np
import pandas as pd
import yfinance as yf

from finrl import config_tickers
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from processor_sinopac import SinopacProcessor
from shioajidownloader import SinopacDownloader

TAI_0050_TICKER = [
    "3008",  # Largan Precision Co., Ltd.
    "1303",  # Nan Ya Plastics Corporation
    "2412",  # Chunghwa Telecom Co., Ltd.
    "1301",  # Formosa Plastics Corporation
    "1216",  # Uni-President Enterprises Corporation
    "2881",  # Fubon Financial Holding Co., Ltd.
    "2882",  # Cathay Financial Holding Co., Ltd.
    "5871",  # China Development Financial Holding Corporation
    "2886",  # Mega Financial Holding Co., Ltd.
    "2891",  # CTBC Financial Holding Co., Ltd.
    "2884",  # E.SUN Financial Holding Co., Ltd.
    "5880",  # Yuanta Financial Holding Co., Ltd.
    "2883",  # China Development Financial Holding Corporation
    "2892",  # First Financial Holding Co., Ltd.
    "2880",  # SinoPac Financial Holdings Company Limited
    "2303",  # United Microelectronics Corporation
    "1326",  # Formosa Chemicals & Fibre Corporation
    "1101",  # Taiwan Cement Corp.
    "3006",  # Advanced Semiconductor Engineering, Inc.
    "3045",  # Compal Electronics Inc.
    "2912",  # President Chain Store Corporation
    "2327",  # ASE Technology Holding Co., Ltd.
    "1304",  # China Petrochemical Development Corporation
    "2379",  # Realtek Semiconductor Corp.
    "2801",  # Chang Hwa Commercial Bank, Ltd.
    "1402",  # Far Eastern New Century Corporation
    "2345",  # Acer Incorporated
    "2301",  # Lite-On Technology Corporation
    "2408",  # AU Optronics Corp.
    "2357",  # Asustek Computer Inc.
    "9910",  # Feng Hsin Iron & Steel Co., Ltd.
    "2395",  # Advantech Co., Ltd.
    "2353",  # Acer Incorporated
    "2354",  # Micro-Star International Co., Ltd.
    "3711",  # ASE Technology Holding Co., Ltd.
    "2890",  # Taishin Financial Holding Co., Ltd.
    "2377",  # Micro-Star International Co., Ltd.
    "4904",  # Far EasTone Telecommunications Co., Ltd.
    "2324",  # Compal Electronics, Inc.
    "2305",  # First International Computer, Inc.
    "1102",  # Asia Cement Corporation
    "9933",  # Mega Financial Holding Co., Ltd.
]

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
