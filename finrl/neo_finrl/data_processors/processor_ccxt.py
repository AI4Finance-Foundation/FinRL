import calendar
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf


class CCXTEngineer:
    def __init__(self):
        self.binance = ccxt.binance()

    def data_fetch(self, start, end, pair_list=["BTC/USDT"], period="1m"):
        def min_ohlcv(dt, pair, limit):
            since = calendar.timegm(dt.utctimetuple()) * 1000
            ohlcv = self.binance.fetch_ohlcv(
                symbol=pair, timeframe="1m", since=since, limit=limit
            )
            return ohlcv

        def ohlcv(dt, pair, period="1d"):
            ohlcv = []
            limit = 1000
            if period == "1m":
                limit = 720
            elif period == "1d":
                limit = 1
            elif period == "1h":
                limit = 24
            elif period == "5m":
                limit = 288
            for i in dt:
                start_dt = i
                since = calendar.timegm(start_dt.utctimetuple()) * 1000
                if period == "1m":
                    ohlcv.extend(min_ohlcv(start_dt, pair, limit))
                else:
                    ohlcv.extend(
                        self.binance.fetch_ohlcv(
                            symbol=pair, timeframe=period, since=since, limit=limit
                        )
                    )
            df = pd.DataFrame(
                ohlcv, columns=["time", "open", "high", "low", "close", "volume"]
            )
            df["time"] = [
                datetime.fromtimestamp(float(time) / 1000) for time in df["time"]
            ]
            df["open"] = df["open"].astype(np.float64)
            df["high"] = df["high"].astype(np.float64)
            df["low"] = df["low"].astype(np.float64)
            df["close"] = df["close"].astype(np.float64)
            df["volume"] = df["volume"].astype(np.float64)
            return df

        crypto_column = pd.MultiIndex.from_product(
            [pair_list, ["open", "high", "low", "close", "volume"]]
        )
        first_time = True
        for pair in pair_list:
            start_dt = datetime.strptime(start, "%Y%m%d %H:%M:%S")
            end_dt = datetime.strptime(end, "%Y%m%d %H:%M:%S")
            start_timestamp = calendar.timegm(start_dt.utctimetuple())
            end_timestamp = calendar.timegm(end_dt.utctimetuple())
            if period == "1m":
                date_list = [
                    datetime.utcfromtimestamp(float(time))
                    for time in range(start_timestamp, end_timestamp, 60 * 720)
                ]
            else:
                date_list = [
                    datetime.utcfromtimestamp(float(time))
                    for time in range(start_timestamp, end_timestamp, 60 * 1440)
                ]
            df = ohlcv(date_list, pair, period)
            if first_time:
                dataset = pd.DataFrame(columns=crypto_column, index=df["time"].values)
                first_time = False
            temp_col = pd.MultiIndex.from_product(
                [[pair], ["open", "high", "low", "close", "volume"]]
            )
            dataset[temp_col] = df[["open", "high", "low", "close", "volume"]].values
        print("Actual end time: " + str(df["time"].values[-1]))
        return dataset

    def add_technical_indicators(
        self,
        df,
        pair_list,
        tech_indicator_list=[
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    ):
        df = df.dropna()
        df = df.copy()
        column_list = [
            pair_list,
            ["open", "high", "low", "close", "volume"] + (tech_indicator_list),
        ]
        column = pd.MultiIndex.from_product(column_list)
        index_list = df.index
        dataset = pd.DataFrame(columns=column, index=index_list)
        for pair in pair_list:
            pair_column = pd.MultiIndex.from_product(
                [[pair], ["open", "high", "low", "close", "volume"]]
            )
            dataset[pair_column] = df[pair]
            temp_df = df[pair].reset_index().sort_values(by=["index"])
            temp_df = temp_df.rename(columns={"index": "date"})
            crypto_df = Sdf.retype(temp_df.copy())
            for indicator in tech_indicator_list:
                temp_indicator = crypto_df[indicator].values.tolist()
                dataset[(pair, indicator)] = temp_indicator
        print("Succesfully add technical indicators")
        return dataset

    def df_to_ary(
        self,
        df,
        pair_list,
        tech_indicator_list=[
            "macd",
            "boll_ub",
            "boll_lb",
            "rsi_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
        ],
    ):
        df = df.dropna()
        date_ary = df.index.values
        price_array = df[pd.MultiIndex.from_product([pair_list, ["close"]])].values
        tech_array = df[
            pd.MultiIndex.from_product([pair_list, tech_indicator_list])
        ].values
        return price_array, tech_array, date_ary
