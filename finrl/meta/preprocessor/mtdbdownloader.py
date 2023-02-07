"""Contains methods and classes to collect data from
Download Forex Data from DB
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime
from.preprocessors import convert_to_datetime

from ..data_processors.fx_history_data.constant import Interval, Exchange
from ..data_processors.fx_history_data.database import BaseDatabase
from ..data_processors.fx_history_data.orm_pymongo import Database
from ..data_processors.fx_history_data.utility import extract_atp_symbol

class MtDbDownloader:
    """Provides methods for retrieving minute Forex data from
    Database

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from DB

    """

    def __init__(self, start_date: datetime, end_date: datetime, ticker_list: list):

        self.start_date = convert_to_datetime(start_date)
        self.end_date = convert_to_datetime(end_date)
        self.ticker_list = ticker_list
        self.interval = Interval.MINUTE

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from DB
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        database: BaseDatabase = Database()

        for tic in self.ticker_list:
            tic, exchange = extract_atp_symbol(tic)
            temp_df = database.load_bar_data(
                symbol=tic,
                exchange=exchange,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval
            )
            temp_df = pd.DataFrame(temp_df)
            data_df = pd.concat([data_df, temp_df])
        # reset the index, we want to use numbers as index instead of dates
        # data_df.reset_index(inplace=True)
        try:
            # convert the column names to standardized names
            data_df= data_df[[
                "time",
                "open",
                "high",
                "low",
                "close",
                "symbol"]
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        # data_df["day"] = data_df["time"].dt.dayofweek
        # convert date to standard string format, easy to filter
        # data_df["time"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        # data_df = data_df.dropna()
        # data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["time", "symbol"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.symbol.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["symbol", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.symbol.value_counts() >= mean_df)
        names = df.symbol.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.symbol.isin(select_stocks_list)]
        return df
