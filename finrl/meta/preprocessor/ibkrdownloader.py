"""Contains methods and classes to collect data from
interactive broker API
"""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
from ib_insync import *


class ibkrdownloader:
    """Provides methods for retrieving daily stock data from
    IBKR API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)
        interval: str
            ime period of one bar. Must be one of:
                '1 secs', '5 secs', '10 secs' 15 secs', '30 secs',
                '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
                '20 mins', '30 mins',
                '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
                '1 day', '1 week', '1 month'.

    Methods
    -------
    fetch_data()
        Fetches data from IBKR API
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker_list: list,
        interval="1 day",
        host="127.0.0.1",
        port=7497,
        clientId=999,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.interval = interval
        self.host = host
        self.port = port
        self.clientId = clientId

    def createContract(self, ticker):
        contract = Stock(ticker, "SMART", "USD")
        return contract

    def connect2ibkr(self):
        self.ib = IB()
        self.ib.connect(self.host, self.port, clientId=self.clientId)

    def calculate_duration(self, start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        duration_days = (end_date - start_date).days

        return duration_days

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from IBKR API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        self.connect2ibkr()
        data_df = pd.DataFrame()
        duration = self.calculate_duration(self.start_date, self.end_date)
        if duration > 365:
            duration = f"{math.ceil(duration / 365)} Y"
        else:
            duration = f"{duration} D"

        end_date_time = datetime.strptime(self.end_date, "%Y-%m-%d")
        for tic in self.ticker_list:
            contract = self.createContract(tic)
            self.ib.qualifyContracts(contract)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date_time,
                durationStr=duration,
                barSizeSetting=self.interval,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )
            temp_df = util.df(bars)
            print(f"************* {tic} *************")
            temp_df["tic"] = [tic] * len(temp_df)
            data_df = pd.concat([data_df, temp_df], ignore_index=True)
        data_df = data_df[["date", "open", "high", "low", "close", "volume", "tic"]]
        filter_df = data_df[data_df["date"] > pd.Timestamp(self.start_date).date()]
        filter_df = filter_df.sort_values(by=["date", "tic"])
        filter_df.index = filter_df["date"].factorize()[0]

        self.disconnect()
        return filter_df

    def disconnect(self):
        self.ib.disconnect()

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


if __name__ == "__main__":
    intr = ibkrdownloader(
        "2023-01-01", "2023-04-12", ["AAPL", "MSFT", "CSCO", "WMT", "TSLA"]
    )
    try:
        df = intr.fetch_data()
        df.to_csv("data.csv", index=False)
    except:
        intr.disconnect()

    intr.disconnect()
