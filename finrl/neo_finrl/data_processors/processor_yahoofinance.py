"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

import numpy as np
import pandas as pd
import pytz
import trading_calendars as tc
import yfinance as yf
from finrl.neo_finrl.data_processors.basic_processor import BasicProcessor

class YahooFinanceProcessor(BasicProcessor):
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)
    """

    def __init__(self):
        pass

    def download_data(
        self, ticker_list: list, start_date: str, end_date: str, time_interval: str
    ) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=start_date, end=end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def clean_data(self, data) -> pd.DataFrame:

        df = data.copy()
        df = df.rename(columns={"date": "time"})
        time_interval = self.time_interval
        # get ticker list
        tic_list = np.unique(df.tic.values)

        # get complete time index
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        if time_interval == "1D":
            times = trading_days
        elif time_interval == "1Min":
            times = []
            for day in trading_days:
                NY = "America/New_York"
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                for i in range(390):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )

        # fill NaN data
        new_df = pd.DataFrame()
        for tic in tic_list:
            print(("Clean data for ") + tic)
            # create empty DataFrame using complete time index
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "adjcp", "volume"], index=times
            )
            # get data for current ticker
            tic_df = df[df.tic == tic]
            # fill empty DataFrame using orginal data
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "adjcp", "volume"]
                ]

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_close = tmp_df.iloc[i]["close"]
                        first_valid_adjclose = tmp_df.iloc[i]["adjcp"]

                tmp_df.iloc[0] = [
                    first_valid_close,
                    first_valid_close,
                    first_valid_close,
                    first_valid_close,
                    first_valid_adjclose,
                    0.0,
                ]

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    previous_adjcp = tmp_df.iloc[i - 1]["adjcp"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_adjcp,
                        0.0,
                    ]

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

            print(("Data clean for ") + tic + (" is finished."))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        print("Data clean all finished!")

        return new_df



    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days
