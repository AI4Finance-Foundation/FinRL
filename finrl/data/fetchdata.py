"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import yfinance as yf
from finrl.exceptions import *
import os
import sys
import glob
from datetime import datetime

class FetchData:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
    start_date : str
        start date of the data (modified from config.py)
        
    end_date : str
        end date of the data (modified from config.py)
            
    ticker_list : list
        a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, config: dict):
        self.config = config
        self.data_list = []
        
    def get_data(self):
        for root, dirs, files in os.walk("./"):
          for file in files:
            if file.endswith(".json"):  
              file_json = os.path.join(root, file)
              self.data_list.append(file_json)
        return self.data_list


    def fetch_data_stock(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        exchange = "yahoo"
        datadir = f'{self.config["user_data_dir"]}/data/{exchange}'
        print(datadir)
        data = self.get_data()
        timeframe = self.config["timeframe"]
        ticker_list = self.config["ticker_list"]
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for i in ticker_list:
            for text in data:
                if f"{datadir}" and f"{i}-{timeframe}" in text:
                    i_df = pd.read_json(text)
                    if not i_df.empty:
                        i_df["tic"] = i
                        data_df = data_df.append(temp_df)
                    else:
                        print(f"Stock {i} from {text} is Data Not Available...")
                  print(f"Stock {i} from {text} Fetched successfully...")
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

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)
        print(data_df.head())
        return data_df

    def fetch_data_crypto(self) -> pd.DataFrame:
        """
        Fetches data from local history directory (default= user_data/data/exchange)
        
        Parameters
        ----------
        config.json ---> Exchange, Whitelist, timeframe 

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        data = self.get_data()
        if self.config.get("timeframes"):
          timeframe = self.config["timeframes"]
        else:
          timeframe = self.config["timeframe"]
        # Check if regex found something and only return these results
        df = pd.DataFrame()
        for i in self.config["pairs"]:
            i = i.replace("/","_")
            for text in data:
                if f"{self.config['datadir']}" and f"{i}-{timeframe}" in text:
                  i_df = pd.read_json(text)
                  if not i_df.empty:
                       i_df["tic"] = i
                       i_df.columns = ["date", "open","high", "low", "close", "volume", "tic"]
                       i_df.date = i_df.date.apply(lambda d: datetime.fromtimestamp(d/1000))
                       df = df.append(i_df)
                  else:
                     print(f"coin {i} from {text} is Empty...")
                  print(f"coin {i} from {text} completed...")
        df = df.sort_values(by=['date','tic']).reset_index(drop=True)
        print(df.shape)
        return df

    def select_equal_rows_stock(self, df,start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df

 
