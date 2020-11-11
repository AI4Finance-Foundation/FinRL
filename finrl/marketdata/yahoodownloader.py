"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import yfinance as yf


class YahooDownloader:
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
    def __init__(self, 
        start_date:str,
        end_date:str,
        ticker_list:list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list


    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            A date, open, high, low, close, adjusted close price, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df['tic'] = tic
            data_df=data_df.append(temp_df)
        # reset the index, we want to use numbers instead of dates
        data_df=data_df.reset_index()
        # convert the column names to standardized names
        data_df.columns = ['date','open','high','low','close','adjcp','volume','tic']
        # convert date to string format, easy to filter
        data_df['date']=data_df.date.apply(lambda x: x.strftime('%Y-%m-%d'))
        # drop missing data 
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        #print("Display DataFrame: ", data_df.head())
        return data_df
