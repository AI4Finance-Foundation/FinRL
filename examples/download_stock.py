# %%
from __future__ import annotations

import os
import time

import pandas as pd
import yfinance as yf

from finrl import config
from finrl import config_tickers

# from alpha_vantage.timeseries import TimeSeries
key = "9GZN1AFPSBK42IM7"
polygon_key = "_mg7FDMEkjIXAh7YvyTCqROWkpLCvJbu"
tickers = config_tickers.SP_500_TICKER
# ts = TimeSeries(key=key, output_format='pandas')

# %%


def get_data(tickers, start_date, end_date):
    # loop over tickers and create a dataframe with prices concatenated by ticker
    df = pd.DataFrame()
    for t in tickers:
        temp = yf.download(t, start_date, end_date)
        data, meta_data = ts.get_daily(symbol="MSFT", outputsize="full")

        df = df.dropna()
        df = df.sort_index()

    # Get json object with the intraday data and another with  the call's metadata
    # data, meta_data = ts.get_intraday('GOOGL')


def read_ticker_list(fp):
    df = pd.read_csv(fp)
    return df.Symbol.to_list()


# %%
# df = get_data(tickers, '2009-01-01', '2022-09-01')
# print(df)
# %%
def fetch_data(tickers, start_date, end_date, proxy) -> pd.DataFrame:
    """Fetches data from Yahoo API
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
    for tic in tickers:
        # temp_df = yf.download(
        #     tic, start=start_date, end=end_date, proxy=proxy, auto_adjust=True, period="max")
        temp_df = yf.download(tic, proxy=proxy, auto_adjust=True, period="10y")

        # temp_df, meta_data = ts.get_daily(symbol=tic, outputsize='full')

        temp_df["tic"] = tic
        # data_df = data_df.append(temp_df)
        data_df = pd.concat([data_df, temp_df])
        time.sleep(0.2)
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
        # use adjusted close price instead of close price
        data_df["close"] = data_df["adjcp"]

        # drop the adjusted close price column
        data_df = data_df.drop(labels="adjcp", axis=1)
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


# %%
# df = fetch_data(tickers, '2009-01-01', '2022-09-01', proxy=None)
# df.to_csv('../datasets/SP500_10yr.csv')
# %%
if __name__ == "__main__":
    fp = "../datasets/tickers/sp-600-index-10-18-2022.csv"
    fp_split = os.path.split(fp)
    # get directory
    dir = fp_split[0]
    # get filename
    filename = fp_split[1]
    # get filename without extension
    filename_no_ext = os.path.splitext(filename)[0]
    # get extension
    ext = os.path.splitext(filename)[1]
    # remove date from filename

    tickers = read_ticker_list(fp)
    print(tickers)
    start_date = config.TRAIN_START_DATE
    end_date = config.TEST_END_DATE
    df = fetch_data(tickers, start_date, end_date, proxy=None)
    df.to_csv(f"../datasets/{filename}_data_{start_date}_{end_date}.csv", index=False)
