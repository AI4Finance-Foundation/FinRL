"""Contains methods and classes to collect data from
shioaji Finance API
"""

from __future__ import annotations

import pandas as pd
import shioaji as sj


class sinopacDownloader:

    def __init__(self, api,start_date: str, end_date: str, ticker_list: list):
        if api is None:
            api = sj.Shioaji()
            api.login(
                api_key="3gNpFbPDW3YC7RhKzXRthtDJ2TDkkuevvNuqsq1Jese2",
                secret_key="NbHaa8brgXNmsckwvXLtCnrgCfBWKumrUbXyNgqsWXK",
                contracts_cb=lambda security_type: print(
                    f"{repr(security_type)} fetch done."
                ),
            )
        else:
            api = api
        self.api = api
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, api,proxy=None) -> pd.DataFrame:
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
        num_failures = 0
        for tic in self.ticker_list:
            kbars =self.api.kbars(
                api.Contracts.Stocks[tic],
                start=self.start_date,
                end=self.end_date,
            )
            temp_df = pd.DataFrame({**kbars})
            temp_df.ts = pd.to_datetime(temp_df.ts)
            temp_df['tic'] = tic
            if len(temp_df) > 0:
                # data_df = data_df.append(temp_df)
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                num_failures += 1
        if num_failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "index",
                "ts",
                "Open",
                "High",
                "Low",
                "Close",    
                "Volume",
                "Amount",
                "contract_symbol",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        data_df = data_df.rename(columns={"ts": "timestamp", "contract_symbol": "tic"})
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["timestamp"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.timestamp.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)

        return data_df

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
    api = sj.Shioaji()
    api.login(
        api_key="3gNpFbPDW3YC7RhKzXRthtDJ2TDkkuevvNuqsq1Jese2",
        secret_key="NbHaa8brgXNmsckwvXLtCnrgCfBWKumrUbXyNgqsWXK",
        contracts_cb=lambda security_type: print(
            f"{repr(security_type)} fetch done."
        ),
    )
    start_date = "2021-01-01"
    end_date = "2021-01-31"
    ticker_list = ["2330", "2317", "2454", "2303", "2412"]
    data = sinopacDownloader(api,start_date, end_date, ticker_list)
    df = data.fetch_data(api)
    print(df)
    print(df.tic.value_counts())
    df = data.select_equal_rows_stock(df)
    print(df.tic.value_counts())
    print(df)
    print(df.shape)