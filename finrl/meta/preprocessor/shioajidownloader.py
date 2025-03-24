"""Contains methods and classes to collect data from
shioaji Finance API
"""

from __future__ import annotations

import pandas as pd
import shioaji as sj


class SinopacDownloader:

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker_list: list = [],
        api: sj.Shioaji = None,
    ):
        if api is None:
            self.api = sj.Shioaji()
            self.api.login(
                api_key="3Tn2BbtCzbaU1KSy8yyqLa4m7LEJJyhkRCDrK2nknbcu",
                secret_key="Epakqh1Nt4inC3hsqowE2XjwQicPNzswkuLjtzj2WKpR",
                contracts_cb=lambda security_type: print(
                    f"{repr(security_type)} fetch done."
                ),
            )
        else:
            self.api = api
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self) -> pd.DataFrame:
        """Fetches data from Shioaji API

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, open, high, low, close, volume, amount, ticker
        """
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            try:
                kbars = self.api.kbars(
                    self.api.Contracts.Stocks[tic],
                    start=self.start_date,
                    end=self.end_date,
                )
                temp_df = pd.DataFrame({**kbars})
                temp_df.ts = pd.to_datetime(temp_df.ts)
                temp_df["tic"] = tic
                data_df = pd.concat([data_df, temp_df], axis=0)
            except Exception as e:
                num_failures += 1
                print(f"Failed to fetch data for ticker {tic}: {e}")

        if num_failures == len(self.ticker_list):
            raise ValueError("No data is fetched.")

        data_df = data_df.reset_index(drop=True)
        print("Original columns:", data_df.columns)
        try:
            data_df.columns = [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "tic",
            ]
        except ValueError as e:
            print(f"Error renaming columns: {e}")

        data_df["day"] = data_df["timestamp"].dt.dayofweek
        data_df["date"] = data_df.timestamp.apply(lambda x: x.strftime("%Y-%m-%d"))
        data_df = data_df.dropna().reset_index(drop=True)
        data_df = data_df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", data_df.shape)
        print("Display DataFrame: ", data_df.head())

        return data_df

    def select_equal_rows_stock(self, df: pd.DataFrame) -> pd.DataFrame:
        df_check = df.ticker.value_counts().reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        select_stocks_list = df_check[df_check.counts >= mean_df]["tic"].tolist()
        df = df[df.ticker.isin(select_stocks_list)]
        return df


if __name__ == "__main__":
    start_date = "2023-04-13"
    end_date = "2024-04-13"
    ticker_list = ["2330", "2317", "2454", "2303", "2412"]

    # 测试 api 为 None 的情况
    downloader = SinopacDownloader(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list, api=None
    )
    df = downloader.fetch_data()
    print(df)
    print(df.ticker.value_counts())
    df = downloader.select_equal_rows_stock(df)
    print(df.ticker.value_counts())
    print(df)
    print(df.shape)
