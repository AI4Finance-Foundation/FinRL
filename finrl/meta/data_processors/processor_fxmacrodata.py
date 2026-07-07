"""FXMacroData data processor for daily FX spot data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from typing import List

from finrl.meta.preprocessor.fxmacrodatadownloader import FXMacroDataDownloader


class FXMacroDataProcessor:
    """Provides daily FX spot data from FXMacroData for FinRL processors."""

    def __init__(self, api_key=None, base_url="https://api.fxmacrodata.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def download_data(
        self,
        ticker_list: List[str],
        start_date: str,
        end_date: str,
        time_interval: str,
    ) -> pd.DataFrame:
        if time_interval.lower() not in {"1d", "1day"}:
            raise ValueError("FXMacroDataProcessor supports daily data only.")

        self.start = start_date
        self.end = end_date
        self.time_interval = "1d"
        data_df = FXMacroDataDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            api_key=self.api_key,
            base_url=self.base_url,
        ).fetch_data()
        data_df["timestamp"] = pd.to_datetime(data_df["date"])
        return data_df[["timestamp", "open", "high", "low", "close", "volume", "tic"]]

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna()
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: List[str]
    ) -> pd.DataFrame:
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for tic in unique_ticker:
                try:
                    temp_indicator = stock[stock.tic == tic][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = tic
                    temp_indicator["timestamp"] = df[df.tic == tic][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as exc:
                    print(exc)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["turbulence"] = 0
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["VIXY"] = 0
        return df

    def add_vixor(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.add_vix(df)

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool
    ) -> List[np.ndarray]:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        turbulence_array = None
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        return price_array, tech_array, turbulence_array
