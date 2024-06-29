from abc import ABC, abstractmethod

import pandas as pd

class AbstractProcessor(ABC):
    @abstractmethod
    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        pass


    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_technical_indicator(
        self,
        df,
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
        pass

    @abstractmethod
    def calculate_turbulence(self, data, time_period=252):
        pass

    @abstractmethod
    def add_turbulence(self, data, time_period=252):
        pass

    @abstractmethod
    def df_to_array(self, df, tech_indicator_list, if_vix):
        pass

    @abstractmethod
    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def add_vix(self, df) -> pd.DataFrame:
        pass

    @abstractmethod
    def close_conn(self):
        pass