import numpy as np
import pandas as pd
from typing import List

class BasicProcessor:
    def __init__(self, data_source: str):
        assert data_source in ["alpaca", "ccxt", "joinquant", "quantconnect", "wrds", "yahoofinance", ], "Data source input is NOT supported yet."
        self.data_source = data_source

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) -> pd.DataFrame:
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def add_technical_indicator(self, df: pd.DataFrame, tech_indicator_list: List[str]) -> pd.DataFrame:
        pass

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
