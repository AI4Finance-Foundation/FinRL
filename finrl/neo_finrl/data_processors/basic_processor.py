import numpy as np
import pandas as pd

class BasicProcessor:
    def __init__(self, data_source, **kwargs):
        assert data_source in ["alpaca", "ccxt", "joinquant", "quantconnect", "wrds", "yahoofinance", ], "Data source input is NOT supported yet."
        self.data_source = data_source

    def download_data(self, ticker_list, start_date, end_date, time_interval) -> pd.DataFrame:
        pass

    def clean_data(self, df) -> pd.DataFrame:
        pass

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        pass

    def add_turbulence(self, df) -> pd.DataFrame:
        pass
