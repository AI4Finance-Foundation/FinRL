import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, data_source, **kwargs):
        assert data_source in ["alpaca", "ccxt", "joinquant", "quantconnect", "wrds", "yahoofinance", ], "Data source input is NOT supported yet."

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array
