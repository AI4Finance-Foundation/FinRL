import os
import logbook
import pandas as pd
import numpy as np
from finrl.config import (
        DATA_SAVE_DIR
)

class FileProcessor:
    def __init__(self) -> None:
        try:
            self.logger = logbook.Logger(self.__class__.__name__)
        except Exception as e:
            self.logger.error(e)


    def download_data(self, ticker_list, start_date, end_date, time_interval, directory_path: str = DATA_SAVE_DIR ) -> pd.DataFrame:
        self.logger.info ( f"Loading data from {file_path}")
        
        dfs = []
        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(directory_path, filename)
                
                # Read the CSV file and append the DataFrame to the list
                dfs.append(pd.read_csv(file_path,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True,
                    index_col=0, 
                    date_parser=lambda x: pd.to_datetime(x, utc=True).tz_convert('America/New_York')
                ))

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df;

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna()
        return df
    
    def add_technical_indicator(self, df: pd.DataFrame, tech_indicator_list: list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df
    
    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df
    
    def add_vixor(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df
    

    def df_to_array(self, df: pd.DataFrame) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array