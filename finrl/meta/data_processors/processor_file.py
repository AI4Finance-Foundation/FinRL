import os
import logbook
import pandas as pd
import numpy as np
from finrl.config import (
        DATA_SAVE_DIR
)

class FileProcessor:
    def __init__(self, directory_path: str) -> None:
        try:
            print ( self)
            self.logger = logbook.Logger(self.__class__.__name__)
            self.directory_path = directory_path
        except Exception as e:
            self.logger.error(e)


    def download_data(self, ticker_list, start_date, end_date, time_interval ) -> pd.DataFrame:
        self.logger.info ( f"Loading data from {self.directory_path}")

        
        dfs = []
        # Loop through all files in the directory
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(self.directory_path, filename)
                self.logger.info ( f"Reading file {file_path}")
                
                # Read the CSV file and append the DataFrame to the list
                dfs.append(pd.read_csv(file_path,
                    parse_dates=['timestamp'],
                    infer_datetime_format=True,
                    index_col=0, 
                    date_parser=lambda x: pd.to_datetime(x, utc=True).tz_convert('America/New_York')
                ))

        combined_df = pd.concat(dfs)
        
        return combined_df;

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = df.dropna()
        return df
    
    def add_technical_indicator(self, df: pd.DataFrame, tech_indicator_list: list) -> pd.DataFrame:
        # df = df.dropna()
        return df
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = df.dropna()
        return df
    
    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = df.dropna()
        return df
    
    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = df.dropna()
        return df
    
    def add_vixor(self, df: pd.DataFrame) -> pd.DataFrame:
        # df = df.dropna()
        return df
    

    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: np.array, if_vix: bool) -> np.array:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
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
        #        self.logger.info("Successfully transformed into array")
        return price_array, tech_array, turbulence_array