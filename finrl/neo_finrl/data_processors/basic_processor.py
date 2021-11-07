import numpy as np
import pandas as pd
from typing import List
from stockstats import StockDataFrame as Sdf

class BasicProcessor:
    def __init__(self, data_source: str, **kwargs):
        assert data_source in ["alpaca", "ccxt", "joinquant", "quantconnect", "wrds", "yahoofinance", ], "Data source input is NOT supported yet."
        self.data_source = data_source

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) \
            -> pd.DataFrame:
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_trading_days(self, start: str, end: str) -> List[str]:
        pass

    def add_technical_indicator(self, data: pd.DataFrame, tech_indicator_list: List[str]) \
            -> pd.DataFrame:
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "time"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["time"] = df[df.tic == unique_ticker[i]][
                        "time"
                    ].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
            )
        df = df.sort_values(by=["time", "tic"])
        return df

    def add_turbulence(self, data: pd.DataFrame) \
            -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data: pd.DataFrame, time_period: int=252) \
            -> pd.DataFrame:
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="time", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"time": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_vix(self, data: pd.DataFrame)\
            -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = self.download_data(
            start_date=df.time.min(),
            end_date=df.time.max(),
            ticker_list=["^VIX"],
            time_interval=self.time_interval,
        )
        df_vix = self.clean_data(df_vix)
        vix = df_vix[["time", "adjcp"]]
        vix.columns = ["time", "vix"]

        df = df.merge(vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df

    # tech_array: technical indicator
    # price_array, close price
    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) \
            -> List[np.array]:
        """transform final df to numpy arrays"""
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["adjcp"]].values
                # price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["vix"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["adjcp"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        assert price_array.shape[0] == tech_array.shape[0]
        assert tech_array.shape[0] == turbulence_array.shape[0]
        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array