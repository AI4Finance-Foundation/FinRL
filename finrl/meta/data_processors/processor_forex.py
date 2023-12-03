from __future__ import annotations

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from stockstats import StockDataFrame as Sdf
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ..data_processors.fx_history_data.techfeatures import ArrayManager
from ..data_processors.fx_history_data.utility import timer
from ..data_processors.fx_history_data.vo import BarData
from ..preprocessor.mtdbdownloader import MtDbDownloader
from ..utils.timefeatures import time_features


class Dataset_MT4(Dataset):
    def __init__(
        self,
        bars,
        feature_cols,
        features,
        flag="train",
        size=None,
        target="close",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.feature_cols = feature_cols
        self.bars = bars
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """

        # ### Using Mt4downloader
        # downloader = MtDbDownloader(
        #     start_date=self.start_date,
        #     end_date=self.end_date,
        #     ticker_list=self.ticker_list
        # )
        # bars_vo = downloader.fetch_data()
        df_raw = pd.DataFrame(self.bars)
        df_raw.rename(columns={"time": "date"}, inplace=True)
        df_raw = df_raw[self.feature_cols]
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ForexEngineer:
    @timer
    def download_data(
        self,
        ticker_list: list,
        time_interval: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[BarData]:
        """"""
        downloader = MtDbDownloader(
            start_date=start_date, end_date=end_date, ticker_list=ticker_list
        )
        bars = downloader.fetch_data()

        return bars

    @timer
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # fill time gap to ensure all symbols has same date, and fill NONE pricing data by last price
        time = df["time"].unique()
        symbol = df["symbol"].unique()
        columns = df.columns.drop(["time", "symbol"])
        index = pd.MultiIndex.from_product([time, symbol])
        df = df.set_index(["time", "symbol"], drop=True)
        df = pd.DataFrame(df, index=index, columns=columns)
        df = df.reset_index()
        df = df.sort_values(by=["level_1", "level_0"])
        df = df.fillna(method="pad")
        df.rename(columns={"level_1": "symbol", "level_0": "time"}, inplace=True)
        return df

    @timer
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_time = time_features(df, timeenc=1, freq="t")
        return df_time

    @timer
    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical indicators
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["symbol", "time"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.symbol.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.symbol == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["symbol"] = unique_ticker[i]
                    temp_indicator["time"] = df[df.symbol == unique_ticker[i]][
                        "time"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["symbol", "time", indicator]],
                on=["symbol", "time"],
                how="left",
            )
        df = df.sort_values(by=["time", "symbol"])
        return df

    @timer
    def add_technical_indicator_by_talib(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        return tech_features(data, tech_indicator_list)

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 24 * 60 * 15
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="time", columns="symbol", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.time.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            # Lear: minute data rolling window set to 24(hour) * 60(min) * 15(day)
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

    @timer
    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 24 * 60 * 15
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="time")
        df = df.sort_values(["time", "symbol"]).reset_index(drop=True)
        return df

    @timer
    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.copy()
        unique_ticker = df.symbol.unique()
        if_first_time = True
        for symbol in unique_ticker:
            if if_first_time:
                price_array = df[df.symbol == symbol][["close"]].values
                tech_array = df[df.symbol == symbol][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.symbol == symbol][["VIXY"]].values
                else:
                    turbulence_array = df[df.symbol == symbol][["turbulence"]].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.symbol == symbol][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.symbol == symbol][tech_indicator_list].values]
                )
                if if_vix:
                    turbulence_array = np.hstack(
                        [turbulence_array, df[df.symbol == symbol][["VIXY"]].values]
                    )
                else:
                    turbulence_array = np.hstack(
                        [
                            turbulence_array,
                            df[df.symbol == symbol][["turbulence"]].values,
                        ]
                    )

        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array
