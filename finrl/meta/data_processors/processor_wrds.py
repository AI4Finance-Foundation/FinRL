from __future__ import annotations

import datetime

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import wrds
from stockstats import StockDataFrame as Sdf

pd.options.mode.chained_assignment = None


class WrdsProcessor:
    def __init__(self, if_offline=False):
        if not if_offline:
            self.db = wrds.Connection()

    def download_data(
        self,
        start_date,
        end_date,
        ticker_list,
        time_interval,
        if_save_tempfile=False,
        filter_shares=0,
    ):
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        def get_trading_days(start, end):
            nyse = tc.get_calendar("NYSE")
            df = nyse.sessions_in_range(
                pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
            )
            trading_days = []
            for day in df:
                trading_days.append(str(day)[:10])

            return trading_days

        def data_fetch_wrds(date="2021-05-01", stock_set=("AAPL"), time_interval=60):
            # start_date, end_date should be in the same year
            current_date = datetime.datetime.strptime(date, "%Y-%m-%d")
            lib = "taqm_" + str(current_date.year)  # taqm_2021
            table = "ctm_" + current_date.strftime("%Y%m%d")  # ctm_20210501

            parm = {"syms": stock_set, "num_shares": filter_shares}
            try:
                data = self.db.raw_sql(
                    "select * from "
                    + lib
                    + "."
                    + table
                    + " where sym_root in %(syms)s and time_m between '9:30:00' and '16:00:00' and size > %(num_shares)s and sym_suffix is null",
                    params=parm,
                )
                if_empty = False
                return data, if_empty
            except BaseException:
                print("Data for date: " + date + " error")
                if_empty = True
                return None, if_empty

        dates = get_trading_days(start_date, end_date)
        print("Trading days: ")
        print(dates)
        first_time = True
        empty = True
        stock_set = tuple(ticker_list)
        for i in dates:
            x = data_fetch_wrds(i, stock_set, time_interval)

            if not x[1]:
                empty = False
                dataset = x[0]
                dataset = self.preprocess_to_ohlcv(
                    dataset, time_interval=(str(time_interval) + "S")
                )
                if first_time:
                    print("Data for date: " + i + " finished")
                    temp = dataset
                    first_time = False
                    if if_save_tempfile:
                        temp.to_csv("./temp.csv")
                else:
                    print("Data for date: " + i + " finished")
                    temp = pd.concat([temp, dataset])
                    if if_save_tempfile:
                        temp.to_csv("./temp.csv")
        if empty:
            raise ValueError("Empty Data under input parameters!")
        else:
            result = temp
            result = result.sort_values(by=["time", "tic"])
            result = result.reset_index(drop=True)
            return result

    def preprocess_to_ohlcv(self, df, time_interval="60S"):
        df = df[["date", "time_m", "sym_root", "size", "price"]]
        tic_list = np.unique(df["sym_root"].values)
        final_df = None
        first_time = True
        for i in range(len(tic_list)):
            tic = tic_list[i]
            time_list = []
            temp_df = df[df["sym_root"] == tic]
            for i in range(0, temp_df.shape[0]):
                date = temp_df["date"].iloc[i]
                time_m = temp_df["time_m"].iloc[i]
                time = str(date) + " " + str(time_m)
                try:
                    time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
                except BaseException:
                    time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
                time_list.append(time)
            temp_df["time"] = time_list
            temp_df = temp_df.set_index("time")
            data_ohlc = temp_df["price"].resample(time_interval).ohlc()
            data_v = temp_df["size"].resample(time_interval).agg({"size": "sum"})
            volume = data_v["size"].values
            data_ohlc["volume"] = volume
            data_ohlc["tic"] = tic
            if first_time:
                final_df = data_ohlc.reset_index()
                first_time = False
            else:
                final_df = final_df.append(data_ohlc.reset_index(), ignore_index=True)
        return final_df

    def clean_data(self, df):
        df = df[["time", "open", "high", "low", "close", "volume", "tic"]]
        # remove 16:00 data
        tic_list = np.unique(df["tic"].values)
        ary = df.values
        rows_1600 = []
        for i in range(ary.shape[0]):
            row = ary[i]
            time = row[0]
            if str(time)[-8:] == "16:00:00":
                rows_1600.append(i)

        df = df.drop(rows_1600)
        df = df.sort_values(by=["tic", "time"])

        # check missing rows
        tic_dic = {}
        for tic in tic_list:
            tic_dic[tic] = [0, 0]
        ary = df.values
        for i in range(ary.shape[0]):
            row = ary[i]
            volume = row[5]
            tic = row[6]
            if volume != 0:
                tic_dic[tic][0] += 1
            tic_dic[tic][1] += 1
        constant = np.unique(df["time"].values).shape[0]
        nan_tics = []
        for tic in tic_dic:
            if tic_dic[tic][1] != constant:
                nan_tics.append(tic)
        # fill missing rows
        normal_time = np.unique(df["time"].values)

        df2 = df.copy()
        for tic in nan_tics:
            tic_time = df[df["tic"] == tic]["time"].values
            missing_time = []
            for i in normal_time:
                if i not in tic_time:
                    missing_time.append(i)
            for time in missing_time:
                temp_df = pd.DataFrame(
                    [[time, np.nan, np.nan, np.nan, np.nan, 0, tic]],
                    columns=["time", "open", "high", "low", "close", "volume", "tic"],
                )
                df2 = df2.append(temp_df, ignore_index=True)

        # fill nan data
        df = df2.sort_values(by=["tic", "time"])
        for i in range(df.shape[0]):
            if float(df.iloc[i]["volume"]) == 0:
                previous_close = df.iloc[i - 1]["close"]
                if str(previous_close) == "nan":
                    raise ValueError("Error nan price")
                df.iloc[i, 1] = previous_close
                df.iloc[i, 2] = previous_close
                df.iloc[i, 3] = previous_close
                df.iloc[i, 4] = previous_close
        # check if nan
        ary = df[["open", "high", "low", "close", "volume"]].values
        assert not np.isnan(np.min(ary))
        # final preprocess
        df = df[["time", "open", "high", "low", "close", "volume", "tic"]]
        df = df.reset_index(drop=True)
        print("Data clean finished")
        return df

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
        df = df.rename(columns={"time": "date"})
        df = df.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()
        tech_indicator_list = tech_indicator_list

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                # print(unique_ticker[i], i)
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                # print(len(df[df.tic == unique_ticker[i]]['date'].to_list()))
                temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                    "date"
                ].to_list()
                indicator_df = indicator_df.append(temp_indicator, ignore_index=True)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        print("Succesfully add technical indicators")
        return df

    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a fixed time period
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
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_vix(self, data):
        vix_df = self.download_data(
            ["vix"], self.start, self.end_date, self.time_interval
        )
        cleaned_vix = self.clean_data(vix_df)
        vix = cleaned_vix[["date", "close"]]

        df = data.copy()
        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)

        return df

    def df_to_array(self, df, tech_indicator_list):
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                # price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array
