import datetime

import numpy as np
import pandas as pd
import pytz
import trading_calendars as tc
import wrds
from stockstats import StockDataFrame as Sdf
from finrl.neo_finrl.data_processors.basic_processor import BasicProcessor

pd.options.mode.chained_assignment = None


class WrdsProcessor(BasicProcessor):
    # def __init__(self, if_offline=False):
    #     if not if_offline:
    #         self.db = wrds.Connection()
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)

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




