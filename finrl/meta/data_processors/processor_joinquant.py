from __future__ import annotations

import copy
import os

import jqdatasdk as jq
import numpy as np
import pandas as pd
from func import calc_all_filenames
from func import date2str
from func import remove_all_files


class JoinQuantEngineer:
    def __init__(self):
        pass

    def auth(self, username, password):
        jq.auth(username, password)

    def data_fetch(self, stock_list, num, unit, end_dt):
        df = jq.get_bars(
            security=stock_list,
            count=num,
            unit=unit,
            fields=["date", "open", "high", "low", "close", "volume"],
            end_dt=end_dt,
        )
        return df

    def preprocess(df, stock_list):
        n = len(stock_list)
        N = df.shape[0]
        assert N % n == 0
        d = int(N / n)
        stock1_ary = df.iloc[0:d, 1:].values
        temp_ary = stock1_ary
        for j in range(1, n):
            stocki_ary = df.iloc[j * d : (j + 1) * d, 1:].values
            temp_ary = np.hstack((temp_ary, stocki_ary))
        return temp_ary

    # start_day: str
    # end_day: str
    # output: list of str_of_trade_day, e.g., ['2021-09-01', '2021-09-02']
    def calc_trade_days_by_joinquant(self, start_day, end_day):
        dates = jq.get_trade_days(start_day, end_day)
        str_dates = [date2str(dt) for dt in dates]
        return str_dates

    # start_day: str
    # end_day: str
    # output: list of dataframes, e.g., [df1, df2]
    def read_data_from_csv(self, path_of_data, start_day, end_day):
        datasets = []
        selected_days = self.calc_trade_days_by_joinquant(start_day, end_day)
        filenames = calc_all_filenames(path_of_data)
        for filename in filenames:
            dataset_orig = pd.read_csv(filename)
            dataset = copy.deepcopy(dataset_orig)
            days = dataset.iloc[:, 0].values.tolist()
            indices_of_rows_to_drop = [d for d in days if d not in selected_days]
            dataset.drop(index=indices_of_rows_to_drop, inplace=True)
            datasets.append(dataset)
        return datasets

    # start_day: str
    # end_day: str
    # read_data_from_local: if it is true, read_data_from_csv, and fetch data from joinquant otherwise.
    # output: list of dataframes, e.g., [df1, df2]
    def data_fetch_for_stocks(
        self, stocknames, start_day, end_day, read_data_from_local, path_of_data
    ):
        assert read_data_from_local in [0, 1]
        if read_data_from_local == 1:
            remove = 0
        else:
            remove = 1
        remove_all_files(remove, path_of_data)
        dfs = []
        if read_data_from_local == 1:
            dfs = self.read_data_from_csv(path_of_data, start_day, end_day)
        else:
            if os.path.exists(path_of_data) is False:
                os.makedirs(path_of_data)
            for stockname in stocknames:
                df = jq.get_price(
                    stockname,
                    start_date=start_day,
                    end_date=end_day,
                    frequency="daily",
                    fields=["open", "close", "high", "low", "volume"],
                )
                dfs.append(df)
                df.to_csv(path_of_data + "/" + stockname + ".csv", float_format="%.4f")
        return dfs


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    # from finrl.neo_finrl.neofinrl_config import TRADE_START_DATE
    # from finrl.neo_finrl.neofinrl_config import TRADE_END_DATE
    # from finrl.neo_finrl.neofinrl_config import READ_DATA_FROM_LOCAL
    # from finrl.neo_finrl.neofinrl_config import PATH_OF_DATA

    # read_data_from_local = READ_DATA_FROM_LOCAL
    # path_of_data = '../' + PATH_OF_DATA

    path_of_data = "../" + "data"

    TRADE_START_DATE = "20210901"
    TRADE_END_DATE = "20210911"
    READ_DATA_FROM_LOCAL = 1

    e = JoinQuantEngineer()
    username = "xxx"  # should input your username
    password = "xxx"  # should input your password
    e.auth(username, password)

    trade_days = e.calc_trade_days_by_joinquant(TRADE_START_DATE, TRADE_END_DATE)
    stocknames = ["000612.XSHE", "601808.XSHG"]
    data = e.data_fetch_for_stocks(
        stocknames, trade_days[0], trade_days[-1], READ_DATA_FROM_LOCAL, path_of_data
    )
