from __future__ import annotations

import copy
import datetime
import os
from datetime import date
from datetime import timedelta
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


# filename: str
# output: stockname
def calc_stockname_from_filename(filename):
    return filename.split("/")[-1].split(".csv")[0]


def calc_all_filenames(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    paths2 = []
    for dir in dir_list:
        filename = os.path.join(os.path.abspath(path), dir)
        if ".csv" in filename and "#" not in filename and "~" not in filename:
            paths2.append(filename)
    return paths2


def calc_stocknames(path):
    filenames = calc_all_filenames(path)
    res = []
    for filename in filenames:
        stockname = calc_stockname_from_filename(filename)
        res.append(stockname)
    return res


def remove_all_files(remove, path_of_data):
    assert remove in [0, 1]
    if remove == 1:
        os.system("rm -f " + path_of_data + "/*")
    dir_list = os.listdir(path_of_data)
    for file in dir_list:
        if "~" in file:
            os.system("rm -f " + path_of_data + "/" + file)
    dir_list = os.listdir(path_of_data)

    if remove == 1:
        if len(dir_list) == 0:
            print(f"dir_list: {dir_list}. Right.")
        else:
            print(
                "dir_list: {}. Wrong. You should remove all files by hands.".format(
                    dir_list
                )
            )
        assert len(dir_list) == 0
    else:
        if len(dir_list) == 0:
            print(f"dir_list: {dir_list}. Wrong. There is not data.")
        else:
            print(f"dir_list: {dir_list}. Right.")
        assert len(dir_list) > 0


def date2str(dat: datetime.date) -> str:
    return datetime.date.strftime(dat, "%Y-%m-%d")


def str2date(dat: str) -> datetime.date:
    return datetime.datetime.strptime(dat, "%Y-%m-%d").date()


# include start_date, inclue end_date. step: delta
def calc_dates(
    start_date: datetime.date, end_date: datetime.date, delta: datetime.timedelta
) -> list[str]:
    dates = []
    dat = copy.deepcopy(start_date)
    while dat <= end_date:
        d = date2str(dat)
        dates.append(d)
        dat += delta
    return dates


# init_train_dates: the init train_dates, but not separated to subsets, e.g.,'2010-01-01', ...'2021-10-01'
# init_trade_dates: the trade_dates, but not separated to subsets
# num_days_if_rolling: the num of days in a subset if trade_dates splits.
# return: train_starts, train_ends, trade_starts, trade_ends, which has the same length num_subsets_if_rolling
# start is include, end is not include. The max of endIndex is len(dates) - 1.
def calc_train_trade_starts_ends_if_rolling(
    init_train_dates: list[str], init_trade_dates: list[str], rolling_window_length: int
) -> tuple[list[str], list[str], list[str], list[str]]:
    trade_dates_length = len(init_trade_dates)
    train_window_length = len(init_train_dates)
    trade_window_length = min(rolling_window_length, trade_dates_length)
    num_subsets_if_rolling = int(np.ceil(trade_dates_length / trade_window_length))
    print("num_subsets_if_rolling: ", num_subsets_if_rolling)
    dates = np.concatenate((init_train_dates, init_trade_dates), axis=0)
    train_starts = []
    train_ends = []
    trade_starts = []
    trade_ends = []

    for i in range(num_subsets_if_rolling):
        trade_start_index = train_window_length + i * trade_window_length
        trade_start = dates[trade_start_index]
        trade_starts.append(trade_start)
        trade_end_index = min(trade_start_index + trade_window_length, len(dates) - 1)
        trade_end = dates[trade_end_index]
        trade_ends.append(trade_end)
        train_start = dates[trade_start_index - train_window_length]
        train_starts.append(train_start)
        train_end = dates[trade_start_index]
        train_ends.append(train_end)
    print("train_starts: ", train_starts)
    print("train_ends__: ", train_ends)
    print("trade_starts: ", trade_starts)
    print("trade_ends__: ", trade_ends)
    return train_starts, train_ends, trade_starts, trade_ends


def calc_train_trade_data(
    i: int,
    train_starts: list[str],
    train_ends: list[str],
    trade_starts: list[str],
    trade_ends: list[str],
    init_train_data: pd.DataFrame(),
    init_trade_data: pd.DataFrame(),
    date_col: str,
) -> tuple[pd.DataFrame(), pd.DataFrame()]:
    train_start = train_starts[i]
    train_end = train_ends[i]
    trade_start = trade_starts[i]
    trade_end = trade_ends[i]
    train_data = init_train_data.loc[
        (init_train_data[date_col] >= train_start)
        & (init_train_data[date_col] < train_end)
    ]
    train_data.index = train_data[date_col].factorize()[0]
    trade_data = init_trade_data.loc[
        (init_trade_data[date_col] >= trade_start)
        & (init_trade_data[date_col] < trade_end)
    ]
    trade_data.index = trade_data[date_col].factorize()[0]
    return train_data, trade_data
