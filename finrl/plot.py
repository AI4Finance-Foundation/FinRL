from __future__ import annotations

import copy
import datetime
from copy import deepcopy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries

from finrl import config
from finrl.meta.data_processors.func import date2str
from finrl.meta.data_processors.func import str2date
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def backtest_stats(account_value, value_col_name="account_value"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    perf_stats_all = timeseries.perf_stats(
        returns=dr_test,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print(perf_stats_all)
    return perf_stats_all


def backtest_plot(
    account_value,
    baseline_start=config.TRADE_START_DATE,
    baseline_end=config.TRADE_END_DATE,
    baseline_ticker="^DJI",
    value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(
            returns=test_returns, benchmark_rets=baseline_returns, set_context=False
        )


def get_baseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
        selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

        tic_plot = df_trade[
            (df_trade["tic"] == df_trx_temp.name)
            & (df_trade["date"].isin(df_trx.index))
        ]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(
            tic_plot,
            "^",
            markersize=10,
            color="m",
            label="buying signal",
            markevery=buying_signal,
        )
        plt.plot(
            tic_plot,
            "v",
            markersize=10,
            color="k",
            label="selling signal",
            markevery=selling_signal,
        )
        plt.title(
            f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])}"
        )
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()


# 2022-01-15 -> 01/15/2022
def transfer_date(str_dat):
    return datetime.datetime.strptime(str_dat, "%Y-%m-%d").date().strftime("%m/%d/%Y")


def plot_result_from_csv(
    csv_file: str,
    column_as_x: str,
    savefig_filename: str = "fig/result.png",
    xlabel: str = "Date",
    ylabel: str = "Result",
    num_days_xticks: int = 20,
    xrotation: int = 0,
):
    result = pd.read_csv(csv_file)
    plot_result(
        result,
        column_as_x,
        savefig_filename,
        xlabel,
        ylabel,
        num_days_xticks,
        xrotation,
    )


# select_start_date: included
# select_end_date: included
# is if_need_calc_return is True, it is account_value, and then transfer it to return
# it is better that column_as_x is the first column, and the other columns are strategies
# xrotation: the rotation of xlabel, may be used in dates. Default=0 (adaptive adjustment)
def plot_result(
    result: pd.DataFrame(),
    column_as_x: str,
    savefig_filename: str = "fig/result.png",
    xlabel: str = "Date",
    ylabel: str = "Result",
    num_days_xticks: int = 20,
    xrotation: int = 0,
):
    columns = result.columns
    columns_strtegy = []
    for i in range(len(columns)):
        col = columns[i]
        if "Unnamed" not in col and col != column_as_x:
            columns_strtegy.append(col)

    result.reindex()

    x = result[column_as_x].values.tolist()
    plt.rcParams["figure.figsize"] = (15, 6)
    # plt.figure()

    fig, ax = plt.subplots()
    colors = [
        "black",
        "red",
        "green",
        "blue",
        "cyan",
        "magenta",
        "yellow",
        "aliceblue",
        "coral",
        "darksalmon",
        "firebrick",
        "honeydew",
    ]
    for i in range(len(columns_strtegy)):
        col = columns_strtegy[i]
        ax.plot(
            x,
            result[col],
            color=colors[i],
            linewidth=1,
            linestyle="-",
        )

    plt.title("", fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    plt.legend(labels=columns_strtegy, loc="best", fontsize=16)

    # set grid
    plt.grid()

    plt.xticks(size=22)  # 设置刻度大小
    plt.yticks(size=22)  # 设置刻度大小

    # #设置每隔多少距离⼀个刻度
    # plt.xticks(x[::60])

    # # 设置每月定位符
    # if if_set_x_monthlocator:
    #     ax.xaxis.set_major_locator(mdates.MonthLocator())  # interval = 1

    # 设置每隔多少距离⼀个刻度
    plt.xticks(x[::num_days_xticks])

    plt.setp(ax.get_xticklabels(), rotation=xrotation, horizontalalignment="center")

    # 为防止x轴label重叠，自动调整label旋转角度
    if xrotation == 0:
        if_overlap = get_if_overlap(fig, ax)

        if if_overlap == True:
            plt.gcf().autofmt_xdate(ha="right")  # ⾃动旋转⽇期标记

    plt.tight_layout()  # 自动调整子图间距

    plt.savefig(savefig_filename)

    plt.show()


def get_if_overlap(fig, ax):
    fig.canvas.draw()
    # 获取日期标签的边界框
    bboxes = [label.get_window_extent() for label in ax.get_xticklabels()]
    # 计算日期标签之间的距离
    distances = [bboxes[i + 1].x0 - bboxes[i].x1 for i in range(len(bboxes) - 1)]
    # 如果有任何距离小于0，说明有重叠
    if any(distance < 0 for distance in distances):
        if_overlap = True
    else:
        if_overlap = False

    return if_overlap


def plot_return(
    result: pd.DataFrame(),
    column_as_x: str,
    if_need_calc_return: bool,
    savefig_filename: str = "fig/result.png",
    xlabel: str = "Date",
    ylabel: str = "Return",
    if_transfer_date: bool = True,
    select_start_date: str = None,
    select_end_date: str = None,
    num_days_xticks: int = 20,
    xrotation: int = 0,
):
    if select_start_date is None:
        select_start_date: str = result[column_as_x].iloc[0]
        select_end_date: str = result[column_as_x].iloc[-1]
    # calc returns if if_need_calc_return is True, so that result stores returns
    select_start_date_index = result[column_as_x].tolist().index(select_start_date)
    columns = result.columns
    columns_strtegy = []
    column_as_x_index = None
    for i in range(len(columns)):
        col = columns[i]
        if col == column_as_x:
            column_as_x_index = i
        elif "Unnamed" not in col:
            columns_strtegy.append(col)
            if if_need_calc_return:
                result[col] = result[col] / result[col][select_start_date_index] - 1

    # select the result between select_start_date and select_end_date
    # if date is 2020-01-15, transfer it to 01/15/2020
    num_rows, num_cols = result.shape
    tmp_result = copy.deepcopy(result)
    result = pd.DataFrame()
    if_first_row = True
    columns = []
    for i in range(num_rows):
        if (
            str2date(select_start_date)
            <= str2date(tmp_result[column_as_x][i])
            <= str2date(select_end_date)
        ):
            if "-" in tmp_result.iloc[i][column_as_x] and if_transfer_date:
                new_date = transfer_date(tmp_result.iloc[i][column_as_x])
            else:
                new_date = tmp_result.iloc[i][column_as_x]
            tmp_result.iloc[i, column_as_x_index] = new_date
            # print("tmp_result.iloc[i]: ", tmp_result.iloc[i])
            # result = result.append(tmp_result.iloc[i])
            if if_first_row:
                columns = tmp_result.iloc[i].index.tolist()
                result = pd.DataFrame(columns=columns)
                # result = pd.concat([result, tmp_result.iloc[i]], axis=1)
                # result = pd.DataFrame(tmp_result.iloc[i])
                # result.columns = tmp_result.iloc[i].index.tolist()
                if_first_row = False
            row = pd.DataFrame([tmp_result.iloc[i].tolist()], columns=columns)
            result = pd.concat([result, row], axis=0)

    # print final return of each strategy
    final_return = {}
    for col in columns_strtegy:
        final_return[col] = result.iloc[-1][col]
    print("final return: ", final_return)

    result.reindex()

    plot_result(
        result=result,
        column_as_x=column_as_x,
        savefig_filename=savefig_filename,
        xlabel=xlabel,
        ylabel=ylabel,
        num_days_xticks=num_days_xticks,
        xrotation=xrotation,
    )


def plot_return_from_csv(
    csv_file: str,
    column_as_x: str,
    if_need_calc_return: bool,
    savefig_filename: str = "fig/result.png",
    xlabel: str = "Date",
    ylabel: str = "Return",
    if_transfer_date: bool = True,
    select_start_date: str = None,
    select_end_date: str = None,
    num_days_xticks: int = 20,
    xrotation: int = 0,
):
    result = pd.read_csv(csv_file)
    plot_return(
        result,
        column_as_x,
        if_need_calc_return,
        savefig_filename,
        xlabel,
        ylabel,
        if_transfer_date,
        select_start_date,
        select_end_date,
        num_days_xticks,
        xrotation,
    )
