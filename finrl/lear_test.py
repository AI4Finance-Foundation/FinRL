# from __future__ import annotations
# from typing import List, Callable
# from finrl.meta.data_processor import DataProcessor
# from finrl.meta.data_processors.fx_history_data.vo import BarData
# from finrl.meta.data_processors.fx_history_data.constant import Interval, Exchange
# from finrl.meta.data_processors.fx_history_data.minutebars import BarGenerator
# from finrl.meta.data_processors.fx_history_data.techfeatures import ArrayManager
# from finrl.meta.data_processors.fx_history_data.utility import timer
# import pandas as pd
# import copy
#
#
# @timer
# def df_to_vo(df: pd.DataFrame) -> List[BarData]:
#     bars = []
#
#     for _,row in df.iterrows():
#         bar = BarData(
#             exchange = Exchange.MT4,
#             gateway_name = "DB",
#             volume = 0,
#             open_interest = 0,
#             turnover = 0,
#             close = row['close'],
#             open = row['open'],
#             high = row['high'],
#             low = row['low'],
#             time = row['time'],
#             symbol = row['symbol']
#         )
#
#         bars.append(bar)
#     return bars
#
#
# class MinuteBar:
#     """
#     使用1m线合成 日线、30分钟、1分钟三种数据
#     """
#
#     def __init__(self, bar_data, window_mn):
#         self.bar_data = bar_data
#         self.bg = BarGenerator(self.on_bar, interval=Interval.MINUTE, window=window_mn,
#                                         on_window_bar=self.on_minute_bar)
#         self.all_bars:List[BarData] = []
#         self.load_bar()
#
#     def on_minute_bar(self, bar):
#         # only called on mins
#         self.all_bars.append(self.bg.window_bar)
#
#     def on_bar(self, bar: BarData):
#         self.bg.update_bar(bar)
#
#     def load_bar(
#         self,
#         callback: Callable = None,
#     ) -> None:
#         """
#         Load historical bar data for initializing strategy.
#         """
#         if not callback:
#             callback: Callable = self.on_bar
#
#         bars: List[BarData] = self.bar_data
#
#         for bar in bars:
#             callback(bar)
#
# if __name__ == "__main__":
#     data_source = "forex"
#     start_date = "2022-11-01"
#     end_date = "2022-11-05"
#     time_interval = "MINUTE"
#     ticker_list = ["EURUSD.MT4"]
#     dp = DataProcessor(data_source)
#     df = dp.download_data(ticker_list, start_date, end_date, time_interval)
#     #df = df.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"})
#     #df.set_index("Date", inplace=True, drop=True)
#     x = df_to_vo(df)
#     min5_bar = MinuteBar(x, window_mn=5).all_bars
#     min30_bar = MinuteBar(x, window_mn=30).all_bars
#
#     print(min5_bar)
from __future__ import annotations

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TimesNet")

    # basic config
    parser.add_argument(
        "--task_name",
        type=str,
        required=False,
        default="long_term_forecast_mt4",
        help="task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection, long_term_forecast_mt4]",
    )
    parser.add_argument(
        "--is_training", type=int, required=False, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=False, default="mt4_train", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="TimesNet",
        help="model name, options: [Autoformer, Transformer, TimesNet]",
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    args = parser.parse_args()
    for ii in range(args.itr):
        print(ii)
