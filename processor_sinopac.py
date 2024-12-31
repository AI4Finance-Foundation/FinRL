from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import shioaji as sj
import talib
from shioaji import Exchange
from shioaji import TickSTKv1
from stockstats import StockDataFrame as Sdf
from talib import abstract

from finrl.meta.preprocessor.shioajidownloader import sinopacDownloader


class SinopacProcessor:
    def __init__(self, API_KEY=None, API_SECRET=None, api=None):
        if api is None:
            try:
                self.api = sj.Shioaji()
                print("API connected")
                print("enter API_KEY and API_SECRET")
                API_KEY = input("API_KEY: ")
                API_SECRET = input("API_SECRET: ")
                self.api.login(
                    API_KEY=API_KEY,
                    API_SECRET=API_SECRET,
                    contracts_cb=lambda security_type: print(
                        f"{repr(security_type)} fetch done."
                    ),
                )
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def download_data(self):
        # 創建 sinopacDonwloader 的ticker_list
        print("enter start date")
        start_date = input("start date: ")
        print("enter end date")
        end_date = input("end date: ")
        print("enter ticker list")
        ticker_list = input("ticker list: ")
        ticker_list = ticker_list.astype(str).split(",")
        # 創建 sinopacDownloader 實例
        downloader = sinopacDownloader(
            api=self.api,
            start_date=self.start_date,
            end_date=self.end_date,
            ticker_list=self.ticker_list,
        )
        # 使用 downloader 獲取數據
        data = downloader.fetch_data(api=self.api)
        return data

    @staticmethod
    def clean_individual_ticker(args):
        tic, df, times = args
        # 筛选特定股票并重设索引
        tic_df = df[df["tic"] == tic].set_index("timestamp")

        # 创建一个新的 DataFrame 以确保所有时间点都被包括
        tmp_df = pd.DataFrame(index=times)
        tmp_df = tmp_df.join(
            tic_df[["Open", "High", "Low", "Close", "Volume", "Amount"]], how="left"
        )

        # 处理 NaN 值，使用前一个可用值填充
        tmp_df.fillna(method="ffill", inplace=True)

        # 附加股票代码和日期
        tmp_df["tic"] = tic
        tmp_df["date"] = tmp_df.index.strftime("%Y-%m-%d")

        return tmp_df

    def clean_data(self, df):
        print("Data cleaning started")
        tic_list = df["tic"].unique()
        n_tickers = len(tic_list)

        # 生成全时间序列
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        times = pd.date_range(start=start, end=end, freq="T")  # 'T' 代表分钟级别的频率

        # 处理每个股票的数据
        results = []
        for tic in tic_list:
            cleaned_data = self.clean_individual_ticker((tic, df, times))
            results.append(cleaned_data)

        # 合并结果
        new_df = pd.concat(results)

        print("Data cleaning finished!")
        return new_df.reset_index(drop=True)

    def add_technical_indicator(self, df):
        print("Started adding Indicators")
        tech_indicator_list = talib.get_functions()  # 获取所有 TA-Lib 可用指标

        # 调整列名以匹配 TA-Lib 的需求
        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # 循环添加每个指标
        for indicator in tech_indicator_list:
            try:
                # 获取指标函数
                indicator_function = getattr(talib.abstract, indicator)
                # 计算指标
                result = indicator_function(df)

                # 如果结果是 Series，转换为 DataFrame 并重命名列
                if isinstance(result, pd.Series):
                    df[indicator.lower()] = result
                else:  # 如果结果是 DataFrame，合并所有列
                    result.columns = [
                        f"{indicator.lower()}_{col}" for col in result.columns
                    ]
                    df = pd.concat([df, result], axis=1)
            except Exception as e:
                print(f"Error calculating {indicator}: {str(e)}")
        print(df.head())
        print(df.shape())
        print(df.tail())
        print("Finished adding Indicators")
        return df

    # Allows to multithread the add_vix function for quicker execution
    def download_and_clean_data(self):
        # VIX_index start at 2023-04-12
        vix_kbars = self.api.kbars(
            contract=self.api.Contracts.Indexs.TAIFEX["TAIFEXTAIWANVIX"],
            start=self.start,
            end=self.end,
        )
        vix_df = pd.DataFrame({**vix_kbars})
        vix_df.ts = pd.to_datetime(vix_df.ts)
        return self.clean_data(vix_df)

    def add_vix(self, data):
        cleaned_vix = self.download_and_clean_data()
        vix = cleaned_vix[["ts", "close"]]
        vix = vix.rename(columns={"ts": "timestamp", "close": "VIXY"})

        data = data.copy()
        data = data.merge(vix, on="timestamp")
        data = data.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        print("Finished adding VIX data")
        return data

    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
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
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )

        # print("turbulence_index\n", turbulence_index)

        return turbulence_index

    def add_turbulence(self, data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
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
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        xtai = tc.get_calendar("XTAI")
        df = xtai.sessions_in_range(
            pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    def on_tick(self, exchange: Exchange, tick: TickSTKv1):
        tick_data = {
            "timestamp": tick.datetime,
            "tic": tick.code,
            "open": float(tick.open),
            "high": float(tick.high),
            "low": float(tick.low),
            "close": float(tick.close),
            "volume": tick.volume,
        }
        self.data = self.data.append(tick_data, ignore_index=True)

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            contract = self.api.Contracts.Stocks[tic]
            self.api.quote.subscribe(
                contract,
                quote_type=sj.constant.QuoteType.Tick,
                version=sj.constant.QuoteVersion.v1,
            )

        def resample_to_kbars(group):
            group.set_index("timestamp", inplace=True)
            ohlc_dict = {"price": "ohlc", "volume": "sum"}
            kbars = group.resample("1T").apply(ohlc_dict)
            kbars.columns = ["open", "high", "low", "close", "volume"]
            return kbars

        kbars_data = []
        for tic in ticker_list:
            tic_data = self.data[self.data.tic == tic]
            kbars = resample_to_kbars(tic_data)
            kbars["tic"] = tic
            kbars_data.append(kbars)

        self.data = pd.concat(kbars_data).reset_index()
        self.data = self.data.sort_values(["timestamp", "tic"]).reset_index(drop=True)

        df = self.add_technical_indicator(self.data, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.kbars(
            contract=self.api.Contracts.Indexs.TAIFEX["TAIFEXTAIWANVIX"],
            start=self.end_date,
            end=self.end_date,
        )
        latest_turb = pd.DataFrame({**turb_df})["close"].values
        return latest_price, latest_tech, latest_turb
