from __future__ import annotations

import numpy as np
import pandas as pd
import logbook

import alpaca_trade_api as tradeapi
import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz

from futu import *
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from stockstats import StockDataFrame as Sdf
from finrl.meta.data_processors.schemas import DownloadDataSchema

class FutuProcessor:
    def __init__(self, host='futu-opend', port=11111, rsa_file='futu.pem', exchange = 'XHKG'):
        
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file( rsa_file)
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        self.logger = logbook.Logger(type(self).__name__)
        self.tz = "America/New_York"

        self.exchange = exchange

        self.if_vix = False

    # 接口限制

    # 分 K 提供最近 8 年数据，日 K 及以上提供最近 10 年的数据。
    # 我们会根据您账户的资产和交易的情况，下发历史 K 线额度。因此，30 天内您只能获取有限只股票的历史 K 线数据。具体规则参见 订阅额度 & 历史 K 线额度。您当日消耗的历史 K 线额度，会在 30 天后自动释放。
    # 每 30 秒内最多请求 60 次历史 K 线接口。注意：如果您是分页获取数据，此限频规则仅适用于每只股票的首页，后续页请求不受限频规则的限制。
    # 换手率，仅提供日 K 及以上级别。
    # 期权，仅提供日K, 1分K，5分K，15分K，60分K。
    # 美股 盘前和盘后 K 线，仅支持 60 分钟及以下级别。由于美股盘前和盘后时段为非常规交易时段，此时段的 K 线数据可能不足 2 年。
    # 美股的 成交额，仅提供 2015-10-12 之后的数据。
    def _fetch_data_for_ticker(self, ticker, start_date, end_date, time_interval):
        self.logger.info(f"Fetching data for {ticker}")
        max_count = 1000
        all_data = pd.DataFrame()
        ret, data, page_req_key = self.quote_ctx.request_history_kline(ticker, start=start_date, end=end_date, max_count=max_count, ktype=KLType.K_1M) # 5 per page, request the first page
        if ret == RET_OK:
            all_data = pd.concat([all_data, data], ignore_index=True)
        else:
            raise ValueError(f"Error fetching data for {ticker}: {data}")

        while page_req_key != None: # Request all results after
            ret, data, page_req_key = self.quote_ctx.request_history_kline(ticker, start=start_date, end=end_date, max_count=max_count ,page_req_key=page_req_key, ktype=KLType.K_1M) # Request the page after turning data
            if ret == RET_OK:
                all_data = pd.concat([all_data, data], ignore_index=True)
            else:
                raise ValueError(f"Error fetching data for {ticker}: {data}")

        self.logger.info('All pages are finished!')
        
        all_data = all_data.rename(columns={'time_key': 'timestamp', 'code': 'tic'})
        all_data = all_data.filter([
            'tic',
            'timestamp',
            'open',
            'close',
            'high',
            'low',
            'volume'
        ])
        all_data["timestamp"] = pd.to_datetime(all_data["timestamp"])
        all_data['timestamp'] = all_data['timestamp'].dt.tz_localize('UTC').dt.tz_convert( self.tz)
        all_data.reset_index(drop=True)

        
        
        self.logger.info(f"Data fetched for {ticker}")
        return DownloadDataSchema.validate(all_data)
        


    def _fetch_latest_data_for_ticker ( self, ticker, time_interval, limit = 100, tech_indicator_list = None):
        if tech_indicator_list is None:
            tech_indicator_list = []
        self.logger.info(f"Fetching latest data for {ticker} limit {limit} time_interval {time_interval} tech_indicator_list {tech_indicator_list}")
        ret, data = self.quote_ctx.get_cur_kline( ticker, limit, KLType.K_1M, AuType.QFQ)  # 获取港股00700最近2个 K 线数据
        if ret == RET_OK:
            data = data.rename(columns={'time_key': 'timestamp', 'code': 'tic'})
            data = data.filter([
                'tic',
                'timestamp',
                'open',
                'close',
                'high',
                'low',
                'volume'
            ])

            # print ( data)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data['timestamp'] = data['timestamp'].dt.tz_localize('UTC').dt.tz_convert( self.tz)
            data.reset_index(drop=True)
            data = self.add_technical_indicator(data, tech_indicator_list)

            data = self.add_vix(data) if self.if_vix else self.add_turbulence(data)
            return DownloadDataSchema.validate(data)
        else:
            raise ValueError(f"Error fetching data for {ticker}: {data}")

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # data = pd.DataFrame()
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    self._fetch_data_for_ticker,
                    ticker,
                    start_date,
                    end_date,
                    time_interval,
                )
                for ticker in ticker_list
            ]
            data_list = [future.result() for future in futures]
        
        # Combine the data
        data_df = pd.concat(data_list, axis=0)
        data_df.set_index("timestamp", inplace=True)

        # Convert the timezone
        data_df = data_df.tz_convert( self.tz)
        

        # If time_interval is less than a day, filter out the times outside of NYSE trading hours
        if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            data_df = data_df.between_time("09:30", "15:59")


        
        # Reset the index and rename the columns for consistency
        data_df = data_df.reset_index().rename(
            columns={"index": "timestamp", "symbol": "tic"}
        )
        
        # Sort the data by both timestamp and tic for consistent ordering
        data_df = data_df.sort_values(by=["tic", "timestamp"])
        
        # Reset the index and drop the old index column
        data_df = data_df.reset_index(drop=True)

        
        validated_df = DownloadDataSchema.validate( data_df)
        return data_df

    def download_latest_data ( self, ticker_list, time_interval, limit = 100, tech_indicator_list = None) -> pd.DataFrame:
        self.logger.info ( f'subscribe {ticker_list}')
        if tech_indicator_list is None:
            tech_indicator_list = []

        ret_sub, err_message = self.quote_ctx.subscribe( ticker_list, [SubType.K_DAY, SubType.K_1M], subscribe_push=False)
        # 先订阅 K 线类型。订阅成功后 OpenD 将持续收到服务器的推送，False 代表暂时不需要推送给脚本, [SubType.K_DAY], subscribe_push=False)

        if ret_sub == RET_OK:  # 订阅成功
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        self._fetch_latest_data_for_ticker,
                        ticker,
                        time_interval,
                        limit,
                        tech_indicator_list
                    )
                    for ticker in ticker_list
                ]
                data_list = [future.result() for future in futures]
            
            
            
            # Combine the data
            data_df = pd.concat(data_list, axis=0)
            data_df.set_index("timestamp", inplace=True)
            
            # Convert the timezone
            data_df = data_df.tz_convert( self.tz)
            

            # If time_interval is less than a day, filter out the times outside of NYSE trading hours
            # if pd.Timedelta(time_interval) < pd.Timedelta(days=1):
            #     data_df = data_df.between_time("09:30", "15:59")

            # Reset the index and rename the columns for consistency
            data_df = data_df.reset_index().rename(
                columns={"index": "timestamp", "symbol": "tic"}
            )
            
            # Sort the data by both timestamp and tic for consistent ordering
            data_df = data_df.sort_values(by=["tic", "timestamp"])
            
            # Reset the index and drop the old index column
            # data_df = data_df.reset_index(drop=True)

            return DownloadDataSchema.validate( data_df)
        else:
            raise ValueError(f"Error subscribing to {ticker_list}: {err_message}")

        


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Data cleaning started")

        df.sort_values(["timestamp", "tic"], inplace=True)
        tic_list = np.unique(df.tic.values)
        n_tickers = len(tic_list)

        self.logger.info("align start and end dates")
        
        grouped = df.groupby("timestamp")
        filter_mask = grouped.transform("count")["tic"] >= n_tickers
        df = df[filter_mask]

        # ... (generating 'times' series, same as in your existing code)


        trading_days = self.get_trading_days(start=self.start, end=self.end)

        # produce full timestamp index
        self.logger.info("produce full timestamp index")
        times = []
        for day in trading_days:
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize( self.tz)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)

        self.logger.info("Start processing tickers")

        future_results = []
        for tic in tic_list:
            result = self.clean_individual_ticker((tic, df.copy(), times))
            future_results.append(result)

        self.logger.info("ticker list complete")
        self.logger.info("Start concat and rename")
        new_df = pd.concat(future_results)
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        self.logger.info("Data clean finished!")
        
        return new_df

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
        self.logger.info("Started adding Indicators")

        # Store the original data type of the 'timestamp' column
        original_timestamp_dtype = df["timestamp"].dtype

        # Convert df to stock data format just once
        stock = Sdf.retype(df)
        unique_ticker = stock.tic.unique()

        # Convert timestamp to a consistent datatype (timezone-naive) before entering the loop
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)

        self.logger.info("Running Loop")
        for indicator in tech_indicator_list:
            indicator_dfs = []
            for tic in unique_ticker:
                tic_data = stock[stock.tic == tic]
                indicator_series = tic_data[indicator]

                tic_timestamps = df.loc[df.tic == tic, "timestamp"]

                indicator_df = pd.DataFrame(
                    {
                        "tic": tic,
                        "date": tic_timestamps.values,
                        indicator: indicator_series.values,
                    }
                )
                indicator_dfs.append(indicator_df)

            # Concatenate all intermediate dataframes at once
            indicator_df = pd.concat(indicator_dfs, ignore_index=True)

            # Merge the indicator data frame
            df = df.merge(
                indicator_df[["tic", "date", indicator]],
                left_on=["tic", "timestamp"],
                right_on=["tic", "date"],
                how="left",
            ).drop(columns="date")

        self.logger.info("Restore Timestamps")
        # Restore the original data type of the 'timestamp' column
        if isinstance(original_timestamp_dtype, pd.DatetimeTZDtype):
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            df["timestamp"] = df["timestamp"].dt.tz_convert(original_timestamp_dtype.tz)
        else:
            df["timestamp"] = df["timestamp"].astype(original_timestamp_dtype)

        self.logger.info("Finished adding Indicators")
        return df


    def download_and_clean_data(self):
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        return self.clean_data(vix_df)

    def add_vix(self, data):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(self.download_and_clean_data)
            cleaned_vix = future.result()

        vix = cleaned_vix[["timestamp", "close"]]

        merge_column = "date" if "date" in data.columns else "timestamp"

        vix = vix.rename(
            columns={"timestamp": merge_column, "close": "VIXY"}
        )  # Change column name dynamically

        data = data.copy()
        data = data.merge(
            vix, on=merge_column
        )  # Use the dynamic column name for merging
        data = data.sort_values([merge_column, "tic"]).reset_index(drop=True)

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

        # self.logger.info("turbulence_index\n", turbulence_index)

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
        df = df.sort_values(["timestamp", "tic"])
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
        self.logger.info("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        calendar_exchange = tc.get_calendar( self.exchange)
        df = calendar_exchange.sessions_in_range(
            pd.Timestamp(start).tz_localize(None), pd.Timestamp(end).tz_localize(None)
            # "2022-01-01", "2022-01-13"
        )

        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    @staticmethod
    def clean_individual_ticker(args):
        
        tic, df, times = args
        tmp_df = pd.DataFrame(index=times)
        tic_df = df[df.tic == tic].set_index("timestamp")

        # Step 1: Merging dataframes to avoid loop
        tmp_df = tmp_df.merge(
            tic_df[["open", "high", "low", "close", "volume"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Step 2: Handling NaN values efficiently
        if pd.isna(tmp_df.iloc[0]["close"]):
            first_valid_index = tmp_df["close"].first_valid_index()
            if first_valid_index is not None:
                first_valid_price = tmp_df.loc[first_valid_index, "close"]
                logbook.info(
                    f"The price of the first row for ticker {tic} is NaN. It will be filled with the first valid price."
                )
                tmp_df.iloc[0] = [first_valid_price] * 4 + [0.0]  # Set volume to zero
            else:
                logbook.info(
                    f"Missing data for ticker: {tic}. The prices are all NaN. Fill with 0."
                )
                tmp_df.iloc[0] = [0.0] * 5

        for i in range(1, tmp_df.shape[0]):
            if pd.isna(tmp_df.iloc[i]["close"]):
                previous_close = tmp_df.iloc[i - 1]["close"]
                tmp_df.iloc[i] = [previous_close] * 4 + [0.0]

        # Setting the volume for the market opening timestamp to zero - Not needed
        # tmp_df.loc[tmp_df.index.time == pd.Timestamp("09:30:00").time(), 'volume'] = 0.0

        # Step 3: Data type conversion
        tmp_df = tmp_df.astype(float)

        tmp_df["tic"] = tic
        return tmp_df

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:
        self.logger.info("Fetching latest data")
        data_df = self.download_latest_data( ticker_list, time_interval, limit, tech_indicator_list)

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    self.logger.info(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})
        
        
        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=self.if_vix
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        # turb_df = self.download_latest_data( ["US.VIXY"], time_interval, limit=1)
        # turb_df = self.api.get_bars(["VIXY"], time_interval, limit=1).df
        
        latest_turb = turbulence_array[-1]
        return latest_price, latest_tech, latest_turb

    def close_conn(self):
        self.quote_ctx.close()