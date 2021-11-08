import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pytz
import trading_calendars as tc
from stockstats import StockDataFrame as Sdf
from finrl.neo_finrl.data_processors.basic_processor import BasicProcessor

class AlpacaProcessor(BasicProcessor):
    def __init__(self, API_KEY=None, API_SECRET=None, APCA_API_BASE_URL=None, api=None):
        if api is None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, "v2")
            except BaseException:
                raise ValueError("Wrong Account Info!")
        else:
            self.api = api

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        NY = "America/New_York"
        start_date = pd.Timestamp(start_date, tz=NY)
        end_date = pd.Timestamp(end_date, tz=NY) + pd.Timedelta(days=1)
        date = start_date
        data_df = pd.DataFrame()
        while date != end_date:
            start_time = (date + pd.Timedelta("09:30:00")).isoformat()
            end_time = (date + pd.Timedelta("15:59:00")).isoformat()
            for tic in ticker_list:
                barset = self.api.get_barset(
                    [tic], time_interval, start=start_time, end=end_time, limit=500
                ).df[tic]
                barset["tic"] = tic
                barset = barset.reset_index()
                data_df = data_df.append(barset)
            print(("Data before ") + end_time + " is successfully fetched")
            date = date + pd.Timedelta(days=1)
            if date.isoformat()[-14:-6] == "01:00:00":
                date = date - pd.Timedelta("01:00:00")
            elif date.isoformat()[-14:-6] == "23:00:00":
                date = date + pd.Timedelta("01:00:00")
            if date.isoformat()[-14:-6] != "00:00:00":
                raise ValueError("Timezone Error")
        """times = data_df['time'].values
        for i in range(len(times)):
            times[i] = str(times[i])
        data_df['time'] = times"""
        return data_df

    def clean_data(self, df):
        tic_list = np.unique(df.tic.values)

        trading_days = self.get_trading_days(start=self.start, end=self.end)

        times = []
        for day in trading_days:
            NY = "America/New_York"
            current_time = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)

        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        print("Data clean finished!")

        return new_df


    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    def fetch_latest_data(
        self, ticker_list, time_interval, tech_indicator_list, limit=100
    ) -> pd.DataFrame:

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_barset([tic], time_interval, limit=limit).df[tic]
            barset["tic"] = tic
            barset = barset.reset_index()
            data_df = data_df.append(barset)

        data_df = data_df.reset_index(drop=True)
        start_time = data_df.time.min()
        end_time = data_df.time.max()
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
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
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

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_barset(["VIXY"], time_interval, limit=1).df["VIXY"]
        latest_turb = turb_df["close"].values
        return latest_price, latest_tech, latest_turb
