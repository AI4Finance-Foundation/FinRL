from __future__ import annotations

import datetime
import os
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
import requests
from stockstats import StockDataFrame as Sdf


class EodhdProcessor:
    def __init__(self, csv_folder="./"):
        self.csv_folder = csv_folder
        pass

    #
    def download(self, api_token, dl_vix=True):
        """Fetches data from EODHD API
        Parameters
        ----------
        api_token: token from the EODHD api  (All-in-one plan)
        data_path: path where to save the csv files (of each ticker)

        Returns
        -------
        none
        """

        API_TOKEN = api_token  # Replace with your API token

        # Step 1: Get NASDAQ 100 components
        nasdaq_100_ticker_url = f"https://eodhd.com/api/fundamentals/NDX.INDX?api_token={API_TOKEN}&fmt=json&filter=Components"

        response = requests.get(nasdaq_100_ticker_url).json()

        # Extract tickers
        tickers = [data["Code"] for data in response.values()]

        print(f"{len(tickers)} tickers data to be downloaded")

        # Step 2: Fetch historical minute data for each ticker
        start_date = datetime.datetime(
            2016, 1, 1
        )  # Earliest available data for NASDAQ at EODHD
        # end_date = datetime.datetime.now()  # Today
        end_date = datetime.datetime(2016, 1, 5)  #
        interval = "1m"  # 1-minute interval

        if dl_vix:
            tickers.append("VIX")

        for ticker in tickers:

            all_ticker_data = []
            print(f"Fetching data for {ticker}...")

            start_timestamp = int(time.mktime(start_date.timetuple()))
            end_timestamp = int(time.mktime(end_date.timetuple()))

            while start_timestamp < end_timestamp:
                next_timestamp = start_timestamp + (
                    120 * 24 * 60 * 60
                )  # 120 days max per request
                if next_timestamp > end_timestamp:
                    next_timestamp = end_timestamp

                if ticker == VIX:
                    url = f"https://eodhd.com/api/intraday/VIX.INDX?interval={interval}&from={start_timestamp}&to={next_timestamp}&api_token={API_TOKEN}&fmt=json"
                else:
                    url = f"https://eodhd.com/api/intraday/{ticker}.US?interval={interval}&api_token={API_TOKEN}&fmt=json&from={start_timestamp}&to={next_timestamp}"

                response = requests.get(url)

                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data)
                        df["ticker"] = ticker  # Add ticker column
                        all_ticker_data.append(df)
                else:
                    print(f"Error fetching data for {ticker}: {response.text}")

                # Move to next 120-day period
                start_timestamp = next_timestamp
                time.sleep(1)  # Respect API rate limits

            # Step 3: Save the data
            if all_ticker_data:

                final_df = pd.concat(all_ticker_data)
                final_df.to_csv(
                    self.csv_folder + "nasdaq_100_minute_data_" + ticker + ".csv",
                    index=False,
                )
                print("Data saved to nasdaq_100_minute_data_" + ticker + ".csv")

            else:
                print("No data retrieved.")

        return

    def add_day_column(self):
        """add a Day column to all csv in csv_folder
        Parameters
        ----------

        Returns
        -------
        the max number of days
        """

        # Step 1: First pass to collect all unique dates
        all_dates = []

        max_days = 0

        for filename in os.listdir(self.csv_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.csv_folder, filename)
                df = pd.read_csv(file_path)

                if "datetime" in df.columns:
                    converted = pd.to_datetime(df["datetime"], errors="coerce")
                    dates = converted.dt.date.dropna().tolist()
                    all_dates.extend(dates)

        # Sort and create date index dictionary
        unique_dates = sorted(set(all_dates))
        date_index_dict = {str(date): idx for idx, date in enumerate(unique_dates)}

        max_days = len(date_index_dict)

        print("Date index dictionary created.\n")

        # Step 2: Second pass to update each file with the 'Day' column
        for filename in os.listdir(self.csv_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.csv_folder, filename)
                df = pd.read_csv(file_path)

                if "datetime" in df.columns:
                    # Convert datetime column to datetime objects
                    converted = pd.to_datetime(df["datetime"], errors="coerce")

                    # Map to day indices
                    day_indices = converted.dt.date.map(
                        lambda d: date_index_dict.get(str(d), None)
                    )

                    # Add the Day column
                    df["Day"] = day_indices

                    # Save the updated file back
                    df.to_csv(file_path, index=False)

                    print(f"Saved updated file: {filename}")

        return max_days

    def tics_in_more_than_90perc_days(self, max_days):

        sup_to_90 = 0
        tics_present_in_more_than_90_per = []

        for filename in os.listdir(self.csv_folder):

            print(f"self.csv_folder {self.csv_folder}")
            print(f"filename {filename}")

            if filename.endswith(".csv"):
                file_path = os.path.join(self.csv_folder, filename)
                try:
                    df = pd.read_csv(file_path)

                    if "Day" not in df.columns:
                        print(f"{filename}: 'Day' column not found.")
                        continue

                    # Drop NA values and ensure the column is integer type
                    unique_days = (
                        pd.to_numeric(df["Day"], errors="coerce")
                        .dropna()
                        .astype(int)
                        .nunique()
                    )
                    percentage = (unique_days / max_days) * 100

                    if percentage > 90:
                        sup_to_90 += 1
                        tics_present_in_more_than_90_per.append(str(df["ticker"][0]))

                    print(f"{filename}: {unique_days} unique days ({percentage:.2f}%)")

                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")

        return tics_present_in_more_than_90_per

    def nber_present_tics_per_day(self, max_days, tics_in_more_than_90perc_days):

        dico_ints = {i: 0 for i in range(max_days + 1)}
        counter = 0

        for filename in os.listdir(self.csv_folder):
            if filename.endswith(".csv"):
                # print(counter)
                # print(self.csv_folder)
                # print(filename)

                file_path = os.path.join(self.csv_folder, filename)
                try:
                    # print("in try 0")
                    df = pd.read_csv(file_path)
                    # print("in try 1")
                    if str(df["ticker"][0]) in tics_in_more_than_90perc_days:
                        # print("in try 2")
                        if "Day" not in df.columns:
                            # print("in try 3")
                            print(f"{filename}: 'Day' column not found.")
                            continue
                        else:
                            # print("in try 4")
                            unique_days = set(df["Day"].tolist())

                            for num in unique_days:
                                dico_ints[num] += 1

                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")
                    exit()

                counter += 1

        return dico_ints

    def process_after_dl(self):

        # add a day column
        max_days = self.add_day_column()

        # find the tics that are present in more than 90% of the days
        tics_in_more_than_90perc_days = self.tics_in_more_than_90perc_days(max_days)

        # print(tics_in_more_than_90perc_days) # ['WDAY', 'ADP', 'XEL', 'VRTX', 'AAPL', 'VRSK', 'ADBE', 'ADI']

        # create the dict of days (keys) and number of present tics (values)
        dico_ints = self.nber_present_tics_per_day(
            max_days, tics_in_more_than_90perc_days
        )

        # for each key (day) if the number of present tics is = tics_in_90%, add the Day id to days_to_keep
        num_of_good_ticks = len(tics_in_more_than_90perc_days)
        days_to_keep = []
        for k, v in dico_ints.items():
            if v == num_of_good_ticks:
                days_to_keep.append(k)

        # loop over each tic CSV and remove non wished days
        df_list = []
        for filename in os.listdir(self.csv_folder):

            if filename.endswith(".csv"):
                print(f"removed uncomplete days from {filename}")
                file_path = os.path.join(self.csv_folder, filename)
                try:
                    df = pd.read_csv(file_path)
                    if df["ticker"].iloc[0] in tics_in_more_than_90perc_days:
                        filtered_df = df[df["Day"].isin(days_to_keep)]
                        df_list.append(filtered_df.sort_values(by="Day"))
                    # if str(df["ticker"][0]) in tics_present_in_more_than_90_per:
                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")

        df.loc[df["ticker"] != "VIX", "volume"] = df.loc[
            df["ticker"] != "VIX", "volume"
        ].astype(int)

        df = pd.concat(df_list, ignore_index=True)

        # Reset the Days integers
        unique_days = df["Day"].unique()
        mapping = {old: new for new, old in enumerate(sorted(unique_days))}
        df["Day"] = df["Day"].map(mapping)

        return df

    def clean_data(self, df, min_24=True):

        df.rename(columns={"ticker": "tic"}, inplace=True)
        df.rename(columns={"datetime": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])

        df = df[["time", "open", "high", "low", "close", "volume", "tic", "Day"]]

        # remove 16:00 data
        df.drop(df[df["time"].astype(str).str.endswith("16:00:00")].index, inplace=True)

        df.sort_values(by=["tic", "time"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        tics = df["tic"].unique()
        days = df["Day"].unique()

        start_time = df["time"].min()
        end_time = df["time"].max()
        start_time = df["time"].min().replace(hour=0, minute=0, second=0)
        end_time = df["time"].max().replace(hour=23, minute=59, second=0)
        time_range = pd.date_range(
            start=start_time, end=end_time, freq="min"
        )  # 'T' is minute frequency
        minute_df = pd.DataFrame({"time": time_range})

        # ADDING MISSING ROWS
        for tic in tics:

            print(f"Adding Missing Rows for tic {tic}")

            for day in days:

                # 0) Create the sub df of the missing times

                times_for_this_tic_and_day = df.loc[
                    (df["Day"] == day) & (df["tic"] == tic), "time"
                ]
                times_for_this_tic_and_day = pd.to_datetime(times_for_this_tic_and_day)

                if min_24:
                    specific_day = (
                        df.loc[(df["Day"] == day) & (df["tic"] == tic), "time"]
                        .iloc[0]
                        .date()
                    )
                    filtered_minute_df = minute_df[
                        minute_df["time"].dt.date == pd.to_datetime(specific_day).date()
                    ]
                    filtered_minute_df["time"] = pd.to_datetime(
                        filtered_minute_df["time"]
                    )
                    missing_times = filtered_minute_df[
                        ~filtered_minute_df["time"].isin(times_for_this_tic_and_day)
                    ]
                else:
                    # all times across all tics, for the day
                    # Filter the DataFrame for the given Day (e.g., day_value = 3)
                    existing_time_values = df[df["Day"] == day]["time"].unique()

                    existing_time_df = pd.DataFrame(
                        {"time": pd.to_datetime(existing_time_values)}
                    )
                    missing_times = existing_time_df[
                        ~existing_time_df["time"].isin(times_for_this_tic_and_day)
                    ]

                missing_times["open"] = np.nan  # float64
                missing_times["high"] = np.nan  # float64
                missing_times["low"] = np.nan  # float64
                missing_times["close"] = np.nan  # float64
                missing_times["volume"] = np.nan  # float64
                missing_times["tic"] = tic  # object (empty string is still an object)
                missing_times["Day"] = day  # int64
                missing_times = missing_times.astype(
                    {
                        "open": "float64",
                        "high": "float64",
                        "low": "float64",
                        "close": "float64",
                        "volume": "float64",
                        "tic": "object",
                        "Day": "int64",
                    }
                )

                # 1) Add the sub df (missing_times) to df
                # Example: insert after the last row where Day = 2 and tic = "AAPL"
                mask = (df["Day"] == day) & (df["tic"] == tic)
                insert_index = df[mask].index.max()

                # Split df_orig and insert df in between
                df_before = df.iloc[: insert_index + 1]
                df_after = df.iloc[insert_index + 1 :]

                df = pd.concat([df_before, missing_times, df_after], ignore_index=True)

                df.sort_values(by=["tic", "time"], inplace=True)
                df.reset_index(drop=True, inplace=True)

        # Replace all 0 volume with a Nan  (to allow for ffill and bfill to work)
        df.loc[df["volume"] == 0, "volume"] = np.nan

        ## FILLING THE MISSING ROWS
        for tic in tics:

            print(f"Filling Missing Rows for tic {tic}")

            cols_to_ffill = ["close", "open", "high", "low", "volume"]
            df.loc[df["tic"] == tic, cols_to_ffill] = (
                df.loc[df["tic"] == tic, cols_to_ffill].ffill().bfill()
            )

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
            print(f"doing indicator {indicator}")
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
                # indicator_df = indicator_df.append(temp_indicator, ignore_index=True)
                indicator_df = pd.concat(
                    [indicator_df, temp_indicator], axis=0, ignore_index=True
                )

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
