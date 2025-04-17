from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz

import requests
import time

class EodhdProcessor:
    def __init__(self, if_offline=False):
        pass


    # 
    def download(self, api_token, data_path="./"):
        """Fetches data from EODHD API
        Parameters
        ----------
        api_token: token from the EODHD api  (All-in-one plan)
        data_path: path where to save the csv files (of each ticker)

        Returns
        -------
        none
        """

        API_TOKEN = api_token # Replace with your API token

        # Step 1: Get NASDAQ 100 components
        nasdaq_100_ticker_url = f"https://eodhd.com/api/fundamentals/NDX.INDX?api_token={API_TOKEN}&fmt=json&filter=Components"

        response = requests.get(nasdaq_100_ticker_url).json()

        # Extract tickers
        tickers = [data["Code"] for data in response.values()]

        print("{} tickers data to be downloaded".format(len(tickers)))

        # Step 2: Fetch historical minute data for each ticker
        start_date = datetime.datetime(2016, 1, 1)  # Earliest available data for NASDAQ at EODHD
        #end_date = datetime.datetime.now()  # Today
        end_date = datetime.datetime(2016, 1, 5)  # 
        interval = "1m"  # 1-minute interval

        for ticker in tockers:


            all_ticker_data = []
            print(f"Fetching data for {ticker}...")

            start_timestamp = int(time.mktime(start_date.timetuple()))
            end_timestamp = int(time.mktime(end_date.timetuple()))

            while start_timestamp < end_timestamp:
                next_timestamp = start_timestamp + (120 * 24 * 60 * 60)  # 120 days max per request
                if next_timestamp > end_timestamp:
                    next_timestamp = end_timestamp
                
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
                final_df.to_csv(data_path+"nasdaq_100_minute_data_"+ticker+".csv", index=False)
                print("Data saved to nasdaq_100_minute_data_"+ticker+".csv")

            else:
                print("No data retrieved.")

        return


    def add_day_column(self, csv_folder="./"):
        """add a Day column to all csv in csv_folder
        Parameters
        ----------
        csv_folder: path where are the csvs file (one for each tic)
        Returns
        -------
        the max number of days
        """

        # Step 1: First pass to collect all unique dates
        all_dates = []

        max_days = 0

        for filename in os.listdir(csv_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_folder, filename)
                df = pd.read_csv(file_path)

                if "datetime" in df.columns:
                    converted = pd.to_datetime(df["datetime"], errors='coerce')
                    dates = converted.dt.date.dropna().tolist()
                    all_dates.extend(dates)

        # Sort and create date index dictionary
        unique_dates = sorted(set(all_dates))
        date_index_dict = {str(date): idx for idx, date in enumerate(unique_dates)}

        max_days = len(date_index_dict)

        print("Date index dictionary created.\n")

        # Step 2: Second pass to update each file with the 'Day' column
        for filename in os.listdir(csv_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(csv_folder, filename)
                df = pd.read_csv(file_path)

                if "datetime" in df.columns:
                    # Convert datetime column to datetime objects
                    converted = pd.to_datetime(df["datetime"], errors='coerce')

                    # Map to day indices
                    day_indices = converted.dt.date.map(lambda d: date_index_dict.get(str(d), None))

                    # Add the Day column
                    df["Day"] = day_indices

                    # Save the updated file back
                    df.to_csv(file_path, index=False)

                    print(f"Saved updated file: {filename}")

        return max_days

    def tics_in_more_than_90perc_days(self, max_days):

        sup_to_90=0
        tics_present_in_more_than_90_per = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)

                    if 'Day' not in df.columns:
                        print(f"{filename}: 'Day' column not found.")
                        continue

                    # Drop NA values and ensure the column is integer type
                    unique_days = pd.to_numeric(df['Day'], errors='coerce').dropna().astype(int).nunique()
                    percentage = (unique_days / max_days) * 100

                    if percentage > 90:
                        sup_to_90+=1
                        tics_present_in_more_than_90_per.append(str(df['ticker'][0])) 

                    print(f"{filename}: {unique_days} unique days ({percentage:.2f}%)")

                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")

        return tics_present_in_more_than_90_per


    def nber_present_tics_per_day(self, max_days, tics_in_more_than_90perc_days, folder_path):

        dico_ints = {i:0 for i in range(max_days+1)}
        counter=0

        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                print(counter)
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    
                    if str(df["ticker"][0]) in tics_present_in_more_than_90_per:
                        
                        if 'Day' not in df.columns:
                            print(f"{filename}: 'Day' column not found.")
                            continue
                        else:
                            unique_days = set(df['Day'].tolist())
            
                            for num in unique_days:
                                dico_ints[num] += 1

                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")

                counter += 1

            
        return dico_ints

    def process_after_dl(self, data_path="./"):


        # add a day column
        max_days = self.add_day_column(csv_folder=data_path)

        # find the tics that are present in more than 90% of the days
        tics_in_more_than_90perc_days = self.tics_in_more_than_90perc_days(max_days)

        # create the dict of days (keys) and number of present tics (values)
        dico_ints = self.nber_present_tics_per_day(max_days, tics_in_more_than_90perc_days, data_path)

        # for each key (day) if the number of present tics is = tics_in_90%, add the Day id to days_to_keep
        num_of_good_ticks = len(tics_present_in_more_than_90_per)
        days_to_keep = []
        for k,v in dico_ints.items():
            if v == num_of_good_ticks:
                days_to_keep.append(k)



        # loop over each tic CSV and remove non wished days
        df_list = []
        for filename in os.listdir(data_path):
            if filename.endswith('.csv'):
                print("removed uncomplete days from {}".format(filename))
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    filtered_df = df[df['Day'].isin(days_to_keep)]
                    df_list.append(filtered_df.sort_values(by='Day'))
                    #if str(df["ticker"][0]) in tics_present_in_more_than_90_per:
                except Exception as e:
                    print(f"{filename}: Error reading file - {e}")


        df = pd.concat(df_list, ignore_index=True)

        return df 


    def clean_data(self, df):

        df.rename(columns={'ticker': 'tic'}, inplace=True)
        df.rename(columns={'datetime': 'time'}, inplace=True)

        df = df[["time", "open", "high", "low", "close", "volume", "tic"]]
        
        # remove 16:00 data
        df.drop(df[df["time"].astype(str).str.endswith("16:00:00")].index, inplace=True)
        df.sort_values(by=["tic", "time"], inplace=True)

        # check missing rows
        #tic_dic[tic][0] tells how many rows the tic have with non zero volume
        #tic_dic[tic][1] tells how many rows the tic have in total
        tic_list = np.unique(df["tic"].values)
        tic_dic = {}
        for tic in tic_list:
            tic_dic[tic] = [0, 0]

        volume_nonzero = df.query("volume != 0")["tic"].value_counts()
        volume_total = df["tic"].value_counts()
        for tic_num, tic in enumerate(tic_dic):
            print("tic num is {}".format(str(tic_num)))
            tic_dic[tic][0] = int(volume_nonzero.get(tic, 0)) # the get function, retrieves from volume_nonzero the value for the tic, if the tic not present, returns 0
            tic_dic[tic][1] = int(volume_total.get(tic, 0))


        # list all tics with nan values
        constant = np.unique(df["time"].values).shape[0]
        nan_tics = []
        for tic in tic_dic:
            if tic_dic[tic][1] != constant: # if the number of rows the tic have in total is different from the total number available times, then we add it as a tic with nans
                nan_tics.append(tic)




        # Fill the Nan values with closest close values

        df.sort_values(by=["tic", "time"], inplace=True)

        tics = df["tic"].unique()

        for tic in tics:

            print("filling Nans in tic {}".format(tic))

            tic_mask = df["tic"] == tic
            tic_indices = df.index[tic_mask]
            
            # Initialize last_val as the first non-NaN value in 'close' for this tic
            first_valid_index = df.loc[tic_indices, "close"].first_valid_index()
            if first_valid_index is not None:
                last_val = df.at[first_valid_index, "close"]
            else:
                # All values are NaN for this tic — skip
                continue

            for i in tic_indices:
                val = df.at[i, "close"]
                if pd.notna(val):
                    last_val = val
                else:
                    #df.at[i, "close"] = last_val
                    df.iloc[i, 1] = last_val
                    df.iloc[i, 2] = last_val
                    df.iloc[i, 3] = last_val
                    df.iloc[i, 4] = last_val
                    
        # return a clean df without nans
        return df



    # def add_technical_indicator(
    #     self,
    #     df,
    #     tech_indicator_list=[
    #         "macd",
    #         "boll_ub",
    #         "boll_lb",
    #         "rsi_30",
    #         "dx_30",
    #         "close_30_sma",
    #         "close_60_sma",
    #     ],
    # ):


    # def calculate_turbulence(self, data, time_period=252):


    # def add_turbulence(self, data, time_period=252):
    #     """
    #     add turbulence index from a precalcualted dataframe
    #     :param data: (df) pandas dataframe
    #     :return: (df) pandas dataframe
    #     """