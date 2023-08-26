from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from finrl import config
from finrl import config_tickers
from finrl.config import DATA_SET
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    # indicator_df = indicator_df.append(
                    #     temp_indicator, ignore_index=True
                    # )
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        advancing_data, declining_data = self.calculate_advancing_declining(df)
        mco = self.calculate_mcclellan_oscillator(advancing_data, declining_data)
        df = df.merge(mco.to_frame(), left_on="date", right_index=True, how="left")
        df.rename(columns={0: "mco"}, inplace=True)
        df["mco"].replace(np.nan, 0, inplace=True)
        df = self.calculate_hindenburg_omen(df)
        df["mco"] = df.apply(self.transform_row, axis=1)
        df = self.update_turbulence_vix(df)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def transform_row(self, row):
        if row["mco"] < 0:
            if row["hindenburg"]:
                factor = 10000
                return abs(row["mco"]) * factor
            else:
                factor = 1000
                return abs(row["mco"]) * factor
        else:
            return -(row["mco"] * 1000)

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def update_turbulence_vix(self, df):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe

        Parameters
        ----------
        df
        """
        # df= df['turbulence', 'mco'].sum(axis = 1, skipna = True)
        df["turbulence"] = df["turbulence"] + df["mco"]
        df["vix"] = df["vix"] + df["mco"]
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

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
        try:
            turbulence_index = pd.DataFrame(
                {"date": df_price_pivot.index, "turbulence": turbulence_index}
            )
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index

    def calculate_hindenburg_omen(self, df_raw):
        """
        Calculates the Hindenburg Omen signal for a given DataFrame of market data.

        Args:
            df (pd.DataFrame): DataFrame containing historical market data.
            threshold (float): Threshold value for the number of new highs and new lows.

        Returns:
            bool: True if the Hindenburg Omen signal is triggered, False otherwise.
        """
        TRAIN_START_DATE = config.TRAIN_START_DATE
        TRADE_END_DATE = config.TRADE_END_DATE
        # df_raw_s_p = self.get_sp_raw() )# VIXTRIAL
        if DATA_SET == "nasdaq":
            stock_ticker = ["^NDX"]
        elif DATA_SET == "dow":
            stock_ticker = ["^DJI"]
        else:
            stock_ticker = ["^GSPC"]
        df_raw["date1"] = pd.to_datetime(df_raw["date"])
        df_raw.sort_values(by=["tic", "date1"], inplace=True)
        df_raw["rollinghigh"] = (
            df_raw.groupby("tic")["high"]
            .rolling(365)
            .max()
            .reset_index(level=0, drop=True)
        )
        df_raw["rollinglow"] = (
            df_raw.groupby("tic")["low"]
            .rolling(365)
            .min()
            .reset_index(level=0, drop=True)
        )
        df_raw["52_week_high"] = (df_raw["high"] >= df_raw["rollinghigh"]).astype(int)
        df_raw["52_week_low"] = (df_raw["low"] <= df_raw["rollinglow"]).astype(int)
        df_raw["52_week_high_tic"] = df_raw.groupby("date")["52_week_high"].transform(
            "sum"
        )
        df_raw["52_week_low_tic"] = df_raw.groupby("date")["52_week_low"].transform(
            "sum"
        )
        df_raw["threshold_high"] = (
            df_raw["52_week_high_tic"] / df_raw["tic"].nunique() * 100
        )
        df_raw["threshold_low"] = (
            df_raw["52_week_low_tic"] / df_raw["tic"].nunique() * 100
        )
        stock_index_raw = YahooDownloader(
            start_date=TRAIN_START_DATE,  # dji= ^DJI or nasdaq= ^NDX
            end_date=TRADE_END_DATE,
            ticker_list=stock_ticker,
        ).fetch_data()  # VIXTRIAL
        # stock_index_raw = YahooDownloader(start_date=TRAIN_START_DATE,  # dji= ^DJI or nasdaq= ^NDX
        #                                   end_date=TRADE_END_DATE,
        #                                   ticker_list=['^GSPC']).fetch_data()
        stock_index_raw["week10_ma"] = (
            stock_index_raw["close"].rolling(window=10 * 5).mean()
        )
        # %%
        stock_index_raw["trend"] = (
            stock_index_raw["close"] > stock_index_raw["week10_ma"]
        )
        # %%
        stock_index_raw["date1"] = pd.to_datetime(stock_index_raw["date"])
        # %%
        final_raw = pd.merge(
            df_raw, stock_index_raw[["trend", "date1"]], on="date1", how="left"
        )
        # %%
        final_raw["hindenburg"] = (
            (final_raw["threshold_high"] > 2.2)
            & (final_raw["threshold_low"] > 2.2)
            & (final_raw["52_week_high_tic"] <= (2 * final_raw["52_week_low_tic"]))
            & (final_raw["52_week_low_tic"] != 0)
            & (final_raw["52_week_high_tic"] != 0)
            & (final_raw["trend"] == True)
            & (final_raw["mco"] < 0)
        )
        final_raw = final_raw.drop(
            [
                "rollinghigh",
                "rollinglow",
                "52_week_high",
                "52_week_low",
                "52_week_high_tic",
                "52_week_low_tic",
                "threshold_high",
                "threshold_low",
                "date1",
            ],
            axis=1,
        )
        return final_raw

    def calculate_advancing_declining(self, data):
        """
        Calculates the number of advancing and declining issues for each period in a DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing historical market data.

        Returns:
            pd.Series: Series containing the number of advancing issues for each period.
            pd.Series: Series containing the number of declining issues for each period.
        """
        # Calculate the number of advancing issues for each period
        data["advancing"] = data["close"] > data["close"].shift(1)

        # Calculate the number of declining issues for each period
        data["declining"] = data["close"] < data["close"].shift(1)

        advancing = data.query("advancing == True").groupby(["date"]).sum()
        declining = data.query("declining == True").groupby(["date"]).sum()
        return advancing, declining

    def calculate_mcclellan_oscillator(
        self, advancing, declining, ema_short=19, ema_long=39
    ):
        """
        Calculates the McClellan Oscillator (MCO) for a given set of advancing and declining issues.

        Args:
            advancing (pd.Series or list): Series or list containing advancing issues for each period.
            declining (pd.Series or list): Series or list containing declining issues for each period.
            ema_short (int): Number of periods for the short-term EMA.
            ema_long (int): Number of periods for the long-term EMA.

        Returns:
            pd.Series: Series containing the McClellan Oscillator values.

        Parameters
        ----------
        declining
        advancing
        data
        """
        advancing = pd.Series(advancing["advancing"]).astype(float)
        declining = pd.Series(declining["declining"]).astype(float)

        ratio_change = (advancing - declining) / (advancing + declining)

        day_19_sma = ratio_change.rolling(window=ema_short).mean()
        day_39_sma = ratio_change.rolling(window=ema_long).mean()
        day_19_ema = pd.Series()
        day_39_ema = pd.Series()
        first_calc = True
        for idx, x in enumerate(day_19_sma):
            index = day_19_sma.index[idx]

            if math.isnan(x):
                day_19_ema = pd.concat([day_19_ema, pd.Series([x], index=[index])])
            else:
                if first_calc:
                    ema = ((ratio_change[idx] - x) * 0.10) + x
                    first_calc = False
                else:
                    ema = (
                        (ratio_change[idx] - day_19_ema.iloc[idx - 1]) * 0.10
                    ) + day_19_ema.iloc[idx - 1]
                day_19_ema = pd.concat([day_19_ema, pd.Series([ema], index=[index])])

        first_calc = True
        for idx, x in enumerate(day_39_sma):
            index = day_39_sma.index[idx]

            if math.isnan(x):
                day_39_ema = pd.concat([day_39_ema, pd.Series([x], index=[index])])
            else:
                if first_calc:
                    ema = ((ratio_change[idx] - x) * 0.05) + x
                    first_calc = False
                else:
                    ema = (
                        (ratio_change[idx] - day_39_ema.iloc[idx - 1]) * 0.05
                    ) + day_39_ema.iloc[idx - 1]
                day_39_ema = pd.concat([day_39_ema, pd.Series([ema], index=[index])])
        # day_39_ema = (ratio_change - day_39_sma.shift(1)) * 0.05 + day_39_sma.shift(1)
        mco = day_19_ema - day_39_ema

        return mco

    def calculate_mcclellan_oscillator_sp(self, df_raw_s_p):
        advancing_data, declining_data = self.calculate_advancing_declining(df_raw_s_p)
        mco = self.calculate_mcclellan_oscillator(advancing_data, declining_data)
        return mco
