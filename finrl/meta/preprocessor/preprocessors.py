from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

# from stockstats import StockDataFrame as Sdf # No longer needed if fully using pandas-ta

try:
    from finrl import config
except ImportError:
    print("Warning: finrl.config not found. Using example defaults for INDICATORS.")

    class ConfigFallback:
        INDICATORS = [
            "macd",
            "rsi_30",
            "cci_30",
            "adx",
        ]  # Match your paper trading script
        # Add other config defaults if used elsewhere in this file

    config = ConfigFallback()

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import pandas_ta as ta  # Crucial import for the .ta accessor


def load_dataset(*, file_name: str) -> pd.DataFrame:
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    if "date" in data.columns:
        data.index = data["date"].factorize()[0]
    elif target_date_col in data.columns and target_date_col != "date":
        data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)
    return time


class FeatureEngineer:
    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=None,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = (
            tech_indicator_list
            if tech_indicator_list is not None
            else config.INDICATORS
        )
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
        df_processed = df.copy()
        df_processed = self.clean_data(df_processed)
        if df_processed.empty:
            print("Warning: DataFrame empty after clean_data().")
            return df_processed

        if self.use_technical_indicator:
            df_processed = self.add_technical_indicator(df_processed)
            print("Successfully added technical indicators")

        if self.use_vix:
            df_processed = self.add_vix(df_processed)
            print("Successfully added vix")

        if self.use_turbulence:
            df_processed = self.add_turbulence(df_processed)
            print("Successfully added turbulence index")

        if self.user_defined_feature:
            df_processed = self.add_user_defined_feature(df_processed)
            print("Successfully added user defined features")

        df_processed = df_processed.ffill().bfill()
        return df_processed

    def clean_data(self, data):
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)

        if "date" in df.columns:
            df.index = df.date.factorize()[0]
        else:
            print(
                "Warning: 'date' column not found in clean_data. Index not changed by factorize."
            )

        try:
            merged_closes = df.pivot_table(index="date", columns="tic", values="close")
            merged_closes = merged_closes.dropna(axis=1, how="all")
            if merged_closes.empty:
                print("Warning: merged_closes is empty in clean_data.")
                return pd.DataFrame(columns=df.columns)

            tics = merged_closes.columns
            df = df[df.tic.isin(tics)].reset_index(drop=True)
        except Exception as e:
            print(
                f"Error during pivoting/cleaning in clean_data: {e}. Returning original df sorted."
            )
            df = data.copy().sort_values(["date", "tic"], ignore_index=True)
            if "date" in df.columns:
                df.index = df.date.factorize()[0]
        return df

    def add_technical_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using pandas_ta.
        :param data: (df) pandas dataframe with 'date', 'tic', 'open', 'high', 'low', 'close', 'volume'
        :return: (df) pandas dataframe with added indicator columns
        """
        df = data.copy()
        if df.empty:
            print("Input DataFrame to add_technical_indicator (pandas-ta) is empty.")
            return df

        try:
            df["date_dt"] = pd.to_datetime(df["date"])
        except Exception as e:
            print(
                f"Error converting 'date' column to datetime: {e}. Ensure 'date' is parsable."
            )
            for indicator_name_spec in self.tech_indicator_list:
                if indicator_name_spec not in df.columns:
                    df[indicator_name_spec] = np.nan
            return df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        for col in ohlcv_cols:
            if col not in df.columns:
                print(
                    f"Warning: Essential column '{col}' missing. Cannot calculate TA indicators fully."
                )
                for indicator_name_spec in self.tech_indicator_list:
                    if indicator_name_spec not in df.columns:
                        df[indicator_name_spec] = np.nan
                return df.sort_values(by=["date", "tic"]).reset_index(drop=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=ohlcv_cols, inplace=True)
        if df.empty:
            print(
                "DataFrame became empty after dropping NaNs in OHLCV. Cannot add indicators."
            )
            # Add NaN columns to original `data` if it was passed in, not the modified `df`
            for indicator_name_spec in self.tech_indicator_list:
                if indicator_name_spec not in data.columns:
                    data[indicator_name_spec] = np.nan
            return data.sort_values(by=["date", "tic"]).reset_index(drop=True)

        all_indicator_data_list = []

        for tic_name in df.tic.unique():
            tic_df = df[df.tic == tic_name].copy()
            if tic_df.empty:
                continue

            tic_df.set_index("date_dt", inplace=True)
            tic_df.sort_index(inplace=True)

            if "macd" in self.tech_indicator_list:
                try:
                    macd_results = tic_df.ta.macd(close="close", append=False)
                    if (
                        macd_results is not None
                        and not macd_results.empty
                        and "MACD_12_26_9" in macd_results.columns
                    ):
                        tic_df["macd"] = macd_results["MACD_12_26_9"]
                    else:
                        tic_df["macd"] = np.nan
                except Exception as e:
                    tic_df["macd"] = np.nan
                    print(f"Err macd {tic_name}: {e}")

            if "rsi_30" in self.tech_indicator_list:
                try:
                    rsi_results = tic_df.ta.rsi(length=30, close="close", append=False)
                    if rsi_results is not None and not rsi_results.empty:
                        tic_df["rsi_30"] = rsi_results
                    else:
                        tic_df["rsi_30"] = np.nan
                except Exception as e:
                    tic_df["rsi_30"] = np.nan
                    print(f"Err rsi_30 {tic_name}: {e}")

            if "cci_30" in self.tech_indicator_list:
                try:
                    cci_results = tic_df.ta.cci(
                        length=30, high="high", low="low", close="close", append=False
                    )
                    if cci_results is not None and not cci_results.empty:
                        tic_df["cci_30"] = cci_results
                    else:
                        tic_df["cci_30"] = np.nan
                except Exception as e:
                    tic_df["cci_30"] = np.nan
                    print(f"Err cci_30 {tic_name}: {e}")

            if "adx_30" in self.tech_indicator_list:  # Check for 'adx_30'
                try:
                    # pandas-ta adx: length for DI period, lensig for ADX smoothing period.
                    # If "adx_30" implies 30 for both, use length=30, lensig=30.
                    # Default for lensig if not provided is same as length.
                    # The output column from pandas-ta.adx is named ADX_<lensig>
                    adx_results = tic_df.ta.adx(
                        length=30,
                        lensig=30,
                        high="high",
                        low="low",
                        close="close",
                        append=False,
                    )
                    if (
                        adx_results is not None and "ADX_30" in adx_results.columns
                    ):  # pandas-ta names it ADX_LENSIG
                        tic_df["adx_30"] = adx_results[
                            "ADX_30"
                        ]  # Create column 'adx_30'
                    else:
                        print(
                            f"Warning: pandas-ta did not produce 'ADX_30' column for {tic_name}. 'adx_30' will be NaN."
                        )
                        tic_df["adx_30"] = np.nan
                except Exception as e:
                    tic_df["adx_30"] = np.nan
                    print(f"Err adx_30 for {tic_name}: {e}")
            # --- END MODIFIED ADX PART ---
            elif (
                "adx" in self.tech_indicator_list
            ):  # Fallback if user just asks for 'adx' (default 14 period)
                try:
                    adx_results = tic_df.ta.adx(
                        length=14, high="high", low="low", close="close", append=False
                    )
                    if adx_results is not None and "ADX_14" in adx_results.columns:
                        tic_df["adx"] = adx_results["ADX_14"]
                    else:
                        tic_df["adx"] = np.nan
                except Exception as e:
                    tic_df["adx"] = np.nan
                    print(f"Err adx for {tic_name}: {e}")

            tic_df.reset_index(inplace=True)
            all_indicator_data_list.append(tic_df)

        if not all_indicator_data_list:
            print(
                "No data processed by pandas-ta. Returning original df with NaN indicators."
            )
            for indicator_name_spec in self.tech_indicator_list:
                if indicator_name_spec not in df.columns:
                    df[indicator_name_spec] = np.nan
            return df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        processed_with_ta_df = pd.concat(all_indicator_data_list, ignore_index=True)

        # Prepare for merge: we need original string 'date' and 'tic' from 'data'
        # and indicator columns from 'processed_with_ta_df'.
        # 'processed_with_ta_df' has 'date_dt' (datetime) and 'date' (original string).

        cols_to_merge_from_ta = ["date", "tic"] + [
            ind
            for ind in self.tech_indicator_list
            if ind in processed_with_ta_df.columns
        ]

        mergeable_indicator_df = processed_with_ta_df[
            cols_to_merge_from_ta
        ].drop_duplicates(subset=["date", "tic"], keep="last")

        if mergeable_indicator_df.empty:
            print(
                "Mergeable indicator df is empty after selection/deduplication. Returning original with NaN indicators."
            )
            for indicator_name_spec in self.tech_indicator_list:
                if indicator_name_spec not in data.columns:
                    data[indicator_name_spec] = np.nan  # Add to original `data`
            return data.sort_values(by=["date", "tic"]).reset_index(
                drop=True
            )  # Return original `data`

        # Start with the original `data` DataFrame passed into the function
        output_df = data.copy()

        # Drop existing indicator columns from output_df to avoid _x, _y suffixes if they existed
        for ind_col in self.tech_indicator_list:
            if ind_col in output_df.columns:
                output_df = output_df.drop(columns=[ind_col])

        output_df = output_df.merge(
            mergeable_indicator_df, on=["date", "tic"], how="left"
        )

        # Drop the temporary date_dt column if it was carried over
        if "date_dt" in output_df.columns:
            output_df.drop(columns=["date_dt"], inplace=True)

        return output_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    def add_user_defined_feature(self, data):
        df = data.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["daily_return"] = df.groupby("tic")["close"].transform(
            lambda x: x.pct_change(1)
        )
        return df

    def add_vix(self, data):
        df = data.copy()
        if df.empty or "date" not in df.columns:
            print("VIX not added: input data is empty or missing 'date' column.")
            df["vix"] = np.nan  # Ensure column exists even if not added
            return df

        min_date_str = pd.to_datetime(df.date.min()).strftime("%Y-%m-%d")
        max_date_str = pd.to_datetime(df.date.max()).strftime("%Y-%m-%d")

        df_vix = YahooDownloader(
            start_date=min_date_str, end_date=max_date_str, ticker_list=["^VIX"]
        ).fetch_data()

        if (
            df_vix.empty
            or "close" not in df_vix.columns
            or "date" not in df_vix.columns
        ):
            print("VIX data empty or malformed. VIX not added.")
            df["vix"] = np.nan
            return df

        vix = df_vix[["date", "close"]].copy()
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date", how="left")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        df = data.copy()
        if df.empty:
            df["turbulence"] = np.nan
            return df
        try:
            turbulence_index = self.calculate_turbulence(df)
            if not turbulence_index.empty:
                df = df.merge(turbulence_index, on="date", how="left")
                df = df.sort_values(["date", "tic"]).reset_index(drop=True)
            else:
                print("Turbulence index is empty. Not merged.")
                df["turbulence"] = np.nan
        except Exception as e:
            print(
                f"Error calculating or merging turbulence: {e}. Turbulence not added."
            )
            if "turbulence" not in df.columns:
                df["turbulence"] = np.nan
        return df

    def calculate_turbulence(self, data):
        df = data.copy()
        if df.empty or not all(col in df.columns for col in ["date", "tic", "close"]):
            print("Insufficient data for turbulence calculation.")
            return pd.DataFrame(columns=["date", "turbulence"])

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df.dropna(subset=["close"], inplace=True)

        try:
            df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        except Exception as e:
            print(f"Error pivoting data for turbulence: {e}")
            return pd.DataFrame(columns=["date", "turbulence"])

        df_price_pivot = df_price_pivot.pct_change()
        unique_date = sorted(df.date.unique())

        start = 252
        if len(unique_date) < start:
            print(
                "Not enough unique dates for turbulence calculation (need at least 252)."
            )
            return pd.DataFrame({"date": unique_date, "turbulence": np.nan})

        turbulence_index = [0.0] * start
        count = 0
        for i in range(start, len(unique_date)):
            current_price_slice = df_price_pivot.loc[[unique_date[i]]]
            hist_start_date = unique_date[i - 252]
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= hist_start_date)
            ]

            common_tickers = hist_price.columns.intersection(
                current_price_slice.columns
            )
            hist_price_aligned = (
                hist_price[common_tickers]
                .dropna(axis=1, how="all")
                .dropna(axis=0, how="any")
            )
            current_price_aligned = current_price_slice[common_tickers]

            if (
                hist_price_aligned.shape[0] < 2
                or hist_price_aligned.shape[1] < 1
                or current_price_aligned.isnull().values.any()
            ):
                turbulence_index.append(0.0)
                continue

            cov_temp = hist_price_aligned.cov()
            try:
                inv_cov_temp = np.linalg.pinv(cov_temp)
            except np.linalg.LinAlgError:
                turbulence_index.append(0.0)
                continue

            current_return_diff = (
                current_price_aligned.iloc[0] - np.mean(hist_price_aligned, axis=0)
            ).values.reshape(1, -1)
            temp = current_return_diff.dot(inv_cov_temp).dot(current_return_diff.T)

            turbulence_temp = 0.0
            if temp.size > 0 and temp[0, 0] > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0, 0]
            turbulence_index.append(turbulence_temp)

        try:
            turbulence_df = pd.DataFrame(
                {"date": unique_date, "turbulence": turbulence_index}
            )
        except Exception as e_turb_df:
            print(
                f"Error creating turbulence DataFrame: {e_turb_df}. Turbulence information could not be added."
            )
            return pd.DataFrame({"date": df.date.unique(), "turbulence": np.nan})

        return turbulence_df
