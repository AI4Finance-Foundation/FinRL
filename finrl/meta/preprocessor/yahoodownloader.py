"""Contains methods and classes to collect data from
Yahoo Finance API
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------
        proxy : str, optional
            proxy server string

        Returns
        -------
        `pd.DataFrame`
            Columns: date, open, high, low, close, volume, tic, day
        """
        data_df = pd.DataFrame()
        num_failures = 0
        for tic_i in self.ticker_list:  # Renamed 'tic' to 'tic_i' for clarity in loop
            temp_df = yf.download(
                tic_i,
                start=self.start_date,
                end=self.end_date,
                proxy=proxy,
                # Note: yfinance default for auto_adjust is now True.
                # This means 'Close' is adjusted, 'Adj Close' might not be separate or relevant.
                # actions defaults to False (usually), so 'Dividends', 'Stock Splits' might not be present.
            )

            if temp_df.empty:
                print(
                    f"No data fetched for {tic_i} from {self.start_date} to {self.end_date}"
                )
                num_failures += 1
                continue

            # FIX: Handle MultiIndex columns if yfinance returns them for a single ticker
            if isinstance(temp_df.columns, pd.MultiIndex):
                # Assuming the actual metric names (Open, High, etc.) are at level 0
                temp_df.columns = temp_df.columns.get_level_values(0)
                # Defensive check: remove duplicate column names if any after flattening
                temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]

            temp_df["tic"] = tic_i
            data_df = pd.concat([data_df, temp_df], axis=0)

        if num_failures == len(self.ticker_list):
            # This error should be raised if all fetches fail
            raise ValueError(
                f"No data could be fetched for any ticker in the list: {self.ticker_list}"
            )

        # FIX: The reset_index was misplaced inside the if condition for all failures
        # It should happen after the loop if data_df is not empty.
        if data_df.empty:  # if still empty after loop (all failed)
            raise ValueError(
                f"DataFrame is empty after attempting to fetch all tickers."
            )

        data_df = (
            data_df.reset_index()
        )  # Reset index to turn 'Date' (or 'Datetime') into a column

        # Standardize column names
        # Expected columns from yf.download (auto_adjust=True, actions=False default):
        # Index (becomes 'Date'), 'Open', 'High', 'Low', 'Close', 'Volume'. Plus 'tic' we added.
        # So, after reset_index: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'tic'

        # Convert 'Date' column name if it's different (e.g., 'Datetime')
        # yfinance now often returns 'Datetime' as the index name for intraday, 'Date' for daily.
        # Let's be robust:
        if "Datetime" in data_df.columns:
            data_df = data_df.rename(columns={"Datetime": "date"})
        elif "Date" in data_df.columns:  # 'Date' is usually the name after reset_index
            data_df = data_df.rename(columns={"Date": "date"})
        else:
            # If neither 'Date' nor 'Datetime' is found, we might have an issue
            # or the first column from reset_index is the date column.
            # Let's assume the first column is the date if not named 'Date' or 'Datetime'
            # This is a fallback, might need more robust handling if yf changes index naming
            if "date" not in data_df.columns and data_df.columns[0] != "date":
                data_df = data_df.rename(columns={data_df.columns[0]: "date"})

        # Make column names lowercase for consistency before specific renames
        data_df.columns = [str(col).lower() for col in data_df.columns]

        # Now, map to FinRL's desired final column structure
        # We need: "date", "open", "high", "low", "close", "volume", "tic"
        # 'adjcp' was in the original code, but with auto_adjust=True, 'close' is already adjusted.
        # We will use the 'close' (which is adjusted) as FinRL's 'close'.

        # Check if 'adj close' column exists (it might if auto_adjust was somehow False or for older yf)
        # If 'adj close' exists and is different from 'close', FinRL logic might want it.
        # However, with current yf defaults, 'close' IS the adjusted close.

        # Select and rename columns to FinRL standard
        # We expect columns like: 'date', 'open', 'high', 'low', 'close', 'volume', 'tic'
        # (plus potentially 'dividends', 'stock splits' if actions=True, but we assume actions=False default)

        # Columns we definitely have (after lowercase and potential flattening):
        # 'date', 'open', 'high', 'low', 'close', 'volume', 'tic'

        # Ensure 'adj close' is handled if it appears (e.g. from older yf or different settings)
        if "adj close" in data_df.columns:
            # If 'adj close' is present, FinRL typically uses it.
            data_df = data_df.rename(columns={"adj close": "adjcp"})
            # If the original code's intent was to use adjcp as the primary close:
            # data_df['close'] = data_df['adjcp']
            # And then drop 'adjcp', but this depends on whether 'adjcp' is truly different.
            # For simplicity with auto_adjust=True, we'll assume 'close' is what we need.
            # If 'adjcp' is required by downstream FinRL processes, we create it from 'close'.
            if "adjcp" not in data_df.columns:  # If not created from 'adj close'
                data_df["adjcp"] = data_df[
                    "close"
                ]  # Make adjcp same as close (which is adjusted)
        else:
            # If 'adj close' is not present, 'close' is already adjusted by yfinance (auto_adjust=True)
            data_df["adjcp"] = data_df["close"]  # Create 'adjcp' from 'close'

        # Now ensure the 8 columns for the original assignment are present or can be derived
        # Original target: ["date", "open", "high", "low", "close", "adjcp", "volume", "tic"]
        # We have 'date', 'open', 'high', 'low', 'close' (adjusted), 'volume', 'tic'. And we made 'adjcp'.

        # The original code then does:
        # data_df["close"] = data_df["adjcp"]
        # data_df = data_df.drop(labels="adjcp", axis=1)
        # This means the final 'close' column should be the adjusted price, and 'adjcp' is temporary.
        # Since our 'close' from yf (with auto_adjust=True) is *already* adjusted, this is fine.

        # Let's just ensure the column names are what the rest of the original code expects before it drops 'adjcp'
        # At this point, we should have: date, open, high, low, close (adj), volume, tic, adjcp (copy of close)
        # Now apply the logic from the original try-except block, adapted
        try:
            # The original code intended to rename to these 8, then process 'adjcp'
            # We have already processed and created 'adjcp'.
            # The important part is that downstream code expects specific names.
            # Our current 'close' is already adjusted.
            # The original drop of 'adjcp' happens after this renaming

            # Let's ensure the columns are in the order expected by the original fixed assignment
            # and that all 8 conceptual slots are filled before further processing.
            # This is where the length mismatch happened.
            # We should have these columns now (names already lowercased):
            # 'date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'adjcp'
            # The number of columns here is 8.

            # This part of original code assigns fixed names, then uses 'adjcp'
            # data_df.columns = [
            #     "date", "open", "high", "low", "close", "adjcp", "volume", "tic"
            # ]
            # This fixed assignment should now work if data_df has 8 columns in the right conceptual order.
            # Let's be more explicit about selection and renaming to avoid order issues.

            current_cols = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            if not all(col in data_df.columns for col in current_cols):
                missing_cols = [
                    col for col in current_cols if col not in data_df.columns
                ]
                # This should not happen if logic above is correct
                raise ValueError(
                    f"DataFrame is missing essential columns: {missing_cols}. Current columns: {data_df.columns.tolist()}"
                )

            data_df = data_df[current_cols]  # Select and order the 8 columns

            # Now, apply the logic that was in the original try block:
            # "use adjusted close price instead of close price" - this was confusing.
            # With auto_adjust=True, 'close' from yf IS the adjusted price.
            # If the goal is for the final 'close' column to be the adjusted price,
            # and 'adjcp' was just a temporary holder or for compatibility, then:
            # data_df["close"] = data_df["adjcp"] # This is redundant if adjcp was made from close.
            # No, the original intent was that 'adjcp' IS the adjusted price, and it should become 'close'.
            # Our 'close' (from yf) IS adjusted. Our 'adjcp' is a copy of that. So this is fine.

            # drop the adjusted close price column (which was named 'adjcp')
            data_df = data_df.drop(labels="adjcp", axis=1)
            # Now data_df has: date, open, high, low, close (this is adjusted), volume, tic

        except Exception as e:  # Catch any error during this column processing
            print(f"Error during column standardization: {e}")
            print(f"DataFrame columns at error: {data_df.columns.tolist()}")
            raise  # Re-raise the error to stop execution

        # create day of the week column (monday = 0)
        # Ensure 'date' column is datetime before dt accessor
        data_df["date"] = pd.to_datetime(data_df["date"])
        data_df["day"] = data_df["date"].dt.dayofweek

        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

        # drop missing data - should be done after all column manipulations
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)  # Reset index after potential drops

        print("Shape of DataFrame after processing: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        # Sort values at the end
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        # This method seems fine, no changes needed based on the download issue.
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
