# finrl/plot.py
from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import pyfolio as pf  # pyfolio is a common dependency for FinRL's plotting

# Attempt to import YahooDownloader locally if get_baseline uses it
try:
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
except ImportError:
    print(
        "Warning: YahooDownloader not found for get_baseline. Ensure it's in the correct path or install FinRL correctly."
    )

    # Define a dummy YahooDownloader if needed for get_baseline to not crash immediately
    class YahooDownloader:
        def __init__(self, start_date, end_date, ticker_list):
            self.start_date = start_date
            self.end_date = end_date
            self.ticker_list = ticker_list
            print(
                "Dummy YahooDownloader initialized because actual could not be imported."
            )

        def fetch_data(self):
            print(
                f"Dummy YahooDownloader: Would fetch {self.ticker_list} from {self.start_date} to {self.end_date}"
            )
            return pd.DataFrame()  # Return empty DataFrame


# Define default dates and ticker if config is not used/imported
# These should match the defaults expected or used elsewhere in your project if not using finrl.config
DEFAULT_TRADE_START_DATE = "2000-01-01"
DEFAULT_TRADE_END_DATE = pd.Timestamp.today().strftime(
    "%Y-%m-%d"
)  # Use current date as a sensible default
DEFAULT_BASELINE_TICKER = "^DJI"  # Dow Jones Industrial Average as a common default


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    # Ensure 'date' column exists before trying to convert it
    if "date" not in df.columns:
        if df.index.name == "date" and isinstance(df.index, pd.DatetimeIndex):
            # If 'date' is already the index and is datetime, reset it to be a column
            df = df.reset_index()
        else:
            raise KeyError(
                f"DataFrame is missing the 'date' column for get_daily_return. Columns: {df.columns}"
            )

    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    # It's good practice to ensure the index is sorted if not already
    df.sort_index(inplace=True)
    # Pyfolio often expects UTC-localized or tz-naive. Let's try tz-naive for broader compatibility first.
    # If pyfolio complains, you might need to localize, e.g., df.index = df.index.tz_localize("UTC")
    # but ensure all series passed to pyfolio are consistent.
    if (
        df.index.tz is not None
    ):  # If already localized, pyfolio might handle it or prefer naive
        pass  # df.index = df.index.tz_convert("UTC") # Or convert to UTC
    # else:
    # df.index = df.index.tz_localize("UTC") # Localize if naive, but sometimes causes issues if other data is naive
    return df["daily_return"]


def get_baseline(
    ticker,
    start=DEFAULT_TRADE_START_DATE,
    end=DEFAULT_TRADE_END_DATE,
):
    """
    Returns the DataFrame of a baseline ticker from Yahoo Finanace
    Parameters:
    ------------
    ticker : str
        The ticker symbol of the baseline, e.g. '^DJI'
    start : str
        The start date for collecting the baseline data,
        format: 'YYYY-MM-DD'
    end : str
        The end date for collecting the baseline data,
        format: 'YYYY-MM-DD'
    Returns:
    ------------
    `pd.DataFrame`
        A DataFrame containing the baseline data with columns
        ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    """
    try:
        baseline_df = YahooDownloader(
            start_date=start, end_date=end, ticker_list=[ticker]
        ).fetch_data()
        # YahooDownloader should return 'date' as a column of strings
    except Exception as e:
        print(f"Error fetching baseline data for {ticker}: {e}")
        baseline_df = pd.DataFrame()  # Return empty if error
    return baseline_df


def backtest_plot(
    account_value,  # This is the DataFrame you pass from your notebook
    baseline_start=DEFAULT_TRADE_START_DATE,
    baseline_end=DEFAULT_TRADE_END_DATE,
    baseline_ticker=DEFAULT_BASELINE_TICKER,
    value_col_name="account_value",
):
    df = deepcopy(account_value)

    # --- START OF THE FIX for "ambiguous date" ---
    # Ensure 'date' refers only to the column, not a potentially named index
    if df.index.name == "date":
        df.index.name = None  # Remove the name 'date' from the index
    # --- END OF THE FIX ---

    # Ensure 'date' column exists and is a column
    if "date" not in df.columns:
        raise KeyError(
            f"Input DataFrame to backtest_plot is missing 'date' column. Columns: {df.columns}"
        )
    if value_col_name not in df.columns:
        raise KeyError(
            f"Input DataFrame to backtest_plot is missing value column '{value_col_name}'. Columns: {df.columns}"
        )

    df["date"] = pd.to_datetime(df["date"])  # Convert 'date' column to datetime

    # get_daily_return expects a DataFrame with 'date' and value_col_name as columns.
    # It will set 'date' as DatetimeIndex and return a Series of returns.
    test_returns = get_daily_return(
        df.copy(), value_col_name=value_col_name
    )  # Pass a copy to get_daily_return

    baseline_returns = None  # Initialize
    if baseline_ticker is not None:
        baseline_df_raw = get_baseline(
            ticker=baseline_ticker, start=baseline_start, end=baseline_end
        )
        if (
            not baseline_df_raw.empty
            and "date" in baseline_df_raw.columns
            and "close" in baseline_df_raw.columns
        ):
            # get_daily_return for baseline also expects 'date' and 'close' as columns
            baseline_returns_series = get_daily_return(
                baseline_df_raw.copy(), value_col_name="close"
            )
            baseline_returns = baseline_returns_series.to_frame(
                name=baseline_ticker
            )  # Convert to DataFrame for pyfolio
            # Align baseline returns with test_returns
            baseline_returns = baseline_returns.reindex(test_returns.index).fillna(0.0)
        else:
            print(
                f"Warning: Baseline ticker {baseline_ticker} data is empty or malformed. Plotting without baseline."
            )

    # Plotting with Pyfolio
    try:
        # Ensure returns are Series and tz-naive or consistently tz-aware for Pyfolio
        pyfolio_test_returns = test_returns.copy()
        if pyfolio_test_returns.index.tz is not None:
            pyfolio_test_returns.index = pyfolio_test_returns.index.tz_localize(None)

        if baseline_returns is not None:
            pyfolio_baseline_returns = baseline_returns[
                baseline_ticker
            ].copy()  # Get Series
            if pyfolio_baseline_returns.index.tz is not None:
                pyfolio_baseline_returns.index = (
                    pyfolio_baseline_returns.index.tz_localize(None)
                )

            pf.plot_rolling_returns(
                returns=pyfolio_test_returns,
                factor_returns=pyfolio_baseline_returns,
                live_start_date=None,
                cone_std=None,
            )
        else:
            pf.plot_rolling_returns(
                returns=pyfolio_test_returns,
                live_start_date=None,
                cone_std=None,
            )
        fig = plt.gcf()
        return fig
    except ImportError:
        print("Pyfolio not installed. Plotting a simple line graph.")
        plt.figure(figsize=(10, 6))
        # For simple plot, use the df with 'date' as index and 'account_value'
        df_for_simple_plot = df.copy().set_index(
            "date"
        )  # df already has 'date' as datetime column
        df_for_simple_plot[value_col_name].plot(label="Agent Account Value")

        if baseline_returns is not None and not baseline_df_raw.empty:
            # To plot baseline values, we need to reconstruct them from returns or re-fetch prices.
            # For simplicity here, we won't reconstruct the baseline value line for the simple plot.
            # We could plot baseline *returns* on a secondary axis if desired.
            print(
                "Simple plot does not include baseline value line, only agent's account value."
            )
            pass

        plt.title("Account Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Account Value")
        plt.legend()
        fig = plt.gcf()
        return fig


# You might have other functions like backtest_stats here.
# Ensure get_daily_return is defined before backtest_stats if it's used there.
def backtest_stats(account_value, value_col_name="account_value"):
    """
    Calculates and prints performance stats from a DataFrame of account values.
    account_value: pd.DataFrame with 'date' and 'account_value' (or value_col_name) columns.
    """
    # get_daily_return expects a DataFrame with 'date' and value_col_name as columns.
    # It returns a Series of daily returns with a DatetimeIndex.
    # Ensure input account_value is a DataFrame with required columns
    if not isinstance(account_value, pd.DataFrame):
        raise TypeError(
            "Input 'account_value' to backtest_stats must be a pandas DataFrame."
        )
    if "date" not in account_value.columns:
        raise KeyError(
            f"Input DataFrame to backtest_stats is missing 'date' column. Columns: {account_value.columns}"
        )
    if value_col_name not in account_value.columns:
        raise KeyError(
            f"Input DataFrame to backtest_stats is missing value column '{value_col_name}'. Columns: {account_value.columns}"
        )

    dr_test = get_daily_return(account_value.copy(), value_col_name=value_col_name)

    # Pyfolio's perf_stats expects returns as a Series with DatetimeIndex.
    # dr_test is already in this format.
    # Ensure dr_test is tz-naive or consistently tz-aware for pyfolio
    pyfolio_dr_test = dr_test.copy()
    if pyfolio_dr_test.index.tz is not None:
        pyfolio_dr_test.index = pyfolio_dr_test.index.tz_localize(None)

    try:
        perf_stats_all = pf.timeseries.perf_stats(
            returns=pyfolio_dr_test,
            positions=None,  # Not typically available from just account value
            transactions=None,  # Not typically available
            turnover_denom="AGB",  # Default, can be 'equity'
        )
        print(perf_stats_all)
        return perf_stats_all
    except ImportError:
        print("Pyfolio not installed. Cannot calculate performance stats.")
        return None
    except Exception as e:
        print(f"Error during perf_stats calculation: {e}")
        return None
