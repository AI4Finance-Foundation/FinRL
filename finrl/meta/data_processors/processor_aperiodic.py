"""FinRL data processor for Aperiodic perpetual-futures data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from finrl.meta.preprocessor.aperiodicdownloader import AperiodicDownloader
from finrl.meta.preprocessor.aperiodicdownloader import list_aperiodic_tickers
from finrl.meta.preprocessor.aperiodicdownloader import resolve_exchange
from finrl.meta.preprocessor.aperiodicdownloader import to_feature_list

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class AperiodicProcessor:
    """Provides perpetual-futures bars from Aperiodic (https://aperiodic.io).

    Covers Binance, OKX and Hyperliquid perpetuals, including Hyperliquid
    HIP-3 markets (tokenized stocks, indices, commodities) and delisted
    contracts. Requires ``pip install "aperiodic[pandas]"``.

    Parameters
    ----------
    api_key : str, optional
        Aperiodic API key. If omitted, the SDK uses ``APERIODIC_API_KEY`` or
        the key saved by ``aperiodic.login()``.
    exchange : str
        Venue for tickers without an ``@venue`` suffix: ``binance``, ``okx``
        or ``hyperliquid``.
    derivative_features : list, optional
        Columns joined to each bar on time: ``funding_rate``,
        ``open_interest``, ``open_interest_notional``, ``basis_bps``. Off by
        default, so the frame matches the other processors.
    preview : bool
        Read the public preview sample, which needs no key.
    """

    def __init__(
        self,
        api_key: str = None,
        exchange: str = "binance",
        derivative_features: list[str] = None,
        preview: bool = False,
    ):
        self.api_key = api_key
        self.exchange = resolve_exchange(exchange)
        self.derivative_features = to_feature_list(derivative_features)
        self.preview = preview

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
    ) -> pd.DataFrame:
        downloader = AperiodicDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list,
            time_interval=time_interval,
            exchange=self.exchange,
            derivative_features=self.derivative_features,
            api_key=self.api_key,
            preview=self.preview,
        )
        self.start = start_date
        self.end = end_date
        self.time_interval = downloader.time_interval
        data_df = downloader.fetch_data()
        data_df["timestamp"] = pd.to_datetime(data_df["date"])
        return data_df[
            ["timestamp"] + OHLCV_COLUMNS + ["tic"] + self.derivative_features
        ]

    def list_tickers(self) -> list[str]:
        """Every perpetual on ``exchange``, delisted contracts included."""
        return list_aperiodic_tickers(exchange=self.exchange, api_key=self.api_key)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aligns every ticker on the union of timestamps.

        A missing bar takes the previous close for open, high, low and close
        and a volume of 0; other columns, such as derivative features, carry
        their last value. Bars before a contract's first bar take its first
        close, as in the other FinRL processors.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.drop_duplicates(subset=["tic", "timestamp"], keep="last")
        extra_columns = [
            column
            for column in df.columns
            if column not in ["timestamp", "tic"] + OHLCV_COLUMNS
        ]
        times = pd.DatetimeIndex(df["timestamp"].unique(), name="timestamp")
        times = times.sort_values()
        frames = []
        for tic, tic_df in df.groupby("tic", sort=False):
            tic_df = tic_df.set_index("timestamp").reindex(times)
            tic_df["close"] = tic_df["close"].ffill().bfill()
            for column in ["open", "high", "low"]:
                tic_df[column] = tic_df[column].fillna(tic_df["close"])
            tic_df["volume"] = tic_df["volume"].fillna(0.0)
            tic_df[extra_columns] = tic_df[extra_columns].ffill().bfill()
            tic_df["tic"] = tic
            frames.append(tic_df.reset_index())
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df[["timestamp"] + OHLCV_COLUMNS + ["tic"] + extra_columns]

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ) -> pd.DataFrame:
        """Adds stockstats indicators per ticker.

        Names that are already columns, such as ``funding_rate``, are kept as
        they are, so derivative features can be listed alongside indicators
        and end up in the state array built by ``df_to_array``.
        """
        df = data.copy().sort_values(["tic", "timestamp"]).reset_index(drop=True)
        indicators = [name for name in tech_indicator_list if name not in df.columns]
        frames = []
        for _, tic_df in df.groupby("tic", sort=False):
            tic_df = tic_df.copy()
            stock = Sdf.retype(tic_df.copy())
            for indicator in indicators:
                tic_df[indicator] = stock[indicator].to_numpy()
            frames.append(tic_df)
        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """Adds the turbulence index: the Mahalanobis distance of each bar's
        returns from the previous ``time_period`` bars (0 during warm-up)."""
        df = data.copy()
        prices = df.pivot(index="timestamp", columns="tic", values="close")
        returns = (prices / prices.shift(1) - 1).fillna(0.0)
        turbulence = np.zeros(len(returns))
        for i in range(time_period, len(returns)):
            history = returns.iloc[i - time_period : i]
            deviation = returns.iloc[i] - history.mean()
            covariance = np.linalg.pinv(history.cov().to_numpy())
            turbulence[i] = max(
                float(deviation.to_numpy() @ covariance @ deviation.to_numpy()), 0.0
            )
        turbulence_df = pd.DataFrame(
            {"timestamp": returns.index, "turbulence": turbulence}
        )
        df = df.merge(turbulence_df, on="timestamp")
        return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perpetual futures have no VIX; adds ``VIXY`` = 0 so pipelines that
        pass ``if_vix=True`` still run. Prefer ``add_turbulence``."""
        warnings.warn(
            "Aperiodic data has no VIX series; VIXY is set to 0. "
            "Use add_turbulence and if_vix=False for a risk signal.",
            stacklevel=2,
        )
        df = data.copy()
        df["VIXY"] = 0.0
        return df

    def add_vixor(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.add_vix(data)

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        """Returns price (T, N), tech (T, N * len(tech_indicator_list)) and
        turbulence (T,) arrays; call ``clean_data`` first to align tickers."""
        risk_column = "VIXY" if if_vix else "turbulence"
        columns = list(dict.fromkeys(["close", risk_column, *tech_indicator_list]))
        wide = df.pivot(index="timestamp", columns="tic", values=columns)
        if wide["close"].isna().to_numpy().any():
            raise ValueError("Tickers are not aligned in time; call clean_data first.")
        tics = df.sort_values(["timestamp", "tic"])["tic"].unique()
        price_array = wide["close"][tics].to_numpy()
        tech_array = np.hstack(
            [
                wide.xs(tic, axis=1, level="tic")[list(tech_indicator_list)].to_numpy()
                for tic in tics
            ]
        )
        turbulence_array = wide[risk_column][tics[0]].to_numpy()
        return price_array, tech_array, turbulence_array
