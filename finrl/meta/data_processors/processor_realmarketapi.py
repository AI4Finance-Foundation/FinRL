"""RealMarketAPI data processor for FinRL.

RealMarketAPI (https://realmarketapi.com) serves OHLCV candles for forex
pairs, metals and crypto (for example ``EURUSD``, ``XAUUSD``, ``BTCUSD``).
It does not cover equities.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests
from stockstats import StockDataFrame as Sdf

DEFAULT_BASE_URL = "https://api.realmarketapi.com"
API_KEY_ENV_VAR = "REALMARKETAPI_API_KEY"
HISTORY_PAGE_SIZE = 1000

# FinRL time interval -> RealMarketAPI timeframe
TIME_INTERVALS = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "1h": "H1",
    "4h": "H4",
    "1d": "D1",
}
# Intervals served from /api/v1/history (H1 candles), with the pandas rule
# used to resample H1 up to the interval. None means no resampling.
HISTORY_RESAMPLE_RULES = {"1h": None, "4h": "240min", "1d": "1D"}
TIME_INTERVAL_ALIASES = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "60m": "1h",
    "60min": "1h",
    "240m": "4h",
    "240min": "4h",
    "1day": "1d",
    "d": "1d",
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class RealMarketAPIError(RuntimeError):
    """Raised when RealMarketAPI returns an error response.

    ``error_codes`` holds the API error codes, such as
    ``ERR_0005_INVALID_API_KEY`` or ``ERR_0017_REQUEST_QUOTA_EXCEEDED``.
    """

    def __init__(self, message, status_code=None, error_codes=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_codes = list(error_codes or [])


class RealMarketAPIProcessor:
    """Downloads forex, metal and crypto candles from RealMarketAPI.

    Coverage is forex pairs, metals and crypto (e.g. ``EURUSD``, ``XAUUSD``,
    ``BTCUSD``), not equities. Tickers are RealMarketAPI symbol codes.

    How each ``time_interval`` is served:

    * ``1h``, ``4h``, ``1d``: paged from ``/api/v1/history``, which returns
      1-hour candles for any date range (up to 1,000 per request, about 41
      days of round-the-clock data). ``4h`` and ``1d`` are resampled from
      the 1-hour candles in UTC (open=first, high=max, low=min, close=last,
      volume=sum), so these daily bars run 00:00-24:00 UTC. That can differ
      from a broker's D1 candle, which often opens at the broker's
      server midnight (for example 21:00 or 22:00 UTC).
    * ``1m``, ``5m``, ``15m``: taken from ``/api/v1/candle``, which returns
      only a recent window at that timeframe. The window is filtered to the
      requested range, and a ``ValueError`` is raised if ``start_date`` is
      older than the oldest candle in the window. Use these intervals for
      recent data (e.g. paper trading), not long backtests.

    Every request counts against the API key's quota.

    Timestamps are returned as timezone-naive UTC in the ``timestamp`` column.
    ``end_date`` is exclusive.

    VIX and the turbulence index are equity-market measures, so
    ``add_vix`` and ``add_turbulence`` fill their columns with 0, as the
    FXMacroData processor does.

    Parameters
    ----------
    api_key : str, optional
        RealMarketAPI key. Falls back to the ``REALMARKETAPI_API_KEY``
        environment variable.
    base_url : str
        API base URL.
    timeout : float
        HTTP timeout in seconds.
    session : requests.Session, optional
        Session used for HTTP requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    @staticmethod
    def _normalize_interval(time_interval: str) -> str:
        interval = str(time_interval).strip().lower()
        interval = TIME_INTERVAL_ALIASES.get(interval, interval)
        if interval not in TIME_INTERVALS:
            raise ValueError(
                f"Unsupported time_interval {time_interval!r} for RealMarketAPI. "
                f"Use one of {list(TIME_INTERVALS)}."
            )
        return interval

    @staticmethod
    def _to_utc_iso(value) -> str:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _get(self, route: str, params: dict) -> dict:
        if not self.api_key:
            raise ValueError(
                "A RealMarketAPI key is required. Pass api_key or set the "
                f"{API_KEY_ENV_VAR} environment variable."
            )
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/{route}",
                params={"apiKey": self.api_key, **params},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            # The key is a query parameter, so keep the request URL out of the error.
            raise RealMarketAPIError(
                f"Request to RealMarketAPI failed ({type(e).__name__})"
            ) from None
        if response.status_code >= 400:
            raise self._error_from_response(response)
        return response.json()

    @staticmethod
    def _error_from_response(response) -> RealMarketAPIError:
        codes, details = [], []
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            for error in body.get("errors") or []:
                if not isinstance(error, dict):
                    continue
                code = error.get("errorCode") or error.get("code")
                if code:
                    codes.append(str(code))
                reason = error.get("reason") or error.get("message")
                if reason:
                    details.append(str(reason))
            for key in ("detail", "title"):
                if body.get(key) and not details:
                    details.append(str(body[key]))
        message = f"RealMarketAPI request failed with HTTP {response.status_code}"
        if codes:
            message += f" ({', '.join(codes)})"
        if details:
            message += f": {'; '.join(details)}"
        return RealMarketAPIError(
            message, status_code=response.status_code, error_codes=codes
        )

    @staticmethod
    def _rows_to_frame(rows: list[dict], tic: str) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["timestamp"] + OHLCV_COLUMNS + ["tic"])
        df = df.rename(
            columns={
                "openTime": "timestamp",
                "openPrice": "open",
                "highPrice": "high",
                "lowPrice": "low",
                "closePrice": "close",
            }
        )
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].astype(np.float64)
        df["tic"] = tic
        return df[["timestamp"] + OHLCV_COLUMNS + ["tic"]]

    def _fetch_history(self, tic: str, start: str, end: str) -> pd.DataFrame:
        rows = []
        page_number = 1
        while True:
            payload = self._get(
                "history",
                {
                    "symbolCode": tic,
                    "startTime": start,
                    "endTime": end,
                    "pageNumber": page_number,
                    "pageSize": HISTORY_PAGE_SIZE,
                },
            )
            data = payload.get("data") or []
            rows.extend(data)
            total_pages = int(payload.get("totalPages") or 0)
            if not data or page_number >= total_pages:
                break
            page_number += 1
        df = self._rows_to_frame(rows, tic)
        # Rows arrive newest first; new candles can shift pages while paging.
        df = df.drop_duplicates(subset=["timestamp"], keep="first")
        return df.sort_values("timestamp").reset_index(drop=True)

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str, tic: str) -> pd.DataFrame:
        if df.empty:
            return df
        resampled = (
            df.set_index("timestamp")[OHLCV_COLUMNS]
            .resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        resampled["tic"] = tic
        return resampled[["timestamp"] + OHLCV_COLUMNS + ["tic"]]

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
    ) -> pd.DataFrame:
        """Download OHLCV candles for ``ticker_list`` between the two dates.

        ``time_interval`` is one of ``1m``, ``5m``, ``15m``, ``1h``, ``4h``,
        ``1d``. See the class docstring for which endpoint serves each one.
        """
        interval = self._normalize_interval(time_interval)
        start = self._to_utc_iso(start_date)
        end = self._to_utc_iso(end_date)
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError("start_date must be earlier than end_date.")

        self.start = start_date
        self.end = end_date
        self.time_interval = interval

        frames = []
        for tic in ticker_list:
            if interval in HISTORY_RESAMPLE_RULES:
                df = self._fetch_history(tic, start, end)
                rule = HISTORY_RESAMPLE_RULES[interval]
                if rule is not None:
                    df = self._resample(df, rule, tic)
            else:
                df = self._fetch_recent_range(tic, interval, start, end)
            frames.append(df)
        data_df = pd.concat(frames, ignore_index=True)
        return data_df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def _fetch_candles(self, tic: str, interval: str) -> pd.DataFrame:
        payload = self._get(
            "candle", {"symbolCode": tic, "timeFrame": TIME_INTERVALS[interval]}
        )
        df = self._rows_to_frame(payload.get("data") or [], tic)
        df = df.drop_duplicates(subset=["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def _fetch_recent_range(
        self, tic: str, interval: str, start: str, end: str
    ) -> pd.DataFrame:
        df = self._fetch_candles(tic, interval)
        start_ts = pd.Timestamp(start).tz_localize(None)
        end_ts = pd.Timestamp(end).tz_localize(None)
        oldest = df["timestamp"].min() if not df.empty else None
        if oldest is None or start_ts < oldest:
            available = "no candles" if oldest is None else f"candles from {oldest}"
            raise ValueError(
                f"RealMarketAPI only serves a recent window of {interval} candles "
                f"for {tic} ({available} UTC), so start_date {start} is out of "
                "range. Use a later start_date, or 1h, 4h or 1d for longer "
                "history."
            )
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
        return df.loc[mask].reset_index(drop=True)

    def download_latest_candles(
        self, ticker_list: list[str], time_interval: str
    ) -> pd.DataFrame:
        """Download the most recent window of candles from ``/api/v1/candle``.

        The API decides the window length; no date range is sent.
        """
        interval = self._normalize_interval(time_interval)
        frames = [self._fetch_candles(tic, interval) for tic in ticker_list]
        data_df = pd.concat(frames, ignore_index=True)
        return data_df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align all tickers on one shared timestamp index.

        Forex and metals close at weekends while crypto does not, so mixed
        ticker lists have gaps. Missing bars are filled with the previous
        close (open=high=low=close, volume=0). Timestamps before every ticker
        has its first bar are dropped.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df.drop_duplicates(subset=["tic", "timestamp"], keep="last")
        if df.empty:
            return df.reset_index(drop=True)

        start = df.groupby("tic")["timestamp"].min().max()
        times = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
        times = times[times >= start]

        frames = []
        for tic, group in df.groupby("tic"):
            group = group.set_index("timestamp").reindex(times)
            group["close"] = group["close"].ffill()
            for col in ("open", "high", "low"):
                group[col] = group[col].fillna(group["close"])
            group["volume"] = group["volume"].fillna(0.0)
            group["tic"] = tic
            group.index.name = "timestamp"
            frames.append(group.reset_index())
        cleaned = pd.concat(frames, ignore_index=True)
        cleaned = cleaned[["timestamp"] + OHLCV_COLUMNS + ["tic"]]
        return cleaned.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ) -> pd.DataFrame:
        df = data.copy().sort_values(["tic", "timestamp"]).reset_index(drop=True)
        for tic in df["tic"].unique():
            mask = df["tic"] == tic
            stock = Sdf.retype(df.loc[mask].copy())
            for indicator in tech_indicator_list:
                df.loc[mask, indicator] = np.asarray(stock[indicator])
        return df.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def add_turbulence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turbulence is an equity-market measure; the column is set to 0."""
        df = df.copy()
        df["turbulence"] = 0
        return df

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        """VIX is an equity-market index; the ``VIXY`` column is set to 0."""
        df = df.copy()
        df["VIXY"] = 0
        return df

    def add_vixor(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.add_vix(df)

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.sort_values(["timestamp", "tic"])
        price_array = tech_array = turbulence_array = None
        for tic in df["tic"].unique():
            tic_df = df[df["tic"] == tic]
            if price_array is None:
                price_array = tic_df[["close"]].values
                tech_array = tic_df[tech_indicator_list].values
                turbulence_array = tic_df["VIXY" if if_vix else "turbulence"].values
            else:
                price_array = np.hstack([price_array, tic_df[["close"]].values])
                tech_array = np.hstack([tech_array, tic_df[tech_indicator_list].values])
        return price_array, tech_array, turbulence_array
