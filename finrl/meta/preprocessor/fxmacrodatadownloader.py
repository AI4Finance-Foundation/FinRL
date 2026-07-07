"""Contains methods and classes to collect daily FX spot data from FXMacroData."""

from __future__ import annotations

import json
import os
from typing import List
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd


class FXMacroDataDownloader:
    """Provides methods for retrieving daily FX spot data from FXMacroData.

    Parameters
    ----------
    start_date : str
        Start date of the data in ``YYYY-MM-DD`` format.
    end_date : str
        End date of the data in ``YYYY-MM-DD`` format.
    ticker_list : list
        FX pairs such as ``EURUSD`` or ``EUR/USD``.
    api_key : str, optional
        FXMacroData API key. If omitted, ``FXMACRODATA_API_KEY`` and
        ``FXMD_API_KEY`` environment variables are checked.
    base_url : str
        FXMacroData API base URL.
    timeout : float
        HTTP timeout in seconds.
    """

    _api_key_env_vars = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker_list: List[str],
        api_key: str = None,
        base_url: str = "https://api.fxmacrodata.com/v1",
        timeout: float = 30,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.api_key = api_key or self._get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_data(self) -> pd.DataFrame:
        """Fetches daily FX spot data from FXMacroData.

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume, tic, day.
            FX spot rates are mapped to OHLC using the same daily rate and
            ``volume`` is set to 0.
        """
        data_df = pd.DataFrame()
        failures = 0
        for ticker in self.ticker_list:
            base, quote = self._parse_pair(ticker)
            rows = self._request_rows(base, quote)
            temp_df = self._rows_to_dataframe(ticker, rows)
            if len(temp_df) > 0:
                data_df = pd.concat([data_df, temp_df], axis=0)
            else:
                failures += 1

        if failures == len(self.ticker_list):
            raise ValueError("no data is fetched.")

        data_df = data_df.dropna().reset_index(drop=True)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        return data_df

    @classmethod
    def _get_env_api_key(cls):
        for name in cls._api_key_env_vars:
            api_key = os.getenv(name)
            if api_key:
                return api_key
        return None

    @staticmethod
    def _parse_pair(ticker: str):
        pair = ticker.strip().upper()
        if pair.endswith("=X"):
            pair = pair[:-2]
        pair = pair.replace("/", "").replace("-", "").replace("_", "")
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError(
                "FXMacroData tickers must be currency pairs such as "
                "'EURUSD' or 'EUR/USD'."
            )
        return pair[:3].lower(), pair[3:].lower()

    def _request_rows(self, base: str, quote: str):
        params = urlencode({"start_date": self.start_date, "end_date": self.end_date})
        url = f"{self.base_url}/forex/{base}/{quote}?{params}"
        request = Request(url)
        if self.api_key:
            request.add_header("X-API-Key", self.api_key)
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            return payload.get("data", [])
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def _rows_to_dataframe(ticker: str, rows: list) -> pd.DataFrame:
        data = []
        for row in rows:
            date = row.get("date")
            rate = FXMacroDataDownloader._get_rate(row)
            if date is None or rate is None:
                continue
            data.append(
                {
                    "date": pd.to_datetime(date),
                    "open": rate,
                    "high": rate,
                    "low": rate,
                    "close": rate,
                    "volume": 0.0,
                    "tic": ticker,
                }
            )
        if not data:
            return pd.DataFrame(
                columns=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "tic",
                    "day",
                ]
            )
        data_df = pd.DataFrame(data)
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        return data_df

    @staticmethod
    def _get_rate(row: dict):
        for key in ("val", "value", "close", "rate"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return None
