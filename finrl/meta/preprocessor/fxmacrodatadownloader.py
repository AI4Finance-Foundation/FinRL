"""Contains methods and classes to collect daily FX spot data from FXMacroData."""

from __future__ import annotations

import json
import os
from typing import List
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

import pandas as pd

DEFAULT_BASE_URL = "https://api.fxmacrodata.com/v1"
API_KEY_ENV_VARS = ("FXMACRODATA_API_KEY", "FXMD_API_KEY")


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

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker_list: List[str],
        api_key: str = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.api_key = api_key or get_env_api_key()
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
        return request_rows(
            self.base_url,
            f"forex/{base}/{quote}",
            {"start_date": self.start_date, "end_date": self.end_date},
            self.api_key,
            self.timeout,
        )

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


class FXMacroDataMacroDownloader:
    """Download macro announcements, release calendars, and forecast groups."""

    def __init__(
        self,
        currency: str,
        indicator_list: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        api_key: str = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30,
    ):
        self.currency = currency.lower()
        self.indicator_list = [
            indicator.lower() for indicator in (indicator_list or [])
        ]
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key or get_env_api_key()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_catalogue(self, include_coverage: bool = True) -> dict:
        return request_json(
            self.base_url,
            f"data_catalogue/{self.currency}",
            {"include_coverage": str(include_coverage).lower()},
            self.api_key,
            self.timeout,
        )

    def fetch_announcements(self) -> pd.DataFrame:
        frames = []
        for indicator in self.indicator_list:
            rows = request_rows(
                self.base_url,
                f"announcements/{self.currency}/{indicator}",
                self._date_params(),
                self.api_key,
                self.timeout,
            )
            frames.append(self._rows_to_frame(indicator, rows, "announcements"))
        return concat_frames(frames)

    def fetch_calendar(self) -> pd.DataFrame:
        indicators = self.indicator_list or [None]
        frames = []
        for indicator in indicators:
            params = self._date_params()
            if indicator:
                params["indicator"] = indicator
            rows = request_rows(
                self.base_url,
                f"calendar/{self.currency}",
                params,
                self.api_key,
                self.timeout,
            )
            frames.append(self._rows_to_frame(indicator, rows, "calendar"))
        return concat_frames(frames)

    def fetch_predictions(self) -> pd.DataFrame:
        frames = []
        for indicator in self.indicator_list:
            rows = request_rows(
                self.base_url,
                f"predictions/{self.currency}/{indicator}",
                self._date_params(),
                self.api_key,
                self.timeout,
            )
            frames.append(self._rows_to_frame(indicator, rows, "predictions"))
        return concat_frames(frames)

    def _date_params(self) -> dict:
        return {"start_date": self.start_date, "end_date": self.end_date}

    def _rows_to_frame(self, indicator: str, rows: list, dataset: str) -> pd.DataFrame:
        data = []
        for row in rows:
            release = indicator or row.get("release") or row.get("indicator")
            date = row.get("date") or row.get("release_date")
            if date is None:
                continue
            prediction, prediction_count = prediction_summary(row)
            actual = number(row.get("actual"))
            value = number(row.get("val") or row.get("value"))
            if value is None:
                value = actual
            data.append(
                {
                    "date": pd.to_datetime(date),
                    "currency": self.currency,
                    "indicator": release,
                    "dataset": dataset,
                    "announcement_datetime": int(row.get("announcement_datetime") or 0),
                    "value": value,
                    "actual": actual if actual is not None else value,
                    "previous": number(row.get("previous")),
                    "revised_previous": number(row.get("revised_previous")),
                    "consensus": number(
                        row.get("consensus")
                        or row.get("expected")
                        or row.get("estimate")
                    ),
                    "forecast": number(row.get("forecast")),
                    "surprise": number(row.get("surprise")),
                    "surprise_zscore": number(row.get("surprise_zscore")),
                    "prediction": prediction,
                    "prediction_count": prediction_count,
                    "is_future": bool(
                        row.get("announcement_timing") == "future"
                        or row.get("actual_available") is False
                    ),
                    "source": row.get("source"),
                    "announcement_id": row.get("announcement_id"),
                }
            )
        if not data:
            return empty_macro_frame()
        frame = pd.DataFrame(data)
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        return frame.sort_values(["date", "currency", "indicator"]).reset_index(
            drop=True
        )


def request_json(base_url: str, path: str, params: dict, api_key: str, timeout: float):
    query = urlencode(
        {key: value for key, value in params.items() if value is not None}
    )
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    request = Request(url)
    if api_key:
        request.add_header("X-API-Key", api_key)
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def request_rows(base_url: str, path: str, params: dict, api_key: str, timeout: float):
    payload = request_json(base_url, path, params, api_key, timeout)
    if isinstance(payload, dict):
        data = payload.get("data", [])
        return data if isinstance(data, list) else []
    if isinstance(payload, list):
        return payload
    return []


def get_env_api_key():
    for name in API_KEY_ENV_VARS:
        api_key = os.getenv(name)
        if api_key:
            return api_key
    return None


def number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prediction_summary(row: dict):
    predictions = row.get("predictions")
    if isinstance(predictions, list) and predictions:
        return number(predictions[0].get("predicted_value")), len(predictions)
    for key in ("forecast_prediction", "consensus_prediction"):
        prediction = row.get(key)
        if isinstance(prediction, dict):
            return number(prediction.get("predicted_value")), 1
    return None, 0


def empty_macro_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "currency",
            "indicator",
            "dataset",
            "announcement_datetime",
            "value",
            "actual",
            "previous",
            "revised_previous",
            "consensus",
            "forecast",
            "surprise",
            "surprise_zscore",
            "prediction",
            "prediction_count",
            "is_future",
            "source",
            "announcement_id",
        ]
    )


def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return empty_macro_frame()
    return pd.concat(frames, ignore_index=True)
