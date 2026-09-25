from __future__ import annotations

import json

import pandas as pd
import pytest

from finrl.meta.data_processors.processor_fxmacrodata import FXMacroDataProcessor
from finrl.meta.preprocessor.fxmacrodatadownloader import FXMacroDataDownloader
from finrl.meta.preprocessor.fxmacrodatadownloader import FXMacroDataMacroDownloader

API_KEY = "api_key"


class FXMacroDataResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_fxmacrodata_downloader_fetch_data(monkeypatch):
    requests = []

    def mock_urlopen(request, timeout):
        requests.append((request, timeout))
        return FXMacroDataResponse(
            {
                "data": [
                    {"date": "2024-01-02", "val": "1.101"},
                    {"date": "2024-01-03", "val": 1.102},
                ]
            }
        )

    monkeypatch.setattr(
        "finrl.meta.preprocessor.fxmacrodatadownloader.urlopen", mock_urlopen
    )

    data = FXMacroDataDownloader(
        start_date="2024-01-02",
        end_date="2024-01-03",
        ticker_list=["EUR/USD"],
        api_key=API_KEY,
        base_url="https://example.com/v1",
    ).fetch_data()

    request, timeout = requests[0]
    assert request.full_url == (
        "https://example.com/v1/forex/eur/usd?"
        "start_date=2024-01-02&end_date=2024-01-03"
    )
    assert dict(request.header_items())["X-api-key"] == API_KEY
    assert timeout == 30
    assert list(data.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tic",
        "day",
    ]
    assert data["date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert data["tic"].tolist() == ["EUR/USD", "EUR/USD"]
    assert data["open"].tolist() == [1.101, 1.102]
    assert data["high"].tolist() == [1.101, 1.102]
    assert data["low"].tolist() == [1.101, 1.102]
    assert data["close"].tolist() == [1.101, 1.102]
    assert data["volume"].tolist() == [0.0, 0.0]


def test_fxmacrodata_processor_download_data(monkeypatch):
    monkeypatch.setattr(
        FXMacroDataDownloader,
        "fetch_data",
        lambda self: pd.DataFrame(
            {
                "date": ["2024-01-02"],
                "open": [1.101],
                "high": [1.101],
                "low": [1.101],
                "close": [1.101],
                "volume": [0.0],
                "tic": ["EURUSD"],
                "day": [1],
            }
        ),
    )

    data = FXMacroDataProcessor(
        api_key=API_KEY, base_url="https://example.com/v1"
    ).download_data(["EURUSD"], "2024-01-02", "2024-01-03", "1D")

    assert list(data.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tic",
    ]
    assert data.loc[0, "timestamp"] == pd.Timestamp("2024-01-02")
    assert data.loc[0, "close"] == 1.101


def test_fxmacrodata_downloader_invalid_pair():
    with pytest.raises(ValueError, match="currency pairs"):
        FXMacroDataDownloader("2024-01-02", "2024-01-03", ["EUR"]).fetch_data()


def test_fxmacrodata_processor_daily_only():
    with pytest.raises(ValueError, match="daily data only"):
        FXMacroDataProcessor().download_data(
            ["EURUSD"], "2024-01-02", "2024-01-03", "1Min"
        )


def test_fxmacrodata_macro_downloader_fetches_announcements(monkeypatch):
    requests = []

    def mock_urlopen(request, timeout):
        requests.append((request, timeout))
        return FXMacroDataResponse(
            {
                "data": [
                    {
                        "announcement_id": "usd_inflation_2026-05-31",
                        "date": "2026-05-31",
                        "val": 4.2,
                        "announcement_datetime": 1781094600,
                        "consensus": 3.9,
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "finrl.meta.preprocessor.fxmacrodatadownloader.urlopen", mock_urlopen
    )

    data = FXMacroDataMacroDownloader(
        currency="USD",
        indicator_list=["inflation"],
        start_date="2026-01-01",
        end_date="2026-06-30",
        api_key=API_KEY,
        base_url="https://example.com/v1",
    ).fetch_announcements()

    request, timeout = requests[0]
    assert request.full_url == (
        "https://example.com/v1/announcements/usd/inflation?"
        "start_date=2026-01-01&end_date=2026-06-30"
    )
    assert dict(request.header_items())["X-api-key"] == API_KEY
    assert timeout == 30
    assert data.loc[0, "currency"] == "usd"
    assert data.loc[0, "indicator"] == "inflation"
    assert data.loc[0, "dataset"] == "announcements"
    assert data.loc[0, "value"] == 4.2
    assert data.loc[0, "consensus"] == 3.9
    assert data.loc[0, "announcement_datetime"] == 1781094600


def test_fxmacrodata_macro_downloader_fetches_calendar_and_predictions(monkeypatch):
    payloads = []

    def mock_urlopen(request, timeout):
        payloads.append(request.full_url)
        if "/calendar/" in request.full_url:
            return FXMacroDataResponse(
                {
                    "data": [
                        {
                            "release": "policy_rate",
                            "date": "2026-07-29",
                            "announcement_datetime": 1785330000,
                            "forecast": 4.25,
                            "actual_available": False,
                        }
                    ]
                }
            )
        return FXMacroDataResponse(
            {
                "data": [
                    {
                        "announcement_id": "usd_inflation_2026-07-31",
                        "date": "2026-07-31",
                        "announcement_datetime": 1786537800,
                        "announcement_timing": "future",
                        "predictions": [
                            {
                                "predicted_value": 3.81,
                                "prediction_type": "fxmacrodata",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "finrl.meta.preprocessor.fxmacrodatadownloader.urlopen", mock_urlopen
    )

    downloader = FXMacroDataMacroDownloader(
        currency="usd",
        indicator_list=["policy_rate"],
        api_key=API_KEY,
        base_url="https://example.com/v1",
    )
    calendar = downloader.fetch_calendar()
    assert calendar.loc[0, "dataset"] == "calendar"
    assert calendar.loc[0, "forecast"] == 4.25
    assert bool(calendar.loc[0, "is_future"]) is True

    predictions = FXMacroDataMacroDownloader(
        currency="usd",
        indicator_list=["inflation"],
        api_key=API_KEY,
        base_url="https://example.com/v1",
    ).fetch_predictions()
    assert predictions.loc[0, "dataset"] == "predictions"
    assert predictions.loc[0, "prediction"] == 3.81
    assert predictions.loc[0, "prediction_count"] == 1
    assert len(payloads) == 2


def test_fxmacrodata_processor_adds_macro_features():
    price_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-30", "2026-05-31", "2026-06-01"]),
            "tic": ["EURUSD", "EURUSD", "EURUSD"],
            "close": [1.1, 1.2, 1.3],
        }
    )
    macro_data = pd.DataFrame(
        {
            "date": ["2026-05-31"],
            "currency": ["usd"],
            "indicator": ["inflation"],
            "value": [4.2],
            "actual": [4.2],
            "consensus": [3.9],
            "forecast": [4.0],
            "surprise": [0.3],
            "prediction": [4.1],
            "announcement_datetime": [1781094600],
        }
    )

    data = FXMacroDataProcessor().add_macro_features(price_data, macro_data)

    assert data["macro_usd_inflation_event"].tolist() == [0.0, 1.0, 0.0]
    assert pd.isna(data.loc[0, "macro_usd_inflation_value"])
    assert data.loc[1, "macro_usd_inflation_value"] == 4.2
    assert data.loc[2, "macro_usd_inflation_value"] == 4.2
