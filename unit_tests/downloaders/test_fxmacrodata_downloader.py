from __future__ import annotations

import json

import pandas as pd
import pytest

from finrl.meta.data_processors.processor_fxmacrodata import FXMacroDataProcessor
from finrl.meta.preprocessor.fxmacrodatadownloader import FXMacroDataDownloader

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
