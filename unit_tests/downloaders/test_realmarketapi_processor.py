from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.meta.data_processors.processor_realmarketapi import RealMarketAPIError
from finrl.meta.data_processors.processor_realmarketapi import (
    RealMarketAPIProcessor,
)

API_KEY = "test_key"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    """Returns queued responses and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return self.responses.pop(0)


def candle(open_time, open_, high, low, close, volume=1.0, symbol="XAUUSD"):
    return {
        "symbolCode": symbol,
        "openPrice": open_,
        "highPrice": high,
        "lowPrice": low,
        "closePrice": close,
        "volume": volume,
        "openTime": open_time,
        "bid": close,
        "ask": close,
    }


def history_page(rows, page, total_pages):
    return FakeResponse(
        {
            "data": rows,
            "totalCount": len(rows),
            "currentPage": page,
            "totalPages": total_pages,
            "pageSize": 1000,
        }
    )


def test_download_data_pages_history_and_sorts(monkeypatch):
    monkeypatch.delenv("REALMARKETAPI_API_KEY", raising=False)
    session = FakeSession(
        [
            # H1 candles, newest first, as the API returns them
            history_page(
                [
                    candle("2024-01-02T02:00:00+00:00", 3, 4, 2, 3.5),
                    candle("2024-01-02T01:00:00+00:00", 2, 3, 1, 2.5),
                ],
                page=1,
                total_pages=2,
            ),
            history_page(
                [
                    # duplicate of a page-1 row, as happens if new candles arrive
                    candle("2024-01-02T01:00:00+00:00", 2, 3, 1, 2.5),
                    candle("2024-01-02T00:00:00+00:00", 1, 2, 0.5, 1.5),
                ],
                page=2,
                total_pages=2,
            ),
        ]
    )
    processor = RealMarketAPIProcessor(
        api_key=API_KEY, base_url="https://example.com/", session=session
    )

    df = processor.download_data(["XAUUSD"], "2024-01-02", "2024-01-03", "1h")

    assert len(session.calls) == 2
    first = session.calls[0]
    assert first["url"] == "https://example.com/api/v1/history"
    assert first["params"] == {
        "apiKey": API_KEY,
        "symbolCode": "XAUUSD",
        "startTime": "2024-01-02T00:00:00Z",
        "endTime": "2024-01-03T00:00:00Z",
        "pageNumber": 1,
        "pageSize": 1000,
    }
    assert first["timeout"] == 30
    assert session.calls[1]["params"]["pageNumber"] == 2

    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tic",
    ]
    assert df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 00:00"),
        pd.Timestamp("2024-01-02 01:00"),
        pd.Timestamp("2024-01-02 02:00"),
    ]
    assert df["close"].tolist() == [1.5, 2.5, 3.5]
    assert df["tic"].unique().tolist() == ["XAUUSD"]


def test_download_data_resamples_h1_to_4h():
    rows = [
        candle("2024-01-02T06:00:00+01:00", 4, 9, 3, 8, volume=4),  # 05:00 UTC
        candle("2024-01-02T04:00:00+00:00", 3, 5, 2, 4, volume=3),
        candle("2024-01-02T03:00:00+00:00", 2, 6, 1, 3, volume=2),
        candle("2024-01-02T00:00:00+00:00", 1, 2, 0.5, 1.5, volume=1),
    ]
    session = FakeSession([history_page(rows, page=1, total_pages=1)])
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    df = processor.download_data(["XAUUSD"], "2024-01-02", "2024-01-03", "4h")

    assert session.calls[0]["url"].endswith("/api/v1/history")
    assert df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 00:00"),
        pd.Timestamp("2024-01-02 04:00"),
    ]
    assert df["open"].tolist() == [1.0, 3.0]
    assert df["high"].tolist() == [6.0, 9.0]
    assert df["low"].tolist() == [0.5, 2.0]
    assert df["close"].tolist() == [3.0, 8.0]
    assert df["volume"].tolist() == [3.0, 7.0]


def test_download_data_resamples_h1_to_1d():
    # Hours spread over two calendar days; only the grouping is checked,
    # not where the daily bar starts.
    rows = [
        candle("2024-01-03T12:00:00Z", 6, 7, 5, 6.5, volume=1),
        candle("2024-01-03T11:00:00Z", 5, 8, 4, 5.5, volume=1),
        candle("2024-01-02T13:00:00Z", 2, 4, 1, 3, volume=2),
        candle("2024-01-02T12:00:00Z", 1, 3, 0.5, 2, volume=2),
    ]
    session = FakeSession([history_page(rows, page=1, total_pages=1)])
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    df = processor.download_data(["XAUUSD"], "2024-01-01", "2024-01-05", "1d")

    assert len(df) == 2
    assert df["open"].tolist() == [1.0, 5.0]
    assert df["high"].tolist() == [4.0, 8.0]
    assert df["low"].tolist() == [0.5, 4.0]
    assert df["close"].tolist() == [3.0, 6.5]
    assert df["volume"].tolist() == [4.0, 2.0]


def candle_window(rows):
    return FakeResponse({"data": rows, "totalCount": len(rows)})


def test_minute_intervals_use_candle_window_filtered_to_range():
    session = FakeSession(
        [
            candle_window(
                [
                    candle("2024-01-02T10:45:00Z", 4, 4, 4, 4),
                    candle("2024-01-02T10:30:00Z", 3, 3, 3, 3),
                    candle("2024-01-02T10:15:00Z", 2, 2, 2, 2),
                    candle("2024-01-02T10:00:00Z", 1, 1, 1, 1),
                ]
            )
        ]
    )
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    df = processor.download_data(
        ["XAUUSD"], "2024-01-02 10:15", "2024-01-02 10:45", "15m"
    )

    assert session.calls[0]["url"].endswith("/api/v1/candle")
    assert session.calls[0]["params"] == {
        "apiKey": API_KEY,
        "symbolCode": "XAUUSD",
        "timeFrame": "M15",
    }
    assert df["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02 10:15"),
        pd.Timestamp("2024-01-02 10:30"),
    ]
    assert df["close"].tolist() == [2.0, 3.0]


@pytest.mark.parametrize(
    "rows",
    [
        [candle("2024-01-02T10:00:00Z", 1, 1, 1, 1)],
        [],
    ],
)
def test_minute_interval_start_older_than_window_raises(rows):
    session = FakeSession([candle_window(rows)])
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    with pytest.raises(ValueError, match="recent window of 1m candles"):
        processor.download_data(
            ["XAUUSD"], "2024-01-02 09:00", "2024-01-02 11:00", "1m"
        )
    assert session.calls[0]["params"]["timeFrame"] == "M1"


def test_api_key_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("REALMARKETAPI_API_KEY", "env_key")
    session = FakeSession([history_page([], page=1, total_pages=0)])
    processor = RealMarketAPIProcessor(session=session)

    df = processor.download_data(["EURUSD"], "2024-01-02", "2024-01-03", "1h")

    assert session.calls[0]["params"]["apiKey"] == "env_key"
    assert df.empty


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("REALMARKETAPI_API_KEY", raising=False)
    processor = RealMarketAPIProcessor(session=FakeSession([]))
    with pytest.raises(ValueError, match="REALMARKETAPI_API_KEY"):
        processor.download_data(["EURUSD"], "2024-01-02", "2024-01-03", "1d")


@pytest.mark.parametrize("interval", ["2m", "1w", "30m"])
def test_unsupported_interval_raises(interval):
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=FakeSession([]))
    with pytest.raises(ValueError, match="Unsupported time_interval"):
        processor.download_data(["EURUSD"], "2024-01-02", "2024-01-03", interval)


def test_start_after_end_raises():
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=FakeSession([]))
    with pytest.raises(ValueError, match="earlier than end_date"):
        processor.download_data(["EURUSD"], "2024-01-03", "2024-01-02", "1d")


def test_problem_details_error_includes_error_code():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "type": "https://tools.ietf.org/html/rfc9110#section-15.5.1",
                    "title": "One or more validation errors occurred.",
                    "status": 400,
                    "errors": [
                        {
                            "name": "symbolCode",
                            "reason": "Symbol is not supported.",
                            "errorCode": "ERR_0008_SYMBOL_NOT_SUPPORTED",
                        }
                    ],
                },
                status_code=400,
            )
        ]
    )
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    with pytest.raises(RealMarketAPIError) as exc_info:
        processor.download_data(["AAPL"], "2024-01-02", "2024-01-03", "1d")

    assert exc_info.value.status_code == 400
    assert exc_info.value.error_codes == ["ERR_0008_SYMBOL_NOT_SUPPORTED"]
    assert "ERR_0008_SYMBOL_NOT_SUPPORTED" in str(exc_info.value)
    assert "Symbol is not supported." in str(exc_info.value)


def test_error_without_json_body():
    session = FakeSession([FakeResponse(ValueError("no json"), status_code=502)])
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)
    with pytest.raises(RealMarketAPIError, match="HTTP 502"):
        processor.download_data(["XAUUSD"], "2024-01-02", "2024-01-03", "1d")


def test_download_latest_candles_uses_timeframe():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "data": [
                        candle("2024-01-02T01:00:00Z", 2, 3, 1, 2.5),
                        candle("2024-01-02T00:00:00Z", 1, 2, 0.5, 1.5),
                    ],
                    "totalCount": 2,
                }
            )
        ]
    )
    processor = RealMarketAPIProcessor(api_key=API_KEY, session=session)

    df = processor.download_latest_candles(["XAUUSD"], "1h")

    assert session.calls[0]["url"].endswith("/api/v1/candle")
    assert session.calls[0]["params"] == {
        "apiKey": API_KEY,
        "symbolCode": "XAUUSD",
        "timeFrame": "H1",
    }
    assert df["close"].tolist() == [1.5, 2.5]


def test_clean_data_aligns_tickers_with_different_sessions():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]
                + ["2024-01-05", "2024-01-08"]
            ),
            "open": [1, 2, 3, 4, 10, 11],
            "high": [1, 2, 3, 4, 10, 11],
            "low": [1, 2, 3, 4, 10, 11],
            "close": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0],
            "volume": [1, 1, 1, 1, 5, 5],
            "tic": ["BTCUSD"] * 4 + ["XAUUSD"] * 2,
        }
    )

    cleaned = RealMarketAPIProcessor(api_key=API_KEY).clean_data(df)

    assert cleaned.groupby("tic").size().tolist() == [4, 4]
    gold = cleaned[cleaned.tic == "XAUUSD"]
    assert gold["close"].tolist() == [10.0, 10.0, 10.0, 11.0]
    assert gold["open"].tolist() == [10.0, 10.0, 10.0, 11.0]
    assert gold["volume"].tolist() == [5.0, 0.0, 0.0, 5.0]


def test_indicators_and_df_to_array_shapes():
    times = pd.date_range("2024-01-01", periods=40, freq="h")
    frames = []
    for i, tic in enumerate(["BTCUSD", "XAUUSD"]):
        close = np.linspace(100, 140, len(times)) * (i + 1)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": times,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1.0,
                    "tic": tic,
                }
            )
        )
    processor = RealMarketAPIProcessor(api_key=API_KEY)
    df = processor.clean_data(pd.concat(frames, ignore_index=True))
    df = processor.add_technical_indicator(df, ["macd", "rsi_30"])
    df = processor.add_turbulence(df)
    df = processor.add_vix(df)

    assert {"macd", "rsi_30", "turbulence", "VIXY"} <= set(df.columns)

    price, tech, turbulence = processor.df_to_array(df, ["macd", "rsi_30"], True)
    assert price.shape == (40, 2)
    assert tech.shape == (40, 4)
    assert turbulence.shape == (40,)
    np.testing.assert_allclose(price[:, 1], price[:, 0] * 2)


def test_data_processor_dispatches_to_realmarketapi():
    data_processor_module = pytest.importorskip("finrl.meta.data_processor")

    dp = data_processor_module.DataProcessor("realmarketapi", api_key=API_KEY)

    assert isinstance(dp.processor, RealMarketAPIProcessor)
    assert dp.processor.api_key == API_KEY
    assert dp.processor.base_url == "https://api.realmarketapi.com"


def test_connection_error_does_not_leak_api_key():
    requests = pytest.importorskip("requests")

    class FailingSession:
        def get(self, url, params=None, timeout=None):
            raise requests.ConnectionError(
                f"Max retries exceeded with url: {url}?apiKey={params['apiKey']}"
            )

    processor = RealMarketAPIProcessor(api_key=API_KEY, session=FailingSession())

    with pytest.raises(RealMarketAPIError) as excinfo:
        processor.download_data(["XAUUSD"], "2024-01-01", "2024-01-02", "1h")

    assert API_KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
