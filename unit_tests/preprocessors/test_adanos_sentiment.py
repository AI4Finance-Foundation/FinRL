from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "finrl"
    / "meta"
    / "preprocessor"
    / "adanos_sentiment.py"
)
SPEC = importlib.util.spec_from_file_location("adanos_sentiment", MODULE_PATH)
ADANOS_SENTIMENT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ADANOS_SENTIMENT
SPEC.loader.exec_module(ADANOS_SENTIMENT)

add_adanos_market_sentiment = ADANOS_SENTIMENT.add_adanos_market_sentiment


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"bad status: {self.status_code}")


class DummySession:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params, headers, timeout):
        self.calls.append((url, params, headers, timeout))
        return self.responses[url]


class FailingSession:
    def get(self, url, params, headers, timeout):
        raise AssertionError("API should not be called without an explicit api_key")


def _simple_ohlcv_frame(ticker="AAPL", dates=None):
    if dates is None:
        dates = ["2026-03-01", "2026-03-02"]
    return pd.DataFrame(
        {
            "date": dates,
            "tic": [ticker] * len(dates),
            "close": [100.0 + idx for idx in range(len(dates))],
        }
    )


def test_add_adanos_market_sentiment_is_noop_without_api_key():
    df = _simple_ohlcv_frame()

    result = add_adanos_market_sentiment(df, api_key=None)

    assert list(result.columns) == ["date", "tic", "close"]
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_add_adanos_market_sentiment_default_no_key_skips_network_call():
    df = _simple_ohlcv_frame()

    result = add_adanos_market_sentiment(df, session=FailingSession())

    pd.testing.assert_frame_equal(result, df)


def test_add_adanos_market_sentiment_merges_lagged_daily_features():
    df = _simple_ohlcv_frame()

    responses = {
        "https://api.adanos.org/reddit/stocks/v1/stock/AAPL": DummyResponse(
            {
                "daily_trend": [
                    {
                        "date": "2026-03-01",
                        "buzz_score": 40.0,
                        "sentiment_score": 0.10,
                    },
                    {
                        "date": "2026-03-02",
                        "buzz_score": 60.0,
                        "sentiment_score": 0.30,
                    },
                ]
            }
        ),
        "https://api.adanos.org/news/stocks/v1/stock/AAPL": DummyResponse(
            {
                "daily_trend": [
                    {
                        "date": "2026-03-01",
                        "buzz_score": 50.0,
                        "sentiment_score": 0.20,
                    },
                    {
                        "date": "2026-03-02",
                        "buzz_score": 70.0,
                        "sentiment_score": 0.40,
                    },
                ]
            }
        ),
    }

    session = DummySession(responses)
    result = add_adanos_market_sentiment(
        df,
        api_key="test-key",
        sources=("reddit", "news"),
        session=session,
    )

    assert result.loc[0, "adanos_buzz_mean_lag1"] == 0.0
    assert result.loc[0, "adanos_source_coverage_lag1"] == 0.0
    assert result.loc[1, "adanos_reddit_buzz_lag1"] == 40.0
    assert result.loc[1, "adanos_news_buzz_lag1"] == 50.0
    assert result.loc[1, "adanos_buzz_mean_lag1"] == 45.0
    assert result.loc[1, "adanos_sentiment_mean_lag1"] == pytest.approx(0.15)
    assert result.loc[1, "adanos_source_coverage_lag1"] == 2.0
    assert len(session.calls) == 2


def test_add_adanos_market_sentiment_ignores_missing_sources():
    df = _simple_ohlcv_frame(ticker="TSLA")

    responses = {
        "https://api.adanos.org/reddit/stocks/v1/stock/TSLA": DummyResponse(
            {"daily_trend": []},
            status_code=404,
        ),
        "https://api.adanos.org/x/stocks/v1/stock/TSLA": DummyResponse(
            {
                "daily_trend": [
                    {
                        "date": "2026-03-01",
                        "buzz_score": 55.0,
                        "sentiment_score": 0.25,
                    }
                ]
            }
        ),
    }

    session = DummySession(responses)
    result = add_adanos_market_sentiment(
        df,
        api_key="test-key",
        sources=("reddit", "x"),
        session=session,
    )

    assert "adanos_x_buzz_lag1" in result.columns
    assert result.loc[1, "adanos_x_buzz_lag1"] == 55.0
    assert result.loc[1, "adanos_source_coverage_lag1"] == 1.0


def test_add_adanos_market_sentiment_supports_conservative_lag():
    df = _simple_ohlcv_frame(
        dates=["2026-03-01", "2026-03-02", "2026-03-03"],
    )

    responses = {
        "https://api.adanos.org/reddit/stocks/v1/stock/AAPL": DummyResponse(
            {
                "daily_trend": [
                    {
                        "date": "2026-03-01",
                        "buzz_score": 40.0,
                        "sentiment_score": 0.10,
                    },
                    {
                        "date": "2026-03-02",
                        "buzz_score": 60.0,
                        "sentiment_score": 0.30,
                    },
                    {
                        "date": "2026-03-03",
                        "buzz_score": 80.0,
                        "sentiment_score": 0.50,
                    },
                ]
            }
        )
    }

    result = add_adanos_market_sentiment(
        df,
        api_key="test-key",
        sources=("reddit",),
        lag=2,
        session=DummySession(responses),
    )

    assert "adanos_buzz_mean_lag2" in result.columns
    assert "adanos_buzz_mean_lag1" not in result.columns
    assert result.loc[0, "adanos_buzz_mean_lag2"] == 0.0
    assert result.loc[1, "adanos_buzz_mean_lag2"] == 0.0
    assert result.loc[2, "adanos_reddit_buzz_lag2"] == 40.0
    assert result.loc[2, "adanos_sentiment_mean_lag2"] == pytest.approx(0.10)
