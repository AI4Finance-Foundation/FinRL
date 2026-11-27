from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytest

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor


API_KEY = "???"
API_SECRET = "???"
API_BASE_URL = "https://paper-api.alpaca.markets"
data_url = "wss://data.alpaca.markets"


def test_trading_days_accept_timezone_aware_dates(monkeypatch):
    calendar = tc.get_calendar("NYSE")
    date_range = calendar.date_range_htf

    def date_range_requiring_naive_bounds(frequency, start, end):
        for bound in (start, end):
            if pd.Timestamp(bound).tz is not None:
                raise ValueError("Date must be timezone naive")
        return date_range(frequency, start, end)

    monkeypatch.setattr(calendar, "date_range_htf", date_range_requiring_naive_bounds)
    monkeypatch.setattr(tc, "get_calendar", lambda name: calendar)
    processor = AlpacaProcessor(client=object())
    assert processor.get_trading_days(
        pd.Timestamp("2024-07-03", tz="UTC"),
        pd.Timestamp("2024-07-05", tz="America/New_York"),
    ) == ["2024-07-03", "2024-07-05"]


def test_clean_data_aligns_duplicate_and_missing_bars(monkeypatch):
    processor = AlpacaProcessor(client=object())
    processor.start = processor.end = pd.Timestamp("2024-07-05", tz="UTC")
    start = pd.Timestamp("2024-07-05 09:30", tz="America/New_York")
    end = start + pd.Timedelta(minutes=1)
    bars = pd.DataFrame(
        [
            (start, "AAA", 10.0),
            (start, "AAA", 11.0),
            (start, "BBB", 20.0),
            (end, "AAA", 12.0),
        ],
        columns=["timestamp", "tic", "close"],
    )
    bars = bars.assign(open=bars.close, high=bars.close, low=bars.close, volume=1)

    cleaned = processor.clean_data(bars)
    vix = (
        cleaned.loc[cleaned["tic"] == "AAA", ["timestamp", "close"]]
        .copy()
        .assign(close=30.0)
    )
    monkeypatch.setattr(processor, "download_and_clean_data", lambda: vix)
    featured = processor.add_technical_indicator(cleaned, ["close_2_sma"])
    prices, features, volatility = processor.df_to_array(
        processor.add_vix(featured), ["close_2_sma", "volume"], if_vix=True
    )

    assert prices.shape == (390, 2)
    assert features.shape == (390, 4)
    np.testing.assert_array_equal(prices[:2], [[11.0, 20.0], [12.0, 20.0]])
    np.testing.assert_array_equal(
        features[:2], [[11.0, 1.0, 20.0, 1.0], [11.5, 1.0, 20.0, 0.0]]
    )
    np.testing.assert_array_equal(volatility, np.full(390, 30.0))


def test_df_to_array_rejects_unaligned_ticker_timestamps():
    processor = AlpacaProcessor(client=object())
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-07-03", "2024-07-05"], utc=True),
            "tic": ["AAA", "BBB"],
            "close": [10.0, 20.0],
            "VIXY": [30.0, 30.0],
        }
    )

    with pytest.raises(ValueError, match="one bar per ticker at each timestamp"):
        processor.df_to_array(data, [], if_vix=True)


def test_fetch_latest_data_aligns_staggered_bars():
    start = pd.Timestamp("2024-07-03 13:30", tz="UTC")
    end = start + pd.Timedelta(minutes=1)

    def get_stock_bars(request):
        symbol = request.symbol_or_symbols
        if isinstance(symbol, list):
            symbol = symbol[0]
        timestamps, prices = {
            "AAA": ([start, end], [10.0, 11.0]),
            "BBB": ([end], [20.0]),
            "VIXY": ([end], [30.0]),
        }[symbol]
        index = pd.MultiIndex.from_arrays(
            [[symbol] * len(timestamps), timestamps], names=["symbol", "timestamp"]
        )
        bars = pd.DataFrame(
            {column: prices for column in ["open", "high", "low", "close", "volume"]},
            index=index,
        )
        return SimpleNamespace(df=bars)

    processor = AlpacaProcessor(client=SimpleNamespace(get_stock_bars=get_stock_bars))
    prices, indicators, volatility = processor.fetch_latest_data(
        ["AAA", "BBB"], "1Min", ["close_2_sma"]
    )
    np.testing.assert_allclose(prices, [11.0, 20.0])
    np.testing.assert_allclose(indicators, [10.5, 20.0])
    np.testing.assert_allclose(volatility, [30.0])


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]


def test_intraDayBar_download(ticker_list):
    # Given
    start_date = "2021-07-29"
    end_date = "2021-07-30"
    time_interval = "1H"
    ticker_list = ["AAPL", "GOOG"]

    # When
    DP = AlpacaProcessor(
        API_KEY=API_KEY, API_SECRET=API_SECRET, API_BASE_URL=API_BASE_URL
    )
    data = DP.download_data(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
        time_interval=time_interval,
    )
    actual_head_1 = data[data["tic"] == "GOOG"].head(1).reset_index(drop=True)
    # Then
    expected_shape = (12, 9)
    expected_head_1 = pd.DataFrame(
        [
            [
                "2021-07-29 14:00:00",
                2732.41,
                2740.0,
                2724.11,
                2734.7525,
                120896,
                8650,
                2731.626021,
                "GOOG",
            ]
        ],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "tic",
        ],
    )

    assert data.shape == expected_shape
    assert (actual_head_1 == expected_head_1).all(axis=None)


def test_dayBar_download(ticker_list):
    # Given
    start_date = "2021-07-29"
    end_date = "2021-07-30"
    time_interval = "1D"
    ticker_list = ["AAPL", "GOOG"]

    # When
    DP = AlpacaProcessor(
        API_KEY=API_KEY, API_SECRET=API_SECRET, API_BASE_URL=API_BASE_URL
    )
    data = DP.download_data(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
        time_interval=time_interval,
    )

    # Then
    expected = pd.DataFrame(
        [
            [
                "2021-07-29 04:00:00",
                144.66,
                146.55,
                144.58,
                145.64,
                56571097,
                414821,
                145.806207,
                "AAPL",
            ],
            [
                "2021-07-30 04:00:00",
                144.49,
                146.33,
                144.11,
                145.86,
                70291908,
                464977,
                145.396798,
                "AAPL",
            ],
            [
                "2021-07-29 04:00:00",
                2722.76,
                2743.03,
                2722.76,
                2730.81,
                962833,
                56703,
                2732.943535,
                "GOOG",
            ],
            [
                "2021-07-30 04:00:00",
                2710.22,
                2715.4273,
                2696.284,
                2704.42,
                1192648,
                55739,
                2704.685882,
                "GOOG",
            ],
        ],
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
            "tic",
        ],
    )

    assert (data == expected).all(axis=None)
