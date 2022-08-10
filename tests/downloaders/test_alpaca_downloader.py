from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor


API_KEY = "???"
API_SECRET = "???"
API_BASE_URL = "https://paper-api.alpaca.markets"
data_url = "wss://data.alpaca.markets"


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
