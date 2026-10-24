from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


def test_scrap_data_sorts_by_day_and_tic():
    processor = YahooFinanceProcessor()
    processor.date_to_unix = MagicMock(side_effect=lambda d: 1000)

    def mock_fetch(stock_name, p1, p2):
        return pd.DataFrame(
            {
                "date": ["2020-01-02", "2020-01-01"],
                "open": [10.0, 9.0],
                "high": [11.0, 10.0],
                "low": [9.0, 8.0],
                "close": [10.5, 9.5],
                "adjcp": [10.5, 9.5],
                "volume": [100, 100],
                "tic": [stock_name, stock_name],
                "day": [1, 0],
            }
        )

    processor.fetch_stock_data = MagicMock(side_effect=mock_fetch)
    result = processor.scrap_data(["MSFT", "AAPL"], "2020-01-01", "2020-01-03")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result["day"]) == [0, 0, 1, 1]
    assert list(result["tic"]) == ["AAPL", "MSFT", "AAPL", "MSFT"]


def test_scrap_data_empty():
    processor = YahooFinanceProcessor()
    processor.date_to_unix = MagicMock(side_effect=lambda d: 1000)

    result = processor.scrap_data([], "2020-01-01", "2020-01-03")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
