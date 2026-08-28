from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


class TestYahooFinanceProcessorScrapData(unittest.TestCase):

    def setUp(self):
        self.processor = YahooFinanceProcessor()

    @patch.object(YahooFinanceProcessor, "fetch_stock_data")
    @patch.object(YahooFinanceProcessor, "date_to_unix")
    def test_scrap_data_sorts_by_day_and_tic(self, mock_date_to_unix, mock_fetch):
        mock_date_to_unix.side_effect = lambda d: 1000 if d == "2020-01-01" else 2000

        # Sample df for AAPL
        df_aapl = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
                "open": [102.0, 100.0],
                "high": [106.0, 105.0],
                "low": [100.0, 98.0],
                "close": [105.0, 104.0],
                "adjcp": [102.9, 100.0],
                "volume": [11000, 10000],
                "tic": ["AAPL", "AAPL"],
                "day": [1, 0],
            }
        )

        # Sample df for MSFT
        df_msft = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
                "open": [152.0, 150.0],
                "high": [156.0, 155.0],
                "low": [150.0, 148.0],
                "close": [155.0, 154.0],
                "adjcp": [152.9, 150.0],
                "volume": [21000, 20000],
                "tic": ["MSFT", "MSFT"],
                "day": [1, 0],
            }
        )

        mock_fetch.side_effect = lambda stock_name, p1, p2: (
            df_aapl if stock_name == "AAPL" else df_msft
        )

        result = self.processor.scrap_data(["AAPL", "MSFT"], "2020-01-01", "2020-01-03")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)
        self.assertIn("tic", result.columns)
        self.assertIn("day", result.columns)

        # Verify sorted by day then tic
        self.assertEqual(result.iloc[0]["day"], 0)
        self.assertEqual(result.iloc[0]["tic"], "AAPL")
        self.assertEqual(result.iloc[1]["day"], 0)
        self.assertEqual(result.iloc[1]["tic"], "MSFT")
        self.assertEqual(result.iloc[2]["day"], 1)
        self.assertEqual(result.iloc[2]["tic"], "AAPL")
        self.assertEqual(result.iloc[3]["day"], 1)
        self.assertEqual(result.iloc[3]["tic"], "MSFT")

    @patch.object(YahooFinanceProcessor, "fetch_stock_data")
    @patch.object(YahooFinanceProcessor, "date_to_unix")
    def test_scrap_data_empty_on_failure(self, mock_date_to_unix, mock_fetch):
        mock_date_to_unix.side_effect = lambda d: 1000
        mock_fetch.side_effect = Exception("Fetch failed")

        result = self.processor.scrap_data(["INVALID"], "2020-01-01", "2020-01-03")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
