from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


class TestYahooDownloaderAdjustPrices(unittest.TestCase):

    def setUp(self):
        """Set up a dummy YahooDownloader instance and test data."""
        # These init params are not used by _adjust_prices but needed for instantiation
        self.downloader = YahooDownloader(
            start_date="2020-01-01",
            end_date="2020-01-03",
            ticker_list=["AAPL"],
        )
        # Create a sample DataFrame similar to what fetch_data might produce before adjustment
        self.raw_data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "open": [100.0, 102.0],
                "high": [105.0, 106.0],
                "low": [98.0, 100.0],
                "close": [104.0, 105.0],
                "adjcp": [
                    100.0,
                    102.9,
                ],  # Adjusted close price - crucial for the method
                "volume": [10000, 11000],
                "tic": ["AAPL", "AAPL"],
            }
        )

    def test_adjust_prices_calculates_correctly(self):
        """Test that prices are adjusted correctly based on adjcp/close ratio."""
        # Explicitly ensure columns exist before passing to the method
        self.assertIn("adjcp", self.raw_data.columns)
        self.assertIn("close", self.raw_data.columns)

        adjusted_df = self.downloader._adjust_prices(self.raw_data.copy())

        # Calculate expected values
        adj_ratio_1 = self.raw_data.loc[0, "adjcp"] / self.raw_data.loc[0, "close"]
        adj_ratio_2 = self.raw_data.loc[1, "adjcp"] / self.raw_data.loc[1, "close"]

        expected_data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "open": [
                    self.raw_data.loc[0, "open"] * adj_ratio_1,
                    self.raw_data.loc[1, "open"] * adj_ratio_2,
                ],
                "high": [
                    self.raw_data.loc[0, "high"] * adj_ratio_1,
                    self.raw_data.loc[1, "high"] * adj_ratio_2,
                ],
                "low": [
                    self.raw_data.loc[0, "low"] * adj_ratio_1,
                    self.raw_data.loc[1, "low"] * adj_ratio_2,
                ],
                "close": [
                    self.raw_data.loc[0, "adjcp"],
                    self.raw_data.loc[1, "adjcp"],
                ],  # close becomes adjcp
                "volume": [10000, 11000],
                "tic": ["AAPL", "AAPL"],
            }
        )

        # Select only the columns present in the expected output for comparison
        # and ensure the same column order and index
        adjusted_df_compare = adjusted_df[expected_data.columns].reset_index(drop=True)
        expected_data = expected_data.reset_index(drop=True)

        # Use pandas testing utility for robust DataFrame comparison
        assert_frame_equal(adjusted_df_compare, expected_data, check_dtype=True)

    def test_adjust_prices_drops_columns(self):
        """Test that 'adjcp' and the temporary 'adj' columns are dropped."""
        # Explicitly ensure columns exist before passing to the method
        self.assertIn("adjcp", self.raw_data.columns)
        self.assertIn("close", self.raw_data.columns)

        adjusted_df = self.downloader._adjust_prices(self.raw_data.copy())

        self.assertNotIn("adjcp", adjusted_df.columns)
        self.assertNotIn("adj", adjusted_df.columns)
        # Ensure other essential columns remain
        self.assertIn("open", adjusted_df.columns)
        self.assertIn(
            "close", adjusted_df.columns
        )  # Note: This is the *new* adjusted close
        self.assertIn("tic", adjusted_df.columns)
        self.assertIn("date", adjusted_df.columns)
        self.assertIn("volume", adjusted_df.columns)


if __name__ == "__main__":
    unittest.main()
