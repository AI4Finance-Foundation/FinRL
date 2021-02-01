import unittest
from finrl.marketdata.yahoodownloader import YahooDownloader
import pandas as pd


class TestDownloader(unittest.TestCase):
    def setUp(cls):
        cls.ticker_list = ["AAPL", "GOOG"]

    def test_download(self):
        df = YahooDownloader(
            start_date="2019-01-01", end_date="2019-02-01", ticker_list=self.ticker_list
        ).fetch_data()

        self.assertIsInstance(df, pd.DataFrame)
