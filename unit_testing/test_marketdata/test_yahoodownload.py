import unittest

import pandas as pd

from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader


class TestDownloader(unittest.TestCase):
    def setUp(cls):
        cls.ticker_list = ["AAPL", "GOOG"]

    def test_download(self):
        df = YahooDownloader(
            start_date="2019-01-01", end_date="2019-02-01", ticker_list=self.ticker_list
        ).fetch_data()

        self.assertIsInstance(df, pd.DataFrame)
