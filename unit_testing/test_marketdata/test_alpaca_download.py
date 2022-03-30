import unittest

import pandas as pd

from finrl.finrl_meta.data_processor import DataProcessor


class TestAlpacaDownload(unittest.TestCase):
    def setUp(cls):
        cls.ticker_list = ["AAPL", "GOOG"]

    @unittest.skip("Skipped cuz it needs APIKEY-APISECRET to connect to alpaca")
    def test_download(self):
        API_KEY = "APPLY YOU API KEY"
        API_SECRET = "APPLY YOU API SECRET"
        APCA_API_BASE_URL = 'https://data.alpaca.markets'

        DP = DataProcessor(data_source='alpaca',
                           API_KEY=API_KEY,
                           API_SECRET=API_SECRET,
                           APCA_API_BASE_URL=APCA_API_BASE_URL)

        data = DP.download_data(start_date="2019-01-01", end_date="2019-02-01", ticker_list=self.ticker_list, time_interval='1Min')

        self.assertIsInstance(data, pd.DataFrame)
