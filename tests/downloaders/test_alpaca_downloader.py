from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor


API_KEY = "PK4OZ7MKSTX1VS37EVAZ"
API_SECRET = "T2SJjYibqbXCxyK29JMPZNG1BZMH6IkTMQtdvrpp"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL", "GOOG"]


def test_intraDay_download(ticker_list):
    # Given
    start_date = '2021-07-29'
    end_date = '2021-07-30'
    time_interval = '1H'
    ticker_list = ["AAPL", "GOOG"]

    # When
    DP = AlpacaProcessor(
                      API_KEY = API_KEY,
                      API_SECRET = API_SECRET,
                      API_BASE_URL = API_BASE_URL
                      )
    data = DP.download_data(start_date = start_date,
                        end_date = end_date,
                        ticker_list = ticker_list,
                        time_interval= time_interval)

    # Then
    expected_shape = (24,9)
    expected_head_1 = pd.DataFrame([['2021-07-29 14:00:00', 2732.41, 2740.0, 2724.11, 2734.7525,
            120896, 8650, 2731.626021, 'GOOG']],columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count',
           'vwap', 'tic'])

    assert data.shape == expected_shape
    assert (data[data['tic'] == 'GOOG'].head(1) == expected_head_1).all(axis=None)
