import datetime
import os
from sqlite3 import Date

import pandas as pd
import pytest
from finrl import config, config_tickers
from finrl.config import (DATA_SAVE_DIR, RESULTS_DIR, TENSORBOARD_LOG_DIR,
                          TRAINED_MODEL_DIR)
from finrl.finrl_meta.preprocessor.preprocessors import (FeatureEngineer,
                                                         data_split)
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.main import check_and_make_directories


@pytest.fixture(scope="session")
def DIRS():
    return [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]

@pytest.fixture(scope="session")
def ticker_list():
    return config_tickers.DOW_30_TICKER

@pytest.fixture(scope="session")
def ticker_list_small():
    return ["AAPL", "GOOG"]

@pytest.fixture(scope="session")
def indicators():
    return config.INDICATORS

@pytest.fixture(scope="session")
def start_date():
    return "2021-01-01"

@pytest.fixture(scope="session")
def end_date():
    return "2021-10-31"

def test_check_and_make_directories(DIRS):
    check_and_make_directories(DIRS)
    for dir in DIRS:
        assert os.path.exists(dir)

def test_download_large(ticker_list, start_date, end_date):
    df = YahooDownloader(start_date = start_date,
                     end_date = end_date,
                     ticker_list = ticker_list).fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (6300, 8)


def test_feature_engineer(ticker_list, indicators):
    assert isinstance(ticker_list, list)
    assert isinstance(indicators, list)

    df = YahooDownloader(start_date = '2009-01-01',
                     end_date = '2021-10-31',
                     ticker_list = ticker_list).fetch_data()
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = indicators,
                    use_vix=True,
                    use_turbulence=True,
                    user_defined_feature = False)
    processed = fe.preprocess_data(df)
    # to be continued
    # assert it's cleaned

    # assert technical indicators are added

    # assert vix is added


    # assert turbulence is added


    # assert user defined features are added

