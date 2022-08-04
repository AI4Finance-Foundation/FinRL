from __future__ import annotations

import os
from typing import List

import pandas as pd
import pytest

from finrl import config
from finrl import config_tickers
from finrl.config import DATA_SAVE_DIR
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


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
def old_start_date():
    return "2009-01-01"


@pytest.fixture(scope="session")
def start_date():
    return "2021-01-01"


@pytest.fixture(scope="session")
def end_date():
    return "2021-10-31"


def test_check_and_make_directories(DIRS: list[str]) -> None:
    """
    Tests the creation of directories
    parameters:
    ----------
    DIRS : a List of str, which indicate the name of the folders to create
    """
    assert isinstance(DIRS, list)
    check_and_make_directories(DIRS)
    for dir in DIRS:
        assert os.path.exists(dir)


def test_download_large(ticker_list: list[str], start_date: str, end_date: str) -> None:
    """
    Tests the Yahoo Downloader and the returned data shape
    """
    assert isinstance(ticker_list, list)
    assert len(ticker_list > 0)
    assert isinstance(ticker_list[0], str)
    assert isinstance(start_date, str)
    assert isinstance(end_date, str)
    df = YahooDownloader(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list
    ).fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (6300, 8) or df.shape == (6270, 8)


def test_feature_engineer_no_turbulence(
    ticker_list: list[str],
    indicators: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """
    Tests the feature_engineer function - WIP
    """
    assert isinstance(ticker_list, list)
    assert len(ticker_list > 0)
    assert isinstance(ticker_list[0], str)
    assert isinstance(start_date, str)
    assert isinstance(end_date, str)
    assert isinstance(indicators, list)
    assert isinstance(indicators[0], str)

    df = YahooDownloader(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=True,
        use_turbulence=False,
        user_defined_feature=False,
    )
    assert isinstance(fe.preprocess_data(df), pd.DataFrame)


def test_feature_engineer_turbulence_less_than_a_year(
    ticker_list: list[str],
    indicators: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """
    Tests the feature_engineer function - with turbulence, start and end date
    are less than 1 year apart.
    the code should raise an error
    """
    assert isinstance(ticker_list, list)
    assert len(ticker_list > 0)
    assert isinstance(ticker_list[0], str)
    assert isinstance(start_date, str)
    assert isinstance(end_date, str)
    assert isinstance(indicators, list)
    assert isinstance(indicators[0], str)

    df = YahooDownloader(
        start_date=start_date, end_date=end_date, ticker_list=ticker_list
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    with pytest.raises(Exception):
        fe.preprocess_data(df)


def test_feature_engineer_turbulence_more_than_a_year(
    ticker_list: list[str],
    indicators: list[str],
    old_start_date: str,
    end_date: str,
) -> None:
    """
    Tests the feature_engineer function - with turbulence, start and end date
    are less than 1 year apart.
    the code should raise an error
    """
    assert isinstance(ticker_list, list)
    assert len(ticker_list > 0)
    assert isinstance(ticker_list[0], str)
    assert isinstance(start_date, str)
    assert isinstance(end_date, str)
    assert isinstance(indicators, list)
    assert isinstance(indicators[0], str)

    df = YahooDownloader(
        start_date=old_start_date, end_date=end_date, ticker_list=ticker_list
    ).fetch_data()
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    assert isinstance(fe.preprocess_data(df), pd.DataFrame)
