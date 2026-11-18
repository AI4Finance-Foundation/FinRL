"""Keep tests that need the network or credentials out of the default run.

The Yahoo Finance downloads take minutes and fail whenever Yahoo changes
something; run them with `pytest --run-network`. The Alpaca downloader tests
call the live Alpaca data API and also need ALPACA_API_KEY and
ALPACA_API_SECRET in the environment.
"""

from __future__ import annotations

import os

NETWORK_TEST_FILES = {
    "test_core.py",  # downloads the 30 Dow tickers, asserts an exact row count
    "test_yahoo_downloader.py",  # downloads AAPL and GOOG
    "test_cash_penalty.py",  # fixture downloads AAPL and GOOG
}

CREDENTIAL_TEST_FILES = {
    "test_alpaca_downloader.py",  # live Alpaca data API, 401 without real keys
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run the tests that download data from Yahoo Finance or Alpaca",
    )


def _have_alpaca_credentials() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("ALPACA_API_SECRET")
    )


def pytest_ignore_collect(collection_path, config):
    run_network = config.getoption("--run-network")
    if collection_path.name in NETWORK_TEST_FILES and not run_network:
        return True
    if collection_path.name in CREDENTIAL_TEST_FILES and not (
        run_network and _have_alpaca_credentials()
    ):
        return True
    return None
