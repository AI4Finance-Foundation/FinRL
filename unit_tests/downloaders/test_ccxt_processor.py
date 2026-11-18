from __future__ import annotations

import os
import time
import warnings

import pandas as pd
import pytest

from finrl.meta.data_processors.processor_ccxt import CCXTEngineer


@pytest.fixture
def engineer():
    def fake_fetch_ohlcv(symbol, timeframe, since, limit):
        return [
            [since + i * 3_600_000, 1.0, 2.0, 0.5, 1.5, 100.0] for i in range(limit)
        ]

    eng = CCXTEngineer()
    eng.binance.fetch_ohlcv = fake_fetch_ohlcv
    return eng


# The pre-fix code converted timestamps with the machine's local time, so a
# non-UTC zone is needed for these tests to fail against it.
@pytest.fixture
def new_york_tz():
    previous_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if previous_tz is None:
        del os.environ["TZ"]
    else:
        os.environ["TZ"] = previous_tz
    time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is Unix-only")
def test_index_is_utc_regardless_of_local_timezone(engineer, new_york_tz):
    dataset = engineer.data_fetch(
        "20240101 00:00:00", "20240102 00:00:00", pair_list=["BTC/USDT"], period="1h"
    )

    # Before the fix this index started at 2023-12-31 19:00 in New York.
    assert dataset.index[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert dataset.index[-1] == pd.Timestamp("2024-01-01 23:00:00")
    assert len(dataset) == 24
    assert dataset.index.tz is None


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is Unix-only")
def test_index_is_monotonic_across_dst_fall_back(engineer, new_york_tz):
    # New York falls back on 2024-11-03, so local-time conversion repeats 01:00.
    dataset = engineer.data_fetch(
        "20241103 00:00:00", "20241104 00:00:00", pair_list=["BTC/USDT"], period="1h"
    )

    assert dataset.index.is_monotonic_increasing
    assert dataset.index.is_unique
    assert len(dataset) == 24


def test_multiple_pairs_share_the_same_index(engineer):
    dataset = engineer.data_fetch(
        "20240101 00:00:00",
        "20240102 00:00:00",
        pair_list=["BTC/USDT", "ETH/USDT"],
        period="1h",
    )

    assert dataset.shape == (24, 10)
    assert dataset[("ETH/USDT", "close")].notna().all()


def test_no_deprecation_warning(engineer):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        engineer.data_fetch(
            "20240101 00:00:00",
            "20240102 00:00:00",
            pair_list=["BTC/USDT"],
            period="1h",
        )
