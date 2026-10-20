from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.preprocessor.dow import add_dow_membership
from finrl.meta.preprocessor.dow import get_dow_constituents
from finrl.meta.preprocessor.preprocessors import FeatureEngineer


@pytest.mark.parametrize(
    "date, added, removed",
    [
        ("2014-01-02", "DD", "AAPL"),
        ("2015-03-19", "AAPL", "T"),
        ("2024-02-26", "AMZN", "WBA"),
        ("2024-11-08", "NVDA", "INTC"),
        ("2026-03-20", "AAPL", "INTC"),
    ],
)
def test_dow_roster_tracks_effective_membership(date, added, removed):
    constituents = get_dow_constituents(date)

    assert len(constituents) == 30
    assert added in constituents
    assert removed not in constituents


def test_membership_annotation_uses_each_row_date():
    data = pd.DataFrame(
        {
            "date": ["2014-01-02", "2015-03-18", "2015-03-19"],
            "tic": ["T", "AAPL", "AAPL"],
        }
    )

    annotated = add_dow_membership(data)

    assert annotated["is_member"].tolist() == [True, False, True]


def test_keep_incomplete_tickers_preserves_former_constituents():
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-02"],
            "tic": ["CURRENT", "CURRENT", "FORMER"],
            "close": [10.0, 11.0, 8.0],
        }
    )

    default_result = FeatureEngineer(use_technical_indicator=False).clean_data(data)
    history_result = FeatureEngineer(
        use_technical_indicator=False,
        keep_incomplete_tickers=True,
    ).clean_data(data, keep_incomplete_tickers=True)

    assert default_result["tic"].unique().tolist() == ["CURRENT"]
    assert set(history_result["tic"].unique()) == {"CURRENT", "FORMER"}


@pytest.mark.parametrize("date", ["2013-12-31", "2026-03-21"])
def test_out_of_range_membership_dates_are_rejected(date):
    with pytest.raises(ValueError, match="2014-01-01 through 2026-03-20"):
        get_dow_constituents(date)
