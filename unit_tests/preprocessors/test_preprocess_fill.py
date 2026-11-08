from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.preprocessor.preprocessors import FeatureEngineer


def _constant_price_frame(levels: dict[str, float], n_days: int = 45) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days).strftime("%Y-%m-%d")
    rows = [
        {
            "date": date,
            "tic": tic,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1.0,
            "day": 0,
        }
        for date in dates
        for tic, price in levels.items()
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("keep_incomplete_tickers", [False, True])
def test_missing_values_are_filled_within_each_ticker(keep_incomplete_tickers):
    # close_30_sma is NaN for the first 29 days of every ticker. Filling the
    # whole frame, which is sorted by date and then ticker, gave every ticker
    # the first ticker's day-30 value during its own warm-up window.
    levels = {"AAA": 10.0, "BBB": 100.0, "CCC": 1000.0}
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["close_30_sma"],
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
        keep_incomplete_tickers=keep_incomplete_tickers,
    )

    processed = fe.preprocess_data(_constant_price_frame(levels))

    assert not processed["close_30_sma"].isna().any()
    for tic, price in levels.items():
        sma = processed.loc[processed["tic"] == tic, "close_30_sma"]
        assert sma.tolist() == pytest.approx([price] * len(sma))
