from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.applications.stock_trading.fundamental_stock_trading import (
    align_fundamentals_to_daily,
)
from finrl.applications.stock_trading.fundamental_stock_trading import (
    fundamentals_available_date,
)


def make_prices(start="2020-01-01", end="2020-07-10", tickers=("AAA", "BBB")):
    """Business days only, like the prices from a downloader."""
    dates = pd.bdate_range(start, end)
    frames = [
        pd.DataFrame(
            {
                "date": dates,
                "tic": tic,
                "close": 100.0 * (i + 1) + np.arange(len(dates)),
            }
        )
        for i, tic in enumerate(tickers)
    ]
    return pd.concat(frames, ignore_index=True)


def make_ratios():
    """AAA and BBB, available 2020-02-15 (a Saturday) and 2020-05-05."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-15", "2020-05-05"] * 2),
            "tic": ["AAA", "AAA", "BBB", "BBB"],
            "EPS": [1.0, 2.0, 10.0, 20.0],
        }
    )


def eps(daily, tic, date):
    row = daily[(daily["tic"] == tic) & (daily["date"] == pd.Timestamp(date))]
    assert len(row) == 1
    return row["EPS"].iloc[0]


def test_available_date_is_the_day_after_the_report():
    available = fundamentals_available_date(
        pd.Series([20200331, 20200630]), pd.Series([20200428.0, 20200728.0])
    )
    assert available.name == "date"
    assert available.tolist() == [
        pd.Timestamp("2020-04-29"),
        pd.Timestamp("2020-07-29"),
    ]


def test_available_date_waits_when_the_report_date_is_missing():
    available = fundamentals_available_date(
        pd.Series([20200331, 20200630]), pd.Series([np.nan, 20200728.0])
    )
    assert available.tolist() == [
        pd.Timestamp("2020-06-29"),
        pd.Timestamp("2020-07-29"),
    ]


def test_ratios_only_appear_once_available():
    daily = align_fundamentals_to_daily(make_prices(), make_ratios())
    assert np.isnan(eps(daily, "AAA", "2020-02-14"))
    assert eps(daily, "AAA", "2020-05-04") == 1.0
    assert eps(daily, "AAA", "2020-05-05") == 2.0
    # Every row's value was published on or before that row's date
    public = make_ratios().assign(public=lambda x: x["date"]).drop(columns="date")
    merged = daily.dropna(subset=["EPS"]).merge(public, on=["tic", "EPS"])
    assert (merged["public"] <= merged["date"]).all()


def test_ratios_do_not_cross_tickers():
    daily = align_fundamentals_to_daily(make_prices(), make_ratios())
    assert set(daily.loc[daily["tic"] == "AAA", "EPS"].dropna()) == {1.0, 2.0}
    assert set(daily.loc[daily["tic"] == "BBB", "EPS"].dropna()) == {10.0, 20.0}
    # After its last report a ticker keeps its own value
    assert eps(daily, "AAA", "2020-07-10") == 2.0


def test_ratios_available_on_a_weekend_reach_the_next_trading_day():
    daily = align_fundamentals_to_daily(make_prices(), make_ratios())
    assert eps(daily, "AAA", "2020-02-17") == 1.0


def test_changing_the_future_does_not_change_the_past():
    prices, ratios = make_prices(), make_ratios()
    cutoff = pd.Timestamp("2020-04-01")
    before = align_fundamentals_to_daily(prices, ratios)

    changed_prices = prices.copy()
    changed_prices.loc[changed_prices["date"] >= cutoff, "close"] *= 2
    changed_ratios = ratios.copy()
    changed_ratios.loc[changed_ratios["date"] >= cutoff, "EPS"] *= 2
    after = align_fundamentals_to_daily(changed_prices, changed_ratios)

    pd.testing.assert_frame_equal(
        before[before["date"] < cutoff], after[after["date"] < cutoff]
    )


def test_only_trading_days_are_kept():
    prices = make_prices()
    daily = align_fundamentals_to_daily(prices, make_ratios())
    assert set(daily["date"]) == set(prices["date"])
    assert (daily["date"].dt.day_of_week < 5).all()
    assert len(daily) == prices["date"].nunique() * prices["tic"].nunique()


def test_prices_are_not_filled_from_the_future():
    prices = make_prices()
    # BBB lists on 2020-03-02, and has no price on 2020-04-01
    listed = (prices["tic"] != "BBB") | (prices["date"] >= "2020-03-02")
    gap = (prices["tic"] == "BBB") & (prices["date"] == "2020-04-01")
    daily = align_fundamentals_to_daily(prices[listed & ~gap], make_ratios())
    bbb = daily[daily["tic"] == "BBB"].set_index("date")["close"]
    assert bbb[bbb.index < "2020-03-02"].isna().all()
    assert bbb["2020-04-01"] == bbb["2020-03-31"]


def test_reports_before_the_first_price_are_kept():
    ratios = pd.DataFrame(
        {"date": pd.to_datetime(["2019-11-05"]), "tic": ["AAA"], "EPS": [5.0]}
    )
    daily = align_fundamentals_to_daily(make_prices(), ratios)
    assert eps(daily, "AAA", "2020-01-01") == 5.0


def test_later_quarter_wins_when_two_are_reported_together():
    ratios = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-03", "2020-02-03"]),
            "tic": ["AAA", "AAA"],
            "EPS": [1.0, 2.0],
        }
    )
    prices = make_prices()
    daily = align_fundamentals_to_daily(prices, ratios)
    assert len(daily) == prices["date"].nunique() * prices["tic"].nunique()
    assert eps(daily, "AAA", "2020-02-03") == 2.0


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_date_resolution_does_not_matter(unit):
    prices = make_prices()
    prices["date"] = prices["date"].astype(f"datetime64[{unit}]")
    daily = align_fundamentals_to_daily(prices, make_ratios())
    assert eps(daily, "AAA", "2020-05-05") == 2.0
