"""Point-in-time DJIA constituents for the 2014--2026 tutorial windows.

Membership changes are effective on the dates published by S&P Dow Jones
Indices. The event list is based on the historical components table and its
linked index announcements:
https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average
"""

from __future__ import annotations

import pandas as pd

from finrl.config_tickers import DOW_30_TICKER

_INITIAL_CONSTITUENTS_2014 = {
    "AXP",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "DD",
    "DIS",
    "GE",
    "GS",
    "HD",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PFE",
    "PG",
    "T",
    "TRV",
    "UNH",
    "UTX",
    "V",
    "VZ",
    "WMT",
    "XOM",
}

# Effective date, added tickers, removed tickers. Separate rows are used for
# changes that share an effective date to keep each replacement auditable.
_DJIA_CHANGES = (
    ("2015-03-19", ("AAPL",), ("T",)),
    ("2017-09-01", ("DWDP",), ("DD",)),
    ("2018-06-26", ("WBA",), ("GE",)),
    ("2019-04-02", ("DOW",), ("DWDP",)),
    ("2020-04-06", ("RTX",), ("UTX",)),
    ("2020-08-31", ("AMGN",), ("XOM",)),
    ("2020-08-31", ("HON",), ("PFE",)),
    ("2020-08-31", ("CRM",), ("RTX",)),
    ("2024-02-26", ("AMZN",), ("WBA",)),
    ("2024-11-08", ("NVDA",), ("INTC",)),
    ("2024-11-08", ("SHW",), ("DOW",)),
)

DOW_30_HISTORICAL_UNIVERSE = tuple(
    sorted(
        _INITIAL_CONSTITUENTS_2014.union(
            *(set(added) for _, added, _ in _DJIA_CHANGES)
        ).union(DOW_30_TICKER)
    )
)


def get_dow_constituents(date: str | pd.Timestamp) -> tuple[str, ...]:
    """Return the 30 DJIA tickers effective on ``date``.

    The event history covers 2014-01-01 through 2025-12-31. The tutorial's
    2026-01-01 through 2026-03-20 trading window uses its dated current roster.
    Dates outside those intervals are rejected.
    """
    date = pd.Timestamp(date).normalize()
    if date < pd.Timestamp("2014-01-01") or date > pd.Timestamp("2026-03-20"):
        raise ValueError("Bundled Dow constituents cover 2014-01-01 through 2026-03-20")
    if date >= pd.Timestamp("2026-01-01"):
        return tuple(sorted(DOW_30_TICKER))

    constituents = set(_INITIAL_CONSTITUENTS_2014)
    for effective_date, added, removed in _DJIA_CHANGES:
        if date >= pd.Timestamp(effective_date):
            constituents.difference_update(removed)
            constituents.update(added)
    if len(constituents) != 30:
        raise RuntimeError(f"Expected 30 Dow constituents, found {len(constituents)}")
    return tuple(sorted(constituents))


def add_dow_membership(data: pd.DataFrame) -> pd.DataFrame:
    """Add an ``is_member`` flag to date/ticker rows using historical membership."""
    required_columns = {"date", "tic"}
    if not required_columns.issubset(data.columns):
        raise ValueError("data must contain 'date' and 'tic' columns")

    result = data.copy()
    dates = pd.to_datetime(result["date"])
    member_sets = {
        date: set(get_dow_constituents(date)) for date in dates.dt.normalize().unique()
    }
    result["is_member"] = [
        tic in member_sets[date.normalize()] for tic, date in zip(result["tic"], dates)
    ]
    return result
