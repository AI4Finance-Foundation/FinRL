"""
Crypto Trading Part 1 — Data  (yfinance edition)

Downloads BTC-USD and ETH-USD 1-minute OHLCV candles via yfinance, computes
technical indicators with stockstats, and saves numpy arrays for training /
backtesting.

Note: yfinance 1-minute data is only available for the most recent ~30 days.
  Train : last ~19 days  (TRAIN_START → TRAIN_END)
  Test  :  last ~11 days (TEST_START  → TEST_END)

Output: crypto_data_arrays.npz
  train_price_array  – shape (N_train, 2)
  train_tech_array   – shape (N_train, 14)   [7 indicators × 2 assets]
  train_date_ary     – shape (N_train,)
  test_price_array   – shape (N_test,  2)
  test_tech_array    – shape (N_test,  14)
  test_date_ary      – shape (N_test,)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from stockstats import StockDataFrame as Sdf

warnings.filterwarnings("ignore")

# Redirect yfinance cache to D: so C: disk-full errors don't block us
_yf_cache = "D:/tmp/yfinance_cache"
os.makedirs(_yf_cache, exist_ok=True)
yf.set_tz_cache_location(_yf_cache)

# ── Configuration ──────────────────────────────────────────────────────────────

TICKERS   = ["BTC-USD", "ETH-USD"]        # yfinance symbols
TIMEFRAME = "1m"

# yfinance 1m data is limited to ~30 days back from today (2026-03-25)
TRAIN_START = "2026-02-24"
TRAIN_END   = "2026-03-14"   # exclusive upper bound for yfinance
TEST_START  = "2026-03-14"
TEST_END    = "2026-03-25"   # exclusive upper bound for yfinance

TECH_INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

OUTPUT_FILE = "crypto_data_arrays.npz"

# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_1m(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download 1-minute OHLCV for a single ticker via yfinance.

    yfinance only returns up to 7 days per request at 1m granularity, so we
    fetch in weekly chunks and concatenate.
    """
    chunks = []
    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end,   tz="UTC")
    chunk_size = pd.Timedelta(days=7)

    while t0 < t1:
        t_end_chunk = min(t0 + chunk_size, t1)
        print(f"  Fetching {ticker}  {t0.date()} → {t_end_chunk.date()} …")
        raw = yf.download(
            ticker,
            start=t0.strftime("%Y-%m-%d"),
            end=t_end_chunk.strftime("%Y-%m-%d"),
            interval="1m",
            progress=False,
            auto_adjust=True,
        )
        if not raw.empty:
            # yfinance with a single ticker returns simple columns
            raw = raw.copy()
            raw.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                           for c in raw.columns]
            raw = raw[["open", "high", "low", "close", "volume"]]
            chunks.append(raw)
        t0 = t_end_chunk

    if not chunks:
        raise RuntimeError(f"No data returned for {ticker}")

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def add_tech_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Compute stockstats technical indicators and append them as columns."""
    work = df.reset_index().rename(columns={"Datetime": "date", "index": "date"})
    if "date" not in work.columns:
        work = work.rename(columns={work.columns[0]: "date"})
    sdf = Sdf.retype(work.copy())
    result = df.copy()
    for ind in indicators:
        result[ind] = sdf[ind].values
    return result


def build_arrays(
    dfs: dict[str, pd.DataFrame],
    tickers: list[str],
    indicators: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align timestamps across tickers and return (price_array, tech_array, dates)."""
    # Align on common timestamps
    common_idx = dfs[tickers[0]].index
    for t in tickers[1:]:
        common_idx = common_idx.intersection(dfs[t].index)
    common_idx = common_idx.sort_values()

    price_cols = [dfs[t].loc[common_idx, "close"].values for t in tickers]
    price_array = np.column_stack(price_cols).astype(np.float32)

    # tech_array: [ticker0_ind0, ticker0_ind1, …, ticker1_ind0, …]
    tech_cols = []
    for t in tickers:
        for ind in indicators:
            tech_cols.append(dfs[t].loc[common_idx, ind].values)
    tech_array = np.column_stack(tech_cols).astype(np.float32)

    dates = common_idx.astype(str).values
    return price_array, tech_array, dates


# ── Main ───────────────────────────────────────────────────────────────────────

os.makedirs("datasets", exist_ok=True)

# ── Part 1: Download raw OHLCV ─────────────────────────────────────────────────

print("=" * 60)
print("Part 1 — Downloading 1m OHLCV from Yahoo Finance")
print("=" * 60)

all_data: dict[str, pd.DataFrame] = {}
for ticker in TICKERS:
    print(f"\n[{ticker}] Train period:")
    df_train_raw = fetch_1m(ticker, TRAIN_START, TRAIN_END)
    print(f"[{ticker}] Test  period:")
    df_test_raw  = fetch_1m(ticker, TEST_START,  TEST_END)

    # Combine for indicator computation (needs continuous history for SMA-60 etc.)
    df_all = pd.concat([df_train_raw, df_test_raw])
    df_all = df_all[~df_all.index.duplicated(keep="first")].sort_index()
    all_data[ticker] = df_all
    print(f"  Total rows : {len(df_all)}")

# ── Part 2: Technical indicators ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("Part 2 — Computing technical indicators")
print("=" * 60)

all_with_ind: dict[str, pd.DataFrame] = {}
for ticker in TICKERS:
    print(f"  {ticker} …")
    all_with_ind[ticker] = add_tech_indicators(all_data[ticker], TECH_INDICATORS)
    all_with_ind[ticker] = all_with_ind[ticker].dropna()

# ── Part 3: Split into train / test and build numpy arrays ────────────────────

print("\n" + "=" * 60)
print("Part 3 — Building numpy arrays")
print("=" * 60)

split_ts = pd.Timestamp(TEST_START, tz="UTC")

train_dfs = {t: df[df.index <  split_ts] for t, df in all_with_ind.items()}
test_dfs  = {t: df[df.index >= split_ts] for t, df in all_with_ind.items()}

train_price, train_tech, train_dates = build_arrays(train_dfs, TICKERS, TECH_INDICATORS)
test_price,  test_tech,  test_dates  = build_arrays(test_dfs,  TICKERS, TECH_INDICATORS)

print(f"\nTrain  price shape : {train_price.shape}")
print(f"Train  tech  shape : {train_tech.shape}")
print(f"Test   price shape : {test_price.shape}")
print(f"Test   tech  shape : {test_tech.shape}")
print(f"\nBTC price range (train): ${train_price[:,0].min():,.0f} – ${train_price[:,0].max():,.0f}")
print(f"ETH price range (train): ${train_price[:,1].min():,.0f} – ${train_price[:,1].max():,.0f}")

# ── Part 4: Save ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Part 4 — Saving")
print("=" * 60)

np.savez(
    OUTPUT_FILE,
    train_price_array = train_price,
    train_tech_array  = train_tech,
    train_date_ary    = train_dates,
    test_price_array  = test_price,
    test_tech_array   = test_tech,
    test_date_ary     = test_dates,
)
print(f"\nSaved to: {OUTPUT_FILE}")
print(f"  Train: {len(train_dates):,} timesteps  ({train_dates[0]} → {train_dates[-1]})")
print(f"  Test : {len(test_dates):,}  timesteps  ({test_dates[0]} → {test_dates[-1]})")
print("\nDone — run FinRL_CryptoTrading_2_train.py next.")
