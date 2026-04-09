from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Mapping
from urllib.parse import quote

import pandas as pd
import requests

ADANOS_SENTIMENT_SOURCES = ("reddit", "x", "news", "polymarket")

ADANOS_SENTIMENT_FEATURES = [
    "adanos_buzz_mean_lag1",
    "adanos_sentiment_mean_lag1",
    "adanos_source_coverage_lag1",
    "adanos_reddit_buzz_lag1",
    "adanos_reddit_sentiment_lag1",
    "adanos_x_buzz_lag1",
    "adanos_x_sentiment_lag1",
    "adanos_news_buzz_lag1",
    "adanos_news_sentiment_lag1",
    "adanos_polymarket_buzz_lag1",
    "adanos_polymarket_sentiment_lag1",
]


@dataclass(frozen=True)
class _DailyFeatureKey:
    ticker: str
    date: str


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def _resolve_days(df: pd.DataFrame, date_column: str, lag: int, max_days: int) -> int:
    dates = pd.to_datetime(df[date_column])
    span_days = max(int((dates.max() - dates.min()).days) + 1 + lag, 7)
    return min(span_days, max_days)


def _build_feature_frame(
    ticker_history: Mapping[_DailyFeatureKey, dict[str, float | None]],
    sources: Iterable[str],
) -> pd.DataFrame:
    source_buzz_columns = [f"adanos_{source}_buzz" for source in sources]
    source_sentiment_columns = [f"adanos_{source}_sentiment" for source in sources]

    frame = pd.DataFrame(ticker_history.values())
    for column in source_buzz_columns + source_sentiment_columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["adanos_buzz_mean"] = frame[source_buzz_columns].mean(axis=1, skipna=True)
    frame["adanos_sentiment_mean"] = frame[source_sentiment_columns].mean(
        axis=1, skipna=True
    )
    frame["adanos_source_coverage"] = (
        frame[source_buzz_columns].notna().sum(axis=1).astype(float)
    )
    return frame


def add_adanos_market_sentiment(
    df: pd.DataFrame,
    *,
    api_key: str | None = None,
    base_url: str = "https://api.adanos.org",
    days: int | None = None,
    sources: Iterable[str] = ADANOS_SENTIMENT_SOURCES,
    lag: int = 1,
    ticker_column: str = "tic",
    date_column: str = "date",
    timeout: int = 15,
    fillna: bool = True,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    Optionally enrich a FinRL dataframe with lagged Adanos market sentiment.

    The function is intentionally fail-open: without an API key, or when remote
    requests fail, the original dataframe is returned unchanged.
    """

    enriched = df.copy()
    if enriched.empty or not api_key:
        return enriched

    sources = tuple(source for source in sources if source in ADANOS_SENTIMENT_SOURCES)
    if not sources:
        return enriched

    normalized_dates = _normalize_date_column(enriched[date_column])
    enriched[date_column] = normalized_dates

    effective_days = days or _resolve_days(enriched, date_column, lag, max_days=365)
    client = session or requests.Session()
    headers = {"X-API-Key": api_key, "Accept": "application/json"}

    ticker_history: dict[_DailyFeatureKey, dict[str, float | None]] = {}

    for ticker in sorted(enriched[ticker_column].dropna().astype(str).unique()):
        for source in sources:
            endpoint = (
                f"{base_url.rstrip('/')}/{source}/stocks/v1/stock/{quote(ticker)}"
            )
            try:
                response = client.get(
                    endpoint,
                    params={"days": effective_days},
                    headers=headers,
                    timeout=timeout,
                )
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException:
                continue

            for item in payload.get("daily_trend") or []:
                date_value = item.get("date")
                if not date_value:
                    continue

                key = _DailyFeatureKey(ticker=ticker, date=str(date_value))
                row = ticker_history.setdefault(
                    key,
                    {
                        ticker_column: ticker,
                        date_column: str(date_value),
                    },
                )
                row[f"adanos_{source}_buzz"] = _safe_float(item.get("buzz_score"))
                row[f"adanos_{source}_sentiment"] = _safe_float(
                    item.get("sentiment_score")
                )

    if not ticker_history:
        return enriched

    feature_frame = _build_feature_frame(ticker_history, sources)
    base_frame = (
        enriched[[ticker_column, date_column]]
        .drop_duplicates()
        .sort_values([ticker_column, date_column])
        .reset_index(drop=True)
    )
    feature_frame = base_frame.merge(
        feature_frame,
        on=[ticker_column, date_column],
        how="left",
    )
    lag_base_columns = [
        "adanos_buzz_mean",
        "adanos_sentiment_mean",
        "adanos_source_coverage",
        *[f"adanos_{source}_buzz" for source in sources],
        *[f"adanos_{source}_sentiment" for source in sources],
    ]

    feature_frame = feature_frame.sort_values([ticker_column, date_column])
    shifted = feature_frame.groupby(ticker_column)[lag_base_columns].shift(lag)
    shifted.columns = [f"{column}_lag{lag}" for column in lag_base_columns]
    feature_frame = pd.concat(
        [feature_frame[[ticker_column, date_column]].reset_index(drop=True), shifted],
        axis=1,
    )

    enriched = enriched.merge(feature_frame, on=[ticker_column, date_column], how="left")

    if fillna:
        lag_columns = [f"{column}_lag{lag}" for column in lag_base_columns]
        enriched[lag_columns] = enriched[lag_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        enriched[lag_columns] = enriched[lag_columns].fillna(0.0)

    return enriched
