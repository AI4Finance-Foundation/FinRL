"""Downloads perpetual-futures bars from Aperiodic (https://aperiodic.io)."""

from __future__ import annotations

import pandas as pd

EXCHANGES = {
    "binance": "binance-futures",
    "okx": "okx-perps",
    "hyperliquid": "hyperliquid-perps",
}
INTERVALS = ("15s", "30s", "1m", "5m", "15m", "30m", "1h", "4h", "1d")
# FinRL column name -> Aperiodic derivative metric that carries it.
DERIVATIVE_FEATURES = {
    "funding_rate": "funding",
    "open_interest": "open_interest",
    "open_interest_notional": "open_interest",
    "basis_bps": "basis",
}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
PERPETUAL_PREFIX = "perpetual-"


def _import_aperiodic():
    try:
        import aperiodic
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Aperiodic data source needs the aperiodic SDK, which is not "
            "installed with FinRL. Install it (Python 3.11+) with: "
            'pip install "aperiodic[pandas]"'
        ) from exc
    return aperiodic


def resolve_exchange(exchange: str) -> str:
    """Maps ``binance``, ``okx`` or ``hyperliquid`` to Aperiodic's venue name."""
    name = exchange.strip().lower()
    if name in EXCHANGES:
        return EXCHANGES[name]
    if name in EXCHANGES.values():
        return name
    raise ValueError(
        f"Unknown Aperiodic exchange {exchange!r}. "
        f"Use one of: {', '.join(EXCHANGES)}."
    )


def to_atlas_symbol(ticker: str) -> str:
    """Maps a FinRL ticker to an Aperiodic (Atlas) perpetual symbol.

    ``BTC-USDT`` -> ``perpetual-BTC-USDT:USDT`` (linear, settled in the quote),
    ``XYZ:TSLA-USDC`` -> ``perpetual-XYZ:TSLA-USDC:USDC`` (Hyperliquid HIP-3),
    ``BTC-USD:BTC`` -> ``perpetual-BTC-USD:BTC`` (explicit settlement currency).
    A full ``perpetual-...`` symbol is returned unchanged.
    """
    symbol = ticker.strip()
    if symbol.startswith(PERPETUAL_PREFIX):
        return symbol
    base, _, quote = symbol.rpartition("-")
    if not base or not quote:
        raise ValueError(
            f"Aperiodic tickers look like 'BTC-USDT' or 'XYZ:TSLA-USDC', "
            f"got {ticker!r}."
        )
    if ":" in quote:
        return PERPETUAL_PREFIX + symbol
    return f"{PERPETUAL_PREFIX}{symbol}:{quote}"


def to_finrl_ticker(symbol: str) -> str:
    """Inverse of :func:`to_atlas_symbol` for perpetual symbols."""
    body = symbol[len(PERPETUAL_PREFIX) :]
    market, _, settle = body.rpartition(":")
    quote = market.rpartition("-")[2]
    return market if settle == quote else body


def split_ticker(ticker: str, default_exchange: str):
    """Returns ``(exchange, symbol)`` for ``BTC-USDT`` or ``BTC-USDT@okx``."""
    name, at, venue = ticker.rpartition("@")
    if not at:
        return default_exchange, to_atlas_symbol(ticker)
    return resolve_exchange(venue), to_atlas_symbol(name)


def to_aperiodic_interval(time_interval: str) -> str:
    """Accepts Aperiodic intervals and FinRL's ``1Min``/``5Min``/``1H``/``1D``."""
    interval = str(time_interval).strip()
    if interval.endswith("Min"):
        interval = interval[: -len("Min")] + "m"
    elif interval.endswith(("H", "D")):
        interval = interval.lower()
    if interval == "60m":
        interval = "1h"
    if interval not in INTERVALS:
        raise ValueError(
            f"Unsupported time_interval {time_interval!r} for Aperiodic. "
            f"Use one of: {', '.join(INTERVALS)}."
        )
    return interval


def to_feature_list(derivative_features) -> list[str]:
    if derivative_features is None:
        return []
    if isinstance(derivative_features, str):
        derivative_features = [derivative_features]
    features = list(derivative_features)
    unknown = [name for name in features if name not in DERIVATIVE_FEATURES]
    if unknown:
        raise ValueError(
            f"Unknown derivative_features {unknown}. "
            f"Supported: {', '.join(DERIVATIVE_FEATURES)}."
        )
    return features


def list_aperiodic_tickers(exchange: str = "binance", api_key: str = None):
    """Lists every perpetual on a venue, delisted contracts included, as FinRL
    tickers. Needs an API key (``api_key`` or ``APERIODIC_API_KEY``)."""
    aperiodic = _import_aperiodic()
    symbols = aperiodic.get_symbols(
        api_key=api_key, exchange=resolve_exchange(exchange)
    )
    return sorted(
        to_finrl_ticker(symbol)
        for symbol in symbols
        if symbol.startswith(PERPETUAL_PREFIX)
    )


class AperiodicDownloader:
    """Provides methods for retrieving perpetual-futures bars from Aperiodic.

    Parameters
    ----------
    start_date : str
        First day of the data (UTC, inclusive), ``YYYY-MM-DD``.
    end_date : str
        Last day of the data (UTC, inclusive), ``YYYY-MM-DD``.
    ticker_list : list
        Tickers such as ``BTC-USDT`` or ``XYZ:TSLA-USDC``. Append ``@okx``,
        ``@binance`` or ``@hyperliquid`` to override ``exchange`` per ticker.
    time_interval : str
        ``1m``, ``5m``, ``15m``, ``30m``, ``1h``, ``4h`` or ``1d`` (FinRL's
        ``1Min``/``5Min``/``1H``/``1D`` are accepted).
    exchange : str
        Default venue: ``binance``, ``okx`` or ``hyperliquid``.
    derivative_features : list, optional
        Columns joined to each bar on time: ``funding_rate``,
        ``open_interest``, ``open_interest_notional``, ``basis_bps``.
    api_key : str, optional
        Aperiodic API key. If omitted, the SDK uses ``APERIODIC_API_KEY`` or
        the key saved by ``aperiodic.login()``.
    preview : bool
        Read the public preview sample, which needs no key.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        ticker_list: list[str],
        time_interval: str = "1d",
        exchange: str = "binance",
        derivative_features: list[str] = None,
        api_key: str = None,
        preview: bool = False,
    ):
        self.start_date = pd.Timestamp(start_date).date()
        self.end_date = pd.Timestamp(end_date).date()
        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date.")
        self.ticker_list = ticker_list
        self.time_interval = to_aperiodic_interval(time_interval)
        self.exchange = resolve_exchange(exchange)
        self.derivative_features = to_feature_list(derivative_features)
        self.api_key = api_key
        self.preview = preview

    def fetch_data(self) -> pd.DataFrame:
        """Fetches bars for every ticker.

        Returns
        -------
        pd.DataFrame
            Columns: date, open, high, low, close, volume, tic, day, then any
            ``derivative_features``. ``date`` is the bar's open time in UTC,
            formatted ``YYYY-MM-DD`` for daily bars and
            ``YYYY-MM-DD HH:MM:SS`` otherwise.
        """
        aperiodic = _import_aperiodic()
        frames = [self._fetch_ticker(aperiodic, tic) for tic in self.ticker_list]
        data_df = pd.concat(frames, ignore_index=True)
        data_df["day"] = data_df["date"].dt.dayofweek
        date_format = "%Y-%m-%d" if self.time_interval == "1d" else "%Y-%m-%d %H:%M:%S"
        data_df["date"] = data_df["date"].dt.strftime(date_format)
        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        return data_df[
            ["date"] + OHLCV_COLUMNS + ["tic", "day"] + self.derivative_features
        ]

    def _fetch_ticker(self, aperiodic, ticker: str) -> pd.DataFrame:
        exchange, symbol = split_ticker(ticker, self.exchange)
        bars = self._request(aperiodic, aperiodic.get_ohlcv, ticker, exchange, symbol)
        bars = bars[["time"] + OHLCV_COLUMNS].astype(
            {column: "float64" for column in OHLCV_COLUMNS}
        )
        metrics = {}
        for feature in self.derivative_features:
            metrics.setdefault(DERIVATIVE_FEATURES[feature], []).append(feature)
        for metric, columns in metrics.items():
            values = self._request(
                aperiodic,
                aperiodic.get_derivative_metrics,
                ticker,
                exchange,
                symbol,
                metric=metric,
            )
            bars = bars.merge(values[["time"] + columns], on="time", how="left")
        bars = bars.rename(columns={"time": "date"})
        bars["tic"] = ticker
        return bars

    def _request(self, aperiodic, fetch, ticker, exchange, symbol, **kwargs):
        dataset = kwargs.get("metric", "ohlcv")
        try:
            data = fetch(
                api_key=self.api_key,
                exchange=exchange,
                symbol=symbol,
                interval=self.time_interval,
                timestamp="exchange",
                start_date=self.start_date,
                end_date=self.end_date,
                output="pandas",
                show_progress=False,
                preview=self.preview,
                **kwargs,
            )
        except (
            aperiodic.APIError,
            aperiodic.AperiodicDataError,
            aperiodic.DownloadError,
        ) as exc:
            hint = (
                " (preview requests must match https://aperiodic.io/catalog#preview)"
                if self.preview
                else ""
            )
            raise RuntimeError(
                f"Aperiodic {dataset} request for {ticker} ({symbol} on "
                f"{exchange}) failed{hint}: {exc}"
            ) from exc
        if data.empty:
            raise ValueError(
                f"Aperiodic returned no {dataset} data for {ticker} ({symbol} on "
                f"{exchange}, {self.time_interval}, {self.start_date} to "
                f"{self.end_date}). Check the ticker with "
                "list_aperiodic_tickers(), that the dates fall within the "
                "contract's history and that your plan covers them "
                "(https://aperiodic.io/pricing)."
            )
        return data
