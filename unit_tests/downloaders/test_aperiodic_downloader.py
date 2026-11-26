from __future__ import annotations

import datetime
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_aperiodic import AperiodicProcessor
from finrl.meta.preprocessor import aperiodicdownloader
from finrl.meta.preprocessor.aperiodicdownloader import AperiodicDownloader
from finrl.meta.preprocessor.aperiodicdownloader import list_aperiodic_tickers
from finrl.meta.preprocessor.aperiodicdownloader import to_aperiodic_interval
from finrl.meta.preprocessor.aperiodicdownloader import to_atlas_symbol
from finrl.meta.preprocessor.aperiodicdownloader import to_finrl_ticker

TIMES = pd.date_range("2026-08-01", periods=3, freq="1h")


class APIError(Exception):
    pass


def ohlcv_frame(times=TIMES, close=100.0):
    closes = close + np.arange(len(times), dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "open": closes - 0.5,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(len(times), 10.0, dtype="float32"),
            "volume_notional": np.full(len(times), 1000.0, dtype="float32"),
        }
    )


def derivative_frame(metric, times=TIMES):
    columns = {
        "funding": {"funding_rate": 0.0001, "funding_rate_update_frequency": 1.0},
        "open_interest": {"open_interest": 5000.0, "open_interest_notional": 5e5},
        "basis": {"basis": 1.0, "basis_bps": 0.5},
    }[metric]
    data = {"time": times}
    data.update({name: np.full(len(times), value) for name, value in columns.items()})
    return pd.DataFrame(data)


def fake_sdk(ohlcv=None, derivatives=None):
    return SimpleNamespace(
        get_ohlcv=mock.MagicMock(side_effect=ohlcv or (lambda **kw: ohlcv_frame())),
        get_derivative_metrics=mock.MagicMock(
            side_effect=derivatives or (lambda **kw: derivative_frame(kw["metric"]))
        ),
        get_symbols=mock.MagicMock(),
        APIError=APIError,
        AperiodicDataError=type("AperiodicDataError", (Exception,), {}),
        DownloadError=type("DownloadError", (Exception,), {}),
    )


def patch_sdk(sdk):
    return mock.patch.object(aperiodicdownloader, "_import_aperiodic", return_value=sdk)


class TestAperiodicTickers(unittest.TestCase):
    def test_to_atlas_symbol(self):
        self.assertEqual(to_atlas_symbol("BTC-USDT"), "perpetual-BTC-USDT:USDT")
        self.assertEqual(
            to_atlas_symbol("XYZ:TSLA-USDC"), "perpetual-XYZ:TSLA-USDC:USDC"
        )
        self.assertEqual(to_atlas_symbol("BTC-USD:BTC"), "perpetual-BTC-USD:BTC")
        self.assertEqual(
            to_atlas_symbol("perpetual-ETH-USDT:USDT"), "perpetual-ETH-USDT:USDT"
        )
        with self.assertRaisesRegex(ValueError, "BTC-USDT"):
            to_atlas_symbol("BTC")

    def test_to_finrl_ticker_round_trips(self):
        for symbol in [
            "perpetual-BTC-USDT:USDT",
            "perpetual-XYZ:TSLA-USDC:USDC",
            "perpetual-CASH:GOLD-USDT0:USDT0",
            "perpetual-BTC-USD:BTC",
        ]:
            self.assertEqual(to_atlas_symbol(to_finrl_ticker(symbol)), symbol)
        self.assertEqual(
            to_finrl_ticker("perpetual-XYZ:TSLA-USDC:USDC"), "XYZ:TSLA-USDC"
        )

    def test_to_aperiodic_interval(self):
        cases = {"1D": "1d", "1d": "1d", "1H": "1h", "5Min": "5m", "60m": "1h"}
        for finrl_interval, expected in cases.items():
            self.assertEqual(to_aperiodic_interval(finrl_interval), expected)
        for invalid in ["1M", "2h", "1wk"]:
            with self.assertRaises(ValueError):
                to_aperiodic_interval(invalid)

    def test_list_tickers_returns_perpetuals_only(self):
        sdk = fake_sdk()
        sdk.get_symbols.return_value = [
            "perpetual-BTC-USDT:USDT",
            "future-BTC-USDT:USDT-20250926",
            "perpetual-XYZ:TSLA-USDC:USDC",
        ]
        with patch_sdk(sdk):
            tickers = list_aperiodic_tickers("hyperliquid", api_key="key")
        sdk.get_symbols.assert_called_once_with(
            api_key="key", exchange="hyperliquid-perps"
        )
        self.assertEqual(tickers, ["BTC-USDT", "XYZ:TSLA-USDC"])


class TestAperiodicDownloader(unittest.TestCase):
    def test_fetch_data_calls_sdk_and_returns_finrl_frame(self):
        sdk = fake_sdk()
        with patch_sdk(sdk):
            data = AperiodicDownloader(
                start_date="2026-08-01",
                end_date="2026-08-01",
                ticker_list=["BTC-USDT"],
                time_interval="1H",
                exchange="okx",
                api_key="key",
            ).fetch_data()

        sdk.get_ohlcv.assert_called_once_with(
            api_key="key",
            exchange="okx-perps",
            symbol="perpetual-BTC-USDT:USDT",
            interval="1h",
            timestamp="exchange",
            start_date=datetime.date(2026, 8, 1),
            end_date=datetime.date(2026, 8, 1),
            output="pandas",
            show_progress=False,
            preview=False,
        )
        sdk.get_derivative_metrics.assert_not_called()
        self.assertEqual(
            list(data.columns),
            ["date", "open", "high", "low", "close", "volume", "tic", "day"],
        )
        self.assertEqual(data["date"].iloc[0], "2026-08-01 00:00:00")
        self.assertEqual(data["volume"].dtype, np.float64)
        self.assertEqual(data["tic"].unique().tolist(), ["BTC-USDT"])

    def test_venue_suffix_routes_each_ticker(self):
        sdk = fake_sdk()
        with patch_sdk(sdk):
            data = AperiodicDownloader(
                "2026-08-01",
                "2026-08-01",
                ["BTC-USDT", "XYZ:TSLA-USDC@hyperliquid"],
                time_interval="1d",
            ).fetch_data()

        routed = [
            (call.kwargs["exchange"], call.kwargs["symbol"])
            for call in sdk.get_ohlcv.call_args_list
        ]
        self.assertEqual(
            routed,
            [
                ("binance-futures", "perpetual-BTC-USDT:USDT"),
                ("hyperliquid-perps", "perpetual-XYZ:TSLA-USDC:USDC"),
            ],
        )
        self.assertEqual(
            sorted(data["tic"].unique()), ["BTC-USDT", "XYZ:TSLA-USDC@hyperliquid"]
        )
        self.assertEqual(data["date"].iloc[0], "2026-08-01")

    def test_derivative_features_join_on_time(self):
        sdk = fake_sdk(
            derivatives=lambda **kw: derivative_frame(kw["metric"], times=TIMES[1:])
        )
        with patch_sdk(sdk):
            data = AperiodicDownloader(
                "2026-08-01",
                "2026-08-01",
                ["BTC-USDT"],
                time_interval="1h",
                derivative_features=[
                    "funding_rate",
                    "open_interest",
                    "open_interest_notional",
                ],
            ).fetch_data()

        metrics = [
            c.kwargs["metric"] for c in sdk.get_derivative_metrics.call_args_list
        ]
        self.assertEqual(metrics, ["funding", "open_interest"])
        self.assertEqual(
            list(data.columns)[-3:],
            ["funding_rate", "open_interest", "open_interest_notional"],
        )
        self.assertTrue(np.isnan(data["funding_rate"].iloc[0]))
        self.assertEqual(data["open_interest"].iloc[1], 5000.0)

    def test_empty_response_raises(self):
        sdk = fake_sdk(ohlcv=lambda **kw: pd.DataFrame())
        with patch_sdk(sdk), self.assertRaisesRegex(ValueError, "NOPE-USDT"):
            AperiodicDownloader("2026-08-01", "2026-08-01", ["NOPE-USDT"]).fetch_data()

    def test_api_error_text_is_surfaced(self):
        def refuse(**kwargs):
            raise APIError("403: Symbol not in your plan")

        sdk = fake_sdk(ohlcv=refuse)
        with patch_sdk(sdk), self.assertRaisesRegex(
            RuntimeError, "BTC-USDT.*403: Symbol not in your plan"
        ):
            AperiodicDownloader("2026-08-01", "2026-08-01", ["BTC-USDT"]).fetch_data()

    def test_preview_error_points_to_preview_catalog(self):
        def refuse(**kwargs):
            raise APIError("400: Requested parameters do not match any preview data")

        sdk = fake_sdk(ohlcv=refuse)
        with patch_sdk(sdk), self.assertRaisesRegex(
            RuntimeError, "catalog#preview.*do not match any preview data"
        ):
            AperiodicDownloader(
                "2025-05-01", "2025-05-31", ["ETH-USDT"], preview=True
            ).fetch_data()
        self.assertTrue(sdk.get_ohlcv.call_args.kwargs["preview"])

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "exchange"):
            AperiodicDownloader(
                "2026-08-01", "2026-08-02", ["BTC-USDT"], exchange="kraken"
            )
        with self.assertRaisesRegex(ValueError, "derivative_features"):
            AperiodicDownloader(
                "2026-08-01", "2026-08-02", ["BTC-USDT"], derivative_features=["flow"]
            )
        with self.assertRaises(ValueError):
            AperiodicDownloader("2026-08-02", "2026-08-01", ["BTC-USDT"])

    def test_missing_sdk_explains_install(self):
        with mock.patch.dict(sys.modules, {"aperiodic": None}):
            with self.assertRaisesRegex(ModuleNotFoundError, r"aperiodic\[pandas\]"):
                AperiodicDownloader(
                    "2026-08-01", "2026-08-01", ["BTC-USDT"]
                ).fetch_data()


class TestAperiodicProcessor(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2026-08-01", periods=40, freq="1h")
        rng = np.random.default_rng(0)
        rows = []
        for tic, start in [("BTC-USDT", 100.0), ("XYZ:TSLA-USDC@hyperliquid", 50.0)]:
            closes = start * np.exp(np.cumsum(rng.normal(0, 0.01, len(times))))
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": times,
                        "open": closes,
                        "high": closes * 1.01,
                        "low": closes * 0.99,
                        "close": closes,
                        "volume": 1.0,
                        "tic": tic,
                        "funding_rate": 0.0001,
                    }
                )
            )
        self.data = pd.concat(rows, ignore_index=True)

    def test_download_data_uses_timestamp_column(self):
        sdk = fake_sdk()
        with patch_sdk(sdk):
            processor = AperiodicProcessor(
                api_key="key", derivative_features=["funding_rate"]
            )
            data = processor.download_data(
                ["BTC-USDT"], "2026-08-01", "2026-08-01", "1h"
            )
        self.assertEqual(
            list(data.columns),
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tic",
                "funding_rate",
            ],
        )
        self.assertEqual(data["timestamp"].iloc[0], pd.Timestamp("2026-08-01 00:00"))

    def test_clean_data_aligns_listed_and_delisted_contracts(self):
        data = self.data.drop(index=[3, 4]).copy()  # BTC-USDT misses two bars
        data = data[~((data.tic != "BTC-USDT") & (data.index % 40 >= 30))]  # delisted
        data.loc[data.index[5], "funding_rate"] = np.nan
        clean = AperiodicProcessor().clean_data(data)

        self.assertEqual(clean.groupby("tic").size().tolist(), [40, 40])
        self.assertFalse(clean.isna().any().any())
        btc = clean[clean.tic == "BTC-USDT"].reset_index(drop=True)
        self.assertEqual(btc.loc[3, "close"], btc.loc[2, "close"])
        self.assertEqual(btc.loc[4, "open"], btc.loc[2, "close"])
        self.assertEqual(btc.loc[3, "volume"], 0.0)
        tsla = clean[clean.tic != "BTC-USDT"].reset_index(drop=True)
        self.assertEqual(tsla.loc[35, "close"], tsla.loc[29, "close"])

    def test_indicators_keep_feature_columns_and_build_arrays(self):
        processor = AperiodicProcessor()
        data = processor.clean_data(self.data)
        tech = ["macd", "rsi_30", "funding_rate"]
        data = processor.add_technical_indicator(data, tech)
        data = processor.add_turbulence(data, time_period=10)

        self.assertEqual(list(data.columns).count("funding_rate"), 1)
        self.assertIn("macd", data.columns)
        self.assertTrue((data["turbulence"].iloc[:20] == 0).all())
        self.assertGreater(data["turbulence"].max(), 0)

        price, tech_array, turbulence = processor.df_to_array(data, tech, if_vix=False)
        self.assertEqual(price.shape, (40, 2))
        self.assertEqual(tech_array.shape, (40, 6))
        self.assertEqual(turbulence.shape, (40,))
        np.testing.assert_allclose(
            price[:, 0], data[data.tic == "BTC-USDT"]["close"].to_numpy()
        )

    def test_add_vix_warns_and_fills_zero(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = AperiodicProcessor().add_vix(self.data)
        self.assertTrue((data["VIXY"] == 0).all())
        self.assertIn("no VIX", str(caught[0].message))

    def test_data_processor_dispatch(self):
        from finrl.meta.data_processor import DataProcessor

        dp = DataProcessor(
            "aperiodic",
            API_KEY="key",
            exchange="hyperliquid",
            derivative_features=["open_interest"],
        )
        self.assertIsInstance(dp.processor, AperiodicProcessor)
        self.assertEqual(dp.processor.api_key, "key")
        self.assertEqual(dp.processor.exchange, "hyperliquid-perps")
        self.assertEqual(dp.processor.derivative_features, ["open_interest"])


if __name__ == "__main__":
    unittest.main()
