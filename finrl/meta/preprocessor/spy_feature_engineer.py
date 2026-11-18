"""Enhanced feature engineer for SPY options trading with Greeks and advanced indicators.

This module extends the standard FeatureEngineer to add options Greeks,
advanced technical indicators, and multi-timeframe analysis for SPY trading.
"""

from __future__ import annotations

import warnings
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

warnings.filterwarnings("ignore")

from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.options_processor import OptionsProcessor


class SPYFeatureEngineer(FeatureEngineer):
    """Enhanced feature engineer for SPY options trading.

    Extends FeatureEngineer with:
    - Options Greeks (Delta, Gamma, Theta, Vega, Rho)
    - Advanced momentum indicators
    - Volume analysis
    - Multi-timeframe features
    - Market regime detection

    Attributes
    ----------
    use_options_greeks : bool
        Include options Greeks in features
    use_advanced_indicators : bool
        Include advanced technical indicators
    use_volume_features : bool
        Include volume-based features
    use_regime_detection : bool
        Include market regime features
    timeframes : List[str]
        List of timeframes for multi-timeframe analysis
    options_processor : OptionsProcessor
        Options data processor

    Methods
    -------
    preprocess_spy_data()
        Main method to add all SPY-specific features
    add_options_greeks()
        Add options Greeks features
    add_advanced_indicators()
        Add advanced technical indicators
    add_volume_features()
        Add volume analysis features
    add_regime_features()
        Add market regime detection features
    """

    def __init__(
        self,
        use_technical_indicator: bool = True,
        tech_indicator_list: list[str] = None,
        use_vix: bool = True,
        use_turbulence: bool = True,
        use_options_greeks: bool = True,
        use_advanced_indicators: bool = True,
        use_volume_features: bool = True,
        use_regime_detection: bool = True,
        timeframes: list[str] = None,
        risk_free_rate: float = 0.05,
    ):
        # Default technical indicators
        if tech_indicator_list is None:
            tech_indicator_list = [
                "macd",
                "boll_ub",
                "boll_lb",
                "rsi_30",
                "cci_30",
                "dx_30",
                "close_30_sma",
                "close_60_sma",
            ]

        # Initialize parent class
        super().__init__(
            use_technical_indicator=use_technical_indicator,
            tech_indicator_list=tech_indicator_list,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
            user_defined_feature=True,
        )

        # SPY-specific features
        self.use_options_greeks = use_options_greeks
        self.use_advanced_indicators = use_advanced_indicators
        self.use_volume_features = use_volume_features
        self.use_regime_detection = use_regime_detection
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "1d"]
        self.options_processor = OptionsProcessor(
            ticker="SPY", risk_free_rate=risk_free_rate
        )

    def preprocess_spy_data(
        self, df: pd.DataFrame, include_options: bool = True
    ) -> pd.DataFrame:
        """Main method to add all SPY-specific features.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with OHLCV data
        include_options : bool
            Whether to include options Greeks (requires real-time data)

        Returns
        -------
        pd.DataFrame
            Enhanced dataframe with all features
        """
        # Start with standard preprocessing
        df = self.preprocess_data(df)

        # Add advanced indicators
        if self.use_advanced_indicators:
            df = self.add_advanced_indicators(df)
            print("Successfully added advanced indicators")

        # Add volume features
        if self.use_volume_features:
            df = self.add_volume_features(df)
            print("Successfully added volume features")

        # Add regime detection
        if self.use_regime_detection:
            df = self.add_regime_features(df)
            print("Successfully added regime features")

        # Add options Greeks (if real-time data available)
        if self.use_options_greeks and include_options:
            df = self.add_options_greeks(df)
            print("Successfully added options Greeks")

        # Fill any remaining missing values
        df = df.ffill().bfill()

        return df

    def add_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators.

        Indicators:
        - RSI divergence
        - MACD histogram
        - ADX (Average Directional Index)
        - ATR (Average True Range)
        - Stochastic oscillator
        - Williams %R
        - Rate of Change (ROC)

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with advanced indicators
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        advanced_indicators = {
            "rsi_14": "rsi_14",  # 14-period RSI
            "rsi_7": "rsi_7",  # 7-period RSI for divergence
            "macdh": "macdh",  # MACD histogram
            "atr": "atr",  # Average True Range
            "adx": "adx",  # Average Directional Index
            "kdjk": "kdjk",  # Stochastic K
            "kdjd": "kdjd",  # Stochastic D
            "wr_14": "wr_14",  # Williams %R
        }

        for indicator_name, indicator_key in advanced_indicators.items():
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator_key]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], axis=0, ignore_index=True
                    )
                except Exception as e:
                    print(f"Error adding {indicator_name}: {e}")

            if not indicator_df.empty:
                df = df.merge(
                    indicator_df[["tic", "date", indicator_key]],
                    on=["tic", "date"],
                    how="left",
                )

        # Add Rate of Change (ROC)
        df = df.sort_values(by=["tic", "date"])
        for ticker in unique_ticker:
            mask = df["tic"] == ticker
            df.loc[mask, "roc_5"] = df.loc[mask, "close"].pct_change(5) * 100
            df.loc[mask, "roc_10"] = df.loc[mask, "close"].pct_change(10) * 100

        df = df.sort_values(by=["date", "tic"])
        return df

    def add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.

        Features:
        - Volume SMA (20, 50 periods)
        - Volume ratio (current / SMA)
        - On-Balance Volume (OBV)
        - Money Flow Index (MFI)
        - Volume Rate of Change

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with volume features
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        unique_ticker = df.tic.unique()

        for ticker in unique_ticker:
            mask = df["tic"] == ticker
            ticker_df = df[mask].copy()

            # Volume moving averages
            ticker_df["volume_sma_20"] = ticker_df["volume"].rolling(window=20).mean()
            ticker_df["volume_sma_50"] = ticker_df["volume"].rolling(window=50).mean()

            # Volume ratio
            ticker_df["volume_ratio"] = ticker_df["volume"] / (
                ticker_df["volume_sma_20"] + 1e-10
            )

            # On-Balance Volume (OBV)
            ticker_df["obv"] = (
                (np.sign(ticker_df["close"].diff()) * ticker_df["volume"])
                .fillna(0)
                .cumsum()
            )

            # Volume Rate of Change
            ticker_df["volume_roc"] = ticker_df["volume"].pct_change(5) * 100

            # Money Flow Index (simplified)
            typical_price = (
                ticker_df["high"] + ticker_df["low"] + ticker_df["close"]
            ) / 3
            money_flow = typical_price * ticker_df["volume"]
            ticker_df["mfi"] = money_flow.rolling(window=14).mean()

            # Update main dataframe
            df.loc[mask, "volume_sma_20"] = ticker_df["volume_sma_20"].values
            df.loc[mask, "volume_sma_50"] = ticker_df["volume_sma_50"].values
            df.loc[mask, "volume_ratio"] = ticker_df["volume_ratio"].values
            df.loc[mask, "obv"] = ticker_df["obv"].values
            df.loc[mask, "volume_roc"] = ticker_df["volume_roc"].values
            df.loc[mask, "mfi"] = ticker_df["mfi"].values

        df = df.sort_values(by=["date", "tic"])
        return df

    def add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features.

        Features:
        - Trend regime (uptrend, downtrend, sideways)
        - Volatility regime (high, medium, low)
        - Market state (bull, bear, neutral)

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe

        Returns
        -------
        pd.DataFrame
            Dataframe with regime features
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        unique_ticker = df.tic.unique()

        for ticker in unique_ticker:
            mask = df["tic"] == ticker
            ticker_df = df[mask].copy()

            # Trend detection using moving averages
            sma_20 = ticker_df["close"].rolling(window=20).mean()
            sma_50 = ticker_df["close"].rolling(window=50).mean()

            # Trend regime: 1 = uptrend, -1 = downtrend, 0 = sideways
            trend = np.where(sma_20 > sma_50, 1, -1)
            ticker_df["trend_regime"] = trend

            # Volatility regime using ATR
            high_low = ticker_df["high"] - ticker_df["low"]
            high_close = np.abs(ticker_df["high"] - ticker_df["close"].shift())
            low_close = np.abs(ticker_df["low"] - ticker_df["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr_14 = true_range.rolling(window=14).mean()

            # Normalize ATR by price
            atr_percent = (atr_14 / ticker_df["close"]) * 100

            # Volatility regime: 2 = high, 1 = medium, 0 = low
            volatility_regime = pd.cut(
                atr_percent,
                bins=[
                    0,
                    atr_percent.quantile(0.33),
                    atr_percent.quantile(0.67),
                    float("inf"),
                ],
                labels=[0, 1, 2],
            ).astype(float)
            ticker_df["volatility_regime"] = volatility_regime

            # Market state (combines trend and momentum)
            rsi = ticker_df.get("rsi_30", ticker_df.get("rsi_14", 50))
            market_state = np.where(
                (trend == 1) & (rsi > 50),
                1,  # Bull
                np.where((trend == -1) & (rsi < 50), -1, 0),  # Bear or Neutral
            )
            ticker_df["market_state"] = market_state

            # Update main dataframe
            df.loc[mask, "trend_regime"] = ticker_df["trend_regime"].values
            df.loc[mask, "volatility_regime"] = ticker_df["volatility_regime"].values
            df.loc[mask, "market_state"] = ticker_df["market_state"].values

        df = df.sort_values(by=["date", "tic"])
        return df

    def add_options_greeks(
        self, data: pd.DataFrame, expiration_date: str | None = None
    ) -> pd.DataFrame:
        """Add options Greeks features.

        Note: This requires real-time options data and should be used
        during live trading or with recent historical data.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        expiration_date : str, optional
            Specific options expiration date

        Returns
        -------
        pd.DataFrame
            Dataframe with Greeks features
        """
        df = data.copy()

        try:
            # Get current price
            current_price = self.options_processor.get_current_price()

            if current_price == 0:
                print("Could not fetch current price for Greeks calculation")
                return df

            # Fetch options chain
            options_df = self.options_processor.fetch_options_chain(expiration_date)

            if options_df.empty:
                print("No options data available")
                return df

            # Calculate Greeks
            options_with_greeks = self.options_processor.calculate_greeks(
                options_df, current_price
            )

            # Select optimal calls and puts
            best_call = self.options_processor.select_optimal_strike(
                options_with_greeks, strategy="balanced", option_type="call"
            )
            best_put = self.options_processor.select_optimal_strike(
                options_with_greeks, strategy="balanced", option_type="put"
            )

            # Add Greeks as features (aggregate values)
            if best_call:
                df["call_delta"] = best_call.get("delta", 0)
                df["call_gamma"] = best_call.get("gamma", 0)
                df["call_theta"] = best_call.get("theta", 0)
                df["call_vega"] = best_call.get("vega", 0)
                df["call_iv"] = best_call.get("calc_iv", 0)
                df["call_strike"] = best_call.get("strike", 0)

            if best_put:
                df["put_delta"] = best_put.get("delta", 0)
                df["put_gamma"] = best_put.get("gamma", 0)
                df["put_theta"] = best_put.get("theta", 0)
                df["put_vega"] = best_put.get("vega", 0)
                df["put_iv"] = best_put.get("calc_iv", 0)
                df["put_strike"] = best_put.get("strike", 0)

            # Calculate put-call ratio and implied volatility spread
            call_volume = options_with_greeks[
                options_with_greeks["optionType"] == "call"
            ]["volume"].sum()
            put_volume = options_with_greeks[
                options_with_greeks["optionType"] == "put"
            ]["volume"].sum()
            df["put_call_ratio"] = put_volume / (call_volume + 1e-10)

            # Average IV across ATM options
            atm_strikes = options_with_greeks[
                (options_with_greeks["strike"] >= current_price * 0.98)
                & (options_with_greeks["strike"] <= current_price * 1.02)
            ]
            df["atm_iv"] = atm_strikes["calc_iv"].mean() if not atm_strikes.empty else 0

        except Exception as e:
            print(f"Error adding options Greeks: {e}")

        return df

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names.

        Returns
        -------
        List[str]
            List of feature column names
        """
        features = []

        # Standard technical indicators
        if self.use_technical_indicator:
            features.extend(self.tech_indicator_list)

        # Advanced indicators
        if self.use_advanced_indicators:
            features.extend(
                [
                    "rsi_14",
                    "rsi_7",
                    "macdh",
                    "atr",
                    "adx",
                    "kdjk",
                    "kdjd",
                    "wr_14",
                    "roc_5",
                    "roc_10",
                ]
            )

        # Volume features
        if self.use_volume_features:
            features.extend(
                [
                    "volume_sma_20",
                    "volume_sma_50",
                    "volume_ratio",
                    "obv",
                    "volume_roc",
                    "mfi",
                ]
            )

        # Regime features
        if self.use_regime_detection:
            features.extend(["trend_regime", "volatility_regime", "market_state"])

        # Options Greeks
        if self.use_options_greeks:
            features.extend(
                [
                    "call_delta",
                    "call_gamma",
                    "call_theta",
                    "call_vega",
                    "call_iv",
                    "call_strike",
                    "put_delta",
                    "put_gamma",
                    "put_theta",
                    "put_vega",
                    "put_iv",
                    "put_strike",
                    "put_call_ratio",
                    "atm_iv",
                ]
            )

        # VIX and turbulence
        if self.use_vix:
            features.append("vix")
        if self.use_turbulence:
            features.append("turbulence")

        return features
