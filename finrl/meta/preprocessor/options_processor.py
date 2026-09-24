"""Options data processor with Greeks calculation for SPY trading.

This module provides functionality to fetch real-time options data,
calculate options Greeks, and prepare data for reinforcement learning agents.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")


class OptionsProcessor:
    """Fetches options data and calculates Greeks for options trading.

    Attributes
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY')
    risk_free_rate : float
        Risk-free interest rate for Greeks calculation (default: 0.05)

    Methods
    -------
    fetch_options_chain()
        Fetches the complete options chain for the ticker
    calculate_greeks()
        Calculates all Greeks (Delta, Gamma, Theta, Vega, Rho)
    select_optimal_strike()
        Selects the best strike price based on Greeks and strategy
    get_real_time_data()
        Gets real-time options data with Greeks
    """

    def __init__(self, ticker: str = "SPY", risk_free_rate: float = 0.05):
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.stock = yf.Ticker(ticker)

    def _black_scholes_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate Black-Scholes option price.

        Parameters
        ----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'

        Returns
        -------
        float
            Option price
        """
        if T <= 0 or sigma <= 0:
            return max(0, S - K) if option_type == "call" else max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def _calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        max_iterations: int = 100,
        tolerance: float = 1e-5,
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method.

        Parameters
        ----------
        market_price : float
            Current market price of the option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance

        Returns
        -------
        float
            Implied volatility
        """
        if T <= 0:
            return 0.0

        sigma = 0.3  # Initial guess

        for _ in range(max_iterations):
            price = self._black_scholes_price(S, K, T, r, sigma, option_type)
            vega = self._calculate_vega(S, K, T, r, sigma)

            diff = market_price - price

            if abs(diff) < tolerance:
                return sigma

            if vega < 1e-10:
                break

            sigma = sigma + diff / vega

            # Keep sigma in reasonable bounds
            sigma = max(0.01, min(5.0, sigma))

        return sigma

    def _calculate_delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate Delta (rate of change of option price with respect to stock price).

        Returns
        -------
        float
            Delta value
        """
        if T <= 0 or sigma <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def _calculate_gamma(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate Gamma (rate of change of Delta with respect to stock price).

        Returns
        -------
        float
            Gamma value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def _calculate_theta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate Theta (rate of change of option price with respect to time).

        Returns
        -------
        float
            Theta value (per day)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == "call":
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 - term2) / 365  # Convert to per day
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365  # Convert to per day

        return theta

    def _calculate_vega(
        self, S: float, K: float, T: float, r: float, sigma: float
    ) -> float:
        """Calculate Vega (rate of change of option price with respect to volatility).

        Returns
        -------
        float
            Vega value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% change

    def _calculate_rho(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call",
    ) -> float:
        """Calculate Rho (rate of change of option price with respect to interest rate).

        Returns
        -------
        float
            Rho value
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    def fetch_options_chain(self, expiration_date: str | None = None) -> pd.DataFrame:
        """Fetch options chain for the ticker.

        Parameters
        ----------
        expiration_date : str, optional
            Specific expiration date (format: 'YYYY-MM-DD')
            If None, uses the nearest expiration date

        Returns
        -------
        pd.DataFrame
            Options chain data with columns: strike, lastPrice, bid, ask, volume,
            openInterest, impliedVolatility, inTheMoney, contractSymbol, lastTradeDate,
            expiration, optionType
        """
        try:
            # Get available expiration dates
            expirations = self.stock.options

            if len(expirations) == 0:
                raise ValueError(f"No options available for {self.ticker}")

            # Use specified expiration or the nearest one
            if expiration_date is None:
                expiration = expirations[0]  # Nearest expiration
            else:
                expiration = expiration_date

            # Get options chain
            opt_chain = self.stock.option_chain(expiration)

            # Combine calls and puts
            calls = opt_chain.calls.copy()
            calls["optionType"] = "call"
            calls["expiration"] = expiration

            puts = opt_chain.puts.copy()
            puts["optionType"] = "put"
            puts["expiration"] = expiration

            # Combine both
            options_df = pd.concat([calls, puts], ignore_index=True)

            return options_df

        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return pd.DataFrame()

    def calculate_greeks(
        self, options_df: pd.DataFrame, current_price: float
    ) -> pd.DataFrame:
        """Calculate all Greeks for options chain.

        Parameters
        ----------
        options_df : pd.DataFrame
            Options chain data from fetch_options_chain()
        current_price : float
            Current stock price

        Returns
        -------
        pd.DataFrame
            Options data with Greeks columns added
        """
        if options_df.empty:
            return options_df

        df = options_df.copy()

        # Calculate time to expiration in years
        today = datetime.now()
        df["expiration_dt"] = pd.to_datetime(df["expiration"])
        df["days_to_expiry"] = (df["expiration_dt"] - today).dt.days
        df["time_to_expiry"] = df["days_to_expiry"] / 365.0

        # Initialize Greeks columns
        df["delta"] = 0.0
        df["gamma"] = 0.0
        df["theta"] = 0.0
        df["vega"] = 0.0
        df["rho"] = 0.0
        df["calc_iv"] = 0.0

        # Calculate Greeks for each option
        for idx, row in df.iterrows():
            S = current_price
            K = row["strike"]
            T = row["time_to_expiry"]
            r = self.risk_free_rate
            option_type = row["optionType"]

            # Get or calculate implied volatility
            if (
                "impliedVolatility" in row
                and pd.notna(row["impliedVolatility"])
                and row["impliedVolatility"] > 0
            ):
                sigma = row["impliedVolatility"]
            else:
                # Calculate IV from lastPrice
                if pd.notna(row["lastPrice"]) and row["lastPrice"] > 0:
                    sigma = self._calculate_implied_volatility(
                        row["lastPrice"], S, K, T, r, option_type
                    )
                else:
                    sigma = 0.3  # Default

            df.at[idx, "calc_iv"] = sigma

            if T > 0 and sigma > 0:
                df.at[idx, "delta"] = self._calculate_delta(
                    S, K, T, r, sigma, option_type
                )
                df.at[idx, "gamma"] = self._calculate_gamma(S, K, T, r, sigma)
                df.at[idx, "theta"] = self._calculate_theta(
                    S, K, T, r, sigma, option_type
                )
                df.at[idx, "vega"] = self._calculate_vega(S, K, T, r, sigma)
                df.at[idx, "rho"] = self._calculate_rho(S, K, T, r, sigma, option_type)

        return df

    def select_optimal_strike(
        self,
        options_df: pd.DataFrame,
        strategy: str = "balanced",
        option_type: str = "call",
    ) -> dict:
        """Select optimal strike based on Greeks and strategy.

        Parameters
        ----------
        options_df : pd.DataFrame
            Options data with Greeks
        strategy : str
            Strategy type: 'aggressive' (high delta), 'balanced' (moderate delta),
            'conservative' (low delta, high theta)
        option_type : str
            'call' or 'put'

        Returns
        -------
        Dict
            Selected option details
        """
        if options_df.empty:
            return {}

        # Filter by option type
        df = options_df[options_df["optionType"] == option_type].copy()

        if df.empty:
            return {}

        # Filter liquid options (volume > 0 or openInterest > 100)
        df = df[(df["volume"] > 0) | (df["openInterest"] > 100)]

        if df.empty:
            return {}

        # Strategy-based selection
        if strategy == "aggressive":
            # High delta (close to ATM or ITM), high gamma
            df["score"] = abs(df["delta"]) * 0.6 + df["gamma"] * 1000 * 0.4
        elif strategy == "balanced":
            # Moderate delta, good gamma/theta ratio
            df["score"] = (
                abs(df["delta"]) * 0.4
                + df["gamma"] * 1000 * 0.3
                - abs(df["theta"]) * 10 * 0.3
            )
        else:  # conservative
            # Lower delta, positive theta (for selling), high vega
            df["score"] = (1 - abs(df["delta"])) * 0.5 + df["vega"] * 0.5

        # Select best strike
        best_idx = df["score"].idxmax()
        best_option = df.loc[best_idx].to_dict()

        return best_option

    def get_real_time_data(
        self, timeframe: str = "1m", period: str = "1d"
    ) -> pd.DataFrame:
        """Get real-time stock data with options Greeks.

        Parameters
        ----------
        timeframe : str
            Data interval: '1m', '5m', '15m', '1h', '1d'
        period : str
            Data period: '1d', '5d', '1mo', etc.

        Returns
        -------
        pd.DataFrame
            Real-time stock data with timestamp
        """
        try:
            # Get real-time stock data
            stock_data = self.stock.history(period=period, interval=timeframe)

            if stock_data.empty:
                print(f"No data available for {self.ticker}")
                return pd.DataFrame()

            # Reset index to make timestamp a column
            stock_data = stock_data.reset_index()
            stock_data.rename(
                columns={"Datetime": "timestamp", "Date": "timestamp"}, inplace=True
            )

            # Standardize column names
            stock_data.columns = [col.lower() for col in stock_data.columns]

            return stock_data

        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return pd.DataFrame()

    def get_current_price(self) -> float:
        """Get current stock price.

        Returns
        -------
        float
            Current price
        """
        try:
            data = self.stock.history(period="1d", interval="1m")
            if not data.empty:
                return data["Close"].iloc[-1]
            else:
                # Fallback to info
                info = self.stock.info
                return info.get("currentPrice", info.get("regularMarketPrice", 0))
        except Exception as e:
            print(f"Error getting current price: {e}")
            return 0.0
