"""Multi-timeframe optimizer using CAGR for SPY trading.

This module optimizes trading strategy across multiple timeframes
using CAGR (Compound Annual Growth Rate) as the primary metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TimeframeOptimizer:
    """Optimizes trading strategy across multiple timeframes using CAGR.

    This optimizer:
    - Tests strategy on different timeframes (1m, 5m, 15m, 1h, 1d)
    - Calculates CAGR, Sharpe ratio, and max drawdown for each
    - Selects optimal timeframe based on risk-adjusted returns
    - Adaptively switches timeframes based on market conditions

    Attributes
    ----------
    timeframes : List[str]
        List of timeframes to optimize
    lookback_period : int
        Number of periods to use for optimization
    reoptimize_freq : int
        How often to reoptimize (in minutes)
    current_best_tf : str
        Currently selected best timeframe
    performance_history : Dict
        Historical performance for each timeframe
    """

    def __init__(
        self,
        timeframes: List[str] = None,
        lookback_period: int = 100,
        reoptimize_freq: int = 60,
        risk_free_rate: float = 0.02,
    ):
        """Initialize the timeframe optimizer.

        Parameters
        ----------
        timeframes : List[str], optional
            List of timeframes to test
        lookback_period : int
            Number of periods for backtesting
        reoptimize_freq : int
            Reoptimization frequency in minutes
        risk_free_rate : float
            Annual risk-free rate for Sharpe calculation
        """
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '1d']
        self.lookback_period = lookback_period
        self.reoptimize_freq = reoptimize_freq
        self.risk_free_rate = risk_free_rate

        self.current_best_tf = self.timeframes[0]
        self.performance_history = {tf: [] for tf in self.timeframes}
        self.last_optimization_time = None
        self.optimization_results = {}

    def calculate_cagr(
        self,
        portfolio_values: np.ndarray,
        periods_per_year: int
    ) -> float:
        """Calculate Compound Annual Growth Rate.

        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values over time
        periods_per_year : int
            Number of periods per year for this timeframe

        Returns
        -------
        float
            CAGR as a decimal (e.g., 0.15 for 15%)
        """
        if len(portfolio_values) < 2:
            return 0.0

        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        n_periods = len(portfolio_values)

        if initial_value <= 0 or final_value <= 0:
            return 0.0

        # CAGR = (Final/Initial)^(1/years) - 1
        years = n_periods / periods_per_year
        if years <= 0:
            return 0.0

        cagr = (final_value / initial_value) ** (1 / years) - 1

        return cagr

    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int
    ) -> float:
        """Calculate annualized Sharpe ratio.

        Parameters
        ----------
        returns : np.ndarray
            Array of period returns
        periods_per_year : int
            Number of periods per year

        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Calculate excess returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualize
        annual_return = mean_return * periods_per_year
        annual_std = std_return * np.sqrt(periods_per_year)
        risk_free_period = self.risk_free_rate / periods_per_year

        sharpe = (annual_return - risk_free_period * periods_per_year) / annual_std

        return sharpe

    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown.

        Parameters
        ----------
        portfolio_values : np.ndarray
            Array of portfolio values

        Returns
        -------
        float
            Maximum drawdown as a decimal (e.g., -0.20 for 20% drawdown)
        """
        if len(portfolio_values) < 2:
            return 0.0

        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_dd = np.min(drawdown)

        return max_dd

    def calculate_calmar_ratio(self, cagr: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio (CAGR / abs(max drawdown)).

        Parameters
        ----------
        cagr : float
            Compound annual growth rate
        max_drawdown : float
            Maximum drawdown

        Returns
        -------
        float
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0

        return cagr / abs(max_drawdown)

    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns).

        Parameters
        ----------
        returns : np.ndarray
            Array of returns

        Returns
        -------
        float
            Win rate (0 to 1)
        """
        if len(returns) == 0:
            return 0.0

        wins = np.sum(returns > 0)
        return wins / len(returns)

    def get_periods_per_year(self, timeframe: str) -> int:
        """Get number of periods per year for a timeframe.

        Parameters
        ----------
        timeframe : str
            Timeframe string (e.g., '1m', '5m', '1h', '1d')

        Returns
        -------
        int
            Number of periods per year
        """
        # Trading days per year: ~252
        # Trading hours per day: ~6.5 (9:30 AM - 4:00 PM ET)
        timeframe_to_periods = {
            '1m': 252 * 6.5 * 60,      # ~98,280 minutes per year
            '5m': 252 * 6.5 * 12,      # ~19,656 5-min periods
            '15m': 252 * 6.5 * 4,      # ~6,552 15-min periods
            '30m': 252 * 6.5 * 2,      # ~3,276 30-min periods
            '1h': 252 * 6.5,           # ~1,638 hours
            '1d': 252,                 # 252 trading days
        }

        return timeframe_to_periods.get(timeframe, 252)

    def evaluate_timeframe(
        self,
        portfolio_values: np.ndarray,
        timeframe: str
    ) -> Dict:
        """Evaluate performance metrics for a timeframe.

        Parameters
        ----------
        portfolio_values : np.ndarray
            Portfolio values over time
        timeframe : str
            Timeframe being evaluated

        Returns
        -------
        Dict
            Performance metrics
        """
        if len(portfolio_values) < 2:
            return {
                'cagr': 0,
                'sharpe': 0,
                'max_drawdown': 0,
                'calmar': 0,
                'win_rate': 0,
                'total_return': 0,
                'score': 0,
            }

        periods_per_year = self.get_periods_per_year(timeframe)

        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Calculate metrics
        cagr = self.calculate_cagr(portfolio_values, periods_per_year)
        sharpe = self.calculate_sharpe_ratio(returns, periods_per_year)
        max_dd = self.calculate_max_drawdown(portfolio_values)
        calmar = self.calculate_calmar_ratio(cagr, max_dd)
        win_rate = self.calculate_win_rate(returns)
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]

        # Calculate composite score
        # Weight: CAGR (40%), Sharpe (30%), Calmar (20%), Win Rate (10%)
        score = (
            cagr * 0.4 +
            sharpe * 0.05 * 0.3 +  # Normalize Sharpe to ~0-1 range
            calmar * 0.1 * 0.2 +    # Normalize Calmar
            win_rate * 0.1
        )

        return {
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'win_rate': win_rate,
            'total_return': total_return,
            'score': score,
        }

    def optimize(
        self,
        timeframe_data: Dict[str, np.ndarray]
    ) -> Tuple[str, Dict]:
        """Optimize across timeframes and select the best one.

        Parameters
        ----------
        timeframe_data : Dict[str, np.ndarray]
            Dictionary mapping timeframe to portfolio values

        Returns
        -------
        Tuple[str, Dict]
            Best timeframe and its metrics
        """
        results = {}

        # Evaluate each timeframe
        for tf in self.timeframes:
            if tf not in timeframe_data or len(timeframe_data[tf]) < 2:
                results[tf] = {
                    'cagr': 0,
                    'sharpe': 0,
                    'max_drawdown': 0,
                    'calmar': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'score': 0,
                }
                continue

            portfolio_values = timeframe_data[tf]
            results[tf] = self.evaluate_timeframe(portfolio_values, tf)

        # Select best timeframe based on score
        best_tf = max(results.keys(), key=lambda k: results[k]['score'])
        self.current_best_tf = best_tf
        self.optimization_results = results
        self.last_optimization_time = datetime.now()

        return best_tf, results[best_tf]

    def should_reoptimize(self) -> bool:
        """Check if it's time to reoptimize.

        Returns
        -------
        bool
            True if should reoptimize
        """
        if self.last_optimization_time is None:
            return True

        time_elapsed = (datetime.now() - self.last_optimization_time).total_seconds() / 60
        return time_elapsed >= self.reoptimize_freq

    def get_current_best_timeframe(self) -> str:
        """Get the currently selected best timeframe.

        Returns
        -------
        str
            Best timeframe
        """
        return self.current_best_tf

    def get_optimization_summary(self) -> pd.DataFrame:
        """Get summary of optimization results.

        Returns
        -------
        pd.DataFrame
            Summary table of all timeframes
        """
        if not self.optimization_results:
            return pd.DataFrame()

        df = pd.DataFrame(self.optimization_results).T
        df = df.round(4)
        df = df.sort_values('score', ascending=False)

        return df

    def add_performance_record(self, timeframe: str, value: float):
        """Add a performance record for a timeframe.

        Parameters
        ----------
        timeframe : str
            Timeframe
        value : float
            Portfolio value
        """
        if timeframe in self.performance_history:
            self.performance_history[timeframe].append(value)

            # Keep only lookback period
            if len(self.performance_history[timeframe]) > self.lookback_period:
                self.performance_history[timeframe] = \
                    self.performance_history[timeframe][-self.lookback_period:]

    def get_recommended_action(
        self,
        current_metrics: Dict
    ) -> str:
        """Get recommended action based on current metrics.

        Parameters
        ----------
        current_metrics : Dict
            Current performance metrics

        Returns
        -------
        str
            Recommendation: 'continue', 'switch_timeframe', 'reduce_risk'
        """
        # If max drawdown is too large, reduce risk
        if current_metrics.get('max_drawdown', 0) < -0.15:
            return 'reduce_risk'

        # If Sharpe ratio is negative, consider switching
        if current_metrics.get('sharpe', 0) < 0:
            return 'switch_timeframe'

        # If win rate is too low
        if current_metrics.get('win_rate', 0) < 0.4:
            return 'switch_timeframe'

        return 'continue'

    def generate_report(self) -> str:
        """Generate a text report of optimization results.

        Returns
        -------
        str
            Formatted report
        """
        if not self.optimization_results:
            return "No optimization results available."

        report = "\n" + "="*60 + "\n"
        report += "TIMEFRAME OPTIMIZATION REPORT\n"
        report += "="*60 + "\n\n"

        report += f"Best Timeframe: {self.current_best_tf}\n"
        report += f"Optimization Time: {self.last_optimization_time}\n\n"

        report += "Performance by Timeframe:\n"
        report += "-"*60 + "\n"

        for tf in sorted(self.optimization_results.keys(),
                        key=lambda k: self.optimization_results[k]['score'],
                        reverse=True):
            metrics = self.optimization_results[tf]
            report += f"\n{tf}:\n"
            report += f"  CAGR:         {metrics['cagr']*100:6.2f}%\n"
            report += f"  Sharpe Ratio: {metrics['sharpe']:6.2f}\n"
            report += f"  Max Drawdown: {metrics['max_drawdown']*100:6.2f}%\n"
            report += f"  Calmar Ratio: {metrics['calmar']:6.2f}\n"
            report += f"  Win Rate:     {metrics['win_rate']*100:6.2f}%\n"
            report += f"  Total Return: {metrics['total_return']*100:6.2f}%\n"
            report += f"  Score:        {metrics['score']:6.4f}\n"

        report += "\n" + "="*60 + "\n"

        return report
