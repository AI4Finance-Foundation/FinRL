"""Performance tracking and visualization for SPY trading tool.

This module provides comprehensive performance tracking, metrics calculation,
and visualization capabilities for the trading system.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class PerformanceTracker:
    """Tracks and analyzes trading performance.

    Features:
    - Real-time performance metrics
    - Trade-by-trade logging
    - Equity curve tracking
    - Drawdown analysis
    - Risk metrics (Sharpe, Sortino, Calmar)
    - Win/loss statistics
    - Visualization tools

    Attributes
    ----------
    trades : List[Dict]
        List of all trades
    equity_curve : List[float]
        Portfolio values over time
    timestamps : List[datetime]
        Timestamps for each update
    initial_capital : float
        Starting capital
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        log_file: str | None = None,
        risk_free_rate: float = 0.02,
    ):
        """Initialize performance tracker.

        Parameters
        ----------
        initial_capital : float
            Initial trading capital
        log_file : str, optional
            Path to save trade log
        risk_free_rate : float
            Annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.log_file = log_file
        self.risk_free_rate = risk_free_rate

        # Trading records
        self.trades = []
        self.equity_curve = [initial_capital]
        self.timestamps = [datetime.now()]
        self.signals = []

        # Performance metrics
        self.metrics = {}
        self.daily_metrics = []

    def log_trade(self, trade: dict):
        """Log a completed trade.

        Parameters
        ----------
        trade : Dict
            Trade information
        """
        trade["timestamp"] = datetime.now()
        self.trades.append(trade)

        # Save to file if specified
        if self.log_file:
            self._save_trade_to_file(trade)

    def log_signal(self, signal: dict):
        """Log a trading signal.

        Parameters
        ----------
        signal : Dict
            Signal information
        """
        signal["timestamp"] = datetime.now()
        self.signals.append(signal)

    def update_equity(self, portfolio_value: float):
        """Update equity curve with current portfolio value.

        Parameters
        ----------
        portfolio_value : float
            Current portfolio value
        """
        self.equity_curve.append(portfolio_value)
        self.timestamps.append(datetime.now())

    def calculate_metrics(self) -> dict:
        """Calculate comprehensive performance metrics.

        Returns
        -------
        Dict
            Performance metrics
        """
        if len(self.equity_curve) < 2:
            return {}

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Basic metrics
        total_return = (equity[-1] - equity[0]) / equity[0]
        num_trades = len(self.trades)

        # Win rate
        if num_trades > 0:
            winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
            win_rate = len(winning_trades) / num_trades
            avg_win = (
                np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0
            )
            losing_trades = [t for t in self.trades if t.get("pnl", 0) < 0]
            avg_loss = (
                np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0
            )
            profit_factor = (
                abs(
                    sum([t["pnl"] for t in winning_trades])
                    / sum([t["pnl"] for t in losing_trades])
                )
                if losing_trades
                else float("inf")
            )
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = (
                (np.mean(returns) - self.risk_free_rate / 252)
                / np.std(returns)
                * np.sqrt(252)
            )
        else:
            sharpe = 0

        # Sortino ratio (uses only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino = (
                (np.mean(returns) - self.risk_free_rate / 252)
                / np.std(downside_returns)
                * np.sqrt(252)
            )
        else:
            sortino = 0

        # Maximum drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = np.min(drawdown)
        max_drawdown_pct = max_drawdown * 100

        # Calmar ratio
        if max_drawdown != 0:
            cagr = self._calculate_cagr(equity)
            calmar = cagr / abs(max_drawdown)
        else:
            calmar = 0

        # Current drawdown
        current_drawdown = (equity[-1] - cummax[-1]) / cummax[-1] * 100

        self.metrics = {
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "current_drawdown_pct": current_drawdown,
            "calmar_ratio": calmar,
            "current_equity": equity[-1],
        }

        return self.metrics

    def _calculate_cagr(self, equity: np.ndarray) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity) < 2 or len(self.timestamps) < 2:
            return 0

        initial_value = equity[0]
        final_value = equity[-1]
        years = (self.timestamps[-1] - self.timestamps[0]).days / 365.25

        if years <= 0 or initial_value <= 0:
            return 0

        cagr = (final_value / initial_value) ** (1 / years) - 1
        return cagr

    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades.

        Returns
        -------
        pd.DataFrame
            Trade summary
        """
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)
        return df

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame.

        Returns
        -------
        pd.DataFrame
            Equity curve with timestamps
        """
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "equity": self.equity_curve,
            }
        )

    def plot_equity_curve(self, save_path: str | None = None, show: bool = True):
        """Plot equity curve.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        show : bool
            Display the plot
        """
        if len(self.equity_curve) < 2:
            print("Not enough data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot equity curve
        ax.plot(
            self.timestamps, self.equity_curve, linewidth=2, label="Portfolio Value"
        )
        ax.axhline(
            y=self.initial_capital,
            color="gray",
            linestyle="--",
            label="Initial Capital",
        )

        # Fill area
        ax.fill_between(
            self.timestamps,
            self.equity_curve,
            self.initial_capital,
            where=np.array(self.equity_curve) >= self.initial_capital,
            interpolate=True,
            alpha=0.3,
            color="green",
            label="Profit",
        )
        ax.fill_between(
            self.timestamps,
            self.equity_curve,
            self.initial_capital,
            where=np.array(self.equity_curve) < self.initial_capital,
            interpolate=True,
            alpha=0.3,
            color="red",
            label="Loss",
        )

        # Formatting
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.set_title("SPY Trading - Equity Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Equity curve saved to {save_path}")

        if show:
            plt.show()

        plt.close()

    def plot_drawdown(self, save_path: str | None = None, show: bool = True):
        """Plot drawdown curve.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        show : bool
            Display the plot
        """
        if len(self.equity_curve) < 2:
            print("Not enough data to plot")
            return

        equity = np.array(self.equity_curve)
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot drawdown
        ax.fill_between(self.timestamps, drawdown, 0, alpha=0.3, color="red")
        ax.plot(self.timestamps, drawdown, linewidth=2, color="darkred")

        # Formatting
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Drawdown (%)", fontsize=12)
        ax.set_title("SPY Trading - Drawdown", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Drawdown plot saved to {save_path}")

        if show:
            plt.show()

        plt.close()

    def plot_returns_distribution(
        self, save_path: str | None = None, show: bool = True
    ):
        """Plot returns distribution histogram.

        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        show : bool
            Display the plot
        """
        if len(self.equity_curve) < 2:
            print("Not enough data to plot")
            return

        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1] * 100

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot histogram
        n, bins, patches = ax.hist(returns, bins=50, alpha=0.7, edgecolor="black")

        # Color bars
        for i in range(len(patches)):
            if bins[i] < 0:
                patches[i].set_facecolor("red")
            else:
                patches[i].set_facecolor("green")

        # Add vertical line at mean
        ax.axvline(
            x=np.mean(returns),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(returns):.3f}%",
        )

        # Formatting
        ax.set_xlabel("Return (%)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(
            "SPY Trading - Returns Distribution", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Returns distribution saved to {save_path}")

        if show:
            plt.show()

        plt.close()

    def generate_report(self, save_path: str | None = None) -> str:
        """Generate comprehensive performance report.

        Parameters
        ----------
        save_path : str, optional
            Path to save report

        Returns
        -------
        str
            Formatted report
        """
        metrics = self.calculate_metrics()

        report = "\n" + "=" * 70 + "\n"
        report += "SPY OPTIONS TRADING - PERFORMANCE REPORT\n"
        report += "=" * 70 + "\n\n"

        report += f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Trading Period: {self.timestamps[0].strftime('%Y-%m-%d')} to {self.timestamps[-1].strftime('%Y-%m-%d')}\n\n"

        report += "PORTFOLIO SUMMARY\n"
        report += "-" * 70 + "\n"
        report += f"  Initial Capital:        ${self.initial_capital:,.2f}\n"
        report += (
            f"  Current Equity:         ${metrics.get('current_equity', 0):,.2f}\n"
        )
        report += (
            f"  Total Return:           {metrics.get('total_return_pct', 0):+.2f}%\n"
        )
        report += f"  Profit/Loss:            ${metrics.get('current_equity', 0) - self.initial_capital:+,.2f}\n\n"

        report += "TRADING STATISTICS\n"
        report += "-" * 70 + "\n"
        report += f"  Total Trades:           {metrics.get('num_trades', 0)}\n"
        report += f"  Win Rate:               {metrics.get('win_rate', 0)*100:.2f}%\n"
        report += f"  Average Win:            ${metrics.get('avg_win', 0):,.2f}\n"
        report += f"  Average Loss:           ${metrics.get('avg_loss', 0):,.2f}\n"
        report += f"  Profit Factor:          {metrics.get('profit_factor', 0):.2f}\n\n"

        report += "RISK METRICS\n"
        report += "-" * 70 + "\n"
        report += f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}\n"
        report += f"  Sortino Ratio:          {metrics.get('sortino_ratio', 0):.2f}\n"
        report += f"  Calmar Ratio:           {metrics.get('calmar_ratio', 0):.2f}\n"
        report += (
            f"  Maximum Drawdown:       {metrics.get('max_drawdown_pct', 0):.2f}%\n"
        )
        report += f"  Current Drawdown:       {metrics.get('current_drawdown_pct', 0):.2f}%\n\n"

        report += "=" * 70 + "\n"

        if save_path:
            with open(save_path, "w") as f:
                f.write(report)
            print(f"Report saved to {save_path}")

        return report

    def _save_trade_to_file(self, trade: dict):
        """Save trade to log file."""
        if not self.log_file:
            return

        # Convert datetime to string
        trade_copy = trade.copy()
        if "timestamp" in trade_copy and isinstance(trade_copy["timestamp"], datetime):
            trade_copy["timestamp"] = trade_copy["timestamp"].isoformat()

        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(trade_copy) + "\n")

    def export_to_csv(self, directory: str = "./results"):
        """Export all data to CSV files.

        Parameters
        ----------
        directory : str
            Directory to save CSV files
        """
        os.makedirs(directory, exist_ok=True)

        # Export trades
        if self.trades:
            trades_df = self.get_trade_summary()
            trades_path = os.path.join(directory, "trades.csv")
            trades_df.to_csv(trades_path, index=False)
            print(f"Trades exported to {trades_path}")

        # Export equity curve
        equity_df = self.get_equity_curve_df()
        equity_path = os.path.join(directory, "equity_curve.csv")
        equity_df.to_csv(equity_path, index=False)
        print(f"Equity curve exported to {equity_path}")

        # Export metrics
        metrics = self.calculate_metrics()
        metrics_df = pd.DataFrame([metrics])
        metrics_path = os.path.join(directory, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics exported to {metrics_path}")
