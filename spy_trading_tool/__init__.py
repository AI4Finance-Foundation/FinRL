"""SPY Options Trading Tool with Real-time Learning.

A comprehensive trading system for SPY options that combines:
- Options Greeks analysis
- Multi-timeframe optimization using CAGR
- Real-time continuous learning
- Price target predictions
- Minute-by-minute updates
"""

from __future__ import annotations

from spy_trading_tool.learning_agent import ContinuousLearningAgent
from spy_trading_tool.spy_trader import SPYTradingTool
from spy_trading_tool.timeframe_optimizer import TimeframeOptimizer

__version__ = "1.0.0"
__all__ = ["SPYTradingTool", "ContinuousLearningAgent", "TimeframeOptimizer"]
