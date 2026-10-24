"""Regression tests for paper-trading order dispatch."""

from __future__ import annotations

import ast
from pathlib import Path


SOURCE = (
    Path(__file__).parents[2]
    / "finrl"
    / "meta"
    / "env_stock_trading"
    / "env_stock_papertrading.py"
)


def test_trade_starts_submit_order_threads_without_calling_submit_order_early():
    """Order dispatch must pass the callable and arguments separately to Thread."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    thread_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Thread"
    ]

    order_threads = [
        node
        for node in thread_calls
        if any(
            keyword.arg == "target"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "submitOrder"
            for keyword in node.keywords
        )
    ]

    assert len(order_threads) == 3
    assert all(
        any(keyword.arg == "args" for keyword in node.keywords)
        for node in order_threads
    )
