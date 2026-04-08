from __future__ import annotations

import ast
from pathlib import Path

ALPACA_PATH = (
    Path(__file__).resolve().parents[1]
    / "finrl"
    / "meta"
    / "paper_trading"
    / "alpaca.py"
)


def _thread_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "Thread":
            calls.append(node)
    return calls


def _is_submit_order_attr(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "submitOrder"


def test_submit_order_thread_target_is_not_called_immediately() -> None:
    source = ALPACA_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    thread_calls = _thread_calls(tree)
    assert thread_calls, "Expected threading.Thread usage in alpaca.py"

    for call in thread_calls:
        target_kw = next((kw for kw in call.keywords if kw.arg == "target"), None)
        if target_kw is None:
            continue

        target = target_kw.value

        # Regression check: ensure submitOrder is not invoked when creating Thread.
        if isinstance(target, ast.Call) and _is_submit_order_attr(target.func):
            raise AssertionError(
                "submitOrder should be passed as target, not called immediately"
            )

        if _is_submit_order_attr(target):
            args_kw = next((kw for kw in call.keywords if kw.arg == "args"), None)
            assert args_kw is not None, "Thread target submitOrder should pass args=..."
