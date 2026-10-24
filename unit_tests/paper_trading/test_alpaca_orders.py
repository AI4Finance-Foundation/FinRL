"""Regression tests for order dispatch in Alpaca paper trading.

``submitOrder`` reports its outcome by appending to the response list it is
handed, so a response list only means anything once the thread that owns it has
been joined. Every dispatch site used to join its threads and then drop the
response lists on the floor, which made a rejected order indistinguishable
from a filled one (issue #1414).

These tests read the source instead of importing it, because
``finrl.meta.paper_trading.common`` cannot be imported: it uses ``np`` in
``Config.__init__`` but only imports numpy several hundred lines further down.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "finrl" / "meta" / "paper_trading" / "alpaca.py"

# The market-close liquidation plus the sell, buy and turbulence branches of
# ``trade``.
DISPATCH_SITES = 4


def _tree():
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _calls(tree, attr):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def _order_threads(tree):
    """Every ``Thread`` call that dispatches an order through ``submitOrder``."""
    return [
        call
        for call in _calls(tree, "Thread")
        if any(
            keyword.arg == "target"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "submitOrder"
            for keyword in call.keywords
        )
    ]


def _response_lists(tree):
    """The response list each order thread hands to ``submitOrder``."""
    response_lists = []
    for call in _order_threads(tree):
        args = next(k.value for k in call.keywords if k.arg == "args")
        assert isinstance(args, ast.Tuple), "submitOrder args must be a tuple"
        assert len(args.elts) == 4, "submitOrder takes (qty, stock, side, resp)"
        response = args.elts[3]
        assert isinstance(response, ast.Name), "the response list must be a name"
        response_lists.append(response.id)
    return response_lists


def test_order_dispatch_hands_submit_order_a_response_list():
    """All four dispatch sites pass a per-order response list to the thread."""
    tree = _tree()

    assert len(_order_threads(tree)) == DISPATCH_SITES
    assert len(_response_lists(tree)) == DISPATCH_SITES


def test_no_response_list_is_dropped_after_the_join():
    """Each response list is recorded, so it can be read once the thread ends."""
    tree = _tree()

    recorded = set()
    for call in _calls(tree, "append"):
        if (
            isinstance(call.func.value, ast.Name)
            and call.func.value.id == "submitted"
            and call.args
        ):
            recorded.update(
                node.id for node in ast.walk(call.args[0]) if isinstance(node, ast.Name)
            )

    assert set(_response_lists(tree)) <= recorded


def test_failures_are_reported_after_the_threads_are_joined():
    """Reporting a response before its thread ends would race the append."""
    tree = _tree()

    reports = _calls(tree, "reportFailedOrders")
    assert len(reports) == DISPATCH_SITES

    joins = _calls(tree, "join")
    for report in reports:
        assert isinstance(report.args[0], ast.Name)
        assert report.args[0].id == "submitted"
        assert any(join.lineno < report.lineno for join in joins)
