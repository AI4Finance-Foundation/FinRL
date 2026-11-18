from __future__ import annotations

import importlib

__all__ = ["test", "trade", "train"]


def __getattr__(name):
    # finrl.train / finrl.test / finrl.trade pull in the data-vendor and
    # paper-trading stacks (Alpaca, WRDS, Selenium, torch). Import them only
    # when they are asked for, so `import finrl` works with the core
    # dependencies alone.
    if name in __all__:
        function = getattr(importlib.import_module(f"finrl.{name}"), name)
        globals()[name] = function
        return function
    raise AttributeError(f"module 'finrl' has no attribute {name!r}")
