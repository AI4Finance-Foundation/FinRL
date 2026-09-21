from __future__ import annotations

# finrl.trade pulls in finrl.meta.data_processors.processor_alpaca, which
# imports optional trading-platform dependencies (e.g. pandas_market_calendars)
# that are not required for most FinRL use cases and may not be installed in
# every environment. Guard these top-level convenience re-exports so a bare
# `import finrl` (and, transitively, any `finrl.<subpackage>` import) still
# succeeds when those optional extras are absent, instead of failing outright.
try:
    from finrl.test import test
    from finrl.trade import trade
    from finrl.train import train
except ImportError:
    pass
