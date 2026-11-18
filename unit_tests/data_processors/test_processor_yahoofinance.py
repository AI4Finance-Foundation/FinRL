from __future__ import annotations

from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

_PROCESSOR_PATH = (
    Path(__file__).parents[2]
    / "finrl"
    / "meta"
    / "data_processors"
    / "processor_yahoofinance.py"
)
_SPEC = spec_from_file_location("processor_yahoofinance_under_test", _PROCESSOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load {_PROCESSOR_PATH}")
_PROCESSOR_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROCESSOR_MODULE)
YahooFinanceProcessor = _PROCESSOR_MODULE.YahooFinanceProcessor


class TestYahooFinanceProcessor(TestCase):
    def test_scrap_data_sorts_by_day_and_tic(self):
        processor = YahooFinanceProcessor()
        stock_data = {
            "BBB": pd.DataFrame({"day": [1, 0], "tic": ["BBB", "BBB"]}),
            "AAA": pd.DataFrame({"day": [1, 0], "tic": ["AAA", "AAA"]}),
        }

        def fake_fetch_stock_data(stock_name, period1, period2):
            return stock_data[stock_name]

        with patch.object(
            processor, "fetch_stock_data", side_effect=fake_fetch_stock_data
        ):
            result = processor.scrap_data(
                stock_names=["BBB", "AAA"],
                start_date="2020-01-01",
                end_date="2020-01-03",
            )

        self.assertEqual(
            result[["day", "tic"]].to_dict("records"),
            [
                {"day": 0, "tic": "AAA"},
                {"day": 0, "tic": "BBB"},
                {"day": 1, "tic": "AAA"},
                {"day": 1, "tic": "BBB"},
            ],
        )
