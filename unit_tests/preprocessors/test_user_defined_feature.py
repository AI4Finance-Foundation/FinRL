from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.preprocessor.preprocessors import FeatureEngineer


def test_daily_return_is_computed_per_ticker():
    df = pd.DataFrame(
        {"tic": ["AAA", "BBB", "AAA", "BBB"], "close": [10.0, 100.0, 11.0, 90.0]}
    )
    returns = FeatureEngineer().add_user_defined_feature(df)["daily_return"]
    assert returns.tolist()[2:] == pytest.approx([0.1, -0.1])
