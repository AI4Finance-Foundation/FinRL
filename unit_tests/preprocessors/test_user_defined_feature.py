from __future__ import annotations

import pandas as pd
import pytest

from finrl.meta.preprocessor.preprocessors import FeatureEngineer


def test_daily_return_is_computed_per_ticker():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            + ["2024-01-04", "2024-01-04"],
            "tic": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 100.0, 11.0, 110.0, 12.1, 99.0],
        }
    )

    result = FeatureEngineer().add_user_defined_feature(df)

    returns = result.set_index(["date", "tic"])["daily_return"]
    assert pd.isna(returns[("2024-01-02", "AAA")])
    assert pd.isna(returns[("2024-01-02", "BBB")])
    assert returns[("2024-01-03", "AAA")] == pytest.approx(0.1)
    assert returns[("2024-01-04", "AAA")] == pytest.approx(0.1)
    assert returns[("2024-01-03", "BBB")] == pytest.approx(0.1)
    assert returns[("2024-01-04", "BBB")] == pytest.approx(-0.1)
