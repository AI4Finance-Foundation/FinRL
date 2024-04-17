from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from finrl.meta.preprocessor.preprocessors import GroupByScaler

test_dataframe = pd.DataFrame(
    {
        "tic": ["A", "B", "A", "B", "A", "B"],
        "feature_1": [5.0, 3.0, 9.0, 12.0, 0.0, 5.0],
        "feature_2": [9.0, 11.0, 7.0, 3.0, 9.0, 13.0],
    }
)


def test_fit_transform():
    scaler = GroupByScaler(by="tic")
    transformed_df = scaler.fit_transform(test_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        5 / 9,
        1 / 4,
        1.0,
        1.0,
        0.0,
        5 / 12,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        1.0,
        11 / 13,
        7 / 9,
        3 / 13,
        1.0,
        1.0,
    ]


def test_fit_transform_specific_column():
    scaler = GroupByScaler(by="tic", columns=["feature_1"])
    transformed_df = scaler.fit_transform(test_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        5 / 9,
        1 / 4,
        1.0,
        1.0,
        0.0,
        5 / 12,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        9.0,
        11.0,
        7.0,
        3.0,
        9.0,
        13.0,
    ]


def test_fit_transform_other_df():
    scaler = GroupByScaler(by="tic")
    scaler.fit(test_dataframe)
    another_dataframe = pd.DataFrame(
        {
            "tic": ["A", "B", "A", "B"],
            "feature_1": [7.0, 5.0, 8.0, 10.0],
            "feature_2": [1.0, 3.0, 2.0, 5.0],
        }
    )
    transformed_df = scaler.transform(another_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        7 / 9,
        5 / 12,
        8 / 9,
        5 / 6,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        1 / 9,
        3 / 13,
        2 / 9,
        5 / 13,
    ]


def test_minmax_fit_transform():
    scaler = GroupByScaler(by="tic", scaler=MinMaxScaler)
    transformed_df = scaler.fit_transform(test_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        5 / 9,
        0.0,
        1.0,
        1.0,
        0.0,
        2 / 9,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        1.0,
        4 / 5,
        0.0,
        0.0,
        1.0,
        1.0,
    ]


def test_minmax_fit_transform_specific_column():
    scaler = GroupByScaler(by="tic", scaler=MinMaxScaler, columns=["feature_1"])
    transformed_df = scaler.fit_transform(test_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        5 / 9,
        0.0,
        1.0,
        1.0,
        0.0,
        2 / 9,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        9.0,
        11.0,
        7.0,
        3.0,
        9.0,
        13.0,
    ]


def test_minmax_fit_transform_other_df():
    scaler = GroupByScaler(by="tic", scaler=MinMaxScaler)
    scaler.fit(test_dataframe)
    another_dataframe = pd.DataFrame(
        {
            "tic": ["A", "B", "A", "B"],
            "feature_1": [7.0, 5.0, 8.0, 10.0],
            "feature_2": [1.0, 3.0, 2.0, 5.0],
        }
    )
    transformed_df = scaler.transform(another_dataframe)
    assert pytest.approx(transformed_df["feature_1"].tolist()) == [
        7 / 9,
        2 / 9,
        8 / 9,
        7 / 9,
    ]
    assert pytest.approx(transformed_df["feature_2"].tolist()) == [
        -3.0,
        0.0,
        -5 / 2,
        0.2,
    ]
