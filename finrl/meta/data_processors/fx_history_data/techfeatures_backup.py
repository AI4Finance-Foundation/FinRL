from typing import List

import numpy as np
import pandas as pd
import talib as ta

class TechFeature:
    def __init__(self):
        pass

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class macd(TechFeature):
    """macd"""

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        return ta.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)

class boll(TechFeature):
    """boll"""

    def __call__(self, df: pd.DataFrame) -> (np.ndarray):
        upperband, middleband, lowerband = ta.BBANDS(df.close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        return (upperband, middleband, lowerband)


def tech_features_from_feature_list() -> List[TechFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features = [
        macd,
        boll
        # "rsi_30",
        # "cci_30",
        # "dx_30",
        # "close_30_sma",
        # "close_60_sma",
    ]

    return [cls() for cls in features]



def tech_features(df, feature_list):
    x = np.vstack([feat(df) for feat in tech_features_from_feature_list()]).transpose(1, 0)

    return x
