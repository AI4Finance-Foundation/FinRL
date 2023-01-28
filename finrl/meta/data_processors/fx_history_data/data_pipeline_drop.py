from functools import lru_cache
from typing import List
from datetime import datetime

from .fx_history_data.data_pipeline import get_database, BaseDatabase
from .fx_history_data.data_pipeline import BarData
from .fx_history_data.data_pipeline import Interval, Exchange


@lru_cache(maxsize=256)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime
) -> List[BarData]:
    """"""
    database: BaseDatabase = get_database()

    return database.load_bar_data(
        symbol, exchange, interval, start, end
    )


if __name__ == "__main__":
    df = load_bar_data(
        symbol="EURUSD",
        interval=Interval.MINUTE,
        exchange=Exchange.MT4,
        start=datetime(2022, 12, 15),
        end=datetime(2022, 12, 30),
    )
    print(df)
