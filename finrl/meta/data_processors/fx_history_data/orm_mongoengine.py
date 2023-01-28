import datetime
from typing import List
from mongoengine import (
    Document,
    DateTimeField,
    FloatField,
    StringField,
    IntField,
    connect,
    QuerySet
)
from mongoengine.errors import DoesNotExist
from .database import BaseDatabase
from .vo import BarData, BarOverview
from .config import SETTINGS
from .constant import Interval, Exchange
from .utility import ZoneInfo


class HistMarketData(Document):
    """
    Mongoengine should use DB column name
    """

    symbol: str = StringField()
    time: datetime = DateTimeField()
    time_frame: str = StringField()
    open_price: float = FloatField()
    high: float = FloatField()
    low: float = FloatField()
    close_price: float = FloatField()

    meta = {
        'collection': 'HistMarketData',
        "indexes": [
            {
                "fields": ("symbol", "time", "time_frame"),
                "unique": True
            }
        ]
    }


class DbBarOverview(Document):
    """"""

    symbol: str = StringField()
    exchange: str = StringField()
    interval: str = StringField()
    count: int = IntField()
    start: datetime = DateTimeField()
    end: datetime = DateTimeField()

    meta = {
        "indexes": [
            {
                "fields": ("symbol", "exchange", "interval"),
                "unique": True,
            }
        ],
    }


class Database(BaseDatabase):
    """"""

    def __init__(self) -> None:
        """
        database = "market_data"
        host = "localhost"
        port = 27017
        username = 'atp_user'
        password = 'qwer!@#ZXCVjkl;'
        authentication_source = db_name
        """
        self.database = SETTINGS["database.database"]
        self.host = SETTINGS["database.host"]
        self.port = SETTINGS["database.port"]
        self.username = SETTINGS["database.username"]
        self.password = SETTINGS["database.password"]
        self.authentication_source = SETTINGS["database.authentication_source"]

        if not self.username:
            self.username = None
            self.password = None
            self.authentication_source = None

        connect(
            db=self.database,
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            authentication_source=self.authentication_source,
        )

    def save_bar_data(self, bars: List[BarData]) -> bool:
        """"""
        # Store key parameters
        bar = bars[0]
        symbol = bar.symbol
        interval = bar.interval

        # Upsert data into mongodb
        for bar in bars:
            bar.time = convert_tz(bar.time)

            d = bar.__dict__
            d["time_frame"] = d["time_frame"].value
            d.pop("gateway_name")
            d.pop("vt_symbol")
            param = to_update_param(d)

            DbBarData.objects(
                symbol=d["symbol"],
                time_frame=d["time_frame"],
                time=d["time"],
            ).update_one(upsert=True, **param)

        # Update bar overview
        try:
            overview: DbBarOverview = DbBarOverview.objects(
                symbol=symbol,
                time_frame=interval.value
            ).get()
        except DoesNotExist:
            overview: DbBarOverview = DbBarOverview(
                symbol=symbol,
                time_frame=interval.value
            )

        if not overview.start:
            overview.start = bars[0].time
            overview.end = bars[-1].time
            overview.count = len(bars)
        else:
            overview.start = min(bars[0].time, overview.start)
            overview.end = max(bars[-1].time, overview.end)
            overview.count = DbBarData.objects(
                symbol=symbol,
                time_frame=interval.value
            ).count()

        overview.save()

    def load_bar_data(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime
    ) -> List[BarData]:
        """"""
        s: QuerySet = HistMarketData.objects(
            symbol=symbol,
            time_frame=interval.value,
            time__gte=convert_tz(start),
            time__lte=convert_tz(end)
        )
        atp_symbol = f"{symbol}.{exchange.value}"
        bars: List[BarData] = []
        for db_bar in s:
            # db_bar.time = DB_TZ.localize(db_bar.time)
            db_bar.interval = Interval(db_bar.time_frame)
            db_bar.high_price = db_bar.high
            db_bar.low_price = db_bar.low
            db_bar.gateway_name = "DB"
            db_bar.atp_symbol = atp_symbol
            bars.append(db_bar)

        return bars

    def delete_bar_data(
            self,
            symbol: str,
            time_frame: Interval
    ) -> int:
        """"""
        count = DbBarData.objects(
            symbol=symbol,
            time_frame=interval.value
        ).delete()

        # Delete bar overview
        DbBarOverview.objects(
            symbol=symbol,
            time_frame=interval.value
        ).delete()

        return count

    def get_bar_overview(self) -> List[BarOverview]:
        """
        Return data avaible in database.
        """
        # Init bar overview for old version database
        data_count = HistMarketData.objects.count()
        overview_count = DbBarOverview.objects.count()
        if data_count and not overview_count:
            self.init_bar_overview()

        s: QuerySet = DbBarOverview.objects()
        overviews = []
        for overview in s:
            overview.time_frame = Interval(overview.interval)
            overviews.append(overview)
        return overviews

    def init_bar_overview(self) -> None:
        """
        Init overview table if not exists.
        """
        s: QuerySet = (
            HistMarketData.objects.aggregate({
                "$group": {
                    "_id": {
                        "symbol": "$symbol",
                        "time_frame": "$interval",
                    },
                    "count": {"$sum": 1}
                }
            })
        )

        for d in s:
            id_data = d["_id"]

            overview = DbBarOverview()
            overview.symbol = id_data["symbol"]
            overview.time_frame = id_data["time_frame"]
            overview.count = d["count"]

            start_bar: DbBarData = (
                DbBarData.objects(
                    symbol=id_data["symbol"],
                    time_frame=id_data["time_frame"],
                )
                .order_by("+datetime")
                .first()
            )
            overview.start = start_bar.datetime

            end_bar: DbBarData = (
                DbBarData.objects(
                    symbol=id_data["symbol"],
                    time_frame=id_data["time_frame"],
                )
                .order_by("-datetime")
                .first()
            )
            overview.end = end_bar.time

            overview.save()


def to_update_param(d: dict) -> dict:
    """
    Convert data dict to update parameters.
    """
    param = {f"set__{k}": v for k, v in d.items()}
    return param


DB_TZ = ZoneInfo(SETTINGS["database.timezone"])


def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    """
    dt: datetime = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)
