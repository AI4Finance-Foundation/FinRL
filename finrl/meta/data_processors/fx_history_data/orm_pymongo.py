""""""
from datetime import datetime
from typing import List
import pandas as pd

from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.database import Database
from pymongo.cursor import Cursor
from pymongo.collection import Collection
from pymongo.results import DeleteResult

from .constant import Exchange, Interval
from .vo import BarData, TickData
from .database import BaseDatabase, BarOverview, TickOverview, DB_TZ
from .config import SETTINGS


class Database(BaseDatabase):
    """MongoDB数据库接口"""

    def __init__(self) -> None:
        """"""
        # 读取配置
        self.database: str = SETTINGS["database.database"]
        self.host: str = SETTINGS["database.host"]
        self.port: int = SETTINGS["database.port"]
        self.username: str = SETTINGS["database.username"]
        self.password: str = SETTINGS["database.password"]
        self.authSource: str = SETTINGS["database.authentication_source"]

        # 创建客户端
        if self.username and self.password:
            self.client: MongoClient = MongoClient(
                host=self.host,
                port=self.port,
                tz_aware=True,
                username=self.username,
                password=self.password,
                tzinfo=DB_TZ,
                authSource=self.authSource
            )
        else:
            self.client: MongoClient = MongoClient(
                host=self.host,
                port=self.port,
                tz_aware=True,
                tzinfo=DB_TZ
            )

        # 初始化数据库
        self.db: Database = self.client[self.database]

        # 初始化K线数据表
        self.bar_collection: Collection = self.db["HistBarData"]
        self.bar_collection.create_index(
            [
                # ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("time_frame", ASCENDING),
                ("time", ASCENDING),
            ],
            unique=True
        )

        # 初始化Tick数据表
        self.tick_collection: Collection = self.db["tickdata"]
        self.tick_collection.create_index(
            [
                # ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("time", ASCENDING),
            ],
            unique=True
        )

        # 初始化K线概览表
        self.bar_overview_collection: Collection = self.db["bar_overview"]
        self.bar_overview_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
            ],
            unique=True
        )

        # 初始化Tick概览表
        self.tick_overview_collection: Collection = self.db["tick_overview"]
        self.tick_overview_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
            ],
            unique=True
        )

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据"""
        requests: List[ReplaceOne] = []

        for bar in bars:
            # 逐个插入
            filter: dict = {
                "symbol": bar.symbol,
                # "exchange": bar.exchange.value,
                "time": bar.time,
                "interval": bar.interval.value,
            }

            d: dict = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "time": bar.time,
                "interval": bar.interval.value,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest,
                "open_price": bar.open_price,
                "high_price": bar.high_price,
                "low_price": bar.low_price,
                "close_price": bar.close_price,
            }

            requests.append(ReplaceOne(filter, d, upsert=True))

        self.bar_collection.bulk_write(requests, ordered=False)

        # 更新汇总
        filter: dict = {
            "symbol": bar.symbol,
            "exchange": bar.exchange.value,
            "interval": bar.interval.value
        }

        overview: dict = self.bar_overview_collection.find_one(filter)

        if not overview:
            overview = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "interval": bar.interval.value,
                "count": len(bars),
                "start": bars[0].datetime,
                "end": bars[-1].datetime
            }
        elif stream:
            overview["end"] = bars[-1].datetime
            overview["count"] += len(bars)
        else:
            overview["start"] = min(bars[0].datetime, overview["start"])
            overview["end"] = max(bars[-1].datetime, overview["end"])
            overview["count"] = self.bar_collection.count_documents(filter)

        self.bar_overview_collection.update_one(filter, {"$set": overview}, upsert=True)

        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存TICK数据"""
        requests: List[ReplaceOne] = []

        for tick in ticks:
            filter: dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
            }

            d: dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
                "name": tick.name,
                "volume": tick.volume,
                "turnover": tick.turnover,
                "open_interest": tick.open_interest,
                "last_price": tick.last_price,
                "last_volume": tick.last_volume,
                "limit_up": tick.limit_up,
                "limit_down": tick.limit_down,
                "open_price": tick.open_price,
                "high_price": tick.high_price,
                "low_price": tick.low_price,
                "pre_close": tick.pre_close,
                "bid_price_1": tick.bid_price_1,
                "bid_price_2": tick.bid_price_2,
                "bid_price_3": tick.bid_price_3,
                "bid_price_4": tick.bid_price_4,
                "bid_price_5": tick.bid_price_5,
                "ask_price_1": tick.ask_price_1,
                "ask_price_2": tick.ask_price_2,
                "ask_price_3": tick.ask_price_3,
                "ask_price_4": tick.ask_price_4,
                "ask_price_5": tick.ask_price_5,
                "bid_volume_1": tick.bid_volume_1,
                "bid_volume_2": tick.bid_volume_2,
                "bid_volume_3": tick.bid_volume_3,
                "bid_volume_4": tick.bid_volume_4,
                "bid_volume_5": tick.bid_volume_5,
                "ask_volume_1": tick.ask_volume_1,
                "ask_volume_2": tick.ask_volume_2,
                "ask_volume_3": tick.ask_volume_3,
                "ask_volume_4": tick.ask_volume_4,
                "ask_volume_5": tick.ask_volume_5,
                "localtime": tick.localtime,
            }

            requests.append(ReplaceOne(filter, d, upsert=True))

        self.tick_collection.bulk_write(requests, ordered=False)

        # 更新Tick汇总
        filter: dict = {
            "symbol": tick.symbol,
            "exchange": tick.exchange.value
        }

        overview: dict = self.tick_overview_collection.find_one(filter)

        if not overview:
            overview = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "count": len(ticks),
                "start": ticks[0].datetime,
                "end": ticks[-1].datetime
            }
        elif stream:
            overview["end"] = ticks[-1].datetime
            overview["count"] += len(ticks)
        else:
            overview["start"] = min(ticks[0].datetime, overview["start"])
            overview["end"] = max(ticks[-1].datetime, overview["end"])
            overview["count"] = self.bar_collection.count_documents(filter)

        self.tick_overview_collection.update_one(filter, {"$set": overview}, upsert=True)

        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """读取K线数据"""
        filter: dict = {
            "symbol": symbol,
            # "exchange": exchange.value,
            "time_frame": interval.value,
            "time": {
                "$gte": start.astimezone(DB_TZ),
                "$lte": end.astimezone(DB_TZ)
            }
        }

        c: Cursor = self.bar_collection.find(filter)

        bars: List[BarData] = []
        for d in c:
            d["exchange"] = exchange
            d["interval"] = Interval(d["time_frame"]).value
            d["gateway_name"] = "DB"
            d["close"] = d["close_price"]
            d["open"] = d["open_price"]
            d.pop("_id")
            d.pop("close_price")
            d.pop("open_price")
            d.pop("time_frame")
            bar = BarData(**d)
            bars.append(bar)

        return bars

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """读取TICK数据"""
        filter: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "datetime": {
                "$gte": start.astimezone(DB_TZ),
                "$lte": end.astimezone(DB_TZ)
            }
        }

        c: Cursor = self.tick_collection.find(filter)

        ticks: List[TickData] = []
        for d in c:
            d["exchange"] = Exchange(d["exchange"])
            d["gateway_name"] = "DB"
            d.pop("_id")

            tick: TickData = TickData(**d)
            ticks.append(tick)

        return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        filter: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
        }

        result: DeleteResult = self.bar_collection.delete_many(filter)
        self.bar_overview_collection.delete_one(filter)

        return result.deleted_count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        filter: dict = {
            "symbol": symbol,
            "exchange": exchange.value
        }

        result: DeleteResult = self.tick_collection.delete_many(filter)
        self.tick_overview_collection.delete_one(filter)

        return result.deleted_count

    def get_bar_overview(self) -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        c: Cursor = self.bar_overview_collection.find()

        overviews: List[BarOverview] = []
        for d in c:
            d["exchange"] = Exchange(d["exchange"])
            d["interval"] = Interval(d["interval"])
            d.pop("_id")

            overview: BarOverview = BarOverview(**d)
            overviews.append(overview)

        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """查询数据库中的Tick汇总信息"""
        c: Cursor = self.tick_overview_collection.find()

        overviews: List[TickOverview] = []
        for d in c:
            d["exchange"] = Exchange(d["exchange"])
            d.pop("_id")

            overview: TickOverview = TickOverview(**d)
            overviews.append(overview)

        return overviews
