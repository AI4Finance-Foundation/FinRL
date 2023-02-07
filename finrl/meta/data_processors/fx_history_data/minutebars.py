from datetime import datetime
from typing import Callable, Optional, List
from .constant import Interval
from .vo import BarData, TickData


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data
    Notice:
    1. for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
    2. for x hour bar, x can be any number
    """

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable = None,
        interval: Interval = Interval.MINUTE
    ) -> None:
        """Constructor"""
        self.bar: BarData = None
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.hour_bar: BarData = None

        self.window: int = window
        self.window_bar: BarData = None
        self.on_window_bar: Callable = on_window_bar

        self.last_tick: TickData = None

    def update_tick(self, tick: TickData) -> None:
        """
        Update new tick data into generator.
        """
        new_minute: bool = False

        # Filter tick data with 0 last price
        if not tick.last_price:
            return

        # Filter tick data with older timestamp
        if self.last_tick and tick.datetime < self.last_tick.datetime:
            return

        if not self.bar:
            new_minute = True
        elif (
            (self.bar.datetime.minute != tick.datetime.minute)
            or (self.bar.datetime.hour != tick.datetime.hour)
        ):
            self.bar.datetime = self.bar.datetime.replace(
                second=0, microsecond=0
            )
            self.on_bar(self.bar)

            new_minute = True

        if new_minute:
            self.bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            )
        else:
            self.bar.high_price = max(self.bar.high_price, tick.last_price)
            if tick.high_price > self.last_tick.high_price:
                self.bar.high_price = max(self.bar.high_price, tick.high_price)

            self.bar.low_price = min(self.bar.low_price, tick.last_price)
            if tick.low_price < self.last_tick.low_price:
                self.bar.low_price = min(self.bar.low_price, tick.low_price)

            self.bar.close_price = tick.last_price
            self.bar.open_interest = tick.open_interest
            self.bar.datetime = tick.datetime

        if self.last_tick:
            volume_change: float = tick.volume - self.last_tick.volume
            self.bar.volume += max(volume_change, 0)

            turnover_change: float = tick.turnover - self.last_tick.turnover
            self.bar.turnover += max(turnover_change, 0)

        self.last_tick = tick

    def update_bar(self, bar: BarData) -> None:
        """
        Update 1 minute bar into generator
        """
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bar)
        else:
            self.update_bar_hour_window(bar)

    def update_bar_minute_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.window_bar:
            dt: datetime = bar.time.replace(second=0, microsecond=0)
            self.window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                time=dt,
                gateway_name=bar.gateway_name,
                open=bar.open,
                high=bar.high,
                low=bar.low
            )
        # Otherwise, update high/low price into window bar
        else:
            self.window_bar.high = max(
                self.window_bar.high,
                bar.high
            )
            self.window_bar.low = min(
                self.window_bar.low,
                bar.low
            )

        # Update close price/volume/turnover into window bar
        self.window_bar.close = bar.close
        self.window_bar.volume += bar.volume
        self.window_bar.turnover += bar.turnover
        self.window_bar.open_interest = bar.open_interest

        # Check if window bar completed
        if not (bar.time.minute + 1) % self.window:
            self.on_window_bar(self.window_bar) #get tech
            # why 29 min set window_bar = None?
            self.window_bar = None

    def update_bar_hour_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create window bar object
        if not self.hour_bar:
            dt: datetime = bar.time.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                time=dt,
                gateway_name=bar.gateway_name,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
            return

        finished_bar: BarData = None

        # If minute is 59, update minute bar into window bar and push
        if bar.time.minute == 59:
            self.hour_bar.high = max(
                self.hour_bar.high,
                bar.high
            )
            self.hour_bar.low = min(
                self.hour_bar.low,
                bar.low
            )

            self.hour_bar.close = bar.close
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

            finished_bar = self.hour_bar
            self.hour_bar = None

        # If minute bar of new hour, then push existing window bar
        elif bar.time.hour != self.hour_bar.time.hour:
            finished_bar = self.hour_bar

            dt: datetime = bar.time.replace(minute=0, second=0, microsecond=0)
            self.hour_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                time=dt,
                gateway_name=bar.gateway_name,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest
            )
        # Otherwise only update minute bar
        else:
            self.hour_bar.high = max(
                self.hour_bar.high,
                bar.high
            )
            self.hour_bar.low = min(
                self.hour_bar.low,
                bar.low
            )

            self.hour_bar.close = bar.close
            self.hour_bar.volume += bar.volume
            self.hour_bar.turnover += bar.turnover
            self.hour_bar.open_interest = bar.open_interest

        # Push finished window bar
        if finished_bar:
            self.on_hour_bar(finished_bar)

    def on_hour_bar(self, bar: BarData) -> None:
        """"""
        if self.window == 1:
            self.on_window_bar(bar)
        else:
            if not self.window_bar:
                self.window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    time=bar.time,
                    gateway_name=bar.gateway_name,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low
                )
            else:
                self.window_bar.high = max(
                    self.window_bar.high,
                    bar.high
                )
                self.window_bar.low = min(
                    self.window_bar.low,
                    bar.low
                )

            self.window_bar.close = bar.close
            self.window_bar.volume += bar.volume
            self.window_bar.turnover += bar.turnover
            self.window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bar)
                self.window_bar = None

    def generate(self) -> Optional[BarData]:
        """
        Generate the bar data and call callback immediately.
        """
        bar: BarData = self.bar

        if self.bar:
            bar.time = bar.time.replace(second=0, microsecond=0)
            self.on_bar(bar)

        self.bar = None
        return bar

class MinuteBar:
    """
    使用1m线合成 日线、30分钟、1分钟三种数据
    """

    def __init__(self, bar_data, window_mn):
        self.bar_data = bar_data
        self.bg = BarGenerator(self.on_bar, interval=Interval.MINUTE, window=window_mn, on_window_bar=self.on_minute_bar)
        self.all_bars:List[BarData] = []
        self.load_bar()

    def on_minute_bar(self, bar):
        # only called on mins
        self.all_bars.append(self.bg.window_bar)

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def load_bar(
        self,
        callback: Callable = None,
    ) -> None:
        """
        Load historical bar data for initializing strategy.
        """
        if not callback:
            callback: Callable = self.on_bar

        bars: List[BarData] = self.bar_data

        for bar in bars:
            callback(bar)



