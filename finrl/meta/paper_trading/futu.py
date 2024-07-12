import attr
import datetime
import threading
import time
import logbook

# import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd
import torch

from finrl.meta.paper_trading.broker import IBroker
from finrl.meta.data_processors.processor_futu import FutuProcessor
from futu import RET_OK, RET_ERROR, TrdSide, OrderType, TrdEnv, ModifyOrderOp, OpenSecTradeContext, SysConfig, SecurityFirm, TrdMarket
from exchange_calendars import get_calendar

class PaperTradingFutu(IBroker):
    def __init__(self, host = 'futu-opend', port = 11111, pwd_unlock = '', rsa_file='futu.pem', exchange = 'XHKG'):
        
        self.logger = logbook.Logger(self.__class__.__name__)
        self.trd_env = TrdEnv.SIMULATE # important!
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file( rsa_file)
        self.logger.info ( f'PaperTradingFutu {host} {port}' )
        self.trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host=host, port=port, security_firm=SecurityFirm.FUTUSECURITIES)
        self.processor = FutuProcessor(host=host, port=port, rsa_file=rsa_file)

        self.pwd_unlock = pwd_unlock

        # https://pypi.org/project/exchange-calendars/#Calendars
        self.exchange = exchange # exchange ISO code
        

    # https://openapi.futunn.com/futu-api-doc/en/trade/get-order-list.html

    # order status list
    # https://openapi.futunn.com/futu-api-doc/en/trade/trade.html#8074
    def list_orders(self, status="open"):


        self.logger.info ( f"list order status:{status}")
        ret, data = self.trd_ctx.order_list_query(status_filter_list=[])
        try:
            if ret == RET_OK:
                
                data['id'] = data["order_id"]
                return data.iterrows()
            else:
                raise Exception('order_list_query error: ', data)
        except Exception as e:
            self.logger.error ( e)
            
        
        

    def cancel_order(self, order_id):
        self.logger.info ( f"cancel order {order_id}")
        ret, data = trd_ctx.unlock_trade(self.pwd_unlock)  # If you use a live trading account to modify or cancel an order, you need to unlock the account first. The example here is to cancel an order on a paper trading account, and unlocking is not necessary.
        if ret == RET_OK:
            ret, data = trd_ctx.modify_order(ModifyOrderOp.CANCEL, order_id, 0, 0)
            if ret == RET_OK:
                self.logger.info ( data)
                self.logger.info ( data['order_id'][0])  # Get the order ID of the modified order
                self.logger.info ( data['order_id'].values.tolist())  # Convert to list
            else:
                self.logger.info ( 'modify_order error: ', data)
        else:
            self.logger.info ( 'unlock_trade failed: ', data)
        

    def get_clock(self):
        now = datetime.now()
        # now = datetime.strptime('2024-07-10 05:45:00', '%Y-%m-%d %H:%M:%S')
        is_open = False

        # xnys = xcals.get_calendar("XNYS")  # New York Stock Exchange
        # xhkg = xcals.get_calendar("XHKG")  # Hong Kong Stock Exchange
        # https://pypi.org/project/exchange-calendars/ # calendar list
        exchange_cal = get_calendar( self.exchange)

        is_session = exchange_cal.is_session( now.strftime('%Y-%m-%d'))

        if is_session:
            is_open = exchange_cal.is_trading_minute( now.strftime('%Y-%m-%d %H:%M'))
        
            today_sess = exchange_cal.session_open( now.strftime('%Y-%m-%d'))
            is_break = exchange_cal.is_break_minute( now.strftime('%Y-%m-%d %H:%M'))

            next_minute = exchange_cal.next_minute( now.strftime('%Y-%m-%d %H:%M'))
            
            # Get the next close timestamp
            next_close = exchange_cal.next_close(today_sess)
            
            data = {
                "timestamp": now,
                "is_open": is_open,
                "is_open_on_minute": exchange_cal.is_open_on_minute( now.strftime('%Y-%m-%d %H:%M')),
                "is_session": is_session,
                "is_break": is_break,
                "next_open": next_minute, # if is break return break end, else return next open
                "next_close": next_close
            }
        else:
            data = {
                "timestamp": now,
                "is_open": is_open,
                "is_session": is_session,
                "next_open": exchange_cal.next_open( now.strftime('%Y-%m-%d')),
                "next_close": exchange_cal.next_close( now.strftime('%Y-%m-%d'))
            }

        return ExchangeClock.from_dict( data)

    # Define a function to create dynamic objects from a DataFrame row
    def _create_position(self, row):
        class Position:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            # def __repr__(self):
            #     return f"{self.__class__.__name__}({', '.join(f'{k}="{v}"' for k, v in self.__dict__.items())})"

        return Position(**row)

        
    def list_positions(self):
        self.logger.info ( f"list postitions")

        
        ret, data = self.trd_ctx.position_list_query()
        if ret == RET_OK:
            data = data.rename(columns={'code': 'symbol', 'position_side': 'side'})
            data['side'].str.lower()
            return data.apply(self._create_position, axis=1).tolist()
        else:
            self.logger.info ( 'position_list_query error: ', data)
    
    def get_account(self):
        # https://openapi.futunn.com/futu-api-doc/en/trade/get-funds.html
        ret, data = self.trd_ctx.accinfo_query(trd_env=self.trd_env)
        if ret == RET_OK:
            data['last_equity'] = data['market_val']
            return data
        else:
            raise Exception(f"accinfo_query error: {data}")

    
        

    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list):
        self.logger.info ( f"fetch latest data")
        return self.processor.fetch_latest_data(
            ticker_list=ticker_list,
            time_interval=time_interval, # "1Min", "5Min", "15Min", "1H", "1D
            tech_indicator_list=tech_indicator_list,
            limit=300
        )

    def submit_order(self, stock: str, qty: int, side: str, order_type: str, time_in_force: str):
        ret, data = self.trd_ctx.unlock_trade(self.pwd_unlock)  # If you use a live trading account to place an order, you need to unlock the account first. The example here is to place an order on a paper trading account, and unlocking is not necessary.
        if ret in (RET_OK, RET_ERROR):
            # https://openapi.futunn.com/futu-api-doc/en/trade/place-order.html
            # place market order
            trd_side = TrdSide.BUY if side == "buy" else TrdSide.SELL
            order_type = OrderType.MARKET if order_type == "market" else OrderType.NORMAL
            ret, data = self.trd_ctx.place_order(price=0.0, qty=qty, code=stock, trd_side=trd_side, trd_env=self.trd_env, order_type=order_type)
            if ret == RET_OK:
                self.logger.info ( data)
                self.logger.info ( data['order_id'][0])  # Get the order ID of the placed order
                self.logger.info ( data['order_id'].values.tolist())  # Convert to list
            else:
                raise Exception(f"place_order error: {data}")
                
        else:
            raise Exception(f"unlock_trade failed: {data}")
        

    def close_conn(self):
        self.trd_ctx.close()
        self.processor.close_conn()


@attr.s(auto_attribs=True)
class ExchangeClock:
    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime

    @staticmethod
    def from_dict(data):
        return ExchangeClock(
            timestamp= data.get("timestamp", None),
            # timestamp = datetime.utcnow(),
            is_open=data["is_open"],
            next_open= data.get("next_open", None),
            next_close= data.get("next_close", None),
        )


@attr.s(auto_attribs=True)
class Position:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)