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
from futu import *
from exchange_calendars import get_calendar

class PaperTradingFutu(IBroker):
    def __init__(self, host = 'futu-opend', port = 11111):
        
        self.logger = logbook.Logger(self.__class__.__name__)
        self.trd_env = TrdEnv.SIMULATE # important!
        SysConfig.enable_proto_encrypt(True)
        SysConfig.set_init_rsa_file("futu.pem")
        self.trd_ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.HK, host=host, port=port, security_firm=SecurityFirm.FUTUSECURITIES)
        self.processor = FutuProcessor()    

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
        

    def get_clock(self):
        self.logger.info ( f"get clock")
        now = datetime.now()
        is_open = False

        # xnys = xcals.get_calendar("XNYS")  # New York Stock Exchange
        # xhkg = xcals.get_calendar("XHKG")  # Hong Kong Stock Exchange
        # https://pypi.org/project/exchange-calendars/ # calendar list
        exchange_cal = get_calendar('XHKG')

        is_open = exchange_cal.is_trading_minute( now.strftime('%Y-%m-%d %H:%M'))
        
        today_sess = exchange_cal.session_open( now.strftime('%Y-%m-%d'))
        
        # Get the next open timestamp
        next_open = exchange_cal.next_open(today_sess)


        # Get the next close timestamp
        next_close = exchange_cal.next_close(today_sess)
        
        data = {
            "timestamp": now,
            "is_open": is_open,
            "next_open": next_open, 
            "next_close": next_close
        }

        print ( data)
        

        return ExchangeClock.from_dict( data)


        
    def list_positions(self):
        self.logger.info ( f"list postitions")
    
    def get_account(self):
        self.logger.info ( f"get account")
        
        ret, data = self.trd_ctx.accinfo_query()
        if ret == RET_OK:
            self.logger.info ( data)
            self.logger.info ( type( data))
            self.logger.info ( data.dtypes)
            self.logger.info ( data['cash'])
            return data 
        else:
            self.logger.info('accinfo_query error: ', data)

    def close_conn(self):
        self.trd_ctx.close()
        self.processor.close_conn()
        

    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list):
        self.logger.info ( f"fetch latest data")
        return self.processor.fetch_latest_data(
            ticker_list=ticker_list,
            time_interval=time_interval, # "1Min", "5Min", "15Min", "1H", "1D
            tech_indicator_list=tech_indicator_list,
        )

    def submit_order(self, stock, qty, order_type, time_in_force):
        self.logger.info ( f"submit order {stock} {qty} {order_type} {time_in_force}")



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