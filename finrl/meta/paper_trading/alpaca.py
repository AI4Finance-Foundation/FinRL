# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# Setup Alpaca Paper trading environment
from __future__ import annotations


import logbook

import alpaca_trade_api as tradeapi

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor
from finrl.meta.paper_trading.broker import IBroker


class PaperTradingAlpaca(IBroker):
    def __init__( self, alpaca_api_key, alpaca_api_secret, alpaca_api_base_url):
        self.logger = logbook.Logger(self.__class__.__name__)

        self.alpaca = tradeapi.REST(alpaca_api_key, alpaca_api_secret, alpaca_api_base_url, "v2")
        self.processor = AlpacaProcessor(alpaca_api_key, alpaca_api_secret, alpaca_api_base_url)
    

    def list_orders(self, status="open"):
        # self.logger.info ( f"list order {status}")
        return self.alpaca.list_orders(status=status)

    def cancel_order(self, order_id):
        self.logger.info ( f"cancel order {order_id}")
        return self.alpaca.cancel_order( order_id)

    def get_clock(self):
        # self.logger.info ( f"get clock")
        return self.alpaca.get_clock()
        
    def list_positions(self):
        # self.logger.info ( f"list postitions")
        return self.alpaca.list_positions()
    

    
    def get_account(self):
        return self.alpaca.get_account()

    def close_conn(self):
        self.logger.info ( f"close connection")
        

    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list):
        self.logger.info ( f"fetch latest data!!!")
        (array([214.12]), array([ 5.70769136e-03,  2.14227649e+02,  2.14112931e+02,  4.99760911e+01,
       -1.93160228e+01,  5.72512439e+01,  2.14190193e+02,  2.14109930e+02]), array([10.78]))
        data = self.processor.fetch_latest_data(
            ticker_list=ticker_list,
            time_interval=time_interval, # "1Min", "5Min", "15Min", "1H", "1D
            tech_indicator_list=tech_indicator_list,
        )
    
        self.logger.info ( f"{data} !!!")
        return data

    def submit_order(self, stock, qty, order_type, time_in_force):
        self.logger.info ( f"submit order {stock} {qty} {order_type} {time_in_force}")
        return self.alpaca.submit_order(
            symbol=stock,
            qty=qty,
            side="buy",
            type=order_type, # market
            time_in_force=time_in_force, #day
        )