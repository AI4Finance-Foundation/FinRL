'''Reference: https://github.com/alpacahq/alpaca-trade-api-python/tree/master/examples'''

import datetime
import threading
from neo_finrl.alpaca.alpaca_engineer import AlpacaEngineer 
import alpaca_trade_api as tradeapi
import time
import pandas as pd
import numpy as np
import torch

'''please input your own account info'''
API_KEY = ""
API_SECRET = ""
APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

'''load prepared model'''
action_dim = 5
state_dim = 1+ 1 + 1+ 2*5+ 5*7
from elegantrl.agent import AgentPPO
agent = AgentPPO()
net_dim = 2 ** 7
cwd = './AgentPPO/test-v1'
agent.init(net_dim, state_dim, action_dim)
agent.save_load_model(cwd=cwd, if_save=False)
act = agent.act
device = agent.device

'''paper trading class'''
class PPO_PaperTrading:
    def __init__(self):
        self.alpaca = tradeapi.REST(API_KEY,API_SECRET,APCA_API_BASE_URL, 'v2')
        stockUniverse = [
            'AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX'
        ]
        self.stocks = np.asarray([0] * len(stockUniverse))
        self.cash = None
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index = stockUniverse)
        self.stockUniverse = stockUniverse
        self.price = np.asarray([0] * len(stockUniverse))
        self.turb_bool = 0
        self.equities = []
        
    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
          self.alpaca.cancel_order(order.id)
    
        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")
        while True:

          # Figure out when the market will close so we can prepare to sell beforehand.
          clock = self.alpaca.get_clock()
          closingTime = clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          self.timeToClose = closingTime - currTime
    
          if(self.timeToClose < (60 * 15)):
            # Close all positions when 15 minutes til market close.
            print("Market closing soon.  Closing positions.")
    
            positions = self.alpaca.list_positions()
            for position in positions:
              if(position.side == 'long'):
                orderSide = 'sell'
              else:
                orderSide = 'buy'
              qty = abs(int(float(position.qty)))
              respSO = []
              tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
              tSubmitOrder.start()
              tSubmitOrder.join()
    
            # Run script again after market close for next trading day.
            print("Sleeping until market close (15 minutes).")
            time.sleep(60 * 15)
          else:
            # Trade and save equity records
            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time,last_equity])
            np.save('./equity.npy', np.asarray(self.equities, dtype = float))
            time.sleep(60)
            
    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while(not isOpen):
          clock = self.alpaca.get_clock()
          openingTime = clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
          currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
          timeToOpen = int((openingTime - currTime) / 60)
          print(str(timeToOpen) + " minutes til market open.")
          time.sleep(60)
          isOpen = self.alpaca.get_clock().is_open
    
    def trade(self):
        state = self.get_state()
        with torch.no_grad():
            s_tensor = torch.as_tensor((state,), device=device)
            a_tensor = act(s_tensor)  
            action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        action = (action * 100).astype(int)
        if self.turb_bool == 0:
            min_action = 10  
            for index in np.where(action < -min_action)[0]:  # sell:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty =  abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)

            for index in np.where(action > min_action)[0]:  # buy:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(tmp_cash // self.price[index], abs(int(action[index])))
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                
        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if(position.side == 'long'):
                    orderSide = 'sell'
                else:
                    orderSide = 'buy'
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                tSubmitOrder.start()
                tSubmitOrder.join()
    

    def get_state(self):
        AE = AlpacaEngineer(api=self.alpaca)
        df = self.alpaca.get_barset(self.stockUniverse, '1Min', limit=1000).df
        df = AE.add_technical_indicators(df, self.stockUniverse)
        time = df.index
        first_time = True
        for stock in self.stockUniverse:
            if first_time == True:
                closes = df[(stock,'close')].values
                ary = np.vstack([time,closes]).T
                tmp_df = pd.DataFrame(ary, columns = ['date','close'])
                tmp_df['tic'] = stock
                first_time = False
            else:
                closes = df[(stock,'close')]
                ary = np.vstack([time,closes]).T
                tmp_tmp_df = pd.DataFrame(ary, columns = ['date','close'])
                tmp_tmp_df['tic'] = stock
                tmp_df = tmp_df.append(tmp_tmp_df)
        
        tmp_df = AE.add_turbulence(tmp_df)
        turbulence_ary = tmp_df[tmp_df.tic==self.stockUniverse[0]]['turbulence'].values
        turbulence_bool = (turbulence_ary > int(1e4)).astype(np.float32)
        turbulence_ary = (turbulence_ary * 2 ** -7).clip((int(1e4)) * 2)
        price_array, tech_array = AE.df_to_ary(df, self.stockUniverse)
        price = price_array[-1]
        self.price = price
        tech = tech_array[-1]
        turb = turbulence_ary[-1]
        turb_bool = turbulence_bool[-1]
        self.turb_bool = turb_bool
        positions = self.alpaca.list_positions()
        stocks = [0] * 5
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = ( abs(int(float(position.qty))))
        self.stocks = stocks
        stocks = np.asarray(stocks, dtype = float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        state = np.hstack((max(cash, 1e4) * (2 ** -17),
                           price * (2 ** -9),
                           turb,
                           turb_bool,
                           stocks * (2 ** -5),
                           tech * (2 **-9),
                           )).astype(np.float32) 
        return state
        
    def submitOrder(self, qty, stock, side, resp):
        if(qty > 0):
          try:
            self.alpaca.submit_order(stock, qty, side, "market", "day")
            print("Market order of | " + str(qty) + " " + stock + " " + side + " | completed.")
            resp.append(True)
          except:
            print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
            resp.append(False)
        else:
          print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
          resp.append(True)
          
drl = PPO_PaperTrading()
drl.run()
