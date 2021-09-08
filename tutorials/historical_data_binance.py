'''
Below Code is taken from the YouTube tutorial
https://www.youtube.com/watch?v=kolnmZTQev0 by Quant Nomad
It is just modified and formatted for use
The final dataframe generated can be directly used for FinRL environments
'''

import requests
import json
import pandas as pd
from datetime import datetime,timedelta

class get_binance_data:
    '''
    It takes in 5 arguments
    symbol: Symbol for the cryptocurrency, like BTCUSDT denotes Bitcoin Tether coin 
    Tether coin is pegged at $1, hence same as USD
    interval: It scrapes data for the specified interval, like 1d is for one day
    start_time: For start time
    end_time: For end time
    limit: Number of data is scrape at an instant for rate limit
    '''
    def __init__(self,symbol:str,interval:str,start_time:str,end_time:str,limit:str):
        self.url = "https://api.binance.com/api/v3/klines"

        startTime = datetime.strptime(start_time, '%Y-%m-%d')
        endTime = datetime.strptime(end_time, '%Y-%m-%d')
        
        self.start_time = self.stringify_dates(startTime)
        self.end_time = self.stringify_dates(endTime)
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def stringify_dates(self,date:datetime):
        return str(int(date.timestamp()*1000))

    def get_binance_bars(self,last_datetime):
        req_params = {"symbol": self.symbol, 'interval': self.interval,
                      'startTime': last_datetime, 'endTime': self.end_time, 'limit': self.limit}
        # For debugging purposes, uncomment these lines and if they throw an error
        # then you may have an error in req_params
        # r = requests.get(self.url, params=req_params)
        # print(r.text) 
        df = pd.DataFrame(json.loads(requests.get(self.url, params=req_params).text))
        if (len(df.index) == 0):
            return None
        
        df = df.iloc[:,0:6]
        df.columns = ['datetime','open','high','low','close','volume']

        df.open = df.open.astype("float")
        df.high = df.high.astype("float")
        df.low = df.low.astype("float")
        df.close = df.close.astype("float")
        df.volume = df.volume.astype("float")

        # No stock split and dividend announcement, hence close is same as adjusted close
        df['adj_close'] = df['close']

        df['datetime'] = [datetime.fromtimestamp(
            x / 1000.0) for x in df.datetime
        ]
        df.index = [x for x in range(len(df))]
        return df
    
    def dataframe_with_limit(self):
        df_list = []
        last_datetime = self.start_time
        while True:
            new_df = self.get_binance_bars(last_datetime)
            if new_df is None:
                break
            df_list.append(new_df)
            last_datetime = max(new_df.datetime) + timedelta(days=1)
            last_datetime = self.stringify_dates(last_datetime)
            
        final_df = pd.concat(df_list)
        date_value = [x.strftime('%Y-%m-%d') for x in final_df['datetime']]
        final_df.insert(0,'date',date_value)
        final_df.drop('datetime',inplace=True,axis=1)
        
        return final_df

if __name__ == '__main__':
    hist_data = get_binance_data('BTCUSDT','1d','2012-01-01','2021-09-01','1000')
    final_df = hist_data.dataframe_with_limit()
    final_df.to_csv('Bitcoin.csv')

        

