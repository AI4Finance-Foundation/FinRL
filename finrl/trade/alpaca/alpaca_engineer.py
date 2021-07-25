import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf

class AlpacaEngineer():
    def __init__(self, API_KEY, API_SECRET, APCA_API_BASE_URL):
        try:
            self.api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
        except:
            raise ValueError('Wrong Account Info!')
            
    def data_fetch(self,stock_list=['AAPL'], start_date='2021-05-10',
                   end_date='2021-05-10',time_interval='15Min'):
        NY = 'America/New_York'
        start_date = pd.Timestamp(start_date, tz=NY)
        end_date = pd.Timestamp(end_date, tz=NY) + pd.Timedelta(days=1)
        date = start_date
        dataset = None
        if_first_time = True
        while date != end_date:
            start_time=(date + pd.Timedelta('09:30:00')).isoformat()
            end_time=(date + pd.Timedelta('16:00:00')).isoformat()
            print(('Data before ') + end_time + ' is successfully fetched')
            barset = self.api.get_barset(stock_list, time_interval, start=start_time,
                                    end=end_time, limit=500)
            if if_first_time:
                dataset = barset.df
                if_first_time = False
            else:
                dataset = dataset.append(barset.df)
            date = date + pd.Timedelta(days=1)
            if date.isoformat()[-14:-6] == '01:00:00':
                date = date - pd.Timedelta('01:00:00')
            elif date.isoformat()[-14:-6] == '23:00:00':
                date = date + pd.Timedelta('01:00:00')
            if date.isoformat()[-14:-6] != '00:00:00':
                raise ValueError('Timezone Error')
            
        return dataset
    
    def add_technical_indicators(self, df, stock_list, tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']):
        df = df.dropna()
        df = df.copy()
        column_list = [stock_list, ['open','high','low','close','volume']+(tech_indicator_list)]
        column = pd.MultiIndex.from_product(column_list)
        index_list = df.index
        dataset = pd.DataFrame(columns=column,index=index_list)
        for stock in stock_list:
            stock_column = pd.MultiIndex.from_product([[stock],['open','high','low','close','volume']])
            dataset[stock_column] = df[stock]
            temp_df = df[stock].reset_index().sort_values(by=['time'])
            temp_df = temp_df.rename(columns={'time':'date'})
            stock_df = Sdf.retype(temp_df.copy())  
            for indicator in tech_indicator_list:
                temp_indicator = stock_df[indicator].values.tolist()
                dataset[(stock,indicator)] = temp_indicator
        print('Succesfully add technical indicators')
        return dataset
    
    def df_to_ary(self, df, stock_list, tech_indicator_list):
        df = df.dropna()
        price_array = df[pd.MultiIndex.from_product([stock_list,['close']])].values
        tech_array = df[pd.MultiIndex.from_product([stock_list,tech_indicator_list])].values
        return price_array, tech_array
        