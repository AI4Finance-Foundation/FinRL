import alpaca_trade_api as tradeapi
import pandas as pd

def data_fetch(API_KEY, API_SECRET, APCA_API_BASE_URL,stock_list=['AAPL']
               , start_date='2021-05-10',
               end_date='2021-05-10',time_interval='15Min'):
    api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
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
        barset = api.get_barset(stock_list, time_interval, start=start_time,
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


