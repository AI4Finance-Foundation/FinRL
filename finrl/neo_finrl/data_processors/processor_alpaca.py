import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
import trading_calendars as tc
import pytz

class AlpacaProcessor():
    def __init__(self, API_KEY=None, API_SECRET=None, APCA_API_BASE_URL=None, api=None):
        if api == None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
            except:
                raise ValueError('Wrong Account Info!')
        else:
            self.api = api
            
    def download_data(self, ticker_list, start_date, end_date, 
                   time_interval) -> pd.DataFrame:
        
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval
        
        NY = 'America/New_York'
        start_date = pd.Timestamp(start_date, tz=NY)
        end_date = pd.Timestamp(end_date, tz=NY) + pd.Timedelta(days=1)
        date = start_date
        data_df = pd.DataFrame()
        while date != end_date:
            start_time=(date + pd.Timedelta('09:30:00')).isoformat()
            end_time=(date + pd.Timedelta('15:59:00')).isoformat()
            for tic in ticker_list:
                barset = self.api.get_barset([tic], time_interval, 
                                             start=start_time, end=end_time, 
                                             limit=500).df[tic]
                barset['tic'] = tic
                barset = barset.reset_index()
                data_df = data_df.append(barset)
            print(('Data before ') + end_time + ' is successfully fetched')
            date = date + pd.Timedelta(days=1)
            if date.isoformat()[-14:-6] == '01:00:00':
                date = date - pd.Timedelta('01:00:00')
            elif date.isoformat()[-14:-6] == '23:00:00':
                date = date + pd.Timedelta('01:00:00')
            if date.isoformat()[-14:-6] != '00:00:00':
                raise ValueError('Timezone Error')
        '''times = data_df['time'].values
        for i in range(len(times)):
            times[i] = str(times[i])
        data_df['time'] = times'''
        return data_df
            
    def clean_data(self, df):
        tic_list = np.unique(df.tic.values)
    
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        
        times = []
        for day in trading_days:
            NY = 'America/New_York'
            current_time = pd.Timestamp(day+' 09:30:00').tz_localize(NY)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)
        
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(columns=['open','high','low','close','volume'], 
                                  index=times)
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]['time']] = tic_df.iloc[i][['open','high',
                                                                     'low','close',
                                                                     'volume']]
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]['close']) == 'nan':
                    previous_close = tmp_df.iloc[i-1]['close']
                    if str(previous_close) == 'nan':
                        raise ValueError
                    tmp_df.iloc[i] = [previous_close, previous_close, previous_close,
                                      previous_close, 0.0]
            tmp_df = tmp_df.astype(float)
            tmp_df['tic'] = tic
            new_df = new_df.append(tmp_df)
        
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={'index':'time'})
        
        print('Data clean finished!')
        
        return new_df
      
    def add_technical_indicator(self, df, tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']):
        df = df.rename(columns={'time':'date'})
        df = df.copy()
        df = df.sort_values(by=['tic', 'date'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()
        tech_indicator_list = tech_indicator_list
        
        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                # print(unique_ticker[i], i)
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator['tic'] = unique_ticker[i]
                # print(len(df[df.tic == unique_ticker[i]]['date'].to_list()))
                temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                indicator_df = indicator_df.append(
                    temp_indicator, ignore_index=True
                )
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        df = df.sort_values(by=['date', 'tic'])
        df = df.rename(columns={'date':'time'})
        print('Succesfully add technical indicators')
        return df
    
    def add_vix(self, data):
        vix_df = self.download_data(['VIXY'], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        vix = cleaned_vix[['time','close']]
        vix = vix.rename(columns={'close':'VIXY'})
        
        df = data.copy()
        df = df.merge(vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df
    
    def calculate_turbulence(self,data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()
    
        unique_date = df.date.unique()
        # start after a fixed time period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)
    
            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
    
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
    
    def add_turbulence(self,data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic==tic][['close']].values
                tech_array = df[df.tic==tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic==tic]['VIXY'].values
                else:
                    turbulence_array = df[df.tic==tic]['turbulence'].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic==tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic==tic][tech_indicator_list].values])
        print('Successfully transformed into array')
        return price_array,tech_array,turbulence_array
    
    def get_trading_days(self, start, end):
        nyse = tc.get_calendar('NYSE')
        df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC),
                                    pd.Timestamp(end,tz=pytz.UTC))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
    
        return trading_days
    
    def fetch_latest_data(self, ticker_list, time_interval, tech_indicator_list,
                          limit=100) -> pd.DataFrame: 
        
        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = self.api.get_barset([tic], time_interval, 
                                         limit=limit).df[tic]
            barset['tic'] = tic
            barset = barset.reset_index()
            data_df = data_df.append(barset)
        
        data_df = data_df.reset_index(drop=True)
        start_time = data_df.time.min()
        end_time = data_df.time.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)
        
        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(columns=['open','high','low','close','volume'], 
                                  index=times)
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]['time']] = tic_df.iloc[i][['open','high',
                                                                     'low','close',
                                                                     'volume']]
            if str(tmp_df.iloc[0]['close']) == 'nan':
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]['close']) != 'nan':
                        first_valid_close = tmp_df.iloc[i]['close']
                        
                tmp_df.iloc[0] = [first_valid_close, first_valid_close, 
                                  first_valid_close, first_valid_close, 0.0]
                
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]['close']) == 'nan':
                    previous_close = tmp_df.iloc[i-1]['close']
                    if str(previous_close) == 'nan':
                        raise ValueError
                    tmp_df.iloc[i] = [previous_close, previous_close, previous_close,
                                      previous_close, 0.0]
            tmp_df = tmp_df.astype(float)
            tmp_df['tic'] = tic
            new_df = new_df.append(tmp_df)
        
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={'index':'time'})
        
        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df['VIXY'] = 0
        
        price_array, tech_array, turbulence_array = self.df_to_array(df, tech_indicator_list, if_vix=True)
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        turb_df = self.api.get_barset(['VIXY'], time_interval, 
                                         limit=1).df['VIXY']
        latest_turb = turb_df['close'].values
        return latest_price, latest_tech, latest_turb
        
