'''Reference: https://github.com/AI4Finance-LLC/FinRL'''

import pandas as pd
import yfinance as yf
import numpy as np
from stockstats import StockDataFrame as Sdf
import trading_calendars as tc
import pytz

class YahooFinanceProcessor():
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)
    Methods
    -------
    fetch_data()
        Fetches data from yahoo API
    """

    def __init__(self):
        pass
    
    def download_data(self, start_date: str, end_date: str, ticker_list: list,
                      time_interval: str) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval
        
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(tic, start=start_date, end=end_date)
            temp_df["tic"] = tic
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        return data_df
    
    def clean_data(self, data) -> pd.DataFrame:
        
        df = data.copy()
        df = df.rename(columns={'date':'time'})
        time_interval = self.time_interval
        #get ticker list
        tic_list = np.unique(df.tic.values)
    
        #get complete time index
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        if time_interval == '1D':
            times = trading_days
        elif time_interval == '1Min':
            times = []
            for day in trading_days:
                NY = 'America/New_York'
                current_time = pd.Timestamp(day+' 09:30:00').tz_localize(NY)
                for i in range(390):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError('Data clean at given time interval is not supported for YahooFinance data.')
            
        #fill NaN data
        new_df = pd.DataFrame()
        for tic in tic_list:
            print (('Clean data for ') + tic)
            #create empty DataFrame using complete time index
            tmp_df = pd.DataFrame(columns=['open','high','low','close',
                                           'adjcp','volume'], 
                                  index=times)
            #get data for current ticker
            tic_df = df[df.tic == tic]
            #fill empty DataFrame using orginal data
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]['time']] = tic_df.iloc[i]\
                    [['open','high','low','close','adjcp','volume']]
            
            #if close on start date is NaN, fill data with first valid close 
            #and set volume to 0.
            if str(tmp_df.iloc[0]['close']) == 'nan':
                print('NaN data on start date, fill using first valid data.')
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]['close']) != 'nan':
                        first_valid_close = tmp_df.iloc[i]['close']
                        first_valid_adjclose = tmp_df.iloc[i]['adjcp']
                        
                tmp_df.iloc[0] = [first_valid_close, first_valid_close, 
                                  first_valid_close, first_valid_close,
                                  first_valid_adjclose, 0.0]
                
            #fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]['close']) == 'nan':
                    previous_close = tmp_df.iloc[i-1]['close']
                    previous_adjcp = tmp_df.iloc[i-1]['adjcp']
                    if str(previous_close) == 'nan':
                        raise ValueError
                    tmp_df.iloc[i] = [previous_close, previous_close, previous_close,
                                      previous_close, previous_adjcp, 0.0]
            
            #merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df['tic'] = tic
            new_df = new_df.append(tmp_df)
        
            print (('Data clean for ') + tic + (' is finished.'))
            
        #reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={'index':'time'})
        
        print('Data clean all finished!')
        
        return new_df
    
    def add_technical_indicator(self, data, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['tic','time'])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator['tic'] = unique_ticker[i]
                    temp_indicator['time'] = df[df.tic == unique_ticker[i]]['time'].to_list()
                    indicator_df = indicator_df.append(
                        temp_indicator, ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[['tic','time',indicator]],on=['tic','time'],how='left')
        df = df.sort_values(by=['time','tic'])
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data, time_period = 252):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="time", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
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
            #cov_temp = hist_price.cov()
            #current_temp=(current_price - np.mean(hist_price,axis=0))
            
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
            {"time": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
    
    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = self.download_data(start_date= df.time.min(), 
                                    end_date= df.time.max(),
                                    ticker_list = ["^VIX"],
                                    time_interval = self.time_interval)
        df_vix = self.clean_data(df_vix)
        vix = df_vix[['time','adjcp']]
        vix.columns = ['time','vix']

        df = df.merge(vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df
    
    def df_to_array(self, df, tech_indicator_list, if_vix):
        """transform final df to numpy arrays"""
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic==tic][['adjcp']].values
                #price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic==tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic==tic]['vix'].values 
                else:
                    turbulence_array = df[df.tic==tic]['turbulence'].values 
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic==tic][['adjcp']].values])
                tech_array = np.hstack([tech_array, df[df.tic==tic][tech_indicator_list].values])
        assert price_array.shape[0] == tech_array.shape[0]
        assert tech_array.shape[0] == turbulence_array.shape[0]
        print('Successfully transformed into array')
        return price_array, tech_array, turbulence_array
        

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar('NYSE')
        df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC),
                                    pd.Timestamp(end,tz=pytz.UTC))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
    
        return trading_days