import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf


def calcualte_price(df):
    """
    calcualte adjusted close price, open-high-low price and volume
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df.copy()
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['ajexdi'] = data['ajexdi'].apply(lambda x: 1 if x == 0 else x)

    data['adjcp'] = data['prccd'] / data['ajexdi']
    data['open'] = data['prcod'] / data['ajexdi']
    data['high'] = data['prchd'] / data['ajexdi']
    data['low'] = data['prcld'] / data['ajexdi']
    data['volume'] = data['cshtrd']

    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    data = data.sort_values(['tic', 'datadate'], ignore_index=True)
    return data

def add_technical_indicator(df):
        """
        calcualte technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        stock = Sdf.retype(df.copy())

        stock['close'] = stock['adjcp']
        unique_ticker = stock.tic.unique()

        macd = pd.DataFrame()
        rsi = pd.DataFrame()
        cci = pd.DataFrame()
        dx = pd.DataFrame()

        #temp = stock[stock.tic == unique_ticker[0]]['macd']
        for i in range(len(unique_ticker)):
            ## macd
            temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
            temp_macd = pd.DataFrame(temp_macd)
            macd = macd.append(temp_macd,ignore_index=True)
            ## rsi
            temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
            temp_rsi = pd.DataFrame(temp_rsi)
            rsi = rsi.append(temp_rsi,ignore_index=True)
            ## cci
            temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
            temp_cci = pd.DataFrame(temp_cci)
            cci = cci.append(temp_cci,ignore_index=True)
            ## adx
            temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
            temp_dx = pd.DataFrame(temp_dx)
            dx = dx.append(temp_dx,ignore_index=True)


        df['macd'] = macd
        df['rsi'] = rsi
        df['cci'] = cci
        df['adx'] = dx

        return df



def preprocess_data(filename, choose_stock):
    df_2020 = pd.read_csv(filename)
    # only take the columns we want
    df_2020=df_2020[['datadate','tic','prccd','ajexdi','prcod','prchd','prcld','cshtrd']]
    # filter stocks
    #choose_stock = ['SPY','QQQ','DIA']
    df_2020_2 = df_2020[df_2020.tic.isin(choose_stock)]
    # check if all the stocks have the same data length (very important)
    df_2020_2[df_2020_2.datadate>=20090000].tic.value_counts()
    df_2020_3 = df_2020_2[df_2020_2.datadate>=20090000]

    df_preprocess = calcualte_price(df_2020_3)
    df_final=add_technical_indicator(df_preprocess)
    #fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final