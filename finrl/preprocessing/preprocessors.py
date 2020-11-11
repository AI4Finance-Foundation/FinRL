import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
from config import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    #_data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data

def data_split(df,start,end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.datadate > start) & (df.datadate < end)]
    data=data.sort_values(['datadate','tic'],ignore_index=True)
    #data  = data[final_columns]
    data.index = data.datadate.factorize()[0]
    return data

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
            macd = macd.append(temp_macd, ignore_index=True)
            ## rsi
            temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
            temp_rsi = pd.DataFrame(temp_rsi)
            rsi = rsi.append(temp_rsi, ignore_index=True)
            ## cci
            temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
            temp_cci = pd.DataFrame(temp_cci)
            cci = cci.append(temp_cci, ignore_index=True)
            ## adx
            temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
            temp_dx = pd.DataFrame(temp_dx)
            dx = dx.append(temp_dx, ignore_index=True)


        df['macd'] = macd
        df['rsi'] = rsi
        df['cci'] = cci
        df['adx'] = dx

        return df



def preprocess_data():
    """data preprocessing pipeline"""

    df = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # get data after 2009
    df = df[df.datadate>=20090000]
    # calcualte adjusted price
    df_preprocess = calcualte_price(df)
    # add technical indicators using stockstats
    df_final=add_technical_indicator(df_preprocess)
    # fill the missing values at the beginning
    df_final.fillna(method='bfill',inplace=True)
    return df_final

def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    df_turbulence = pd.read_csv(config.TURBULENCE_DATA)
    df = df.merge(df_turbulence, on='datadate')
    return df


    
def get_type_list(feature_number):
    """
    :param feature_number: an int indicates the number of features
    :return: a list of features n
    """
    if feature_number == 1:
        type_list = ["close"]
    elif feature_number == 2:
        type_list = ["close", "volume"]
        raise NotImplementedError("the feature volume is not supported currently")
    elif feature_number == 3:
        type_list = ["close", "high", "low"]
    elif feature_number == 4:
        type_list = ["close", "high", "low", "open"]
    else:
        raise ValueError("feature number could not be %s" % feature_number)
    return type_list