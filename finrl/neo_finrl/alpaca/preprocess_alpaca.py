import numpy as np

def preprocess(alpaca_df, stock_list):
    alpaca_df = alpaca_df.fillna(axis=0,method='ffill')
    alpaca_df = alpaca_df.fillna(axis=0,method='bfill')
    alpaca_df = alpaca_df.dropna()
    if_first_time = True
    for stock in stock_list:
        df = alpaca_df[stock]
        ary = df.values
        if if_first_time:
            dataset = ary
            if_first_time = False
        else:
            dataset = np.hstack((dataset,ary))
    return dataset