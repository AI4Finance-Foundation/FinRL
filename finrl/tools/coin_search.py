#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:02:02 2020

@author: kaizentech
Coin Research into txt List (Binance & Bittrex)
"""

import time
import os
import ccxt
import re
import pandas as pd
import json

####################### Coin Search ###############################################
######################### Binance ################################################
coin="XRP"
# symbols = binance.id, binance.symbols
# print(symbols)
def coinSearch(coin, top=50):
    ex = ccxt.binance({'enableRateLimit': True,}) # Loads Binance
    ex.load_markets ()
    symbols = ex.symbols
    coin_search = re.compile(".*"+coin+".*")
    coinlist = list(filter(coin_search.match, symbols))

    pd_tickers = pd.DataFrame()
    pair = ex.fetch_tickers(coinlist)
    for i in pair:
        pd_tickers = pd_tickers.append(pd.json_normalize(pair[i]), ignore_index=True)
        highest_coins = pd_tickers[["symbol","bidVolume"]].nlargest(top, "bidVolume")
        
    highest_coins.duplicated()
    highest_coins = highest_coins.drop_duplicates()
    coins = highest_coins["symbol"].tolist()
    return coins


def coins_to_json(config, coinslist):
    with open(config, 'r+') as f:
        data = json.load(f)
        data['exchange']["pair_whitelist"] = coinslist
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4,separators=(", ", ": "))
        f.truncate()     # remove remaining part
def stocks_to_json(config, coinslist):
    with open(config, 'r+') as f:
        data = json.load(f)
        data['ticker_list'] = coinslist
        f.seek(0)        # <--- should reset file position to the beginning.
        json.dump(data, f, indent=4,separators=(", ", ": "))
        f.truncate()     # remove remaining part

def coins_to_txt(coins, Path):
    with open(Path, 'w') as filehandle:
        json.dump(coins, filehandle)
    
######################### Bittrex ################################################


# coin="XRP"
# # symbols = binance.id, binance.symbols
# # print(symbols)

# bittrex = ccxt.bittrex({'enableRateLimit': True,}) # Loads Bittrex
# bittrex.load_markets ()
# symbols = bittrex.symbols
# coin_search = re.compile(".*"+coin+".*")
# coinlist = list(filter(coin_search.match, symbols))

# if bittrex.has['fetchOHLCV']:
#     for symbol in bittrex.markets:
#         time.sleep (bittrex.rateLimit / 1000) # time.sleep wants seconds
#         print (symbol, bittrex.fetch_ohlcv (symbol, '1d')) # one day

# coins = bittrex.fetch_ohlcv(bittrex.markets)

# for i in coins:
#     print(i)
    
# pd_tickers = pd.DataFrame()
# pair = bittrex.fetch_tickers(coinlist)
# for i in pair:
#     pd_tickers = pd_tickers.append(pd.json_normalize(pair[i]), ignore_index=True)
#     highest_coins = pd_tickers[["symbol","bidVolume"]].nlargest(100, "bidVolume")
    
# highest_coins.duplicated()
# highest_coins = highest_coins.drop_duplicates()
# coins = highest_coins["symbol"].tolist()


# with open('coinlist'+coin+'.txt', 'w') as filehandle:
#     json.dump(coins, filehandle)