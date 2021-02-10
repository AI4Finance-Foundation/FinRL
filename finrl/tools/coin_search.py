#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 12:02:02 2020

@author: kaizentech
Coin Research into txt List (Binance & Bittrex)

COIN SEARCH FOR TOP COINS TO TRAIN ON AND TRADE
"""
import time
import os
import ccxt
import re
import pandas as pd
import json

####################### Coin Search ###############################################
######################### Binance ################################################

def coinSearchBinance(coin):
    binance = ccxt.binance({'enableRateLimit': True,}) # Loads Binance
    binance.load_markets ()
    symbols = binance.symbols
    coin_search = re.compile(".*"+coin+".*")
    coinlist = list(filter(coin_search.match, symbols))

    pd_tickers = pd.DataFrame()
    pair = binance.fetch_tickers(coinlist)
    for i in pair:
        pd_tickers = pd_tickers.append(pd.json_normalize(pair[i]), ignore_index=True)
        highest_coins = pd_tickers[["symbol","bidVolume"]].nlargest(100, "bidVolume")
        
    highest_coins.duplicated()
    highest_coins = highest_coins.drop_duplicates()
    coins = highest_coins["symbol"].tolist()
    return coins

def updateConfig(config, coinlist):
    a_file = open(config, "r")
    json_object = json.load(a_file)
    a_file.close()
    print(json_object["exchange"]["pair_whitelist"])
    json_object["exchange"]["pair_whitelist"] = coinlist
    print(json_object["exchange"]["pair_whitelist"])
    a_file = open(config, "w")
    json.dump(json_object, a_file)
    a_file.close()

def write_coins_tofile(path, coinlist) -> None:
    with open(path, 'w') as filehandle:
        json.dump(coinlist, filehandle)