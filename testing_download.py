from finrl.config.configuration import Configuration
from finrl.config.directory_operations import create_userdata_dir
from finrl.commands import start_download_cryptodata, start_download_stockdata, start_list_markets
from pathlib import Path
from finrl.data.fetchdata import FetchData
import pandas as pd
from finrl.config import TimeRange
from datetime import datetime, timedelta
import arrow

#### CREATE USER DATA DIRECTORY IN DESIGNATED PATH, IF NO NAME INDICATED DEFAULT TO user_data
####### create dir to false if only to check existence of directory

# create_userdata_dir("./user_data",create_dir=True)


# ###### Pull Configuration File (using finrl/config/configuration.py)
config = Configuration.from_files(["config.json"])

##### EXAMPLE
##### if directory path is kept none, default = user_data
# create_userdata_dir("./finrl_testing", create_dir=True)

##### args are the different options that could overide config options

# ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
#                       'user_data_dir': None, 'pairs': None, 'pairs_file': None, 
#                       'days': 160, 'timerange': None, 
#                       'download_trades': False, 'exchange': 'binance', 
#                       'timeframes': ['1d'], 'erase': False, 
#                       'dataformat_ohlcv': None, 'dataformat_trades': None}

# ######## downloads data to our local data repository as dictated by our config, or we could overide it using 'datadir'
# start_download_cryptodata(ARGS_DOWNLOAD_DATA)

# ################# fetches all our local data and outputs a df with the normal format (index:date, open, high, low, close, volume and tick symbol)
# ################ can be modified to get its own ARGS and overide config info, using config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)


# df = FetchData(config).fetch_data_crypto()

# print(df.head())

################## Either input timerange or days for period of download
# ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
#                       'user_data_dir': None, 'days': None, 'timerange': "20200101-20210101",
#                       'timeframes': ['1d'], 'erase': False}


# start_download_stockdata(ARGS_DOWNLOAD_DATA)



# df = FetchData(config).fetch_data_stock()

# print(df.head())

# ARGS_LIST_PAIRS = ["exchange": config["exchange"]["name"], "print_list", "list_pairs_print_json", "print_one_column",
#                    "print_csv", "base_currencies", "quote_currencies", "list_pairs_all"]

ARGS_LIST_PAIRS = {"exchange":config["exchange"]["name"]}

start_list_markets(ARGS_LIST_PAIRS)