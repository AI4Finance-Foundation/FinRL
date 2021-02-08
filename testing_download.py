from finrl.config.configuration import Configuration
from finrl.config.directory_operations import create_userdata_dir
from finrl.commands import start_download_data
from pathlib import Path
from finrl.marketdata.yahoodownloader import YahooDownloader
import pandas as pd


#### CREATE USER DATA DIRECTORY IN DESIGNATED PATH, IF NO NAME INDICATED DEFAULT TO user_data
####### create dir to false if only to check existence of directory

create_userdata_dir("./user_data",create_dir=True)


###### Pull Configuration File (using finrl/config/configuration.py)
config = Configuration.from_files(["config.json"])

##### EXAMPLE
##### if directory path is kept none, default = user_data
# create_userdata_dir("./finrl_testing", create_dir=True)

##### args are the different options that could overide config options

ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
                      'user_data_dir': None, 'pairs': None, 'pairs_file': None, 
                      'days': 160, 'timerange': None, 
                      'download_trades': False, 'exchange': 'binance', 
                      'timeframes': ['1d'], 'erase': False, 
                      'dataformat_ohlcv': None, 'dataformat_trades': None}

######## downloads data to our local data repository as dictated by our config, or we could overide it using 'datadir'
start_download_data(ARGS_DOWNLOAD_DATA)

################# fetches all our local data and outputs a df with the normal format (index:date, open, high, low, close, volume and tick symbol)
################ can be modified to get its own ARGS and overide config info, using config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)


df = YahooDownloader(config).fetch_data_crypto()

print(df.head())