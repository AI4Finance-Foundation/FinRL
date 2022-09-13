from common import *
from finrl.config_tickers import DOW_30_TICKER

from alpaca_trade_api.rest import REST, TimeFrame
api = REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
# time_interval = TimeFrame.Hour
time_interval = '1Min'
data = api.get_bars(DOW_30_TICKER, time_interval, "2021-06-08", "2021-06-08", adjustment='raw').df
print(data)