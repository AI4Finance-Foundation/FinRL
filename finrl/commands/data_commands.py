import logging
import sys
import yfinance
import pandas as pd
import yfinance as yf
import os

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List


from finrl.config import TimeRange, setup_utils_configuration
from finrl.data.converter import convert_ohlcv_format, convert_trades_format
from finrl.data.history import (convert_trades_to_ohlcv, refresh_backtest_ohlcv_data,
                                  refresh_backtest_trades_data)
from finrl.exceptions import OperationalException
from finrl.exchange import timeframe_to_minutes
from finrl.resolvers import ExchangeResolver
from finrl.state import RunMode


logger = logging.getLogger(__name__)


def start_download_cryptodata(args: Dict[str, Any]) -> None:
    """
    Parameters:
    -----------
      ARGS_DOWNLOAD_DATA = {'config': ['config.json'], 'datadir': None, 
                        'user_data_dir': None, 'pairs': None, 'pairs_file': None, 
                        'days': 160, 'timerange': None, 
                        'download_trades': False, 'exchange': 'binance', 
                        'timeframes': ['1d'], 'erase': False, 
                        'dataformat_ohlcv': None, 'dataformat_trades': None}
    
    Returns:
    ----------
      Json files in user_data/data/exchange/*.json
    """
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)
    if 'days' in config and 'timerange' in config:
        raise OperationalException("--days and --timerange are mutually exclusive. "
                                   "You can only specify one or the other.")
    timerange = TimeRange()
    if 'days' in config:
        time_since = (datetime.now() - timedelta(days=config['days'])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')

    if 'timerange' in config:
        timerange = timerange.parse_timerange(config['timerange'])

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config['stake_currency'] = ''

    if 'pairs' not in config:
        raise OperationalException(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this.")

    logger.info(f"About to download pairs: {config['pairs']}, "
                f"intervals: {config['timeframes']} to {config['datadir']}")

    pairs_not_available: List[str] = []

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config['exchange']['name'], config, validate=False)
    # Manual validations of relevant settings
    exchange.validate_pairs(config['pairs'])
    for timeframe in config['timeframes']:
        exchange.validate_timeframes(timeframe)

    try:
        if config.get('download_trades'):
            pairs_not_available = refresh_backtest_trades_data(
                exchange, pairs=config['pairs'], datadir=config['datadir'],
                timerange=timerange, erase=bool(config.get('erase')),
                data_format=config['dataformat_trades'])

            # Convert downloaded trade data to different timeframes
            convert_trades_to_ohlcv(
                pairs=config['pairs'], timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
                data_format_ohlcv=config['dataformat_ohlcv'],
                data_format_trades=config['dataformat_trades'],
                )
        else:
            pairs_not_available = refresh_backtest_ohlcv_data(
                exchange, pairs=config['pairs'], timeframes=config['timeframes'],
                datadir=config['datadir'], timerange=timerange, erase=bool(config.get('erase')),
                data_format=config['dataformat_ohlcv'])

    except KeyboardInterrupt:
        sys.exit("Interrupt received, aborting ...")

    finally:
        if pairs_not_available:
            logger.info(f"Pairs [{','.join(pairs_not_available)}] not available "
                        f"on exchange {exchange.name}.")

def start_download_stockdata(args: Dict[str, Any]) -> None:
    """Fetches data from Yahoo API
    
    Parameters:
    ----------
      ticker_list, timerange, 
      
    Returns:
    ---------
      Json of data
    """
    args["exchange"] = "yahoo"
    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)
    

    if 'days' in config and 'timerange' in config:
        raise OperationalException("--days and --timerange are mutually exclusive. "
                                    "You can only specify one or the other.")

    config["datadir"] = "user_data/data/yahoo"

    timerange = TimeRange()
    if 'days' in config:
        time_since = (datetime.now() - timedelta(days=config['days'])).strftime("%Y%m%d")
        timerange = TimeRange.parse_timerange(f'{time_since}-')
        start = datetime.fromtimestamp(timerange.startts).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")

    if 'timerange' in config:
        timerange = timerange.parse_timerange(config['timerange'])
        start = datetime.fromtimestamp(timerange.startts).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(timerange.stopts).strftime("%Y-%m-%d")
    try:
        data_df = pd.DataFrame()
        for tic in config['ticker_list']:
            temp_df = yf.download(tic, start=start, end=end)
            temp_df.columns = [
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
            ]
            temp_df["close"] = temp_df["adjcp"]
            temp_df = temp_df.drop(["adjcp"], axis=1)
            temp_df.to_json(f'{os.getcwd()}/{config["datadir"]}/{tic}.json')
    except KeyboardInterrupt:
        sys.exit("Interrupt received, aborting ...")



def start_convert_data(args: Dict[str, Any], ohlcv: bool = True) -> None:
    """
    Convert data from one format to another
    """
    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        convert_ohlcv_format(config,
                             convert_from=args['format_from'], convert_to=args['format_to'],
                             erase=args['erase'])
    else:
        convert_trades_format(config,
                              convert_from=args['format_from'], convert_to=args['format_to'],
                              erase=args['erase'])


def start_list_data(args: Dict[str, Any]) -> None:
    """
    List available backtest data
    """

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from tabulate import tabulate

    from freqtrade.data.history.idatahandler import get_datahandler
    dhc = get_datahandler(config['datadir'], config['dataformat_ohlcv'])

    paircombs = dhc.ohlcv_get_available_data(config['datadir'])

    if args['pairs']:
        paircombs = [comb for comb in paircombs if comb[0] in args['pairs']]

    print(f"Found {len(paircombs)} pair / timeframe combinations.")
    groupedpair = defaultdict(list)
    for pair, timeframe in sorted(paircombs, key=lambda x: (x[0], timeframe_to_minutes(x[1]))):
        groupedpair[pair].append(timeframe)

    if groupedpair:
        print(tabulate([(pair, ', '.join(timeframes)) for pair, timeframes in groupedpair.items()],
                       headers=("Pair", "Timeframe"),
                       tablefmt='psql', stralign='right'))
