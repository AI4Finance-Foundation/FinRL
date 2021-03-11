import pytest
from pathlib import Path


def get_default_conf(testdatadir):
    """ Returns validated configuration suitable for most tests """
    configuration = {
    "max_open_trades": 10, 
    "stake_currency": "BTC", 
    "stake_amount": 0.1, 
    "tradable_balance_ratio": 0.99, 
    "fiat_display_currency": "USD", 
    "timeframe": "1d", 
    "dry_run": True, 
    "exchange": {
        "name": "binance", 
        "key": "", 
        "secret": "", 
        "ccxt_config": {
            "enableRateLimit": True
        }, 
        "ccxt_async_config": {
            "enableRateLimit": True, 
            "rateLimit": 200
        }, 
        "pair_whitelist": [
            "SC/BTC", 
            "TROY/BTC", 
            "DREP/BTC", 
            "STMX/BTC", 
            "DOGE/BTC", 
            "TRX/BTC", 
            "IOST/BTC", 
            "XVG/BTC", 
            "REEF/BTC", 
            "CKB/BTC"
        ], 
        "pair_blacklist": []
    }, 
    "ticker_list": [
        "AAPL", 
        "MSFT", 
        "JPM", 
        "V", 
        "RTX", 
        "PG", 
        "GS", 
        "NKE", 
        "DIS", 
        "AXP", 
        "HD", 
        "INTC", 
        "WMT", 
        "IBM", 
        "MRK", 
        "UNH", 
        "KO", 
        "CAT", 
        "TRV", 
        "JNJ", 
        "CVX", 
        "MCD", 
        "VZ", 
        "CSCO", 
        "XOM", 
        "BA", 
        "MMM", 
        "PFE", 
        "WBA", 
        "DD"
    ], 
    "pairlists": [
        {
            "method": "StaticPairList"
        }
    ], 
    "telegram": {
        "enabled": True, 
        "token": "", 
        "chat_id": ""
    }, 
    "api_server": {
        "enabled": False, 
        "listen_ip_address": "127.0.0.1", 
        "listen_port": 8080, 
        "verbosity": "info", 
        "jwt_secret_key": "somethingrandom", 
        "CORS_origins": [], 
        "username": "", 
        "password": ""
    }, 
    "dataformat_ohlcv": "json", 
    "dataformat_trades": "jsongz", 
    "user_data_dir": "./user_data/", 
    "TECHNICAL_INDICATORS_LIST": [
        "macd", 
        "boll_ub", 
        "boll_lb", 
        "rsi_30", 
        "cci_30", 
        "dx_30", 
        "close_30_sma", 
        "close_60_sma"
    ]
}
    return configuration


@pytest.fixture
def testdatadir() -> Path:
    """Return the path where testdata files are stored"""
    return (Path(__file__).parent / "testdata").resolve()

@pytest.fixture(scope="function")
def default_conf(testdatadir):
    return get_default_conf(testdatadir)