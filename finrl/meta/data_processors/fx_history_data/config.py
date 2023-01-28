"""
Global setting of the trading platform.
"""

from logging import CRITICAL
from typing import Dict, Any
from tzlocal import get_localzone_name

from .utility import load_json


SETTINGS: Dict[str, Any] = {
    "font.family": "微软雅黑",
    "font.size": 12,

    "log.active": True,
    "log.level": CRITICAL,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    "datafeed.name": "",
    "datafeed.username": "",
    "datafeed.password": "",

    # "database.timezone": get_localzone_name(),
    "database.timezone": "UTC",
    "database.name": "pymongo",
    "database.database": "market_data",
    "database.host": "localhost",
    "database.port": 27017,
    "database.username": "atp_user",
    "database.password": "qwer!@#ZXCVjkl;",
    "database.authentication_source": "market_data"
}


# Load global setting from json file.
SETTING_FILENAME: str = "atp_settings.json"
SETTINGS.update(load_json(SETTING_FILENAME))


def get_settings(prefix: str = "") -> Dict[str, Any]:
    prefix_length: int = len(prefix)
    return {k[prefix_length:]: v for k, v in SETTINGS.items() if k.startswith(prefix)}
