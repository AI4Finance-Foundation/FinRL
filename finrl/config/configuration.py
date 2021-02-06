"""
This module contains the configuration class
"""
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from finrl import constants
from finrl.config.load_config import load_config_file
from finrl.loggers import setup_logging
from finrl.misc import deep_merge_dicts, json_load
from finrl.state import NON_UTIL_MODES, TRADING_MODES, RunMode


logger = logging.getLogger(__name__)

class Configuration:
    """
    Class to read and init the bot configuration
    Reuse this class for the bot, backtesting, hyperopt and every script that required configuration
    """

    def __init__(self, args: Dict[str, Any], runmode: RunMode = None) -> None:
        self.args = args
        self.config: Optional[Dict[str, Any]] = None
        self.runmode = runmode

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config. Use this method to get the bot config
        :return: Dict: Bot config
        """
        if self.config is None:
            self.config = self.load_config()

        return self.config

    @staticmethod
    def from_files(files: List[str]) -> Dict[str, Any]:
        """
        Iterate through the config files passed in, loading all of them
        and merging their contents.
        Files are loaded in sequence, parameters in later configuration files
        override the same parameter from an earlier file (last definition wins).
        Runs through the whole Configuration initialization, so all expected config entries
        are available to interactive environments.
        :param files: List of file paths
        :return: configuration dictionary
        """
        c = Configuration({'config': files}, RunMode.OTHER)
        return c.get_config()

    def load_from_files(self, files: List[str]) -> Dict[str, Any]:

        # Keep this method as staticmethod, so it can be used from interactive environments
        config: Dict[str, Any] = {}

        if not files:
            return deepcopy(constants.MINIMAL_CONFIG)

        # We expect here a list of config filenames
        for path in files:
            logger.info(f'Using config: {path} ...')

            # Merge config options, overwriting old values
            config = deep_merge_dicts(load_config_file(path), config)

        # Normalize config
        if 'internals' not in config:
            config['internals'] = {}
        # TODO: This can be deleted along with removal of deprecated
        # experimental settings
        if 'ask_strategy' not in config:
            config['ask_strategy'] = {}

        if 'pairlists' not in config:
            config['pairlists'] = []

        return config

    def load_config(self) -> Dict[str, Any]:
        """
        Extract information for sys.argv and load the bot configuration
        :return: Configuration dictionary
        """
        # Load all configs
        config: Dict[str, Any] = self.load_from_files(self.args.get("config", []))

        # Keep a copy of the original configuration file
        config['original_config'] = deepcopy(config)

        self._resolve_pairs_list(config)

        return config

    def _resolve_pairs_list(self, config: Dict[str, Any]) -> None:
        """
        Helper for download script.
        Takes first found:
        * -p (pairs argument)
        * --pairs-file
        * whitelist from config
        """

        if "pairs" in config:
            return

        if "pairs_file" in self.args and self.args["pairs_file"]:
            pairs_file = Path(self.args["pairs_file"])
            logger.info(f'Reading pairs file "{pairs_file}".')
            # Download pairs from the pairs file if no config is specified
            # or if pairs file is specified explicitely
            if not pairs_file.exists():
                raise OperationalException(f'No pairs file found with path "{pairs_file}".')
            with pairs_file.open('r') as f:
                config['pairs'] = json_load(f)
                config['pairs'].sort()
            return

        if 'config' in self.args and self.args['config']:
            logger.info("Using pairlist from configuration.")
            config['pairs'] = config.get('exchange', {}).get('pair_whitelist')
        else:
            # Fall back to /dl_path/pairs.json
            pairs_file = config['datadir'] / 'pairs.json'
            if pairs_file.exists():
                with pairs_file.open('r') as f:
                    config['pairs'] = json_load(f)
                if 'pairs' in config:
                    config['pairs'].sort()