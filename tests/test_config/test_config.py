# pragma pylint: disable=missing-docstring, protected-access, invalid-name
import json
import logging
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonschema import ValidationError

from finrl.config import Configuration
                                     
                                     
def test_print_config(default_conf, mocker, caplog) -> None:
    conf1 = deepcopy(default_conf)
    # Delete non-json elements from default_conf
    del conf1['user_data_dir']
    config_files = [conf1]

    configsmock = MagicMock(side_effect=config_files)
    mocker.patch('finrl.config.configuration.create_datadir', lambda c, x: x)
    mocker.patch('finrl.config.configuration.load_config_file', configsmock)

    validated_conf = Configuration.from_files(['test_conf.json'])

    assert isinstance(validated_conf['user_data_dir'], Path)
    assert "user_data_dir" in validated_conf
    assert "original_config" in validated_conf
    assert isinstance(json.dumps(validated_conf['original_config']), str)

