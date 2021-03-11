from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finrl.config.directory_operations import create_datadir, create_userdata_dir
from tests.test_config.conftest import default_conf


def test_create_datadir(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    create_datadir(default_conf, '/foo/bar')
    assert md.call_args[1]['parents'] is True


def test_create_userdata_dir(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=False))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    x = create_userdata_dir('/tmp/bar', create_dir=True)
    assert md.call_count == 7
    assert md.call_args[1]['parents'] is False
    assert isinstance(x, Path)
    assert str(x) == str(Path("/tmp/bar"))


def test_create_userdata_dir_exists(mocker, default_conf, caplog) -> None:
    mocker.patch.object(Path, "is_dir", MagicMock(return_value=True))
    md = mocker.patch.object(Path, 'mkdir', MagicMock())

    create_userdata_dir('/tmp/bar')
    assert md.call_count == 0
