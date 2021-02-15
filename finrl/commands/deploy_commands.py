import logging
import sys
from pathlib import Path
from typing import Any, Dict

from finrl.config import setup_utils_configuration
from finrl.config.directory_operations import copy_sample_files, create_userdata_dir
from finrl.exceptions import OperationalException
from finrl.misc import render_template, render_template_with_fallback
from finrl.state import RunMode


logger = logging.getLogger(__name__)


def start_create_userdir(args: Dict[str, Any]) -> None:
    """
    Create "user_data" directory to contain user data strategies, hyperopt, ...)
    
    Parameters:
    -----------
    args: 
        Cli args from Arguments()
        
    Return: 
    -------
        None
    """
    if "user_data_dir" in args and args["user_data_dir"]:
        userdir = create_userdata_dir(args["user_data_dir"], create_dir=True)
        copy_sample_files(userdir, overwrite=args["reset"])
    else:
        logger.warning("`create-userdir` requires --userdir to be set.")
        sys.exit(1)
