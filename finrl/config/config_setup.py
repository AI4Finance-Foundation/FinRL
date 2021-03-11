import logging
from typing import Any, Dict

from finrl.state import RunMode

from .check_exchange import remove_credentials
from .config_validation import validate_config_consistency
from .configuration import Configuration

logger = logging.getLogger(__name__)


def setup_utils_configuration(
        args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for utils subcommands

    Parameters:
    -----------
    args:
        Cli args from Arguments()

    Return:
    -------
        Configuration
    """
    configuration = Configuration(args, method)
    config = configuration.get_config()
    # Ensure we do not use Exchange credentials
    remove_credentials(config)
    validate_config_consistency(config)

    return config
