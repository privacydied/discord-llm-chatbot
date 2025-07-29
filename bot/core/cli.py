"""
Handles command-line interface parsing and actions.
"""
import argparse
import sys

from bot.config import load_config, validate_required_env, ConfigurationError
from bot.util.logging import get_logger

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Discord LLM Bot")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')
    parser.add_argument('--config-check', action='store_true', help='Validate configuration and exit.')
    parser.add_argument('--version', action='store_true', help='Show version info and exit.')
    return parser.parse_args()

def show_version_info():
    """Display version and system information."""
    # This can be expanded with more detailed version info
    print("Discord LLM Bot - Version 1.0.0")
    print(f"Python Version: {sys.version}")

def validate_configuration_only():
    """Validate configuration and exit."""
    logger = get_logger(__name__)
    try:
        logger.info("--- Running Configuration-Only Validation ---", extra={'subsys': 'core', 'event': 'config_check_start'})
        validate_required_env()
        config = load_config()
        logger.info("Configuration validation successful. The following settings are active:", extra={'subsys': 'core', 'event': 'config_valid_start'})
        
        # Print a summary of the loaded configuration
        for key, value in config.items():
            # Hide sensitive values like tokens
            if "TOKEN" in key or "SECRET" in key:
                value = '********'
            logger.info(f"  • {key}: {value}", extra={'subsys': 'core', 'event': 'config_valid'})
        
        logger.info(f"  • Log level: {config.get('LOG_LEVEL', 'INFO')}", extra={'subsys': 'core', 'event': 'config_valid'})
        
    except ConfigurationError as e:
        logger = get_logger(__name__)
        logger.critical(f"Configuration error: {e}", exc_info=True, extra={'subsys': 'core', 'event': 'config_fail'})
        sys.exit(1)

