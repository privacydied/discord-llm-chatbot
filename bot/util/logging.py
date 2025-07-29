import logging
import os
import sys
from typing import NoReturn

from rich.logging import RichHandler


def init_logging() -> None:
    """Configure logging using the standard library and RichHandler."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Configure root logger with RichHandler as per user request
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,  # Overwrite any existing configuration
    )

    # Get a logger to announce that logging is initialized.
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized with RichHandler.")


def get_logger(name: str) -> logging.Logger:
    """Get a standard logger instance."""
    return logging.getLogger(name)


def shutdown_logging_and_exit(exit_code: int) -> NoReturn:
    """Safely shuts down the logger and exits the program."""
    logger = logging.getLogger(__name__)
    logger.info(f"Initiating shutdown with exit code {exit_code}.")
    logging.shutdown()
    sys.exit(exit_code)