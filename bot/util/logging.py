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
        format="%(message)s",  # Let RichHandler do all the formatting
        datefmt="[%X]", # This is ignored when format is just message, but good practice
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_time=True)],
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
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Initiating shutdown with exit code {exit_code}.")
        
        # Clean shutdown of rich handlers to prevent shutdown warnings
        cleanup_rich_handlers()
        
        # Standard logging shutdown
        logging.shutdown()
    except Exception:
        # If logging fails during shutdown, just continue with exit
        pass
    finally:
        sys.exit(exit_code)


def cleanup_rich_handlers() -> None:
    """Clean up rich handlers to prevent shutdown warnings."""
    try:
        # Get all loggers and clean up their rich handlers
        root_logger = logging.getLogger()
        
        for handler in root_logger.handlers[:]:
            if isinstance(handler, RichHandler):
                try:
                    # Disable rich tracebacks during shutdown to prevent import errors
                    handler.rich_tracebacks = False
                    handler.close()
                except Exception:
                    # Ignore errors during handler cleanup
                    pass
        
        # Also clean up handlers from other loggers
        for name in list(logging.Logger.manager.loggerDict.keys()):
            try:
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    if isinstance(handler, RichHandler):
                        try:
                            handler.rich_tracebacks = False
                            handler.close()
                        except Exception:
                            pass
            except Exception:
                # Ignore errors accessing loggers during shutdown
                pass
                
    except Exception:
        # If cleanup fails, just continue - we're shutting down anyway
        pass