import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from rich.logging import RichHandler


# Icons per level for pretty console sink
LEVEL_ICONS = {
    "DEBUG": "🛈",
    "INFO": "✔",
    "WARNING": "⚠",
    "ERROR": "✖",
    "CRITICAL": "✖",
}


class IconPrefixFilter(logging.Filter):
    """Prefix log messages with an icon based on level if not already present."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            icon = LEVEL_ICONS.get(record.levelname, "")

            # Safely resolve message once here and neutralize args to avoid
            # downstream formatter crashes on msg % args mismatches.
            if getattr(record, "args", None):
                try:
                    resolved = record.msg % record.args
                    record.args = ()  # prevent re-formatting later
                except Exception:
                    # Fallback: preserve original pieces without crashing
                    resolved = f"{record.msg} | args={record.args}"
                    record.args = ()
            else:
                resolved = str(record.msg)

            if icon and not str(resolved).startswith(icon):
                record.msg = f"{icon} {resolved}"
            else:
                record.msg = resolved
        except Exception:
            # Never break logging
            pass
        return True


class JsonlFormatter(logging.Formatter):
    """Structured JSONL formatter with fixed keys and ms precision timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        # Millisecond precision
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8] + ts.strftime("%z")
        # Guard against formatting errors from msg/args mismatches
        try:
            detail = record.getMessage()
        except Exception:
            # As a last resort, emit raw components without raising
            detail = f"{record.msg} | args={getattr(record, 'args', None)}"

        payload = {
            "ts": ts_str,  # local time with ms
            "level": record.levelname,
            "name": record.name,
            # Optional structured fields (preserved if provided via logger.extra)
            "subsys": getattr(record, "subsys", None),
            "guild_id": getattr(record, "guild_id", None),
            "user_id": getattr(record, "user_id", None),
            "msg_id": getattr(record, "msg_id", None),
            "event": getattr(record, "event", None),
            "detail": detail,
        }
        return json.dumps(payload, separators=(",", ":"))


def _ensure_log_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def init_logging() -> None:
    """Configure dual-sink logging: pretty console + structured JSONL file."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    jsonl_path = Path(os.getenv("LOG_JSONL_PATH", "./logs/bot.jsonl"))

    _ensure_log_dir(jsonl_path)

    # Pretty console sink
    pretty = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,
        markup=True,
        enable_link_path=False,
    )
    pretty.addFilter(IconPrefixFilter())

    # Structured JSONL sink
    jsonl_handler = logging.FileHandler(str(jsonl_path), encoding="utf-8")
    jsonl_handler.setFormatter(JsonlFormatter())

    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # RichHandler handles formatting for console
        handlers=[pretty, jsonl_handler],
        force=True,
    )

    # Enforce both sinks are present
    enforce_dual_logging_handlers()

    # Announce initialization
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized (dual-sink: pretty console + JSONL)")


def enforce_dual_logging_handlers() -> None:
    """Assert two active handlers (pretty + jsonl). Abort if either missing."""
    root = logging.getLogger()
    has_pretty = any(isinstance(h, RichHandler) for h in root.handlers)
    has_jsonl = any(isinstance(h, logging.FileHandler) and isinstance(h.formatter, JsonlFormatter) for h in root.handlers)
    if not (has_pretty and has_jsonl):
        # Hard abort as per user rule
        msg = "Logging misconfigured: both pretty and JSONL handlers are required"
        try:
            root.critical(msg)
        finally:
            # Ensure clean shutdown
            cleanup_rich_handlers()
            logging.shutdown()
            sys.exit(2)


def get_logger(name: str) -> logging.Logger:
    """Get a standard logger instance."""
    return logging.getLogger(name)


def shutdown_logging_and_exit(exit_code: int) -> NoReturn:
    """Safely shuts down the logger and exits the program."""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Initiating shutdown with exit code {exit_code}.")
        cleanup_rich_handlers()
        logging.shutdown()
    except Exception:
        pass
    finally:
        sys.exit(exit_code)


def cleanup_rich_handlers() -> None:
    """Clean up rich handlers to prevent shutdown warnings."""
    try:
        root_logger = logging.getLogger()

        for handler in root_logger.handlers[:]:
            if isinstance(handler, RichHandler):
                try:
                    handler.rich_tracebacks = False
                    handler.close()
                except Exception:
                    pass

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
                pass

    except Exception:
        pass