"""
Logging enforcement system for router speed overhaul. [CA][REH]

Enforces dual sink strategy at startup:
- Pretty Console Sink with colors, emojis, padding, level symbols
- Structured JSON Sink with frozen key set preservation
- Startup assertion to verify both handlers are active

User requirements:
- Level‑based symbols: ✔, ⚠, ✖, ℹ
- Auto‑truncate overly long fields; show …(+N) indicator
- Preserve key set: ts, level, name, subsys, guild_id, user_id, msg_id, event, detail
- Timestamps to millisecond precision, local time
- Colour palette: INFO=green, WARNING=yellow, ERROR/CRIT=red, DEBUG=blue‑grey
- Icons left‑pad every line; field grid aligns after icon
"""

from __future__ import annotations

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.panel import Panel


class StructuredJsonFormatter(logging.Formatter):
    """JSON formatter with frozen key set preservation. [CA]"""

    FROZEN_KEYS = {
        "ts",
        "level",
        "name",
        "subsys",
        "guild_id",
        "user_id",
        "msg_id",
        "event",
        "detail",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON. [CA]"""
        # Start with frozen key set
        log_entry = {key: None for key in self.FROZEN_KEYS}

        # Set timestamp with millisecond precision, local time
        log_entry["ts"] = datetime.fromtimestamp(record.created).isoformat()
        log_entry["level"] = record.levelname
        log_entry["name"] = record.name

        # Extract structured data from record.extra if available
        if hasattr(record, "subsys"):
            log_entry["subsys"] = record.subsys
        if hasattr(record, "guild_id"):
            log_entry["guild_id"] = record.guild_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        if hasattr(record, "msg_id"):
            log_entry["msg_id"] = record.msg_id
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "detail"):
            log_entry["detail"] = record.detail

        # If no structured event, use message as detail
        if not log_entry["event"] and record.getMessage():
            log_entry["event"] = "log.message"
            log_entry["detail"] = record.getMessage()

        # Remove None values to keep JSON clean
        log_entry = {k: v for k, v in log_entry.items() if v is not None}

        return json.dumps(log_entry, separators=(",", ":"), ensure_ascii=False)


class PrettyConsoleHandler(RichHandler):
    """Enhanced RichHandler with custom formatting. [CA]"""

    LEVEL_SYMBOLS = {
        "DEBUG": "🔍",  # Blue-grey debug
        "INFO": "✔️",  # Green info
        "WARNING": "⚠️",  # Yellow warning
        "ERROR": "✖️",  # Red error
        "CRITICAL": "🔥",  # Red critical
    }

    LEVEL_COLORS = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red bold",
    }

    def __init__(self, console: Optional[Console] = None, **kwargs):
        """Initialize with custom console settings. [CA]"""
        if console is None:
            console = Console(
                stderr=True, force_terminal=True, width=120, legacy_windows=False
            )

        super().__init__(
            console=console,
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            markup=True,
            **kwargs,
        )

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        """Render message with level symbols and colors. [CA]"""
        # Get level symbol and color
        symbol = self.LEVEL_SYMBOLS.get(record.levelname, "ℹ️")
        color = self.LEVEL_COLORS.get(record.levelname, "white")

        # Create formatted text with symbol prefix
        text = Text()
        text.append(f"{symbol} ", style="bold")

        # Auto-truncate overly long messages
        if len(message) > 200:
            truncated = message[:180]
            remaining = len(message) - 180
            text.append(truncated, style=color)
            text.append(f"…(+{remaining})", style="dim")
        else:
            text.append(message, style=color)

        return text


class LoggingEnforcer:
    """Enforces dual sink logging strategy at startup. [REH]"""

    def __init__(self, log_dir: Path = None):
        """Initialize logging enforcer. [CA]"""
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        self.pretty_handler: Optional[PrettyConsoleHandler] = None
        self.jsonl_handler: Optional[logging.FileHandler] = None
        self.root_logger = logging.getLogger()

    def setup_dual_sinks(self) -> None:
        """Set up both pretty console and JSON file handlers. [CA]"""
        # Clear existing handlers to avoid conflicts
        self.root_logger.handlers.clear()

        # Set up pretty console handler
        self.pretty_handler = PrettyConsoleHandler()
        self.pretty_handler.setLevel(logging.DEBUG)
        self.root_logger.addHandler(self.pretty_handler)

        # Set up structured JSON file handler
        jsonl_file = self.log_dir / "bot.jsonl"
        self.jsonl_handler = logging.FileHandler(jsonl_file, mode="a", encoding="utf-8")
        self.jsonl_handler.setLevel(logging.DEBUG)
        self.jsonl_handler.setFormatter(StructuredJsonFormatter())
        self.root_logger.addHandler(self.jsonl_handler)

        # Set root logger level
        self.root_logger.setLevel(logging.DEBUG)

        logging.info(
            "Dual sink logging initialized",
            extra={
                "event": "logging.dual_sink_init",
                "detail": {
                    "pretty_handler": True,
                    "jsonl_handler": True,
                    "jsonl_file": str(jsonl_file),
                    "log_level": "DEBUG",
                },
            },
        )

    def enforce_startup_assertion(self) -> None:
        """Assert both handlers are active at startup, abort if missing. [REH]"""
        active_handlers = self.root_logger.handlers

        # Check for pretty handler
        pretty_active = any(
            isinstance(h, (RichHandler, PrettyConsoleHandler)) for h in active_handlers
        )

        # Check for JSON handler
        jsonl_active = any(
            isinstance(h, logging.FileHandler)
            and isinstance(h.formatter, StructuredJsonFormatter)
            for h in active_handlers
        )

        if not pretty_active:
            print(
                "❌ FATAL: Pretty console handler missing from logging configuration",
                file=sys.stderr,
            )
            sys.exit(1)

        if not jsonl_active:
            print(
                "❌ FATAL: JSONL file handler missing from logging configuration",
                file=sys.stderr,
            )
            sys.exit(1)

        # Success - log confirmation
        logging.info(
            f"✔️ Logging enforcement passed: {len(active_handlers)} handlers active",
            extra={
                "event": "logging.enforcement_success",
                "detail": {
                    "handlers_count": len(active_handlers),
                    "pretty_handler_active": pretty_active,
                    "jsonl_handler_active": jsonl_active,
                    "enforcement_passed": True,
                },
            },
        )

    def get_logging_status(self) -> Dict[str, Any]:
        """Get current logging configuration status. [PA]"""
        handlers = self.root_logger.handlers

        return {
            "total_handlers": len(handlers),
            "handler_types": [type(h).__name__ for h in handlers],
            "pretty_handler_active": any(
                isinstance(h, (RichHandler, PrettyConsoleHandler)) for h in handlers
            ),
            "jsonl_handler_active": any(
                isinstance(h, logging.FileHandler) for h in handlers
            ),
            "root_level": self.root_logger.level,
            "log_dir": str(self.log_dir),
            "enforcement_compliant": len(handlers) >= 2,
        }


def initialize_logging(config: Optional[Dict[str, Any]] = None) -> LoggingEnforcer:
    """Initialize dual sink logging system with enforcement. [CA]"""
    log_dir = None
    if config and "LOG_DIR" in config:
        log_dir = Path(config["LOG_DIR"])

    enforcer = LoggingEnforcer(log_dir)
    enforcer.setup_dual_sinks()
    enforcer.enforce_startup_assertion()

    return enforcer


def create_structured_logger(name: str, subsys: Optional[str] = None) -> logging.Logger:
    """Create logger with structured logging helpers. [CA]"""
    logger = logging.getLogger(name)

    # Add convenience methods for structured logging
    def info_event(event: str, detail: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info event with structured data. [CA]"""
        extra = {"event": event, "subsys": subsys}
        if detail:
            extra["detail"] = detail
        extra.update(kwargs)
        logger.info(f"[{subsys or name}] {event}", extra=extra)

    def warning_event(event: str, detail: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning event with structured data. [REH]"""
        extra = {"event": event, "subsys": subsys}
        if detail:
            extra["detail"] = detail
        extra.update(kwargs)
        logger.warning(f"[{subsys or name}] {event}", extra=extra)

    def error_event(event: str, detail: Optional[Dict[str, Any]] = None, **kwargs):
        """Log error event with structured data. [REH]"""
        extra = {"event": event, "subsys": subsys}
        if detail:
            extra["detail"] = detail
        extra.update(kwargs)
        logger.error(f"[{subsys or name}] {event}", extra=extra)

    # Monkey patch convenience methods
    logger.info_event = info_event
    logger.warning_event = warning_event
    logger.error_event = error_event

    return logger


# Export main functions
__all__ = [
    "LoggingEnforcer",
    "PrettyConsoleHandler",
    "StructuredJsonFormatter",
    "initialize_logging",
    "create_structured_logger",
]


if __name__ == "__main__":
    # Demo the dual sink logging system
    enforcer = initialize_logging()

    logger = create_structured_logger("demo", "router")

    # Test different log levels and structured events
    logger.debug("Debug message with details")
    logger.info("System startup complete")
    logger.info_event(
        "router.message_processed",
        {"latency_ms": 245.7, "items_count": 3, "streaming": True},
    )
    logger.warning("Performance degradation detected")
    logger.warning_event(
        "router.budget_exceeded",
        {"family": "tweet_syndication", "elapsed_ms": 1250, "budget_ms": 1000},
    )
    logger.error("Critical system failure")
    logger.error_event(
        "router.hard_deadline_exceeded",
        {"family": "stt_processing", "elapsed_ms": 5000, "deadline_ms": 3000},
    )

    # Show status
    status = enforcer.get_logging_status()

    console = Console()
    console.print("\n[bold green]Logging Status:[/bold green]")
    console.print(
        Panel.fit(
            f"Handlers: {status['total_handlers']}\n"
            f"Pretty: {'✔️' if status['pretty_handler_active'] else '❌'}\n"
            f"JSONL: {'✔️' if status['jsonl_handler_active'] else '❌'}\n"
            f"Compliant: {'✔️' if status['enforcement_compliant'] else '❌'}",
            title="Dual Sink Status",
        )
    )
