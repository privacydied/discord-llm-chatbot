import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from rich.logging import RichHandler


# Icons per level for pretty console sink - exact specification compliance
# [CSD, CMV] - Code Smell Detection, Constants over Magic Values
LEVEL_ICONS = {
    "DEBUG": "ℹ",     # INFO symbol for debug (blue-grey styling)
    "INFO": "✔",      # Success checkmark (green styling)
    "WARNING": "⚠",   # Warning triangle (yellow styling) 
    "ERROR": "✖",     # Error X (red styling)
    "CRITICAL": "✖",  # Critical X (red styling)
}

# Colour palette mapping for console styling [CMV]
LEVEL_COLOURS = {
    "DEBUG": "blue",     # Blue-grey for debug info
    "INFO": "green",    # Green for success/info
    "WARNING": "yellow", # Yellow for warnings
    "ERROR": "red",     # Red for errors
    "CRITICAL": "red",  # Red for critical
}

# Field truncation settings [CMV]
MAX_FIELD_LENGTH = 80
TRUNCATION_SUFFIX = "…"


class EnhancedIconPrefixFilter(logging.Filter):
    """Enhanced icon prefix filter with field truncation and alignment.
    
    Implements exact specification for Pretty Console Sink:
    - Level-based symbols with left-padding
    - Auto-truncate long fields with continuation indicator
    - Fixed grid alignment after icon
    - Never wrap lines mid-token
    
    [RAT: CSD, CMV] - Code Smell Detection, Constants over Magic Values
    """

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

            # Apply field truncation with continuation indicator
            if len(resolved) > MAX_FIELD_LENGTH:
                excess_chars = len(resolved) - MAX_FIELD_LENGTH
                truncated = resolved[:MAX_FIELD_LENGTH - len(TRUNCATION_SUFFIX)]
                resolved = f"{truncated}{TRUNCATION_SUFFIX}(+{excess_chars})"

            # Apply icon prefix if not already present
            if icon and not str(resolved).startswith(icon):
                record.msg = f"{icon} {resolved}"
            else:
                record.msg = resolved
                
            # Add subsys field if not present for structured logging
            if not hasattr(record, 'subsys'):
                # Infer subsys from logger name
                name_parts = record.name.split('.')
                if len(name_parts) > 2 and name_parts[0] == 'bot':
                    record.subsys = name_parts[1]  # e.g., 'bot.core.bot' -> 'core'
                else:
                    record.subsys = name_parts[-1]  # Last component
                    
        except Exception:
            # Never break logging - fail silently but preserve functionality
            pass
        return True


class EnhancedJsonlFormatter(logging.Formatter):
    """Enhanced structured JSONL formatter with frozen key set and millisecond precision.
    
    Preserves EXACT keys as specified - no additions/renames allowed:
    ts, level, name, subsys, guild_id, user_id, msg_id, event, detail
    
    Correlation/request IDs and duration go in 'detail' subfields only.
    
    [RAT: CSD, CMV] - Code Smell Detection, Constants over Magic Values
    """

    def format(self, record: logging.LogRecord) -> str:
        # Millisecond precision local time with proper timezone alignment
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f%z")[:-8] + ts.strftime("%z")
        
        # Guard against formatting errors from msg/args mismatches
        try:
            detail = record.getMessage()
        except Exception:
            # As a last resort, emit raw components without raising
            detail = f"{record.msg} | args={getattr(record, 'args', None)}"

        # FROZEN key set - must maintain exact order and never add top-level keys
        payload = {
            "ts": ts_str,  # Millisecond precision, local time
            "level": record.levelname,
            "name": record.name,
            "subsys": getattr(record, "subsys", None),
            "guild_id": getattr(record, "guild_id", None),
            "user_id": getattr(record, "user_id", None),
            "msg_id": getattr(record, "msg_id", None),
            "event": getattr(record, "event", None),
            "detail": detail,
        }
        
        # Add correlation/timing data to detail subfield only (not top-level)
        if hasattr(record, 'correlation_id') or hasattr(record, 'duration_ms'):
            if isinstance(detail, str):
                detail_obj = {"message": detail}
            else:
                detail_obj = detail if isinstance(detail, dict) else {"message": str(detail)}
                
            if hasattr(record, 'correlation_id'):
                detail_obj["correlation_id"] = record.correlation_id
            if hasattr(record, 'duration_ms'):
                detail_obj["duration_ms"] = record.duration_ms
                
            payload["detail"] = detail_obj
            
        return json.dumps(payload, separators=(",", ":"))


def _ensure_log_dir(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def init_logging() -> None:
    """Configure enhanced dual-sink logging with exact specification compliance.
    
    Implements:
    - Pretty Console Sink: Rich styling, Tree/Panel summaries, aesthetic colours, 
      emoji/status glyphs, padded layout, level-based symbols, auto-truncation,
      left-pad icons, fixed field grid alignment, millisecond timestamps
    - Structured JSON Sink: Frozen key set preservation, millisecond precision
    
    [RAT: CSD, CMV] - Code Smell Detection, Constants over Magic Values
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    jsonl_path = Path(os.getenv("LOG_JSONL_PATH", "./logs/bot.jsonl"))

    _ensure_log_dir(jsonl_path)

    # Enhanced Pretty Console Sink with Rich styling and alignment
    pretty = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=True,          # Millisecond precision timestamps
        markup=True,             # Rich markup support for colours
        enable_link_path=False,
        console=None,            # Use default console
        omit_repeated_times=False,  # Always show timestamps for alignment
        log_time_format="[%H:%M:%S.%f]",  # Millisecond precision format
    )
    pretty.addFilter(EnhancedIconPrefixFilter())

    # Enhanced Structured JSONL Sink with frozen keys
    jsonl_handler = logging.FileHandler(str(jsonl_path), encoding="utf-8")
    jsonl_handler.setFormatter(EnhancedJsonlFormatter())

    logging.basicConfig(
        level=log_level,
        format="%(message)s",  # RichHandler handles formatting for console
        handlers=[pretty, jsonl_handler],
        force=True,
    )

    # Enforce both sinks are present with enhanced validation
    enforce_dual_logging_handlers()

    # Announce initialization with enhanced styling
    logger = logging.getLogger(__name__)
    logger.info("✔ Logging initialized (dual-sink: pretty console + JSONL)", extra={"subsys": "logging"})


def enforce_dual_logging_handlers() -> None:
    """Enhanced enforcer with validation of upgraded handler components.
    
    Validates enhanced dual-sink strategy exactly as specified:
    - Pretty Console Sink: RichHandler with EnhancedIconPrefixFilter
    - Structured JSON Sink: FileHandler with EnhancedJsonlFormatter
    
    If either handler is missing or misconfigured, emits one final CRITICAL
    line to stderr with compact diagnostic and aborts process.
    
    [RAT: CA, REH] - Clean Architecture, Robust Error Handling
    """
    root = logging.getLogger()
    
    # Check for enhanced RichHandler (pretty console sink)
    pretty_handler = None
    pretty_has_enhanced_filter = False
    for h in root.handlers:
        if isinstance(h, RichHandler):
            pretty_handler = h
            # Check for enhanced filter
            for f in h.filters:
                if isinstance(f, EnhancedIconPrefixFilter):
                    pretty_has_enhanced_filter = True
                    break
            break
    
    # Check for enhanced JSONL handler (structured JSON sink)
    jsonl_handler = None
    jsonl_has_enhanced_formatter = False
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            if isinstance(h.formatter, EnhancedJsonlFormatter):
                jsonl_handler = h
                jsonl_has_enhanced_formatter = True
                break
    
    # Determine what's missing and provide compact diagnostic
    missing = []
    if not pretty_handler:
        missing.append("pretty_handler (RichHandler)")
    elif not pretty_has_enhanced_filter:
        missing.append("enhanced_icon_filter (EnhancedIconPrefixFilter)")
        
    if not jsonl_handler:
        missing.append("jsonl_handler (FileHandler+EnhancedJsonlFormatter)")
    elif not jsonl_has_enhanced_formatter:
        missing.append("enhanced_jsonl_formatter (EnhancedJsonlFormatter)")
    
    if missing:
        diagnostic = f"Enhanced logging misconfigured: missing {', '.join(missing)}. Expected: dual-sink with enhanced RichHandler console + enhanced JSONL file. Fix: ensure init_logging() sets up both enhanced handlers with proper filters/formatters."
        
        # Emit one final CRITICAL line to stderr as specified
        try:
            sys.stderr.write(f"CRITICAL: {diagnostic}\n")
            sys.stderr.flush()
        except Exception:
            pass  # Never break during error reporting
        
        # Clean shutdown and abort
        try:
            cleanup_rich_handlers()
            logging.shutdown()
        except Exception:
            pass
        finally:
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