import contextlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from rich.logging import RichHandler


def _rich_tracebacks_supported() -> bool:
    try:
        return True
    except Exception:
        return False


class LevelIconFilter(logging.Filter):
    """Adds a level icon to each record for console output."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.CRITICAL or record.levelno >= logging.ERROR:
            record.level_icon = "✖"
        elif record.levelno >= logging.WARNING:
            record.level_icon = "⚠"
        elif record.levelno >= logging.INFO:
            record.level_icon = "✔"
        else:
            record.level_icon = "ℹ"
        return True


class JsonlFormatter(logging.Formatter):
    """Structured JSONL formatter with a frozen key set."""

    KEYS = (
        "ts",
        "level",
        "name",
        "subsys",
        "guild_id",
        "user_id",
        "msg_id",
        "event",
        "detail",
    )

    def format(self, record: logging.LogRecord) -> str:
        # Local time with millisecond precision
        ts = datetime.fromtimestamp(record.created, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Detail prefers explicit record.detail; otherwise message
        try:
            message = record.getMessage()
        except Exception:
            message = str(getattr(record, "msg", ""))

        detail: Any = getattr(record, "detail", None)
        if detail is None:
            detail = message

        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "name": record.name,
            "subsys": getattr(record, "subsys", None),
            "guild_id": getattr(record, "guild_id", None),
            "user_id": getattr(record, "user_id", None),
            "msg_id": getattr(record, "msg_id", None),
            "event": getattr(record, "event", None),
            "detail": detail,
        }

        # Drop None keys; preserve order of KEYS
        obj = {k: payload[k] for k in self.KEYS if payload.get(k) is not None}
        return json.dumps(obj, ensure_ascii=False)


def _ensure_dir(p: Path) -> None:
    with contextlib.suppress(Exception):
        p.parent.mkdir(parents=True, exist_ok=True)


def init_logging() -> None:
    """Configure dual-sink logging: Rich console + JSONL file (enforced)."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    jsonl_path = Path(os.getenv("LOG_JSONL_PATH", "logs/bot.jsonl"))
    _ensure_dir(jsonl_path)

    root = logging.getLogger()
    root.setLevel(level)
    # Clear to avoid duplicates on reload
    if root.hasHandlers():
        root.handlers.clear()

    # Pretty console sink
    rich_tracebacks = _rich_tracebacks_supported()
    pretty = RichHandler(
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=rich_tracebacks,
        show_path=False,
        show_time=True,
        enable_link_path=False,
        log_time_format="%Y-%m-%d %H:%M:%S.%f",
    )
    pretty.set_name("pretty_handler")
    pretty.addFilter(LevelIconFilter())
    pretty.setFormatter(logging.Formatter(fmt="%(level_icon)s %(message)s"))

    # JSONL sink
    jsonl = logging.FileHandler(str(jsonl_path), encoding="utf-8")
    jsonl.set_name("jsonl_handler")
    jsonl.setFormatter(JsonlFormatter())

    # Attach sensitive data scrubber to handlers
    pretty.addFilter(SensitiveDataFilter())
    jsonl.addFilter(SensitiveDataFilter())

    logging.basicConfig(handlers=[pretty, jsonl], level=level, force=True, format="%(message)s")

    # Enforce exactly the two sinks are present
    names = sorted(h.get_name() for h in logging.getLogger().handlers)
    if names != ["jsonl_handler", "pretty_handler"]:
        try:
            sys.stderr.write(f"[logging-enforcer] expected pretty_handler + jsonl_handler, got {names}\n")
            sys.stderr.flush()
        finally:
            logging.shutdown()
            sys.exit(2)

    # Tame noisy third-party libraries unless explicitly overridden [REH]
    try:
        third_party_level = os.getenv("THIRD_PARTY_LOG_LEVEL", "WARNING").upper()
        for name in ("openai", "httpx", "aiohttp", "urllib3"):
            logging.getLogger(name).setLevel(third_party_level)
    except Exception as exc:
        logging.getLogger(__name__).debug(f"third-party log level set failed: {exc}")

    logging.getLogger(__name__).info("✔ Logging initialized (dual-sink)", extra={"subsys": "logging"})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def shutdown_logging_and_exit(exit_code: int) -> NoReturn:
    try:
        logging.getLogger(__name__).info("Shutting down", extra={"subsys": "logging"})
    except Exception as exc:
        logging.getLogger(__name__).debug(f"shutdown log failed: {exc}")
    finally:
        try:
            cleanup_rich_handlers()
            logging.shutdown()
        finally:
            sys.exit(exit_code)


def cleanup_rich_handlers() -> None:
    try:
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, RichHandler):
                try:
                    h.rich_tracebacks = False
                    h.close()
                except Exception as exc:
                    logging.getLogger(__name__).debug(f"RichHandler cleanup failed: {exc}")
    except Exception as exc:
        logging.getLogger(__name__).debug(f"RichHandler cleanup outer failed: {exc}")


class SensitiveDataFilter(logging.Filter):
    """Filter that scrubs sensitive values in structured extras before emission. [SFT]."""

    SECRET_KEYS = {
        "OPENAI_API_KEY",
        "X_API_BEARER_TOKEN",
        "DISCORD_TOKEN",
        "VISION_API_KEY",
        "SCREENSHOT_API_KEY",
        "SEARCH_API_KEY",
        "YOUTUBE_API_KEY",
        "WHISPER_API_KEY",
        "DDG_API_KEY",
        "CUSTOM_SEARCH_API_KEY",
        "AUTHORIZATION",
        "authorization",
        "api_key",
        "token",
        "bearer",
        "secret",
        "password",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            for _key, value in list(record.__dict__.items()):
                if isinstance(value, dict):
                    self._scrub_dict_inplace(value)
                # Also scrub 'detail' if it's a dict-like stored as attribute
                if hasattr(record, "detail") and isinstance(record.detail, dict):
                    self._scrub_dict_inplace(record.detail)
        except Exception:
            # Never block logging on scrubber errors
            return True
        return True

    def _scrub_dict_inplace(self, obj: dict[str, Any]) -> None:
        for k in list(obj.keys()):
            v = obj[k]
            if isinstance(v, dict):
                self._scrub_dict_inplace(v)
            elif isinstance(v, str):
                if k in self.SECRET_KEYS:
                    obj[k] = "[REDACTED]"
                else:
                    # Also redact known secret values embedded in any string field [S7][SFT]
                    obj[k] = redact_sensitive_values(v)


def redact_sensitive_values(text: str) -> str:
    """Redact known sensitive env-var values from arbitrary text.

    Scans for common secret env-var names and replaces their values
    with '[REDACTED]' so log output or user-facing messages never
    leak credentials. [SFT]
    """
    secret_env_keys = [
        "OPENAI_API_KEY",
        "DISCORD_TOKEN",
        "X_API_BEARER_TOKEN",
        "VISION_API_KEY",
        "SCREENSHOT_API_KEY",
        "SEARCH_API_KEY",
        "YOUTUBE_API_KEY",
        "WHISPER_API_KEY",
        "DDG_API_KEY",
        "CUSTOM_SEARCH_API_KEY",
    ]
    for env_key in secret_env_keys:
        val = os.getenv(env_key, "")
        if val and len(val) >= 4:
            # Replace all occurrences of the full value [SFT]
            text = text.replace(val, "[REDACTED]")
    return text
