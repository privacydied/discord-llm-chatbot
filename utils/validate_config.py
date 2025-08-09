#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bot.config import load_config, audit_env_file, validate_required_env, validate_prompt_files

console = Console()


def _kv_table(title: str, data: dict[str, Any]) -> Table:
    table = Table(title=title, show_lines=False, expand=True)
    table.add_column("Key", style="bold cyan", overflow="fold")
    table.add_column("Value", style="green", overflow="fold")
    for k, v in data.items():
        # Mask secrets
        if any(s in k for s in ("KEY", "TOKEN", "SECRET")) and v:
            v_str = str(v)
            masked = v_str[:4] + "…" + v_str[-2:] if len(v_str) > 8 else "***"
            table.add_row(k, masked)
        else:
            table.add_row(k, str(v))
    return table


def main() -> int:
    console.print(Panel.fit("Config Validation", subtitle="utils/validate_config.py", border_style="blue"))

    # Audit .env
    try:
        audit_env_file()
    except Exception as e:
        console.print(Panel.fit(f".env audit warning: {e}", title="ENV AUDIT", style="yellow"))

    # Validate required env and prompt files
    try:
        validate_required_env()
        validate_prompt_files()
        console.print("✔ Required env and prompt files validated", style="bold green")
    except Exception as e:
        console.print(Panel.fit(str(e), title="Validation Error", style="red"))

    cfg = load_config()

    screenshot = {
        "SCREENSHOT_API_KEY": os.getenv("SCREENSHOT_API_KEY", ""),
        "SCREENSHOT_API_URL": cfg.get("SCREENSHOT_API_URL", os.getenv("SCREENSHOT_API_URL", "")),
        "SCREENSHOT_API_DEVICE": os.getenv("SCREENSHOT_API_DEVICE", ""),
        "SCREENSHOT_API_DIMENSION": os.getenv("SCREENSHOT_API_DIMENSION", ""),
        "SCREENSHOT_API_FORMAT": os.getenv("SCREENSHOT_API_FORMAT", ""),
        "SCREENSHOT_API_DELAY": os.getenv("SCREENSHOT_API_DELAY", ""),
        "SCREENSHOT_API_COOKIES": os.getenv("SCREENSHOT_API_COOKIES", ""),
    }

    search = {
        "SEARCH_PROVIDER": cfg.get("SEARCH_PROVIDER"),
        "SEARCH_MAX_RESULTS": cfg.get("SEARCH_MAX_RESULTS"),
        "SEARCH_SAFE": cfg.get("SEARCH_SAFE"),
        "SEARCH_LOCALE": cfg.get("SEARCH_LOCALE"),
        "DDG_API_ENDPOINT": cfg.get("DDG_API_ENDPOINT"),
        "DDG_API_KEY": cfg.get("DDG_API_KEY"),
        "DDG_TIMEOUT_MS": cfg.get("DDG_TIMEOUT_MS"),
        "CUSTOM_SEARCH_API_ENDPOINT": cfg.get("CUSTOM_SEARCH_API_ENDPOINT"),
        "CUSTOM_SEARCH_API_KEY": cfg.get("CUSTOM_SEARCH_API_KEY"),
        "CUSTOM_SEARCH_HEADERS": cfg.get("CUSTOM_SEARCH_HEADERS"),
        "CUSTOM_SEARCH_TIMEOUT_MS": cfg.get("CUSTOM_SEARCH_TIMEOUT_MS"),
        "CUSTOM_SEARCH_RESULT_PATHS": cfg.get("CUSTOM_SEARCH_RESULT_PATHS"),
        "SEARCH_POOL_MAX_CONNECTIONS": cfg.get("SEARCH_POOL_MAX_CONNECTIONS"),
        "SEARCH_BREAKER_FAILURE_WINDOW": cfg.get("SEARCH_BREAKER_FAILURE_WINDOW"),
        "SEARCH_BREAKER_OPEN_MS": cfg.get("SEARCH_BREAKER_OPEN_MS"),
        "SEARCH_BREAKER_HALFOPEN_PROB": cfg.get("SEARCH_BREAKER_HALFOPEN_PROB"),
        "SEARCH_INLINE_MAX_CONCURRENCY": os.getenv("SEARCH_INLINE_MAX_CONCURRENCY", "")
    }

    streaming = {
        "STREAMING_ENABLE": cfg.get("STREAMING_ENABLE"),
        "STREAMING_EMBED_STYLE": cfg.get("STREAMING_EMBED_STYLE"),
        "STREAMING_TICK_MS": cfg.get("STREAMING_TICK_MS"),
        "STREAMING_MAX_STEPS": cfg.get("STREAMING_MAX_STEPS"),
        "STREAMING_ENABLE_TEXT": cfg.get("STREAMING_ENABLE_TEXT"),
        "STREAMING_ENABLE_SEARCH": cfg.get("STREAMING_ENABLE_SEARCH"),
        "STREAMING_ENABLE_RAG": cfg.get("STREAMING_ENABLE_RAG"),
        "STREAMING_ENABLE_MEDIA": cfg.get("STREAMING_ENABLE_MEDIA"),
    }

    stt = {
        "STT_ENABLE": cfg.get("STT_ENABLE"),
        "STT_MODE": cfg.get("STT_MODE"),
        "STT_ACTIVE_PROVIDERS": ", ".join(cfg.get("STT_ACTIVE_PROVIDERS", []) or []),
        "STT_CONFIDENCE_MIN": cfg.get("STT_CONFIDENCE_MIN"),
        "STT_CACHE_TTL": cfg.get("STT_CACHE_TTL"),
        "STT_LOCAL_CONCURRENCY": cfg.get("STT_LOCAL_CONCURRENCY"),
    }

    console.print(_kv_table("Screenshot API", screenshot))
    console.print(_kv_table("Search", search))
    console.print(_kv_table("Streaming Status Cards", streaming))
    console.print(_kv_table("STT Orchestrator", stt))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
