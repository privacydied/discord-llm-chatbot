#!/usr/bin/env python3
"""
Download Kokoro-ONNX English example assets.

Assets (from upstream english.py):
- https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
- https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

Usage:
    uv run python utils/download_kokoro_assets.py --out-dir runtime/kokoro

Notes:
- Logs to console (pretty Rich) and to logs/download_kokoro_assets.jsonl (structured JSONL).
- On success, prints lines you can paste into .env for TTS paths.

[IV][RM][CMV][PA][REH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.tree import Tree
except Exception:  # [REH]
    RichHandler = None  # type: ignore
    Panel = None  # type: ignore
    Tree = None  # type: ignore

# ------------------------------
# Constants [CMV]
# ------------------------------
DEFAULT_OUT_DIR = Path("tts")
JSONL_LOG_PATH = Path("logs/download_kokoro_assets.jsonl")
SUBSYS = "tts.kokoro.assets"

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)

MODEL_NAME = "kokoro-v1.0.onnx"
VOICES_NAME = "voices-v1.0.bin"

# ------------------------------
# Logging with Dual Sink [RAT]
# ------------------------------
class JsonLineFileHandler(logging.Handler):
    """Structured JSONL sink preserving keyset: ts, level, name, subsys, guild_id, user_id, msg_id, event, detail."""

    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.file_path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:  # [RM]
        try:
            payload = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
                "level": record.levelname,
                "name": record.name,
                "subsys": getattr(record, "subsys", SUBSYS),
                "guild_id": getattr(record, "guild_id", None),
                "user_id": getattr(record, "user_id", None),
                "msg_id": getattr(record, "msg_id", None),
                "event": getattr(record, "event", record.getMessage()),
                "detail": getattr(record, "detail", None),
            }
            self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._fp.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._fp.close()
        finally:
            super().close()


def setup_logger(verbosity: int = 0) -> logging.Logger:
    logger = logging.getLogger("utils.download_kokoro_assets")
    logger.setLevel(logging.DEBUG)

    # Pretty console sink
    if RichHandler is not None:
        console = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
            log_time_format="%Y-%m-%d %H:%M:%S.%f",
        )
        console.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        logger.addHandler(console)
    else:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG if verbosity > 0 else logging.INFO)
        logger.addHandler(sh)

    # JSONL sink
    jsonh = JsonLineFileHandler(JSONL_LOG_PATH)
    jsonh.setLevel(logging.DEBUG)
    logger.addHandler(jsonh)

    # Enforcer: exactly two active handlers (pretty + json) [Style Guide]
    handlers = [h for h in logger.handlers]
    if not any(isinstance(h, (RichHandler, logging.StreamHandler)) for h in handlers) or not any(
        isinstance(h, JsonLineFileHandler) for h in handlers
    ):
        logger.critical("Logging handlers misconfigured; expected pretty + jsonl.")
        sys.exit(2)

    return logger


# ------------------------------
# HTTP session with retries [PA]
# ------------------------------

def build_session(total_retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# ------------------------------
# Helpers
# ------------------------------

def atomic_write(src_fp: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Replace existing atomically
    tmp_final = dest.with_suffix(dest.suffix + ".tmp")
    if os.path.exists(tmp_final):
        os.remove(tmp_final)
    shutil.move(src_fp, tmp_final)
    if os.path.exists(dest):
        os.remove(dest)
    os.replace(tmp_final, dest)


def download_file(url: str, dest: Path, session: requests.Session, timeout: int, logger: logging.Logger) -> None:
    extra = {"subsys": SUBSYS, "event": "download_start", "detail": {"url": url, "dest": str(dest)}}
    logger.info("ℹ Starting download", extra=extra)

    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        hasher = hashlib.sha256()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                hasher.update(chunk)
                tmp.write(chunk)
            tmp_path = tmp.name

    atomic_write(tmp_path, dest)
    extra = {
        "subsys": SUBSYS,
        "event": "download_complete",
        "detail": {
            "url": url,
            "dest": str(dest.resolve()),
            "bytes": dest.stat().st_size,
            "sha256": hasher.hexdigest(),
        },
    }
    logger.info("✔ Downloaded", extra=extra)


# ------------------------------
# Main
# ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Kokoro-ONNX English example assets.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for assets.")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout per request (seconds).")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger(args.verbose)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Show plan [CDiP]
    if Tree and Panel:
        t = Tree("[bold green]Kokoro-ONNX asset download plan")
        t.add(f"Out dir: [cyan]{out_dir.resolve()}")
        t.add(f"Model URL: [white]{MODEL_URL}")
        t.add(f"Voices URL: [white]{VOICES_URL}")
        print(Panel(t, title="Downloader", border_style="green"))

    session = build_session()

    targets = [
        (MODEL_URL, out_dir / MODEL_NAME),
        (VOICES_URL, out_dir / VOICES_NAME),
    ]

    for url, dest in targets:
        if dest.exists() and not args.force:
            extra = {"subsys": SUBSYS, "event": "skip_exists", "detail": {"dest": str(dest)}}
            logger.info("ℹ Skipping existing", extra=extra)
            continue
        try:
            download_file(url, dest, session, args.timeout, logger)
        except requests.HTTPError as e:  # [REH]
            extra = {"subsys": SUBSYS, "event": "http_error", "detail": {"url": url, "status": getattr(e.response, 'status_code', None)}}
            logger.error("✖ HTTP error during download", extra=extra, exc_info=True)
            return 1
        except Exception:
            extra = {"subsys": SUBSYS, "event": "error", "detail": {"url": url}}
            logger.error("✖ Unexpected error during download", extra=extra, exc_info=True)
            return 1

    model_path = (out_dir / MODEL_NAME).resolve()
    voices_path = (out_dir / VOICES_NAME).resolve()

    # Env hints
    print()
    print("Add these to your .env for kokoro-onnx:")
    print(f"TTS_ENGINE=kokoro-onnx")
    print(f"TTS_MODEL_PATH={model_path}")
    print(f"TTS_VOICES_PATH={voices_path}")

    extra = {
        "subsys": SUBSYS,
        "event": "done",
        "detail": {"model_path": str(model_path), "voices_path": str(voices_path)},
    }
    logger.info("✔ Completed", extra=extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
