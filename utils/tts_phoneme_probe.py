#!/usr/bin/env python3
"""
Phoneme probe utility for diagnosing TTS phonemizer behavior on specific inputs.

Usage:
  uv run python utils/tts_phoneme_probe.py "cavalli"
  uv run python utils/tts_phoneme_probe.py "pyrex stirs turned to cavalli furs"

This probes the following, if available:
- Misaki G2P (English)
- phonemizer (espeak backend)
- g2p_en
- espeak / espeak-ng CLI with IPA output

Outputs structured, concise results with RichHandler logging and JSONL sink per project logging rules.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from typing import Optional

# Project logging (dual sink with RichHandler + JSONL)
try:
    from bot.utils.logging import init_logging, enforce_dual_logging_handlers

    init_logging()
    enforce_dual_logging_handlers()
except Exception:
    # Fallback to basic logging if project logging is unavailable
    logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def _safe_preview(s: Optional[str], limit: int = 160) -> str:
    try:
        if s is None:
            return "<None>"
        return s[:limit] + ("…(+%d)" % (len(s) - limit) if len(s) > limit else "")
    except Exception:
        return "<unpreviewable>"


def probe_misaki(text: str) -> Optional[str]:
    try:
        from misaki import en as misaki_en  # type: ignore

        try:
            from misaki import espeak as misaki_espeak  # type: ignore

            fallback = misaki_espeak.EspeakFallback(british=False)
            logger.info("Misaki: using espeak fallback")
        except Exception:
            fallback = None
            logger.info("Misaki: espeak fallback not available")
        g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback)
        out = g2p(text)
        if isinstance(out, tuple) and len(out) >= 1:
            out = out[0]
        return str(out)
    except Exception as e:
        logger.warning("Misaki probe failed: %s", e, exc_info=True)
        return None


def probe_phonemizer(text: str) -> Optional[str]:
    try:
        from phonemizer import phonemize  # type: ignore

        # Use espeak backend for English; fall back gracefully
        ph = phonemize(text, language="en-us", backend="espeak", strip=True, njobs=1)
        return str(ph)
    except Exception as e:
        logger.warning("phonemizer probe failed: %s", e, exc_info=True)
        return None


def probe_g2p_en(text: str) -> Optional[str]:
    try:
        from g2p_en import G2p  # type: ignore

        g2p = G2p()
        toks = g2p(text)
        if isinstance(toks, (list, tuple)):
            return " ".join(map(str, toks))
        return str(toks)
    except Exception as e:
        logger.warning("g2p_en probe failed: %s", e, exc_info=True)
        return None


def _run_cli(cmd: list[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.decode("utf-8", errors="replace").strip()
        return None
    except Exception:
        return None


def probe_espeak_cli(text: str) -> Optional[str]:
    exe = shutil.which("espeak")
    if not exe:
        return None
    # Try IPA first, then phoneme trace
    for args in (["--ipa=3", "-v", "en"], ["-x", "-v", "en"]):
        out = _run_cli([exe, "-q", *args, text])
        if out:
            return out
    return None


def probe_espeak_ng_cli(text: str) -> Optional[str]:
    exe = shutil.which("espeak-ng")
    if not exe:
        return None
    for args in (["--ipa=3", "-v", "en"], ["-x", "-v", "en"]):
        out = _run_cli([exe, "-q", *args, text])
        if out:
            return out
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Probe phonemizers for problematic words/phrases"
    )
    ap.add_argument("text", nargs="?", default="pyrex stirs turned to cavalli furs")
    args = ap.parse_args(argv)

    text = args.text
    logger.info("Phoneme probe start | text=%r", text)

    # Run probes
    results = []

    misaki = probe_misaki(text)
    results.append(("misaki", misaki))

    ph = probe_phonemizer(text)
    results.append(("phonemizer", ph))

    g2p = probe_g2p_en(text)
    results.append(("g2p_en", g2p))

    esp = probe_espeak_cli(text)
    results.append(("espeak_cli", esp))

    espng = probe_espeak_ng_cli(text)
    results.append(("espeak_ng_cli", espng))

    # Report
    for name, val in results:
        empty = (val is None) or (isinstance(val, str) and not val.strip())
        logger.info("[%s] empty=%s | preview=%r", name, empty, _safe_preview(val))

    # Explicit focus on the token 'cavalli'
    target = "cavalli"
    for name, val in results:
        contains = None
        try:
            contains = (val is not None) and ("cavalli" in str(val).lower())
        except Exception:
            contains = False
        logger.info("[%s] contains '%s' token? %s", name, target, contains)

    logger.info("Phoneme probe complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
