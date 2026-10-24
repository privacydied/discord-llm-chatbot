"""Auto-translate foreign-language tweets in embed fixes.

vxtwitter / fxtwitter translate a tweet when the link ends in a language code
(``.../status/123/en``). The bot looks up the tweet language on the public
FxTwitter API and appends the target code only when the tweet is in another
language. English tweets keep a plain link. A lookup failure keeps the plain
link too, so translation never blocks or breaks the fix.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from collections.abc import Mapping
from typing import Any

import aiohttp

from bot.utils.logging import get_logger

from .sites import EmbedFix

logger = get_logger(__name__)

TWEET_API = "https://api.fxtwitter.com/status/{tweet_id}"
TRANSLATE_TIMEOUT_S = 4.0
DEFAULT_TARGET_LANG = "en"
_USER_AGENT = "Mozilla/5.0 (compatible; discord-llm-chatbot embed-fix)"
_TARGET_RE = re.compile(r"^[a-z]{2}$")
# Only a plain status link qualifies: a link that already ends in /en, /ja or
# /original keeps the poster's choice.
_PLAIN_TWEET_RE = re.compile(r"^https://[^/]+/[A-Za-z0-9_]+(?:/web)?/status/(?P<id>\d+)$")
# Twitter's non-language codes: undetermined, no text, media/hashtag/mention/
# cashtag/emoji-only tweets. Translating these does nothing useful.
NON_LANGUAGE_CODES = frozenset({"und", "zxx", "qme", "qht", "qam", "qct", "qst", "art"})
# FxTwitter sometimes reports no language at all (seen on a mostly-Hindi tweet,
# 2026-10-24). Then the text decides: enough non-Latin letters = foreign. [REH]
NON_LATIN_SHARE = 0.3
MIN_LETTERS = 12
SCRIPT_GUESS = "xx"  # "some non-target language"; only used for the needs_translation check


def parse_target(raw: Any) -> str | None:
    """Return a valid 2-letter target code, or None to disable translation."""
    value = str(raw if raw is not None else DEFAULT_TARGET_LANG).strip().lower()
    return value if _TARGET_RE.match(value) else None


def tweet_id(fix: EmbedFix) -> str | None:
    """Tweet id of a plain (no language suffix) twitter fix, else None."""
    if fix.site != "twitter":
        return None
    match = _PLAIN_TWEET_RE.match(fix.rewritten)
    return match["id"] if match else None


def needs_translation(lang: Any, target: str) -> bool:
    """True when *lang* is a real language that differs from *target*."""
    if not isinstance(lang, str) or not lang:
        return False
    code = lang.strip().lower()
    return code not in NON_LANGUAGE_CODES and code.split("-")[0] != target


def guess_from_script(text: Any) -> str | None:
    """SCRIPT_GUESS when a real share of the letters are non-Latin, else None (unknown)."""
    letters = [ch for ch in str(text or "") if ch.isalpha()]
    if len(letters) < MIN_LETTERS:
        return None
    non_latin = sum(1 for ch in letters if ord(ch) > 0x24F)  # beyond Latin Extended-B
    return SCRIPT_GUESS if non_latin / len(letters) >= NON_LATIN_SHARE else None


async def fetch_lang(session: aiohttp.ClientSession, tid: str) -> str | None:
    """Return the tweet language code reported by the FxTwitter API (or a script guess)."""
    try:
        async with session.get(TWEET_API.format(tweet_id=tid)) as resp:
            if resp.status != 200:
                logger.debug(f"embed_fix.translate.api_status | tweet={tid} status={resp.status}")
                return None
            payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        logger.debug(f"embed_fix.translate.api_error | tweet={tid} error={exc}")
        return None
    tweet = payload.get("tweet") if isinstance(payload, Mapping) else None
    if not isinstance(tweet, Mapping):
        return None
    lang = tweet.get("lang")
    if isinstance(lang, str) and lang.strip() and lang.strip().lower() not in NON_LANGUAGE_CODES:
        return lang
    return guess_from_script(tweet.get("text")) or lang


async def apply(fixes: list[EmbedFix], target: str | None) -> list[EmbedFix]:
    """Append ``/<target>`` to plain tweet links whose language is not *target*."""
    candidates = {i: tid for i, fix in enumerate(fixes) if target and (tid := tweet_id(fix))}
    if not candidates:
        return fixes
    timeout = aiohttp.ClientTimeout(total=TRANSLATE_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": _USER_AGENT}) as session:
        langs = await asyncio.gather(*(fetch_lang(session, tid) for tid in candidates.values()))
    result = list(fixes)
    for index, lang in zip(candidates, langs, strict=True):
        if needs_translation(lang, target):
            result[index] = dataclasses.replace(fixes[index], rewritten=f"{fixes[index].rewritten}/{target}")
            logger.debug(f"embed_fix.translate | tweet={candidates[index]} lang={lang} target={target}")
    return result
