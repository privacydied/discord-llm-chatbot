"""Feed parsing and message formatting for RSS/Atom -> Discord posting. [CA][IV]

Pure functions (no network, no Discord) so the poller stays thin and testable.
Replaces the operator's Zapier "RSS -> Discord channel message" zaps.
"""

from __future__ import annotations

import hashlib
import re
# Remote feed XML: expat never resolves external entities; the poller caps the size.
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass
from typing import Any

# Zapier zap template, verbatim: **title**—pubDate, description in a code block, audio URL.
DEFAULT_TEMPLATE = "**{title}**\u2014{pubDate}\n\n```{description}```\n{enclosure_url}"
DISCORD_MESSAGE_LIMIT = 2000
TRIGGER_NEW = "new"  # Zapier "smart": post items with an unseen id
TRIGGER_CHANGED = "changed"  # Zapier "everything": also repost when an item's text changes
TRIGGERS = (TRIGGER_NEW, TRIGGER_CHANGED)
_ATOM = "{http://www.w3.org/2005/Atom}"
_TEMPLATE_FIELDS = ("title", "pubDate", "description", "enclosure_url", "link", "author", "feed_title")


@dataclass(frozen=True)
class FeedItem:
    title: str
    pub_date: str  # raw feed string (RFC 822 / ISO 8601), as Zapier's {{raw__pubDate}}
    description: str  # HTML stripped to text
    enclosure_url: str
    link: str
    guid: str
    author: str = ""

    @property
    def identity(self) -> str:
        """Stable id: guid, else link, else enclosure, else title."""
        return self.guid or self.link or self.enclosure_url or self.title


class FeedParseError(ValueError):
    """The response is not an RSS 2.0 or Atom document."""


def _text(element: ET.Element | None, path: str) -> str:
    if element is None:
        return ""
    return (element.findtext(path) or "").strip()


def html_to_text(raw: str) -> str:
    """Feed description HTML -> readable plain text with paragraph breaks kept."""
    if not raw:
        return ""
    if "<" not in raw and "&" not in raw:
        return raw.strip()
    from bs4 import BeautifulSoup  # lazy: only needed for HTML descriptions

    soup = BeautifulSoup(raw, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4"]):
        block.insert_after("\n")
    text = soup.get_text("")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _rss_item(item: ET.Element) -> FeedItem:
    enclosure = item.find("enclosure")
    return FeedItem(
        title=_text(item, "title"),
        pub_date=_text(item, "pubDate"),
        description=html_to_text(item.findtext("description") or ""),
        enclosure_url=(enclosure.get("url") or "").strip() if enclosure is not None else "",
        link=_text(item, "link"),
        guid=_text(item, "guid"),
        author=_text(item, "author") or _text(item, "{http://purl.org/dc/elements/1.1/}creator"),
    )


def _atom_link(entry: ET.Element, rel: str) -> str:
    for link in entry.findall(f"{_ATOM}link"):
        if (link.get("rel") or "alternate") == rel:
            return (link.get("href") or "").strip()
    return ""


def _atom_entry(entry: ET.Element) -> FeedItem:
    summary = entry.findtext(f"{_ATOM}summary") or entry.findtext(f"{_ATOM}content") or ""
    return FeedItem(
        title=_text(entry, f"{_ATOM}title"),
        pub_date=_text(entry, f"{_ATOM}published") or _text(entry, f"{_ATOM}updated"),
        description=html_to_text(summary),
        enclosure_url=_atom_link(entry, "enclosure"),
        link=_atom_link(entry, "alternate"),
        guid=_text(entry, f"{_ATOM}id"),
        author=_text(entry, f"{_ATOM}author/{_ATOM}name"),
    )


def parse_feed(data: bytes) -> tuple[str, list[FeedItem]]:
    """(feed title, items newest-first as published) for RSS 2.0 or Atom."""
    try:
        root = ET.fromstring(data)  # nosec B314
    except ET.ParseError as exc:
        raise FeedParseError(f"not XML: {exc}") from exc
    if root.tag == f"{_ATOM}feed":
        return _text(root, f"{_ATOM}title"), [_atom_entry(e) for e in root.findall(f"{_ATOM}entry")]
    channel = root.find("channel")
    if root.tag != "rss" or channel is None:
        raise FeedParseError(f"unsupported feed root <{root.tag}>")
    return _text(channel, "title"), [_rss_item(i) for i in channel.findall("item")]


def item_key(item: FeedItem, trigger: str = TRIGGER_NEW) -> str:
    """De-duplication key. ``changed`` also keys on the text, so edits repost."""
    basis = item.identity
    if trigger == TRIGGER_CHANGED:
        basis = "\x1f".join((basis, item.title, item.description, item.enclosure_url))
    return hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:24]


def _defuse(text: str) -> str:
    """Feed text inside our template: no breaking out of the code block, no pings."""
    text = text.replace("```", "\u02cb\u02cb\u02cb")  # noqa: PLR6201 - modifier grave accents look the same
    return text.replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")


def render(template: str, item: FeedItem, feed_title: str = "", limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """Fill the template; shorten the description so the post fits one message."""
    values: dict[str, Any] = {
        "title": _defuse(item.title),
        "pubDate": item.pub_date,
        "description": _defuse(item.description),
        "enclosure_url": item.enclosure_url,
        "link": item.link,
        "author": _defuse(item.author),
        "feed_title": _defuse(feed_title),
    }
    text = template.format_map(_Blank(values))
    overflow = len(text) - limit
    if overflow > 0 and values["description"]:
        keep = max(0, len(values["description"]) - overflow - 1)
        values["description"] = values["description"][:keep].rstrip() + "\u2026"
        text = template.format_map(_Blank(values))
    return text.strip()


class _Blank(dict):
    """format_map dict: unknown placeholders render empty instead of raising."""

    def __missing__(self, key: str) -> str:
        return ""


def validate_template(template: str) -> list[str]:
    """Placeholder names in ``template`` that render() does not provide."""
    import string

    names = {name for _, name, _, _ in string.Formatter().parse(template) if name}
    return sorted(names - set(_TEMPLATE_FIELDS))


_ZAPIER_FIELD = re.compile(r"\{\{\s*\d+__(?:raw__)?([A-Za-z@_]+?)\s*\}\}")
_ZAPIER_TO_FIELD = {
    "title": "title",
    "pubDate": "pubDate",
    "description": "description",
    "enclosure__@url": "enclosure_url",
    "link": "link",
    "author": "author",
}


def zapier_template(content: str) -> str:
    """Zapier ``{{1__raw__title}}`` mappings -> this module's ``{title}`` placeholders.

    Unknown Zapier fields are dropped; literal braces are escaped for str.format.
    """
    parts: list[str] = []
    last = 0
    for match in _ZAPIER_FIELD.finditer(content):
        parts.append(content[last : match.start()].replace("{", "{{").replace("}", "}}"))  # noqa: PLR6201
        field = _ZAPIER_TO_FIELD.get(match.group(1))
        parts.append("{" + field + "}" if field else "")
        last = match.end()
    parts.append(content[last:].replace("{", "{{").replace("}", "}}"))  # noqa: PLR6201
    return "".join(parts)
