"""RSS/Atom feed parsing with a hardened XML parser.
[CA][REH][SFT][CMV].

Feed bytes are untrusted input from an arbitrary host, so parsing uses lxml
with entity resolution and network access disabled -- this blocks entity
expansion ("billion laughs") and external-entity (XXE) attacks. lxml is
already a transitive dependency via trafilatura, so no new package is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from lxml import etree

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# XML namespaces used by RSS extensions and Atom. [CMV]
NS_CONTENT = "http://purl.org/rss/1.0/modules/content/"
NS_DC = "http://purl.org/dc/elements/1.1/"
NS_ATOM = "http://www.w3.org/2005/Atom"

# Path suffixes stripped before comparing two URLs for equality. [CMV]
_EQUIV_SUFFIXES = ("/amp", "/index.html", "/")


def _hardened_parser() -> etree.XMLParser:
    """XML parser with entity resolution and network fetches disabled. [SFT]"""
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        huge_tree=False,
        recover=True,
        load_dtd=False,
    )


def normalize_url(url: str) -> str:
    """Reduce a URL to a comparable form (host + path, no scheme/query/amp)."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return (url or "").strip().lower()
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("amp.")
    path = parsed.path or ""
    for suffix in _EQUIV_SUFFIXES:
        if path != "/" and path.endswith(suffix):
            path = path[: -len(suffix)]
    return f"{host}{path.rstrip('/')}"


def html_to_text(markup: str) -> str:
    """Flatten feed HTML into plain text."""
    if not markup:
        return ""
    try:
        soup = BeautifulSoup(markup, "html.parser")
    except Exception as exc:  # noqa: BLE001 - third-party parser on untrusted markup; logged, markup fallback [REH]
        logger.debug("feed html_to_text failed: %s", exc)
        return markup.strip()
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


@dataclass(frozen=True)
class FeedEntry:
    """One item/entry from a syndication feed."""

    link: str
    title: str | None = None
    body: str = ""
    author: str | None = None
    published: str | None = None


def _text(node: object | None) -> str:
    value = getattr(node, "text", None)
    return (value or "").strip() if node is not None else ""


def _first_nonempty(*values: str) -> str:
    for value in values:
        if value:
            return value
    return ""


def _parse_rss_item(item: etree._Element) -> FeedEntry | None:
    link = _first_nonempty(_text(item.find("link")), _text(item.find("guid")))
    if not link:
        return None
    body_markup = _first_nonempty(
        _text(item.find(f"{{{NS_CONTENT}}}encoded")),
        _text(item.find("description")),
    )
    return FeedEntry(
        link=link,
        title=_text(item.find("title")) or None,
        body=html_to_text(body_markup),
        author=_first_nonempty(_text(item.find(f"{{{NS_DC}}}creator")), _text(item.find("author"))) or None,
        published=_text(item.find("pubDate")) or None,
    )


def _atom_link(entry: etree._Element) -> str:
    for link in entry.findall(f"{{{NS_ATOM}}}link"):
        rel = link.get("rel") or "alternate"
        if rel == "alternate" and link.get("href"):
            return link.get("href", "").strip()
    return ""


def _parse_atom_entry(entry: etree._Element) -> FeedEntry | None:
    link = _atom_link(entry)
    if not link:
        return None
    body_markup = _first_nonempty(
        _text(entry.find(f"{{{NS_ATOM}}}content")),
        _text(entry.find(f"{{{NS_ATOM}}}summary")),
    )
    author_node = entry.find(f"{{{NS_ATOM}}}author")
    author = _text(author_node.find(f"{{{NS_ATOM}}}name")) if author_node is not None else ""
    return FeedEntry(
        link=link,
        title=_text(entry.find(f"{{{NS_ATOM}}}title")) or None,
        body=html_to_text(body_markup),
        author=author or None,
        published=_text(entry.find(f"{{{NS_ATOM}}}updated")) or None,
    )


def parse_feed(raw: bytes) -> list[FeedEntry]:
    """Parse RSS or Atom bytes into entries. Returns [] on any failure. [REH]"""
    if not raw:
        return []
    try:
        # Parser disables entity resolution, DTD loading and network access. [SFT]
        root = etree.fromstring(raw, parser=_hardened_parser())
    except Exception as exc:  # noqa: BLE001 - malformed feed is not an error worth raising; logged, [] fallback [REH]
        logger.debug("feed parse failed: %s", exc)
        return []
    if root is None:
        return []

    entries: list[FeedEntry] = []
    for item in root.iter("item"):
        parsed = _parse_rss_item(item)
        if parsed:
            entries.append(parsed)
    for entry in root.iter(f"{{{NS_ATOM}}}entry"):
        parsed = _parse_atom_entry(entry)
        if parsed:
            entries.append(parsed)
    return entries


def find_entry(entries: list[FeedEntry], url: str) -> FeedEntry | None:
    """Return the entry whose link matches ``url``, ignoring cosmetic differences."""
    target = normalize_url(url)
    if not target:
        return None
    for entry in entries:
        if normalize_url(entry.link) == target:
            return entry
    return None
