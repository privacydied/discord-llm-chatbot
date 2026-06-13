"""Search types and constants.
[CA][CMV][IV].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SafeSearch(StrEnum):
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class SearchCategory(StrEnum):
    """Supported search verticals. Additive and non-breaking.
    [CA][CMV].
    """

    TEXT = "text"  # general web
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"


@dataclass(frozen=True)
class SearchQueryParams:
    query: str
    max_results: int = 5
    safesearch: SafeSearch = SafeSearch.MODERATE
    locale: str | None = None
    timeout_ms: int = 5000
    # Optional category (vertical). Defaults to TEXT if not provided. [CMV]
    category: SearchCategory | None = None


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str | None = None
    favicon: str | None = None


SearchResults = list[SearchResult]
