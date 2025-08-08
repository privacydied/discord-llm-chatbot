"""
Search types and constants.
[CA][CMV][IV]
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class SafeSearch(str, Enum):
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


@dataclass(frozen=True)
class SearchQueryParams:
    query: str
    max_results: int = 5
    safesearch: SafeSearch = SafeSearch.MODERATE
    locale: Optional[str] = None
    timeout_ms: int = 5000


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: Optional[str] = None
    favicon: Optional[str] = None


SearchResults = List[SearchResult]
