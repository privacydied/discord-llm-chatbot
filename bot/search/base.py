"""
Base interfaces and helpers for search providers.
[CA][CMV][IV][RM]
"""

from __future__ import annotations

import abc
from typing import Protocol

from .types import SearchQueryParams, SearchResults


class SearchProvider(Protocol):
    async def search(self, params: SearchQueryParams) -> SearchResults:  # noqa: D401
        """Execute a web search with given parameters and return normalized results."""
        ...


class AbstractSearchProvider(abc.ABC):
    @abc.abstractmethod
    async def search(
        self, params: SearchQueryParams
    ) -> SearchResults:  # pragma: no cover
        raise NotImplementedError
