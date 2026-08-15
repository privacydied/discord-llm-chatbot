"""Provider interface for licensed news sources.
[CA][CMV].

Mirrors the ``bot.search.base`` provider shape so both abstractions read the
same way. A provider is responsible for one publisher-sanctioned channel
(an official content API, a syndication feed) and must never attempt to
defeat an access control.
"""

from __future__ import annotations

import abc
from typing import Protocol

from .types import NewsArticle


class NewsProvider(Protocol):
    """A source that can return an article body for a given article URL."""

    name: str

    def supports(self, url: str) -> bool:
        """Return True when this provider can service ``url``."""
        ...

    async def fetch(self, url: str) -> NewsArticle | None:
        """Return the article, or None when unavailable. Must not raise."""
        ...


class AbstractNewsProvider(abc.ABC):
    """Base class supplying the provider name and a default ``supports``."""

    name: str = "abstract"

    def supports(self, url: str) -> bool:  # pragma: no cover - trivial default
        return bool(url)

    @abc.abstractmethod
    async def fetch(self, url: str) -> NewsArticle | None:  # pragma: no cover
        raise NotImplementedError
