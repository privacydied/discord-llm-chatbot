"""Licensed news-source resolution.
[CA].

Used when generic web extraction returns a subscription stub instead of an
article: retries the lookup against publisher-sanctioned channels (official
content APIs, publisher syndication feeds). Nothing here circumvents an
access control -- when no licensed source carries the article, the caller
reports that the content was unavailable.
"""

from __future__ import annotations

from .service import build_providers, resolve_article
from .thin_content import ThinVerdict, assess
from .types import NewsArticle

__all__ = [
    "NewsArticle",
    "ThinVerdict",
    "assess",
    "build_providers",
    "resolve_article",
]
