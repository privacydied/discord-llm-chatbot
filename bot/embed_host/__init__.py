"""Bot-owned public embed pages: cards with the video playing inside, no proxy.

Discord only plays video inside a card when it unfurls a link whose page has
``og:video`` (a bot-built embed can never hold video). So instead of linking
oginstagram/vxtwitter, the bot downloads the post, stores it here and replies
with a link to its own page, which Discord unfurls.

Only pages the bot stored are served (unguessable tokens; nothing is fetched
on request), and the server is separate from the admin dashboard so exposing
it publicly exposes nothing else.
"""

from .server import EmbedHost, get_embed_host, start_embed_host, stop_embed_host
from .store import EmbedStore, StoredMedia, StoredPost

__all__ = [
    "EmbedHost",
    "EmbedStore",
    "StoredMedia",
    "StoredPost",
    "get_embed_host",
    "start_embed_host",
    "stop_embed_host",
]
