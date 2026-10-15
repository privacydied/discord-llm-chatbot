"""Site registry for link-embed fixing (Python port of SaucyBot's rewrite sites).

Each site owns one anchored regex and a builder that turns a match into an
embed-friendly URL on a proxy domain (x.com -> vxtwitter.com, ...). Everything
here is pure: no Discord objects, no I/O, so it is cheap to test.

Ported from Sn0wCrack/saucybot-discord (v2):
  Twitter (VxTwitter.cs), Instagram (VxInstagram.cs; target switched to oginstagram), FurAffinity
  (XFurAffinity.cs), Reddit media (Reddit.cs), Bluesky (bskx.app, the
  VixBluesky host SaucyBot's client calls).
SaucyBot fetches APIs for Pixiv and DeviantArt; here they use the public
rewrite proxies phixiv.net and fixdeviantart.com instead. TikTok and Reddit
posts are additions (not in SaucyBot) for in-client video playback. ppv.*
stream pages have no proxy, so they are marked ``card`` and get a bot-built
embed (see ppv.py).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from urllib.parse import unquote

# SaucyBot MessageValidator.IgnoreContentRegex: a message that wraps anything in
# <...> (suppressed link) or ||...|| (spoiler) opts out entirely. Mentions,
# channels and custom emoji (<@, <#, <:, <a:) do not count.
IGNORE_CONTENT_RE = re.compile(r"(<|\|\|)(?!@|#|:|a:).*(>|\|\|)", re.IGNORECASE)

# SaucyBot Constants.DefaultMaximumEmbeds
DEFAULT_MAX_LINKS = 8

_DOMAIN_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$")

Builder = Callable[[re.Match[str], str], "str | None"]


@dataclass(frozen=True, slots=True)
class EmbedSite:
    """One rewrite rule: identifier, URL pattern, proxy domain, builder."""

    identifier: str
    pattern: re.Pattern[str]
    default_domain: str
    build: Builder
    nsfw: bool = False
    # card=True: no proxy domain exists; the bot builds its own embed card for
    # the (canonicalised) original URL instead of posting a rewritten link.
    card: bool = False


@dataclass(frozen=True, slots=True)
class EmbedFix:
    """A single matched link and its rewrite."""

    site: str
    original: str
    rewritten: str
    nsfw: bool
    position: int
    card: bool = False


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def _build_twitter(m: re.Match[str], domain: str) -> str:
    url = f"https://{domain}/{m['user']}/status/{m['id']}"
    if m["translate"]:
        url += f"/{m['translate']}"
    return url


def _build_instagram(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/{m['kind'].lower()}/{m['code']}/"


def _build_bluesky(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/profile/{m['user']}/post/{m['id']}"


def _build_furaffinity(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/{m['kind'].lower()}/{m['id']}/"


def _build_reddit_media(m: re.Match[str], _domain: str) -> str | None:
    target = unquote(m["url"])
    # Only hand back real http(s) URLs; anything else is not a media link.
    if not re.match(r"^https?://[^\s/]+\.[^\s/]+", target, re.IGNORECASE):
        return None
    return target


def _build_pixiv(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/artworks/{m['id']}"


def _build_deviantart(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/{m['author']}/art/{m['slug']}"


def _build_tiktok(m: re.Match[str], domain: str) -> str:
    if m["short"]:
        return f"https://{m['short_host'].lower()}.{domain}/{m['short']}/"
    return f"https://{domain}/{m['path']}"


def _build_reddit(m: re.Match[str], domain: str) -> str:
    return f"https://{domain}/{m['path']}"


# ppv.st and the mirrors the site lists itself (Nuxt config apiDomains).
PPV_DOMAINS: tuple[str, ...] = ("ppv.st", "ppv.tj", "ppvs.pk", "ppv.rw", "ppv.ms", "ppv.bi", "ppv.ug", "ppv.to", "ppv.cx")
_PPV_HOSTS = "|".join(re.escape(domain) for domain in PPV_DOMAINS)
# Slugs are one or more path segments, e.g. "nfl-network" or
# "cfb/2026-10-10/tex-ou" (+ "/skycast" for an alternate source). No dots, so
# "../" can never reach the API path.
_PPV_SLUG = r"[A-Za-z0-9-]+(?:/[A-Za-z0-9-]+){0,3}"
_PPV_PAGE_RE = re.compile(rf"^https://(?:{_PPV_HOSTS})/live/(?P<slug>{_PPV_SLUG})$", re.IGNORECASE)


def _build_ppv(m: re.Match[str], _domain: str) -> str:
    # Canonical page URL on the host the user posted; the card links back to it.
    return f"https://{m['host'].lower()}/live/{m['slug']}"


def page_slug(url: str) -> str | None:
    """Slug of a canonical ppv page URL produced by the ppv site, else None."""
    match = _PPV_PAGE_RE.match(url or "")
    return match["slug"] if match else None


# Every pattern is anchored on the scheme and an exact host, so "fox.com",
# "dropbox.com" or an already-fixed "vxtwitter.com" never match.
SITES: tuple[EmbedSite, ...] = (
    EmbedSite(
        identifier="twitter",
        pattern=_compile(
            r"https?://(?:www\.|mobile\.)?(?:twitter|x|nitter)\.(?:com|net)/"
            r"(?P<user>[A-Za-z0-9_]+(?:/web)?)/status(?:es)?/(?P<id>\d+)"
            r"(?:/(?:video|photo)/\d)?"
            r"(?:/(?P<translate>original|[A-Za-z]{2}(?:[-_][A-Za-z]{2})?)(?![\w-]))?"
        ),
        default_domain="vxtwitter.com",
        build=_build_twitter,
    ),
    EmbedSite(
        identifier="instagram",
        pattern=_compile(r"https?://(?:www\.|m\.)?instagram\.com/(?P<kind>p|reels?|tv)/(?P<code>[A-Za-z0-9_-]+)"),
        # oginstagram serves a real og: page (video/photo, author, caption) that
        # Discord unfurls; kkinstagram only 302s to the file and no longer embeds.
        default_domain="www.oginstagram.com",
        build=_build_instagram,
    ),
    EmbedSite(
        identifier="bluesky",
        pattern=_compile(r"https?://(?:www\.)?bsky\.app/profile/(?P<user>[^\s/?#]+)/post/(?P<id>[A-Za-z0-9]+)"),
        default_domain="bskx.app",
        build=_build_bluesky,
    ),
    EmbedSite(
        identifier="furaffinity",
        pattern=_compile(r"https?://(?:www\.)?furaffinity\.net/(?P<kind>view|full)/(?P<id>\d+)"),
        default_domain="xfuraffinity.net",
        build=_build_furaffinity,
        nsfw=True,
    ),
    EmbedSite(
        identifier="reddit_media",
        pattern=_compile(r"https?://(?:www\.)?reddit\.com/media\?url=(?P<url>[A-Za-z0-9%._~-]+)"),
        default_domain="",
        build=_build_reddit_media,
        nsfw=True,
    ),
    EmbedSite(
        identifier="pixiv",
        pattern=_compile(r"https?://(?:www\.)?pixiv\.net/(?:[a-z]{2}/)?artworks/(?P<id>\d+)"),
        default_domain="phixiv.net",
        build=_build_pixiv,
    ),
    EmbedSite(
        identifier="deviantart",
        pattern=_compile(r"https?://(?:www\.)?deviantart\.com/(?P<author>[A-Za-z0-9_-]+)/art/(?P<slug>[A-Za-z0-9_-]+)"),
        default_domain="fixdeviantart.com",
        build=_build_deviantart,
    ),
    EmbedSite(
        identifier="tiktok",
        pattern=_compile(
            r"https?://(?:(?:www\.|m\.)?tiktok\.com/(?P<path>@[\w.-]+/video/\d+)"
            r"|(?P<short_host>vm|vt)\.tiktok\.com/(?P<short>[A-Za-z0-9]+))"
        ),
        default_domain="vxtiktok.com",
        build=_build_tiktok,
    ),
    EmbedSite(
        identifier="reddit",
        pattern=_compile(r"https?://(?:www\.|old\.|new\.)?reddit\.com/(?P<path>r/\w+/(?:comments|s)/[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?)"),
        # rxddit.com returns 502 for every post (checked with the Discordbot UA);
        # vxreddit serves full og: tags incl. og:video and resolves /s/ share links.
        default_domain="vxreddit.com",
        build=_build_reddit,
    ),
    EmbedSite(
        identifier="ppv",
        pattern=_compile(rf"https?://(?:www\.)?(?P<host>{_PPV_HOSTS})/live/(?P<slug>{_PPV_SLUG})"),
        default_domain="",
        build=_build_ppv,
        card=True,
    ),
)

SITE_IDS: frozenset[str] = frozenset(site.identifier for site in SITES)


def is_ignored(content: str) -> bool:
    """True when the message opted out via <link> or ||spoiler|| (SaucyBot parity)."""
    return bool(IGNORE_CONTENT_RE.search(content or ""))


def parse_site_list(raw: str | Iterable[str] | None) -> frozenset[str]:
    """Parse a comma-separated (or iterable) list of site identifiers."""
    if not raw:
        return frozenset()
    items = raw.split(",") if isinstance(raw, str) else raw
    return frozenset(item.strip().lower() for item in items if item and item.strip())


def parse_domain_overrides(raw: str | Mapping[str, str] | None) -> dict[str, str]:
    """Parse ``site=domain,site=domain``; drop unknown sites and invalid hosts."""
    if not raw:
        return {}
    pairs = raw.items() if isinstance(raw, Mapping) else (part.split("=", 1) for part in raw.split(",") if "=" in part)
    overrides: dict[str, str] = {}
    for site, domain in pairs:
        key, host = site.strip().lower(), domain.strip().lower()
        if key in SITE_IDS and _DOMAIN_RE.match(host):
            overrides[key] = host
    return overrides


def _site_fixes(site: EmbedSite, content: str, domain: str) -> Iterable[EmbedFix]:
    for match in site.pattern.finditer(content):
        rewritten = site.build(match, domain)
        if rewritten:
            yield EmbedFix(site.identifier, match.group(0), rewritten, site.nsfw, match.start(), site.card)


def find_fixes(
    content: str,
    *,
    disabled: Iterable[str] = (),
    overrides: Mapping[str, str] | None = None,
    limit: int = DEFAULT_MAX_LINKS,
) -> list[EmbedFix]:
    """Return rewrites for every supported link in *content*, in message order.

    Duplicate targets collapse to one entry; at most *limit* are returned.
    """
    if not content or limit <= 0:
        return []
    disabled_set = {d.lower() for d in disabled}
    overrides = overrides or {}
    found: list[EmbedFix] = []
    for site in SITES:
        if site.identifier in disabled_set:
            continue
        found.extend(_site_fixes(site, content, overrides.get(site.identifier, site.default_domain)))
    found.sort(key=lambda fix: fix.position)
    unique: dict[str, EmbedFix] = {}
    for fix in found:
        unique.setdefault(fix.rewritten, fix)
    return list(unique.values())[:limit]


def rewrite_url(url: str, *, overrides: Mapping[str, str] | None = None) -> str | None:
    """Rewrite a single URL, or ``None`` when no site matches it."""
    fixes = find_fixes(url, overrides=overrides, limit=1)
    return fixes[0].rewritten if fixes else None


_URL_TOKEN_RE = re.compile(r"^<?https?://(?P<host>[^/\s?#:>]+)[^\s>]*>?$", re.IGNORECASE)


def proxy_hosts(overrides: Mapping[str, str] | None = None) -> frozenset[str]:
    """Hosts the bot's embed-fix replies link to (defaults + configured overrides)."""
    hosts = {site.default_domain for site in SITES if site.default_domain}
    hosts.update((overrides or {}).values())
    return frozenset(host.lower().removeprefix("www.") for host in hosts)


def is_proxy_link_reply(content: str, overrides: Mapping[str, str] | None = None) -> bool:
    """True when *content* is nothing but embed-fix proxy links (one or more).

    Regex test the gate uses to recognise embed-fix text posts.
    """
    tokens = (content or "").split()
    if not tokens:
        return False
    hosts = proxy_hosts(overrides)
    for token in tokens:
        match = _URL_TOKEN_RE.match(token)
        if match is None or match["host"].lower().removeprefix("www.") not in hosts:
            return False
    return True


def is_embed_fix_post(message: object, overrides: str | Mapping[str, str] | None = None) -> bool:
    """True when *message* (a bot post) is an embed-fix reply, by regex alone.

    Text posts: content is only proxy links (vxtwitter.com, oginstagram, ...).
    Card posts (no text): a card links to a proxy host or a fixable source URL
    (ppv page, instagram post). Caller checks the author is the bot.
    """
    if message is None:
        return False
    parsed = parse_domain_overrides(overrides)
    content = getattr(message, "content", "") or ""
    if content.strip():
        return is_proxy_link_reply(content, parsed)
    urls = [getattr(embed, "url", None) or "" for embed in getattr(message, "embeds", None) or []]
    return any(is_proxy_link_reply(url, parsed) or find_fixes(url, limit=1) for url in urls if url)
