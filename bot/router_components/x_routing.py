"""X/Twitter routing helper utilities extracted from Router."""

from __future__ import annotations

from html import unescape
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

from bot.x_api_client import XApiClient


def parse_twitter_status_id(url: str) -> Optional[str]:
    """Extract tweet/status ID from a Twitter/X URL."""
    return XApiClient.extract_tweet_id(url)


def extract_primary_tweet_id(url: str) -> Optional[str]:
    """Extract stable primary tweet ID, preferring explicit URL hint fragments."""
    raw_url = str(url or "").strip()
    if not raw_url:
        return None

    try:
        parsed = urlparse(raw_url)
        for params in (parse_qs(parsed.fragment or ""), parse_qs(parsed.query or "")):
            for key in ("ptid", "primary", "tweet_id", "status_id", "id"):
                values = params.get(key) or []
                if not values:
                    continue
                candidate = str(values[0] or "").strip()
                if candidate and candidate.isdigit():
                    return candidate
    except Exception:
        pass

    return parse_twitter_status_id(raw_url)


def canonicalize_twitter_status_url(url: str) -> str:
    """Convert any Twitter status URL to canonical form https://x.com/i/status/{id}."""
    status_id = parse_twitter_status_id(url)
    if status_id:
        return f"https://x.com/{build_syndication_status_path()}/{status_id}"
    return url


def is_twitter_url(url: str) -> bool:
    try:
        u = str(url)
    except Exception:
        return False
    try:
        if parse_twitter_status_id(u):
            return True
    except Exception:
        pass
    try:
        low = u.lower()
    except Exception:
        return False
    return any(
        d in low
        for d in (
            "twitter.com/",
            "x.com/",
            "vxtwitter.com/",
            "fxtwitter.com/",
            "fixupx.com/",
        )
    )


def collect_x_candidate_urls(item: Any) -> List[str]:
    urls: List[str] = []
    try:
        if item.source_type == "url":
            urls.append(str(item.payload))
        elif item.source_type == "embed":
            embed = item.payload
            primary_url = getattr(embed, "url", None)
            if primary_url:
                urls.append(primary_url)
            video = getattr(embed, "video", None)
            if video and getattr(video, "url", None):
                urls.append(video.url)
            image = getattr(embed, "image", None)
            if image and getattr(image, "url", None):
                urls.append(image.url)
            thumb = getattr(embed, "thumbnail", None)
            if thumb and getattr(thumb, "url", None):
                urls.append(thumb.url)
        elif item.source_type == "attachment":
            attachment = item.payload
            url = getattr(attachment, "url", None)
            if url:
                urls.append(url)
            proxy = getattr(attachment, "proxy_url", None)
            if proxy:
                urls.append(proxy)
    except Exception:
        pass
    return [u for u in urls if u]


def is_twitter_thumbnail_url(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
    }


def is_twitter_media_cdn(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return host in {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
        "video.twimg.com",
        "ton.twimg.com",
    }


def is_tweet_media_url(url: str) -> bool:
    """Check if URL is valid tweet media, excluding profile/banner metadata images."""
    try:
        u = str(url).lower()
    except Exception:
        return False
    try:
        path = urlparse(u).path or ""
    except Exception:
        return False

    blocked_prefixes = (
        "/profile_images/",
        "/profile_banners/",
        "/card_img/",
        "/ad_img/",
        "/emoji/",
    )
    if any(path.startswith(prefix) for prefix in blocked_prefixes):
        return False

    poster_prefixes = (
        "/amplify_video_thumb/",
        "/ext_tw_video_thumb/",
        "/tweet_video_thumb/",
    )
    if any(prefix in path for prefix in poster_prefixes):
        return False

    return "/media/" in path


def normalize_x_url(url: str) -> str:
    """Normalize X/Twitter URLs to canonical host/path, dropping query/fragment."""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        aliases = {
            "mobile.twitter.com",
            "www.twitter.com",
            build_syndication_twitter_host(),
            "www.x.com",
            build_syndication_x_host(),
            "fxtwitter.com",
            "www.fxtwitter.com",
            "vxtwitter.com",
            "www.vxtwitter.com",
            "fixupx.com",
            "www.fixupx.com",
        }
        if host in aliases:
            host = build_syndication_x_host()
        path = p.path or ""
        if path.endswith("/"):
            path = path[:-1]
        return urlunparse(("https", host, path, "", "", ""))
    except Exception:
        return url


def unwrap_x_media_url(url: str) -> str:
    """Unwrap fx/vx API proxy URLs back to the media CDN when possible."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host in {"api.fxtwitter.com", "api.vxtwitter.com"}:
            params = parse_qs(parsed.query or "")
            for key in ("url", "media_url", "target", "u"):
                values = params.get(key)
                if values:
                    candidate = unquote(values[0])
                    if candidate.startswith("http"):
                        return candidate
        return url
    except Exception:
        return url


def extract_x_api_primary_tweet(api_data: Any) -> Dict[str, Any]:
    """Extract the primary tweet node from X API payload variants."""
    if not isinstance(api_data, dict):
        return {}
    data = api_data.get("data")
    if isinstance(data, list):
        first = data[0] if data else {}
        return first if isinstance(first, dict) else {}
    if isinstance(data, dict):
        return data
    return {}


def extract_x_api_primary_text(api_data: Any) -> str:
    """Extract canonical tweet text from X API payload variants."""
    try:
        tweet = extract_x_api_primary_tweet(api_data)
        return str((tweet or {}).get("text") or "").strip()
    except Exception:
        return ""


def extract_sparse_media_resolution(
    resolved_sparse: Any, *, default_url: str
) -> tuple[str, List[str], str]:
    """Extract sparse media kind/images/url from resolved payload."""
    if not isinstance(resolved_sparse, dict):
        return ("unknown", [], default_url)
    sparse_kind = (resolved_sparse.get("kind") or "unknown").strip() or "unknown"
    sparse_images = resolved_sparse.get("images") or []
    if not isinstance(sparse_images, list):
        sparse_images = []
    sparse_url = resolved_sparse.get("url") or default_url
    if not isinstance(sparse_url, str) or not sparse_url:
        sparse_url = default_url
    return (sparse_kind, sparse_images, sparse_url)


def extract_fxtwitter_tweet_node(payload: Any) -> Dict[str, Any]:
    """Extract the canonical tweet/status node from fx/vx payloads."""
    if not isinstance(payload, dict):
        return {}
    node = (payload.get("tweet") or payload.get("status")) or {}
    return node if isinstance(node, dict) else {}


def stt_result_has_transcription(stt_result: Any) -> bool:
    """Check whether STT result payload has a transcription field."""
    if not isinstance(stt_result, dict):
        return False
    return bool(stt_result.get("transcription"))


def resolve_twitter_status_id(
    url: str,
    *,
    tweet_id: Optional[str] = None,
    parse_status_id: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Resolve status ID from explicit hint first, otherwise parse from URL."""
    parser = parse_status_id or parse_twitter_status_id
    return tweet_id or parser(url) or ""


def is_twitter_status_url(
    url: str,
    *,
    parse_status_id: Optional[Callable[[str], Optional[str]]] = None,
) -> bool:
    """Check whether URL contains a parseable Twitter/X status id."""
    parser = parse_status_id or parse_twitter_status_id
    return parser(url) is not None


def classify_stt_error_reason(stt_err: Optional[str]) -> str:
    """Map STT status token to canonical fallback reason."""
    return "no_speech" if stt_err != "error" else "error"


def build_stt_fail_log_payload(
    reason: str,
    *,
    media_kind: Optional[str] = None,
    msg_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build structured payload for STT failure breadcrumb logging."""
    detail: Dict[str, Any] = {"reason": reason}
    if media_kind:
        detail["media_kind"] = media_kind
    payload: Dict[str, Any] = {
        "event": "stt.fail",
        "detail": detail,
    }
    if msg_id is not None:
        payload["msg_id"] = msg_id
    return payload


def build_caption_only_fallback_log_payload() -> Dict[str, Any]:
    """Build structured payload for caption-only fallback breadcrumb logging."""
    return {
        "event": "fallback",
        "detail": {"kind": "caption_only"},
    }


def build_x_video_stt_error_result_payload(
    *,
    url: str,
    stt_error: Optional[str],
) -> Dict[str, Any]:
    """Build the canonical STT error payload for video tweet formatting."""
    return {
        "transcription": None,
        "error": stt_error or "transcription_failed",
        "media_kind": "video",
        "url": url,
    }


def resolve_caption_only_base_text(
    *,
    api_text: Optional[str],
    tweet_text: Optional[str],
    base_text: Optional[str],
) -> str:
    """Resolve caption-only base text using legacy precedence and strip semantics."""
    return (api_text or tweet_text or base_text or "").strip()


def resolve_video_stt_error_base_text(
    *,
    tweet_text: Optional[str],
    base_text: Optional[str],
) -> str:
    """Resolve video STT-error base text using legacy precedence and strip semantics."""
    return (tweet_text or base_text or "").strip()


def syndication_article_has_blocks(article_node: Any) -> bool:
    """Check whether a syndication article payload contains at least one non-empty block."""
    if not isinstance(article_node, dict):
        return False
    content = article_node.get("content") or {}
    blocks = content.get("blocks") if isinstance(content, dict) else []
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("text") or "").strip():
            return True
    return False


def extract_x_article_text(article_node: Any) -> str:
    """Extract normalized text from an X article payload."""
    if not isinstance(article_node, dict):
        return ""
    title = str(article_node.get("title") or "").strip()
    preview = str(article_node.get("preview_text") or "").strip()
    blocks: List[str] = []
    content = article_node.get("content") or {}
    if isinstance(content, dict):
        raw_blocks = content.get("blocks") or []
        if isinstance(raw_blocks, list):
            for block in raw_blocks:
                if not isinstance(block, dict):
                    continue
                btxt = unescape(str(block.get("text") or "")).strip()
                if btxt:
                    blocks.append(btxt)
    parts: List[str] = []
    if title:
        parts.append(unescape(title))
    if preview:
        parts.append(unescape(preview))
    for btxt in blocks:
        if btxt not in parts:
            parts.append(btxt)
    merged = "\n\n".join(parts).strip()
    max_chars = 12000
    if len(merged) > max_chars:
        return merged[: max_chars - 1].rstrip() + "…"
    return merged


def syndication_needs_article_hydration(
    syn: Dict[str, Any],
    *,
    allow_tco_pointer: bool = False,
    article_has_blocks: Optional[Callable[[Any], bool]] = None,
) -> bool:
    """Check whether a syndication payload should trigger X article hydration."""
    if not isinstance(syn, dict) or not syn:
        return False
    article = syn.get("article")
    has_blocks = article_has_blocks or syndication_article_has_blocks
    if isinstance(article, dict) and article:
        if has_blocks(article):
            return False
        if any(
            str(article.get(k) or "").strip()
            for k in ("id", "rest_id", "title", "preview_text")
        ):
            return True

    # X article syndication can surface as a t.co pointer and optional news action metadata.
    if str(syn.get("news_action_type") or "").strip():
        return True
    if allow_tco_pointer:
        txt = (
            str(syn.get("text") or "").strip()
            or str(syn.get("full_text") or "").strip()
            or str((syn.get("legacy") or {}).get("full_text") or "").strip()
        )
        if bool(re.fullmatch(r"https?://t\.co/[A-Za-z0-9]+", txt)):
            return True
    return False


def extract_syndication_base_text(node: Any) -> str:
    """Extract base tweet text from syndication payload precedence."""
    if not isinstance(node, dict):
        return ""
    note = node.get("note_tweet") or {}
    base_text = (
        (note.get("text") if isinstance(note, dict) else None)
        or (node.get("legacy", {}) or {}).get("full_text")
        or node.get("full_text")
        or node.get("text")
        or ""
    )
    return (base_text or "").strip()


def merge_syndication_base_with_article(
    *,
    base_text: str,
    article_text: str,
) -> str:
    """Merge syndication base tweet text with hydrated article text."""
    if not article_text:
        return base_text
    if base_text and not re.search(r"https?://t\.co/[A-Za-z0-9]+", base_text):
        if article_text in base_text:
            return base_text
        return f"{base_text}\n\n[Linked X Article]\n{article_text}"
    return article_text


def extract_syndication_text(
    node: Any,
    *,
    extract_article_text: Optional[Callable[[Any], str]] = None,
) -> str:
    """Extract tweet body text from syndication payloads, including X article text."""
    if not isinstance(node, dict):
        return ""
    article_extractor = extract_article_text or extract_x_article_text
    base_text = extract_syndication_base_text(node)
    try:
        article_text = article_extractor(node.get("article"))
    except Exception:
        article_text = ""
    return merge_syndication_base_with_article(
        base_text=base_text,
        article_text=article_text,
    )


def build_x_text_miss_log_payload(url: str) -> Dict[str, Any]:
    """Build structured breadcrumb payload when syndication text is empty."""
    return build_x_text_miss_payload(
        primary=XApiClient.extract_tweet_id(url) or "",
        layer="format",
        reason="empty_text",
    )


def build_x_text_miss_payload(
    *,
    primary: str,
    layer: str,
    reason: str,
) -> Dict[str, Any]:
    """Build structured breadcrumb payload for X text-miss events."""
    return {
        "event": "x.text.miss",
        "detail": {
            "primary": primary,
            "layer": layer,
            "reason": reason,
        },
    }


def build_syndication_non_200_log_payload(
    *,
    tweet_id: str,
    status: int,
    endpoint: str,
) -> Dict[str, Any]:
    """Build structured payload for syndication non-200 breadcrumb logging."""
    return {
        "detail": {
            "tweet_id": tweet_id,
            "status": status,
            "endpoint": endpoint,
        }
    }


def build_syndication_non_200_metric_payload(
    *,
    status: int,
    endpoint: str,
) -> Dict[str, str]:
    """Build metrics payload for syndication non-200 counters."""
    return {
        "status": str(status),
        "endpoint": endpoint,
    }


def build_syndication_fetch_failed_payload(
    *,
    tweet_id: str,
    error: str,
) -> Dict[str, Any]:
    """Build structured payload for syndication fetch-failure breadcrumbs."""
    return {
        "detail": {
            "tweet_id": tweet_id,
            "error": error,
        }
    }


def build_x_text_canon_payload(
    *,
    url: str,
    primary: str,
) -> Dict[str, Any]:
    """Build structured payload for X canonical-text breadcrumbs."""
    return {
        "event": "x.text.canon",
        "detail": {
            "url": url,
            "primary": primary,
        },
    }


def build_x_text_resolve_payload(
    *,
    primary: str,
    source: str,
    chars: int,
) -> Dict[str, Any]:
    """Build structured payload for X text-resolution breadcrumbs."""
    return {
        "event": "x.text.resolve",
        "detail": {
            "primary": primary,
            "source": source,
            "chars": chars,
        },
    }


def extract_oembed_html_text(html: Any) -> str:
    """Convert oEmbed HTML snippet into plain text using legacy normalization."""
    if not html:
        return ""
    txt = re.sub(r"<br\\s*/?>", "\n", html)
    txt = re.sub(r"<[^>]+>", "", txt)
    return unescape(txt).strip()


def build_oembed_text_payload(
    obj: Any,
    *,
    html_to_text: Callable[[Any], str] = extract_oembed_html_text,
) -> Optional[Dict[str, Any]]:
    """Build syndication-like payload from oEmbed response object."""
    if not isinstance(obj, dict):
        return None
    html = obj.get("html")
    if not html:
        return None
    txt = html_to_text(html)
    if not txt:
        return None
    return {
        "text": txt,
        "user": {"name": obj.get("author_name")},
    }


def extract_oembed_payload_from_response(
    response: Any,
    *,
    build_payload: Callable[[Any], Optional[Dict[str, Any]]] = build_oembed_text_payload,
) -> Optional[Dict[str, Any]]:
    """Extract oEmbed text payload from an HTTP response-like object."""
    if response.status_code != 200:
        return None
    try:
        obj = response.json()
    except Exception:
        return None
    return build_payload(obj)


def build_syndication_oembed_url() -> str:
    """Return the publish.twitter oEmbed endpoint used in syndication fallbacks."""
    return build_syndication_oembed_endpoint_url(
        build_syndication_oembed_host(),
        build_syndication_oembed_key(),
    )


def build_syndication_oembed_endpoint_url(host: str, endpoint_key: str) -> str:
    """Return fully-qualified oEmbed endpoint URL for a given host/key pair."""
    return f"https://{host}/{endpoint_key}"


def build_syndication_oembed_key() -> str:
    """Return canonical oEmbed key used in endpoint paths and metric labels."""
    return "oembed"


def build_syndication_oembed_host() -> str:
    """Return canonical host used for oEmbed fallback requests."""
    return "publish.twitter.com"


def build_syndication_cdn_host() -> str:
    """Return canonical host used for syndication CDN API requests."""
    return "cdn.syndication.twimg.com"


def build_syndication_base_url() -> str:
    """Return the canonical CDN syndication base URL."""
    return f"https://{build_syndication_cdn_host()}/"


def build_syndication_fetch_user_agent() -> str:
    """Return canonical user-agent for CDN syndication fetches."""
    return (
        f"Mozilla/5.0 ({build_syndication_user_agent_platform()}) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )


def build_syndication_user_agent_platform() -> str:
    """Return canonical platform token used in syndication User-Agent."""
    return "Windows NT 10.0; Win64; x64"


def build_syndication_fetch_accept_language() -> str:
    """Return canonical Accept-Language for CDN syndication fetches."""
    return (
        f"{build_syndication_accept_language_primary_entry()},"
        f"{build_syndication_accept_language_secondary_entry()}"
    )


def build_syndication_region_locale() -> str:
    """Return canonical region-specific locale used in syndication headers."""
    return "en-US"


def build_syndication_accept_language_primary_entry() -> str:
    """Return canonical primary Accept-Language entry."""
    return build_syndication_region_locale()


def build_syndication_accept_language_pair() -> str:
    """Return canonical locale pair used in Accept-Language headers."""
    return f"{build_syndication_region_locale()},{build_syndication_lang()}"


def build_syndication_lang_quality() -> str:
    """Return canonical quality token for secondary Accept-Language entries."""
    return "q=0.9"


def build_syndication_accept_language_secondary_entry() -> str:
    """Return canonical secondary Accept-Language entry with quality."""
    return f"{build_syndication_lang()};{build_syndication_lang_quality()}"


def build_syndication_fetch_referer() -> str:
    """Return canonical Referer for CDN syndication fetches."""
    return f"https://{build_syndication_platform_host()}/"


def build_syndication_platform_host() -> str:
    """Return canonical platform host used for syndication referer headers."""
    return "platform.twitter.com"


def build_syndication_fetch_accept() -> str:
    """Return canonical Accept header for CDN syndication fetches."""
    return (
        f"{build_syndication_accept_json_mime()}, "
        f"{build_syndication_accept_text_entry()}, "
        f"{build_syndication_accept_any_entry()}"
    )


def build_syndication_accept_primary_mimes() -> str:
    """Return canonical primary MIME list for Accept headers."""
    return f"{build_syndication_accept_json_mime()}, {build_syndication_accept_text_mime()}"


def build_syndication_accept_json_mime() -> str:
    """Return canonical application/json MIME token for Accept headers."""
    return "application/json"


def build_syndication_accept_text_mime() -> str:
    """Return canonical text/javascript MIME token for Accept headers."""
    return "text/javascript"


def build_syndication_accept_text_quality() -> str:
    """Return canonical quality token for text/javascript Accept entries."""
    return "q=0.9"


def build_syndication_accept_text_entry() -> str:
    """Return canonical text/javascript entry with quality for Accept headers."""
    return f"{build_syndication_accept_text_mime()};{build_syndication_accept_text_quality()}"


def build_syndication_accept_any_mime() -> str:
    """Return canonical wildcard MIME token for Accept headers."""
    return "*/*"


def build_syndication_accept_any_quality() -> str:
    """Return canonical quality token for wildcard Accept entries."""
    return "q=0.8"


def build_syndication_accept_any_entry() -> str:
    """Return canonical wildcard entry with quality for Accept headers."""
    return f"{build_syndication_accept_any_mime()};{build_syndication_accept_any_quality()}"


def build_syndication_lang() -> str:
    """Return canonical language code for syndication/oEmbed requests."""
    return "en"


def build_syndication_fetch_headers() -> Dict[str, str]:
    """Return canonical headers for CDN syndication fetches."""
    return build_syndication_fetch_headers_base()


def build_syndication_fetch_headers_base() -> Dict[str, str]:
    """Return canonical base headers map for CDN syndication fetches."""
    user_agent_key, accept_key, accept_language_key, referer_key = (
        build_syndication_fetch_header_keys()
    )
    user_agent, accept, accept_language, referer = (
        build_syndication_fetch_header_values()
    )
    return {
        user_agent_key: user_agent,
        accept_key: accept,
        accept_language_key: accept_language,
        referer_key: referer,
    }


def build_syndication_fetch_header_keys() -> Tuple[str, str, str, str]:
    """Return canonical header key tuple for syndication CDN fetches."""
    return ("User-Agent", "Accept", "Accept-Language", "Referer")


def build_syndication_fetch_header_values() -> Tuple[str, str, str, str]:
    """Return canonical header value tuple for syndication CDN fetches."""
    return (
        build_syndication_fetch_user_agent(),
        build_syndication_fetch_accept(),
        build_syndication_fetch_accept_language(),
        build_syndication_fetch_referer(),
    )


def build_syndication_fetch_params(
    tweet_id: str,
    *,
    include_dnt: bool = False,
) -> Dict[str, str]:
    """Return canonical syndication fetch params for a tweet id."""
    params = build_syndication_fetch_params_core(tweet_id)
    return build_syndication_fetch_params_with_optional_dnt(params, include_dnt)


def build_syndication_fetch_params_with_optional_dnt(
    params: Dict[str, str],
    include_dnt: bool,
) -> Dict[str, str]:
    """Return params map with optional DNT flag mutation applied."""
    if include_dnt:
        params[build_syndication_dnt_key()] = build_syndication_dnt_value()
    return params


def build_syndication_dnt_key() -> str:
    """Return canonical DNT param key used in syndication fetch params."""
    return "dnt"


def build_syndication_id_key() -> str:
    """Return canonical id param key used in syndication fetch params."""
    return "id"


def build_syndication_lang_key() -> str:
    """Return canonical lang param key used in syndication fetch params."""
    return "lang"


def build_syndication_fetch_params_core(tweet_id: str) -> Dict[str, str]:
    """Return core syndication fetch params for a tweet id (without DNT)."""
    return build_syndication_fetch_params_core_map(tweet_id, build_syndication_lang())


def build_syndication_fetch_params_core_map(tweet_id: str, lang: str) -> Dict[str, str]:
    """Return core syndication fetch params map for explicit tweet id/lang values."""
    return {
        build_syndication_id_key(): tweet_id,
        build_syndication_lang_key(): lang,
    }


def build_syndication_fetch_params_variants(tweet_id: str) -> List[Tuple[str, Dict[str, str]]]:
    """Return endpoint+params variants for CDN syndication fetch attempts."""
    return build_syndication_fetch_params_variants_list(tweet_id)


def build_syndication_fetch_params_variants_list(
    tweet_id: str,
) -> List[Tuple[str, Dict[str, str]]]:
    """Return canonical ordered list of syndication fetch param variants."""
    return [
        build_syndication_widgets_params_variant(tweet_id),
        build_syndication_tweet_result_params_variant(tweet_id),
        build_syndication_widgets_params_variant_with_dnt(tweet_id),
    ]


def build_syndication_widgets_params_variant(
    tweet_id: str,
    *,
    include_dnt: bool = False,
) -> Tuple[str, Dict[str, str]]:
    """Return widgets endpoint params variant, optionally with DNT flag."""
    return (
        build_syndication_widgets_endpoint(),
        build_syndication_fetch_params(tweet_id, include_dnt=include_dnt),
    )


def build_syndication_widgets_params_variant_with_dnt(
    tweet_id: str,
) -> Tuple[str, Dict[str, str]]:
    """Return widgets endpoint params variant with DNT flag enabled."""
    return build_syndication_widgets_params_variant(tweet_id, include_dnt=True)


def build_syndication_tweet_result_params_variant(
    tweet_id: str,
) -> Tuple[str, Dict[str, str]]:
    """Return tweet-result endpoint params variant."""
    return (
        build_syndication_tweet_result_endpoint(),
        build_syndication_fetch_params(tweet_id),
    )


def build_syndication_oembed_params(
    tweet_id: str,
    *,
    use_x_host: bool = False,
) -> Dict[str, str]:
    """Build oEmbed request params for syndication fallback lookups."""
    host = build_syndication_oembed_host_for_flag(use_x_host)
    return build_syndication_oembed_params_bundle(host, tweet_id)


def build_syndication_oembed_params_bundle(host: str, tweet_id: str) -> Dict[str, str]:
    """Return merged core+options oEmbed params for a resolved host."""
    return {
        **build_syndication_oembed_params_core(host, tweet_id),
        **build_syndication_oembed_options(),
    }


def build_syndication_oembed_params_core(host: str, tweet_id: str) -> Dict[str, str]:
    """Return core oEmbed params (url + lang) for a resolved host."""
    lang = build_syndication_lang()
    return {
        build_syndication_oembed_url_key(): build_syndication_oembed_status_url(
            host, tweet_id
        ),
        build_syndication_lang_key(): lang,
    }


def build_syndication_oembed_url_key() -> str:
    """Return canonical url param key used in oEmbed request params."""
    return "url"


def build_syndication_oembed_host_for_flag(use_x_host: bool) -> str:
    """Return oEmbed host selected from use_x_host toggle."""
    host = build_syndication_x_host() if use_x_host else build_syndication_twitter_host()
    return (
        build_syndication_twitter_host()
        if is_syndication_twitter_host(host)
        else build_syndication_x_host()
    )


def build_syndication_oembed_status_url(host: str, tweet_id: str) -> str:
    """Return status URL used by oEmbed for the selected host."""
    return build_syndication_status_url(host, tweet_id)


def build_syndication_status_url(host: str, status_id: Any) -> str:
    """Return full canonical syndication status URL for host and status id."""
    return f"{build_syndication_status_url_prefix(host)}{status_id}"


def build_syndication_status_url_prefix(host: str) -> str:
    """Return canonical status URL prefix for a given host."""
    return f"https://{host}/{build_syndication_status_path()}/"


def build_syndication_status_path() -> str:
    """Return canonical status path used for X/Twitter status URLs."""
    return "i/status"


def build_syndication_oembed_hosts() -> Tuple[str, str]:
    """Return ordered oEmbed host fallbacks for tweet lookups."""
    return build_syndication_oembed_hosts_tuple()


def build_syndication_oembed_hosts_tuple() -> Tuple[str, str]:
    """Return canonical ordered host tuple for oEmbed fallback attempts."""
    return (build_syndication_twitter_host(), build_syndication_x_host())


def build_syndication_twitter_host() -> str:
    """Return canonical twitter hostname for syndication/oEmbed lookups."""
    return "twitter.com"


def build_syndication_x_host() -> str:
    """Return canonical x hostname for syndication/oEmbed lookups."""
    return "x.com"


def build_syndication_oembed_options() -> Dict[str, str]:
    """Return canonical oEmbed flags for privacy/script/thread behavior."""
    return build_syndication_oembed_options_map()


def build_syndication_oembed_options_map() -> Dict[str, str]:
    """Return canonical oEmbed option key/value map."""
    dnt_key, omit_script_key, hide_thread_key = build_syndication_oembed_option_keys()
    dnt_value, omit_script_value, hide_thread_value = (
        build_syndication_oembed_option_values()
    )
    return {
        dnt_key: dnt_value,
        omit_script_key: omit_script_value,
        hide_thread_key: hide_thread_value,
    }


def build_syndication_oembed_option_keys() -> Tuple[str, str, str]:
    """Return canonical oEmbed option keys in stable order."""
    return (
        build_syndication_oembed_dnt_key(),
        build_syndication_oembed_omit_script_key(),
        build_syndication_oembed_hide_thread_key(),
    )


def build_syndication_oembed_option_values() -> Tuple[str, str, str]:
    """Return canonical oEmbed option values in stable key order."""
    return (
        build_syndication_dnt_value(),
        build_syndication_omit_script_value(),
        build_syndication_hide_thread_value(),
    )


def build_syndication_oembed_dnt_key() -> str:
    """Return canonical DNT option key used in oEmbed params."""
    return "dnt"


def build_syndication_oembed_omit_script_key() -> str:
    """Return canonical omit_script option key used in oEmbed params."""
    return "omit_script"


def build_syndication_oembed_hide_thread_key() -> str:
    """Return canonical hide_thread option key used in oEmbed params."""
    return "hide_thread"


def build_syndication_dnt_value() -> str:
    """Return canonical DNT option value for syndication/oEmbed requests."""
    return build_syndication_bool_false_value()


def build_syndication_omit_script_value() -> str:
    """Return canonical omit-script option value for oEmbed requests."""
    return build_syndication_bool_true_value()


def build_syndication_hide_thread_value() -> str:
    """Return canonical hide-thread option value for oEmbed requests."""
    return build_syndication_bool_true_value()


def build_syndication_bool_true_value() -> str:
    """Return canonical string value used for enabled syndication flags."""
    return "true"


def build_syndication_bool_false_value() -> str:
    """Return canonical string value used for disabled syndication flags."""
    return "false"


def build_syndication_oembed_metric_endpoint(host: str) -> str:
    """Return metric endpoint label for a given oEmbed host."""
    return (
        build_syndication_oembed_x_metric_endpoint()
        if is_syndication_x_host(host)
        else build_syndication_oembed_metric_default_endpoint()
    )


def is_syndication_x_host(host: str) -> bool:
    """Return True when host is the canonical X hostname."""
    return str(host) == build_syndication_x_host()


def is_syndication_twitter_host(host: str) -> bool:
    """Return True when host is the canonical Twitter hostname."""
    return str(host) == build_syndication_twitter_host()


def build_syndication_oembed_metric_default_endpoint() -> str:
    """Return default metric endpoint label for oEmbed fallback calls."""
    return build_syndication_oembed_key()


def build_syndication_oembed_x_metric_endpoint() -> str:
    """Return X-host metric endpoint label for oEmbed fallback calls."""
    return "oembed_x"


def build_syndication_oembed_fallback_params(
    tweet_id: str,
) -> List[Tuple[str, Dict[str, str]]]:
    """Return ordered oEmbed fallback variants and their metric endpoint labels."""
    items: List[Tuple[str, Dict[str, str]]] = []
    for host in build_syndication_oembed_hosts():
        items.append(build_syndication_oembed_fallback_item(host, tweet_id))
    return items


def build_syndication_oembed_fallback_item(
    host: str,
    tweet_id: str,
) -> Tuple[str, Dict[str, str]]:
    """Return one oEmbed fallback item (metric endpoint + params) for a host."""
    endpoint = build_syndication_oembed_metric_endpoint(host)
    return (
        endpoint,
        build_syndication_oembed_params(
            tweet_id,
            use_x_host=is_syndication_x_host(host),
        ),
    )


def build_syndication_oembed_fallback_plan(
    tweet_id: str,
) -> Tuple[str, List[Tuple[str, Dict[str, str]]]]:
    """Return oEmbed fallback URL and ordered variant params."""
    return build_syndication_oembed_fallback_plan_components(tweet_id)


def build_syndication_oembed_fallback_plan_components(
    tweet_id: str,
) -> Tuple[str, List[Tuple[str, Dict[str, str]]]]:
    """Build canonical oEmbed fallback plan components from tweet id."""
    return build_syndication_oembed_fallback_plan_tuple(
        build_syndication_oembed_url(),
        build_syndication_oembed_fallback_params(tweet_id),
    )


def build_syndication_oembed_fallback_plan_tuple(
    url: str,
    variants: List[Tuple[str, Dict[str, str]]],
) -> Tuple[str, List[Tuple[str, Dict[str, str]]]]:
    """Return canonical oEmbed fallback plan tuple (url, variants)."""
    return url, variants


def build_syndication_fetch_plan(
    tweet_id: str,
) -> Tuple[str, Dict[str, str], List[Tuple[str, Dict[str, str]]]]:
    """Build base URL, headers, and endpoint param variants for syndication fetch."""
    return build_syndication_fetch_plan_components(tweet_id)


def build_syndication_fetch_plan_components(
    tweet_id: str,
) -> Tuple[str, Dict[str, str], List[Tuple[str, Dict[str, str]]]]:
    """Build canonical fetch plan components from tweet id."""
    base = build_syndication_base_url()
    headers = build_syndication_fetch_headers()
    params_variants = build_syndication_fetch_params_variants(tweet_id)
    return build_syndication_fetch_plan_tuple(base, headers, params_variants)


def build_syndication_fetch_plan_tuple(
    base: str,
    headers: Dict[str, str],
    params_variants: List[Tuple[str, Dict[str, str]]],
) -> Tuple[str, Dict[str, str], List[Tuple[str, Dict[str, str]]]]:
    """Return canonical (base, headers, variants) fetch plan tuple."""
    return base, headers, params_variants


def build_syndication_fetch_metric_payload(endpoint: str) -> Dict[str, str]:
    """Build metric labels payload for syndication fetch endpoints."""
    return {build_syndication_metric_endpoint_key(): endpoint}


def build_syndication_metric_endpoint_key() -> str:
    """Return canonical metric key name for endpoint label payloads."""
    return "endpoint"


def build_syndication_widgets_endpoint() -> str:
    """Return canonical widgets endpoint key."""
    return "widgets"


def build_syndication_tweet_result_endpoint() -> str:
    """Return canonical tweet-result endpoint key."""
    return "tweet-result"


def build_syndication_widgets_tweet_path() -> str:
    """Return canonical widgets tweet path suffix."""
    return "widgets/tweet"


def build_syndication_tweet_result_path() -> str:
    """Return canonical tweet-result path suffix."""
    return "tweet-result"


def syndication_cache_ttl_s(default_ttl_s: float, cached: Any) -> float:
    """Compute syndication cache TTL with shorter cap for negative entries."""
    ttl = default_ttl_s
    if cached.get(build_syndication_negative_cache_key()):
        ttl = min(default_ttl_s, build_syndication_negative_cache_ttl_cap_s())
    return ttl


def build_syndication_negative_cache_ttl_cap_s() -> float:
    """Return TTL cap for negative syndication cache entries, in seconds."""
    return 300.0


def build_syndication_cache_ts_key() -> str:
    """Return canonical timestamp key used in syndication cache entries."""
    return "ts"


def build_syndication_negative_cache_key() -> str:
    """Return canonical boolean key used for negative syndication cache entries."""
    return "neg"


def build_syndication_cache_data_key() -> str:
    """Return canonical payload key used for positive syndication cache entries."""
    return "data"


def syndication_cache_is_fresh(now_s: float, default_ttl_s: float, cached: Any) -> bool:
    """Return True when syndication cache entry is still fresh under TTL policy."""
    ttl = syndication_cache_ttl_s(default_ttl_s, cached)
    return (now_s - float(cached.get(build_syndication_cache_ts_key(), 0))) < ttl


def classify_syndication_cache_hit(
    now_s: float,
    default_ttl_s: float,
    cached: Any,
) -> Optional[str]:
    """Classify cache hit kind as `neg`, `data`, or None when stale."""
    if not syndication_cache_is_fresh(now_s, default_ttl_s, cached):
        return None
    return (
        build_syndication_negative_cache_hit_label()
        if cached.get(build_syndication_negative_cache_key())
        else build_syndication_data_cache_hit_label()
    )


def build_syndication_negative_cache_entry(now_s: float) -> Dict[str, Any]:
    """Build negative syndication cache entry with timestamp."""
    return {
        build_syndication_negative_cache_key(): True,
        build_syndication_cache_ts_key(): now_s,
    }


def build_syndication_cache_entry(data: Any, now_s: float) -> Dict[str, Any]:
    """Build positive syndication cache entry with timestamp."""
    return {
        build_syndication_cache_data_key(): data,
        build_syndication_cache_ts_key(): now_s,
    }


def build_syndication_negative_cache_hit_label() -> str:
    """Return cache-hit label for negative syndication cache entries."""
    return build_syndication_negative_cache_key()


def build_syndication_data_cache_hit_label() -> str:
    """Return cache-hit label for data-backed syndication cache entries."""
    return build_syndication_cache_data_key()


def build_syndication_endpoint_url(base: str, endpoint: str) -> str:
    """Build syndication endpoint URL preserving legacy endpoint mapping."""
    return base + build_syndication_endpoint_suffix(endpoint)


def build_syndication_endpoint_suffix(endpoint: str) -> str:
    """Return endpoint path suffix preserving legacy endpoint fallback behavior."""
    return (
        build_syndication_widgets_tweet_path()
        if endpoint == build_syndication_widgets_endpoint()
        else build_syndication_tweet_result_path()
    )


def syndication_has_usable_payload(
    node: Any,
    *,
    extract_text: Callable[[Any], str],
    media_hint_keys: Iterable[str],
) -> bool:
    """Return True when syndication payload includes usable text in current schema."""
    if not isinstance(node, dict):
        return False
    if extract_text(node):
        return True
    return any(k in node for k in media_hint_keys)


def syndication_media_hint_keys() -> Tuple[str, ...]:
    """Canonical syndication media-hint keys used in payload usability checks."""
    return (
        "media",
        "photos",
        "video",
        "video_info",
        "video_variants",
        "video_urls",
        "media_duration",
        "duration_ms",
        "extended_entities",
        "entities",
        "quoted_tweet",
        "quoted_status",
        "retweeted_status",
        "legacy",
        "card",
        "image",
        "article",
    )


def format_syndication_body_text(text: str) -> str:
    """Format syndication body text with legacy size limits and fallback copy."""
    if text and len(text) <= 4000:
        return text
    if text:
        return text[:3990] + "…"
    return "(Tweet text not available. If you want analysis, paste the text or add a screenshot.)"


def format_syndication_header_line(
    *,
    user: Any,
    created_at: Any,
    photos: Any,
    url: str,
) -> str:
    """Format syndication header line preserving legacy field access semantics."""
    username = user.get("screen_name") or user.get("name")
    media_hint = f" • media:{len(photos)}" if photos else ""
    prefix = f"@{username}" if username else "Tweet"
    stamp = f" • {created_at}" if created_at else ""
    return f"{prefix}{stamp}{media_hint} → {url}"


def format_syndication_error_fallback(url: str, syn_data: Any) -> str:
    """Format fallback output when syndication payload formatting fails."""
    return f"Tweet → {url}\n{str(syn_data)[:4000]}"


def extract_syndication_photo_urls(photos: Any) -> List[str]:
    """Extract photo URLs from syndication `photos` payload."""
    urls: List[str] = []
    for p in photos:
        if isinstance(p, dict):
            img_url = p.get("url") or p.get("media_url_https") or p.get("media_url")
            if img_url and isinstance(img_url, str):
                urls.append(img_url)
        elif isinstance(p, str):
            urls.append(p)
    return urls


def x_syn_probe_budget_timeout_s(x_syn_timeout_s: float) -> float:
    """Compute bounded timeout budget for image/media probe calls."""
    return min(float(x_syn_timeout_s) + 1.0, 4.5)


def x_syn_quick_request_timeouts(x_syn_timeout_s: float) -> tuple[float, float, float]:
    """Compute bounded connect/read/total request budgets for quick probes."""
    return (
        min(x_syn_timeout_s, 3.0),
        min(x_syn_timeout_s, 3.0),
        min(x_syn_timeout_s + 0.5, 3.5),
    )


def build_syndication_photo_payload(
    text: Optional[str], image_urls: List[str]
) -> Dict[str, Any]:
    """Build syndication-like payload consumed by the unified VL handler."""
    return {
        "text": text,
        "photos": [{"url": url} for url in image_urls],
    }


def format_twitter_syndication_images_log_line(
    image_urls: List[str], *, msg_id: Optional[int] = None
) -> str:
    """Format canonical breadcrumb line for Twitter image-route detection."""
    first_host = ""
    try:
        if image_urls:
            first_host = urlparse(image_urls[0]).netloc
    except Exception:
        first_host = ""
    suffix = f" | msg_id={msg_id}" if msg_id is not None else ""
    return (
        f"route.twitter.syndication | images={len(image_urls)} | "
        f"{first_host or 'n/a'}{suffix}"
    )


async def resolve_and_probe_twitter_images(
    *,
    url: str,
    tweet_id: Optional[str] = None,
    resolve_status_id: Callable[..., str],
    probe_images: Callable[..., Awaitable[List[str]]],
) -> Tuple[str, List[str]]:
    """Resolve status id and probe syndication image URLs with normalized defaults."""
    status_id = resolve_status_id(url, tweet_id=tweet_id)
    image_urls = await probe_images(url, status_id)
    return status_id, (image_urls or [])


def extract_x_status_urls_from_text(
    text: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> List[str]:
    """Extract canonical X/Twitter status URLs from text preserving order."""
    urls: List[str] = []
    try:
        for m in re.finditer(r"https?://[^\s<>\"'\[\]{}|\\^`]+", text or "", re.IGNORECASE):
            raw = m.group(0)
            if is_status_url(raw):
                cu = canonicalize_status_url(raw)
                if cu not in urls:
                    urls.append(cu)
    except Exception:
        pass
    return urls


def extract_raw_urls_from_texts(texts: Iterable[str]) -> List[str]:
    """Extract raw URLs from multiple text blobs in-order with de-duplication."""
    raw_urls: List[str] = []
    try:
        url_re = re.compile(r"https?://[^\s<>\"'\[\]{}|\\^`]+", re.IGNORECASE)
        for t in texts:
            for m in url_re.finditer(t or ""):
                u = m.group(0)
                if u and u not in raw_urls:
                    raw_urls.append(u)
    except Exception:
        pass
    return raw_urls


def filter_canonical_x_urls(
    raw_urls: Iterable[str],
    *,
    is_x_url: Callable[[str], bool],
    canonicalize_x_url: Callable[[str], str],
) -> List[str]:
    """Filter URL list to X/Twitter URLs and canonicalize with de-duplication."""
    out: List[str] = []
    for u in raw_urls:
        if is_x_url(u):
            cu = canonicalize_x_url(u)
            if cu not in out:
                out.append(cu)
    return out
