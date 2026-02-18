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


def extract_url_host_lower(url: str) -> str:
    """Parse a URL host and normalize to lowercase; return empty string on failure."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def extract_url_path(url: str) -> str:
    """Parse a URL path; return empty string on failure."""
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def is_twitter_thumbnail_url(url: str) -> bool:
    host = extract_url_host_lower(url)
    if not host:
        return False
    return is_twitter_thumbnail_host(host)


def is_twitter_thumbnail_host(host: str) -> bool:
    """Return True when host is a known Twitter thumbnail CDN host."""
    return host in {
        "pbs.twimg.com",
        "pbs-0.twimg.com",
        "pbs-1.twimg.com",
        "pbs-2.twimg.com",
        "pbs-3.twimg.com",
    }


def is_twitter_media_cdn(url: str) -> bool:
    host = extract_url_host_lower(url)
    if not host:
        return False
    return is_twitter_media_cdn_host(host)


def is_twitter_media_cdn_host(host: str) -> bool:
    """Return True when host is a known Twitter media CDN host."""
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
    path = extract_url_path(u)

    if is_blocked_tweet_media_path(path):
        return False

    if is_poster_tweet_media_path(path):
        return False

    return has_tweet_media_path_segment(path)


def has_tweet_media_path_segment(path: str) -> bool:
    """Return True when path contains the canonical /media/ segment."""
    return path_contains_tweet_media_segment(path)


def tweet_media_path_segment() -> str:
    """Return canonical media path segment used for tweet media assets."""
    return "/media/"


def path_contains_tweet_media_segment(path: str) -> bool:
    """Return True when path includes tweet media segment literal."""
    return tweet_media_path_segment() in path


def blocked_tweet_media_prefixes() -> Tuple[str, ...]:
    """Return blocked tweet-media metadata path prefixes."""
    return (
        "/profile_images/",
        "/profile_banners/",
        "/card_img/",
        "/ad_img/",
        "/emoji/",
    )


def is_blocked_tweet_media_path(path: str) -> bool:
    """Return True when path matches a blocked tweet-media metadata prefix."""
    return any(path.startswith(prefix) for prefix in blocked_tweet_media_prefixes())


def poster_tweet_media_prefixes() -> Tuple[str, ...]:
    """Return poster/thumbnail tweet-media path markers."""
    return (
        "/amplify_video_thumb/",
        "/ext_tw_video_thumb/",
        "/tweet_video_thumb/",
    )


def is_poster_tweet_media_path(path: str) -> bool:
    """Return True when path points to poster/thumbnail tweet-media assets."""
    return any(prefix in path for prefix in poster_tweet_media_prefixes())


def normalize_x_url(url: str) -> str:
    """Normalize X/Twitter URLs to canonical host/path, dropping query/fragment."""
    try:
        p = urlparse(url)
        host = normalize_x_host((p.netloc or "").lower())
        path = normalize_x_path(p.path or "")
        return urlunparse(("https", host, path, "", "", ""))
    except Exception:
        return url


def x_host_aliases() -> set[str]:
    """Return hosts that normalize to canonical x.com."""
    return {
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


def normalize_x_host(host: str) -> str:
    """Normalize known Twitter/X alias hosts to canonical x.com host."""
    if host in x_host_aliases():
        return build_syndication_x_host()
    return host


def x_path_has_trailing_slash(path: str) -> bool:
    """Return True when path ends with a slash."""
    return path.endswith("/")


def normalize_x_path(path: str) -> str:
    """Normalize X/Twitter path by trimming one trailing slash."""
    if x_path_has_trailing_slash(path):
        return path[:-1]
    return path


def unwrap_x_media_url(url: str) -> str:
    """Unwrap fx/vx API proxy URLs back to the media CDN when possible."""
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if is_unwrap_x_media_proxy_host(host):
            params = parse_qs(parsed.query or "")
            candidate = first_unwrap_x_media_candidate(params)
            if is_unwrap_x_media_candidate_url(candidate):
                return candidate
        return url
    except Exception:
        return url


def first_unwrap_x_media_candidate(params: Dict[str, List[str]]) -> str:
    """Return first decoded candidate from supported unwrap query params."""
    for key in unwrap_x_media_param_keys():
        values = params.get(key)
        if values:
            return unquote(values[0])
    return ""


def unwrap_x_media_param_keys() -> tuple[str, ...]:
    """Return query parameter keys checked when unwrapping proxy media URLs."""
    return ("url", "media_url", "target", "u")


def is_unwrap_x_media_proxy_host(host: str) -> bool:
    """Return True when host is a supported fx/vx media proxy endpoint."""
    return host in {"api.fxtwitter.com", "api.vxtwitter.com"}


def is_unwrap_x_media_candidate_url(candidate: str) -> bool:
    """Return True when unwrapped candidate looks like an absolute HTTP URL."""
    return candidate.startswith("http")


def extract_x_api_primary_tweet(api_data: Any) -> Dict[str, Any]:
    """Extract the primary tweet node from X API payload variants."""
    if not isinstance(api_data, dict):
        return {}
    data = api_data.get("data")
    if isinstance(data, list):
        first = extract_x_api_first_item(data)
        return first if isinstance(first, dict) else {}
    if isinstance(data, dict):
        return data
    return {}


def extract_x_api_first_item(data: List[Any]) -> Any:
    """Return first item from API data list or empty dict when list is empty."""
    return data[0] if data else {}


def extract_x_api_primary_text(api_data: Any) -> str:
    """Extract canonical tweet text from X API payload variants."""
    try:
        tweet = extract_x_api_primary_tweet(api_data)
        return normalize_x_api_text((tweet or {}).get("text"))
    except Exception:
        return ""


def normalize_x_api_text(text: Any) -> str:
    """Normalize X API text field to a stripped string."""
    return str(text or "").strip()


def extract_sparse_media_resolution(
    resolved_sparse: Any, *, default_url: str
) -> tuple[str, List[str], str]:
    """Extract sparse media kind/images/url from resolved payload."""
    if not isinstance(resolved_sparse, dict):
        return ("unknown", [], default_url)
    sparse_kind = normalize_sparse_kind_value(resolved_sparse.get("kind"))
    sparse_images = normalize_sparse_images_value(resolved_sparse.get("images"))
    sparse_url = normalize_sparse_url_value(resolved_sparse.get("url"), default_url=default_url)
    return (sparse_kind, sparse_images, sparse_url)


def normalize_sparse_kind_value(value: Any) -> str:
    """Normalize sparse media kind field to a non-empty lower-level token."""
    return (value or "unknown").strip() or "unknown"


def normalize_sparse_url_value(value: Any, *, default_url: str) -> str:
    """Normalize sparse media URL field to a non-empty string value."""
    sparse_url = value or default_url
    if not isinstance(sparse_url, str) or not sparse_url:
        return default_url
    return sparse_url


def normalize_sparse_images_value(value: Any) -> List[Any]:
    """Normalize sparse media images field to a list value."""
    sparse_images = value or []
    if not isinstance(sparse_images, list):
        return []
    return sparse_images


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
    return stt_transcription_value_is_present(stt_result.get("transcription"))


def stt_transcription_value_is_present(value: Any) -> bool:
    """Return whether STT transcription value is present/non-empty."""
    return bool(value)


def resolve_twitter_status_id(
    url: str,
    *,
    tweet_id: Optional[str] = None,
    parse_status_id: Optional[Callable[[str], Optional[str]]] = None,
) -> str:
    """Resolve status ID from explicit hint first, otherwise parse from URL."""
    parser = resolve_twitter_status_parser(parse_status_id)
    return tweet_id or parser(url) or ""


def is_twitter_status_url(
    url: str,
    *,
    parse_status_id: Optional[Callable[[str], Optional[str]]] = None,
) -> bool:
    """Check whether URL contains a parseable Twitter/X status id."""
    parser = resolve_twitter_status_parser(parse_status_id)
    return parser(url) is not None


def resolve_twitter_status_parser(
    parse_status_id: Optional[Callable[[str], Optional[str]]] = None,
) -> Callable[[str], Optional[str]]:
    """Resolve status ID parser, defaulting to built-in Twitter status parser."""
    return parse_status_id or parse_twitter_status_id


def classify_stt_error_reason(stt_err: Optional[str]) -> str:
    """Map STT status token to canonical fallback reason."""
    return "error" if is_stt_hard_error(stt_err) else "no_speech"


def is_stt_hard_error(stt_err: Optional[str]) -> bool:
    """Return True when STT status token is the canonical hard-error value."""
    return stt_err == "error"


def build_stt_fail_log_payload(
    reason: str,
    *,
    media_kind: Optional[str] = None,
    msg_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build structured payload for STT failure breadcrumb logging."""
    detail = build_stt_fail_detail(reason, media_kind=media_kind)
    payload: Dict[str, Any] = {
        "event": "stt.fail",
        "detail": detail,
    }
    if msg_id is not None:
        payload["msg_id"] = msg_id
    return payload


def build_stt_fail_detail(reason: str, *, media_kind: Optional[str] = None) -> Dict[str, Any]:
    """Build detail object for STT failure breadcrumb payloads."""
    detail: Dict[str, Any] = {"reason": reason}
    if media_kind:
        detail["media_kind"] = media_kind
    return detail


def build_caption_only_fallback_log_payload() -> Dict[str, Any]:
    """Build structured payload for caption-only fallback breadcrumb logging."""
    return {
        "event": "fallback",
        "detail": build_caption_only_fallback_detail(),
    }


def build_caption_only_fallback_detail() -> Dict[str, Any]:
    """Build detail object for caption-only fallback breadcrumb payloads."""
    return {"kind": "caption_only"}


def build_x_video_stt_error_result_payload(
    *,
    url: str,
    stt_error: Optional[str],
) -> Dict[str, Any]:
    """Build the canonical STT error payload for video tweet formatting."""
    return {
        "transcription": None,
        "error": normalize_stt_error_value(stt_error),
        "media_kind": "video",
        "url": url,
    }


def normalize_stt_error_value(stt_error: Optional[str]) -> str:
    """Normalize STT error token to canonical fallback value."""
    return stt_error or "transcription_failed"


def resolve_caption_only_base_text(
    *,
    api_text: Optional[str],
    tweet_text: Optional[str],
    base_text: Optional[str],
) -> str:
    """Resolve caption-only base text using legacy precedence and strip semantics."""
    return normalize_base_text_value(api_text or tweet_text or base_text or "")


def resolve_video_stt_error_base_text(
    *,
    tweet_text: Optional[str],
    base_text: Optional[str],
) -> str:
    """Resolve video STT-error base text using legacy precedence and strip semantics."""
    return normalize_base_text_value(tweet_text or base_text or "")


def normalize_base_text_value(value: Optional[str]) -> str:
    """Normalize base text value using router strip semantics."""
    return (value or "").strip()


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
        if has_non_empty_block_text(block):
            return True
    return False


def has_non_empty_block_text(block: Dict[str, Any]) -> bool:
    """Return True when article/content block has non-empty text payload."""
    return bool(str(block.get("text") or "").strip())


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
                btxt = normalize_article_block_text(block)
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
    return truncate_x_article_text(merged)


def truncate_x_article_text(text: str, *, max_chars: int = 12000) -> str:
    """Truncate extracted X article text to max_chars with ellipsis suffix."""
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text


def normalize_article_block_text(block: Dict[str, Any]) -> str:
    """Normalize article content-block text by unescaping and stripping."""
    return unescape(str(block.get("text") or "")).strip()


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
        if article_has_metadata_hints(article):
            return True

    # X article syndication can surface as a t.co pointer and optional news action metadata.
    if has_news_action_type(syn):
        return True
    if allow_tco_pointer:
        txt = resolve_syndication_pointer_text(syn)
        if is_tco_pointer_text(txt):
            return True
    return False


def resolve_syndication_pointer_text(syn: Dict[str, Any]) -> str:
    """Resolve pointer-probe text from syndication precedence (text/full_text/legacy.full_text)."""
    return (
        str(syn.get("text") or "").strip()
        or str(syn.get("full_text") or "").strip()
        or str((syn.get("legacy") or {}).get("full_text") or "").strip()
    )


def article_has_metadata_hints(article: Dict[str, Any]) -> bool:
    """Return True when article payload contains identifying metadata fields."""
    return any(
        str(article.get(key) or "").strip()
        for key in ("id", "rest_id", "title", "preview_text")
    )


def has_news_action_type(syn: Dict[str, Any]) -> bool:
    """Return True when syndication payload includes a non-empty news action type."""
    return bool(str(syn.get("news_action_type") or "").strip())


def is_tco_pointer_text(text: str) -> bool:
    """Return True when text is exactly one t.co pointer URL."""
    return bool(re.fullmatch(r"https?://t\.co/[A-Za-z0-9]+", text))


def extract_syndication_base_text(node: Any) -> str:
    """Extract base tweet text from syndication payload precedence."""
    if not isinstance(node, dict):
        return ""
    note = node.get("note_tweet") or {}
    base_text = (
        extract_note_tweet_text(note)
        or (node.get("legacy", {}) or {}).get("full_text")
        or node.get("full_text")
        or node.get("text")
        or ""
    )
    return (base_text or "").strip()


def extract_note_tweet_text(note: Any) -> Optional[str]:
    """Extract text field from note_tweet payload when it is a dict."""
    return note.get("text") if isinstance(note, dict) else None


def merge_syndication_base_with_article(
    *,
    base_text: str,
    article_text: str,
) -> str:
    """Merge syndication base tweet text with hydrated article text."""
    if not article_text:
        return base_text
    if base_text and not base_text_contains_tco_link(base_text):
        if article_text in base_text:
            return base_text
        return f"{base_text}\n\n[Linked X Article]\n{article_text}"
    return article_text


def base_text_contains_tco_link(base_text: str) -> bool:
    """Return True when base text includes a t.co link token."""
    return bool(re.search(r"https?://t\.co/[A-Za-z0-9]+", base_text))


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
    article_text = extract_syndication_article_text(
        node=node,
        article_extractor=article_extractor,
    )
    return merge_syndication_base_with_article(
        base_text=base_text,
        article_text=article_text,
    )


def extract_syndication_article_text(
    *,
    node: Dict[str, Any],
    article_extractor: Callable[[Any], str],
) -> str:
    """Extract hydrated article text from syndication payload; fail open on extractor errors."""
    try:
        return article_extractor(node.get("article"))
    except Exception:
        return ""


def build_x_text_miss_log_payload(url: str) -> Dict[str, Any]:
    """Build structured breadcrumb payload when syndication text is empty."""
    return build_x_text_miss_payload(
        primary=extract_primary_tweet_id(url) or "",
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
    return build_syndication_fetch_header_map(
        keys=build_syndication_fetch_header_keys(),
        values=build_syndication_fetch_header_values(),
    )


def build_syndication_fetch_header_map(
    *,
    keys: Tuple[str, str, str, str],
    values: Tuple[str, str, str, str],
) -> Dict[str, str]:
    """Build syndication header map from ordered keys and values tuples."""
    user_agent_key, accept_key, accept_language_key, referer_key = keys
    user_agent, accept, accept_language, referer = values
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
    return maybe_add_syndication_dnt_param(
        params=params,
        include_dnt=include_dnt,
    )


def maybe_add_syndication_dnt_param(
    *,
    params: Dict[str, str],
    include_dnt: bool,
) -> Dict[str, str]:
    """Mutate params with DNT entry when include_dnt is enabled."""
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
    return build_syndication_oembed_params_core_map(
        host,
        tweet_id,
        build_syndication_lang(),
    )


def build_syndication_oembed_params_core_map(
    host: str,
    tweet_id: str,
    lang: str,
) -> Dict[str, str]:
    """Return core oEmbed params map for explicit host/tweet/lang values."""
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
    return build_syndication_oembed_options_map_from_pairs(
        (dnt_key, omit_script_key, hide_thread_key),
        (dnt_value, omit_script_value, hide_thread_value),
    )


def build_syndication_oembed_options_map_from_pairs(
    keys: Tuple[str, str, str],
    values: Tuple[str, str, str],
) -> Dict[str, str]:
    """Return oEmbed options map from aligned key/value tuples."""
    dnt_key, omit_script_key, hide_thread_key = keys
    dnt_value, omit_script_value, hide_thread_value = values
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
    return build_syndication_oembed_fallback_items_list(tweet_id)


def build_syndication_oembed_fallback_items_list(
    tweet_id: str,
) -> List[Tuple[str, Dict[str, str]]]:
    """Return ordered list of oEmbed fallback items for a tweet id."""
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
    base, headers, params_variants = build_syndication_fetch_plan_values(tweet_id)
    return build_syndication_fetch_plan_tuple(base, headers, params_variants)


def build_syndication_fetch_plan_values(
    tweet_id: str,
) -> Tuple[str, Dict[str, str], List[Tuple[str, Dict[str, str]]]]:
    """Return canonical fetch plan values tuple (base, headers, variants)."""
    base = build_syndication_base_url()
    headers = build_syndication_fetch_headers()
    params_variants = build_syndication_fetch_params_variants(tweet_id)
    return base, headers, params_variants


def build_syndication_fetch_plan_tuple(
    base: str,
    headers: Dict[str, str],
    params_variants: List[Tuple[str, Dict[str, str]]],
) -> Tuple[str, Dict[str, str], List[Tuple[str, Dict[str, str]]]]:
    """Return canonical (base, headers, variants) fetch plan tuple."""
    return base, headers, params_variants


def build_syndication_fetch_metric_payload(endpoint: str) -> Dict[str, str]:
    """Build metric labels payload for syndication fetch endpoints."""
    return build_syndication_metric_payload_map(
        build_syndication_metric_endpoint_key(),
        endpoint,
    )


def build_syndication_metric_payload_map(key: str, value: str) -> Dict[str, str]:
    """Build metric payload map from explicit key/value inputs."""
    return {key: value}


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
        ttl = syndication_negative_cache_ttl_value(default_ttl_s)
    return ttl


def syndication_negative_cache_ttl_value(default_ttl_s: float) -> float:
    """Return effective TTL for negative cache entries under cap policy."""
    return min(default_ttl_s, build_syndication_negative_cache_ttl_cap_s())


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
    return (now_s - syndication_cache_timestamp_value(cached)) < ttl


def syndication_cache_timestamp_value(cached: Any) -> float:
    """Return parsed cache timestamp value for freshness checks."""
    return float(cached.get(build_syndication_cache_ts_key(), 0))


def classify_syndication_cache_hit(
    now_s: float,
    default_ttl_s: float,
    cached: Any,
) -> Optional[str]:
    """Classify cache hit kind as `neg`, `data`, or None when stale."""
    if not syndication_cache_is_fresh(now_s, default_ttl_s, cached):
        return None
    return build_syndication_cache_hit_label(cached)


def build_syndication_cache_hit_label(cached: Any) -> str:
    """Return cache-hit label for a fresh cache payload."""
    return (
        build_syndication_negative_cache_hit_label()
        if cached.get(build_syndication_negative_cache_key())
        else build_syndication_data_cache_hit_label()
    )


def build_syndication_negative_cache_entry(now_s: float) -> Dict[str, Any]:
    """Build negative syndication cache entry with timestamp."""
    return {
        **build_syndication_negative_cache_flag_field(),
        **build_syndication_cache_timestamp_field(now_s),
    }


def build_syndication_cache_entry(data: Any, now_s: float) -> Dict[str, Any]:
    """Build positive syndication cache entry with timestamp."""
    return {
        **build_syndication_cache_data_field(data),
        **build_syndication_cache_timestamp_field(now_s),
    }


def build_syndication_cache_timestamp_field(now_s: float) -> Dict[str, float]:
    """Build canonical cache timestamp field map."""
    return {build_syndication_cache_ts_key(): now_s}


def build_syndication_negative_cache_flag_field() -> Dict[str, bool]:
    """Build canonical negative-cache flag field map."""
    return {build_syndication_negative_cache_key(): True}


def build_syndication_cache_data_field(data: Any) -> Dict[str, Any]:
    """Build canonical positive-cache data field map."""
    return {build_syndication_cache_data_key(): data}


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
    return syndication_node_has_media_hints(node, media_hint_keys)


def syndication_node_has_media_hints(
    node: Dict[str, Any],
    media_hint_keys: Iterable[str],
) -> bool:
    """Return True when a syndication payload contains any media-hint key."""
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
        return format_syndication_truncated_text(text)
    return format_syndication_missing_text_fallback()


def format_syndication_truncated_text(text: str) -> str:
    """Return legacy-truncated syndication body text with ellipsis suffix."""
    return text[:3990] + "…"


def format_syndication_missing_text_fallback() -> str:
    """Return legacy fallback copy when syndication text is unavailable."""
    return "(Tweet text not available. If you want analysis, paste the text or add a screenshot.)"


def format_syndication_header_line(
    *,
    user: Any,
    created_at: Any,
    photos: Any,
    url: str,
) -> str:
    """Format syndication header line preserving legacy field access semantics."""
    username = format_syndication_header_username(user)
    media_hint = format_syndication_header_media_hint(photos)
    prefix = format_syndication_header_prefix(username)
    stamp = format_syndication_header_stamp(created_at)
    return format_syndication_header_compose(
        prefix=prefix,
        stamp=stamp,
        media_hint=media_hint,
        url=url,
    )


def format_syndication_header_username(user: Any) -> Any:
    """Resolve syndication header username from `screen_name` then `name`."""
    return user.get("screen_name") or user.get("name")


def format_syndication_header_media_hint(photos: Any) -> str:
    """Return legacy media-count suffix for syndication header lines."""
    return f" • media:{len(photos)}" if photos else ""


def format_syndication_header_prefix(username: Any) -> str:
    """Return legacy header prefix for resolved syndication username."""
    return f"@{username}" if username else "Tweet"


def format_syndication_header_stamp(created_at: Any) -> str:
    """Return legacy timestamp suffix for syndication header lines."""
    return f" • {created_at}" if created_at else ""


def format_syndication_header_compose(
    *,
    prefix: str,
    stamp: str,
    media_hint: str,
    url: str,
) -> str:
    """Compose a complete syndication header line from normalized fragments."""
    return f"{prefix}{stamp}{media_hint} → {url}"


def format_syndication_error_fallback(url: str, syn_data: Any) -> str:
    """Format fallback output when syndication payload formatting fails."""
    return f"Tweet → {url}\n{format_syndication_error_payload_repr(syn_data)}"


def format_syndication_error_payload_repr(syn_data: Any) -> str:
    """Return truncated string representation for syndication fallback payloads."""
    return str(syn_data)[:format_syndication_error_payload_max_chars()]


def format_syndication_error_payload_max_chars() -> int:
    """Return legacy max payload repr length for syndication fallback formatting."""
    return 4000


def extract_syndication_photo_urls(photos: Any) -> List[str]:
    """Extract photo URLs from syndication `photos` payload."""
    urls: List[str] = []
    for p in photos:
        append_syndication_photo_item_urls(urls=urls, photo=p)
    return urls


def append_syndication_photo_item_urls(*, urls: List[str], photo: Any) -> None:
    """Append URLs extracted from one syndication photo payload item."""
    item_urls = extract_syndication_photo_urls_from_item(photo)
    if item_urls:
        urls.extend(item_urls)


def extract_syndication_photo_url_from_dict(photo: Dict[str, Any]) -> Any:
    """Resolve canonical image URL from a syndication photo dict payload."""
    return photo.get("url") or photo.get("media_url_https") or photo.get("media_url")


def extract_syndication_photo_urls_from_item(photo: Any) -> List[str]:
    """Extract zero-or-more syndication photo URLs from one payload item."""
    if isinstance(photo, dict):
        img_url = extract_syndication_photo_url_from_dict(photo)
        if syndication_photo_url_is_usable(img_url):
            return [img_url]
        return []
    if isinstance(photo, str):
        return [photo]
    return []


def syndication_photo_url_is_usable(img_url: Any) -> bool:
    """Return True when extracted syndication photo URL can be appended."""
    return bool(img_url) and isinstance(img_url, str)


def x_syn_probe_budget_timeout_s(x_syn_timeout_s: float) -> float:
    """Compute bounded timeout budget for image/media probe calls."""
    return x_syn_timeout_with_offset_and_cap(x_syn_timeout_s, 1.0, 4.5)


def x_syn_quick_request_timeouts(x_syn_timeout_s: float) -> tuple[float, float, float]:
    """Compute bounded connect/read/total request budgets for quick probes."""
    return (
        x_syn_connect_read_timeout_s(x_syn_timeout_s),
        x_syn_connect_read_timeout_s(x_syn_timeout_s),
        x_syn_timeout_with_offset_and_cap(x_syn_timeout_s, 0.5, 3.5),
    )


def x_syn_connect_read_timeout_s(x_syn_timeout_s: float) -> float:
    """Compute bounded connect/read timeout shared by quick probe requests."""
    return x_syn_timeout_cap(x_syn_timeout_s, 3.0)


def x_syn_timeout_cap(value: float, cap: float) -> float:
    """Return bounded timeout value under the provided cap."""
    return min(value, cap)


def x_syn_timeout_with_offset_and_cap(value: float, offset: float, cap: float) -> float:
    """Return bounded timeout after applying an additive offset."""
    return x_syn_timeout_cap(float(value) + offset, cap)


def build_syndication_photo_payload(
    text: Optional[str], image_urls: List[str]
) -> Dict[str, Any]:
    """Build syndication-like payload consumed by the unified VL handler."""
    return {
        "text": text,
        "photos": build_syndication_photo_items(image_urls),
    }


def build_syndication_photo_items(image_urls: List[str]) -> List[Dict[str, str]]:
    """Build canonical list of photo item dicts for syndication payloads."""
    return [{"url": url} for url in image_urls]


def format_twitter_syndication_images_log_line(
    image_urls: List[str], *, msg_id: Optional[int] = None
) -> str:
    """Format canonical breadcrumb line for Twitter image-route detection."""
    first_host = resolve_first_image_host(image_urls)
    suffix = format_twitter_syndication_msg_suffix(msg_id)
    host_label = format_twitter_syndication_host_label(first_host)
    image_count = format_twitter_syndication_image_count(image_urls)
    return format_twitter_syndication_images_detail(
        image_count=image_count,
        host_label=host_label,
        suffix=suffix,
    )


def format_twitter_syndication_images_detail(
    *,
    image_count: int,
    host_label: str,
    suffix: str,
) -> str:
    """Compose canonical syndication image-route breadcrumb detail string."""
    return f"route.twitter.syndication | images={image_count} | {host_label}{suffix}"


def format_twitter_syndication_msg_suffix(msg_id: Optional[int]) -> str:
    """Return optional log suffix with message id for syndication breadcrumbs."""
    return f" | msg_id={msg_id}" if msg_id is not None else ""


def format_twitter_syndication_host_label(first_host: str) -> str:
    """Return host label used in syndication image-route breadcrumb lines."""
    return first_host or "n/a"


def format_twitter_syndication_image_count(image_urls: List[str]) -> int:
    """Return image count used in syndication image-route breadcrumb lines."""
    return len(image_urls)


def resolve_first_image_host(image_urls: List[str]) -> str:
    """Resolve first parsed host from image URL list preserving legacy failures."""
    first_host = ""
    try:
        first_image_url = resolve_first_image_url(image_urls)
        if first_image_url:
            first_host = parse_image_host(first_image_url)
    except Exception:
        first_host = ""
    return first_host


def parse_image_host(image_url: str) -> str:
    """Parse netloc host from an image URL string."""
    return urlparse(image_url).netloc


def resolve_first_image_url(image_urls: List[str]) -> str:
    """Resolve first image URL from list preserving legacy failure behavior."""
    return first_list_item_or_empty(image_urls)


def first_list_item_or_empty(items: List[str]) -> str:
    """Return first list item when present; otherwise empty string."""
    try:
        if items:
            return items[0]
    except Exception:
        return ""
    return ""


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
    return build_twitter_image_probe_result(status_id, image_urls)


def normalize_probed_image_urls(image_urls: Optional[List[str]]) -> List[str]:
    """Normalize probed image URL payload to an always-list value."""
    return probed_image_urls_or_empty(image_urls)


def probed_image_urls_or_empty(image_urls: Optional[List[str]]) -> List[str]:
    """Return probed image URLs when present, otherwise empty list."""
    return image_urls or []


def build_twitter_image_probe_result(
    status_id: str,
    image_urls: Optional[List[str]],
) -> Tuple[str, List[str]]:
    """Build normalized `(status_id, image_urls)` tuple for Twitter image probes."""
    return status_id, normalize_probed_image_urls(image_urls)


def extract_x_status_urls_from_text(
    text: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> List[str]:
    """Extract canonical X/Twitter status URLs from text preserving order."""
    urls = status_url_items_buffer()
    collect_status_urls_fail_open(
        items=urls,
        text=text,
        is_status_url=is_status_url,
        canonicalize_status_url=canonicalize_status_url,
    )
    return status_url_items_result(urls)


def collect_status_urls_into_items(
    items: List[str],
    text: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Collect status URLs into provided items list using collector flow."""
    collect_status_urls_from_candidates(
        items,
        text,
        is_status_url=is_status_url,
        canonicalize_status_url=canonicalize_status_url,
    )


def collect_status_urls_fail_open(
    *,
    items: List[str],
    text: str,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Collect status URLs while preserving fail-open behavior on parse errors."""
    try:
        collect_status_urls_into_items(
            items,
            text,
            is_status_url=is_status_url,
            canonicalize_status_url=canonicalize_status_url,
        )
    except Exception:
        pass


def status_url_items_result(urls: List[str]) -> List[str]:
    """Return collected status URL items for extraction call sites."""
    return urls


def status_url_items_buffer() -> List[str]:
    """Build mutable list buffer for extracted status URL collection."""
    return []


def collect_status_urls_from_candidates(
    items: List[str],
    text: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Collect canonical status URLs from candidate URLs found in text."""
    for raw in status_url_raw_candidates(text):
        append_status_url_candidate(
            items=items,
            raw=raw,
            is_status_url=is_status_url,
            canonicalize_status_url=canonicalize_status_url,
        )


def append_status_url_candidate(
    *,
    items: List[str],
    raw: str,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Append one raw status URL candidate when it matches status predicate."""
    append_status_url_if_match(
        items,
        raw,
        is_status_url=is_status_url,
        canonicalize_status_url=canonicalize_status_url,
    )


def status_url_raw_candidates(text: str) -> Iterable[str]:
    """Yield raw URL candidates normalized through candidate-raw-value helper."""
    for raw in iter_status_url_candidate_values(text):
        yield status_url_candidate_raw_value(raw)


def iter_status_url_candidate_values(text: str) -> Iterable[str]:
    """Yield candidate URL values for status collector normalization loops."""
    yield from status_url_candidate_values(text)


def status_url_candidate_values(text: str) -> Iterable[str]:
    """Yield raw status URL candidate values for collector loops."""
    yield from status_url_candidates(text)


def status_url_candidate_raw_value(raw: str) -> str:
    """Return raw candidate URL value used by status URL collector loop."""
    return raw


def status_url_candidates(text: str) -> Iterable[str]:
    """Yield status URL candidates from text for collector loops."""
    yield from iter_status_url_candidates_source(text)


def iter_status_url_candidates_source(text: str) -> Iterable[str]:
    """Yield candidates through source iterator wrapper for status URL extraction."""
    yield from status_url_candidates_source(text)


def status_url_candidates_source(text: str) -> Iterable[str]:
    """Yield source iterator for status URL candidates."""
    yield from status_url_candidates_iter(text)


def status_url_candidates_iter(text: str) -> Iterable[str]:
    """Yield iterator used by status URL candidate source helper."""
    yield from iter_status_url_candidates(text)


def x_url_extract_pattern() -> str:
    """Return canonical URL extraction regex pattern used for X/Twitter scans."""
    return x_url_extract_pattern_source()


def x_url_extract_pattern_source() -> str:
    """Return source pattern string used by X URL extraction helper."""
    return x_url_extract_pattern_value()


def x_url_extract_pattern_value() -> str:
    """Return literal regex pattern string used for URL extraction."""
    return r"https?://[^\s<>\"'\[\]{}|\\^`]+"


def iter_status_url_candidates(text: str) -> Iterable[str]:
    """Yield raw URL candidates for status extraction from one text blob."""
    yield from iter_status_url_candidates_from_text(text)


def iter_status_url_candidates_from_text(text: str) -> Iterable[str]:
    """Yield status URL candidates via from-text iterator wrapper."""
    yield from status_url_candidates_from_text(text)


def status_url_candidates_from_text(text: str) -> Iterable[str]:
    """Yield status URL candidates using canonical status regex source."""
    yield from iter_text_urls(text, url_re=status_url_candidates_regex_value())


def status_url_candidates_regex_value() -> Any:
    """Return regex value used by status URL candidates-from-text helper."""
    return status_url_candidates_regex_for_extraction()


def status_url_candidates_regex_for_extraction() -> Any:
    """Return candidate-extraction regex via status URL regex helper chain."""
    return status_url_candidates_regex()


def status_url_candidates_regex() -> Any:
    """Return regex used when extracting status URL candidates from text."""
    return status_url_candidates_regex_source()


def status_url_candidates_regex_source() -> Any:
    """Return underlying regex used for status URL candidate extraction."""
    return status_url_candidates_regex_value_source()


def status_url_candidates_regex_value_source() -> Any:
    """Return source regex value used by status candidates regex helper."""
    return status_url_extract_regex()


def status_url_extract_regex() -> Any:
    """Return compiled regex used for status URL candidate extraction."""
    return status_url_extract_regex_result(status_url_extract_regex_source_call())


def status_url_extract_regex_source_call() -> Any:
    """Return regex source call result used by status URL extraction helper."""
    return status_url_extract_regex_source_for_call()


def status_url_extract_regex_source_for_call() -> Any:
    """Return status URL extraction regex source used by source-call helper."""
    return status_url_extract_regex_source()


def status_url_extract_regex_result(regex: Any) -> Any:
    """Return status URL extraction regex result for call-site symmetry."""
    return status_url_extract_regex_result_value(regex)


def status_url_extract_regex_result_value(regex: Any) -> Any:
    """Return value used by status URL extraction regex result helper."""
    return regex


def status_url_extract_regex_source() -> Any:
    """Return underlying regex object used for status URL extraction."""
    return status_url_extract_regex_source_result(status_url_extract_regex_source_value())


def status_url_extract_regex_source_result(regex: Any) -> Any:
    """Return regex result value for status URL extraction source helper."""
    return regex


def status_url_extract_regex_source_value() -> Any:
    """Return value used by status URL extraction regex source helper."""
    return status_url_extract_regex_source_input()


def status_url_extract_regex_source_input() -> Any:
    """Return upstream regex source used by status regex source value helper."""
    return x_url_extract_regex_source()


def x_url_extract_regex_source() -> Any:
    """Return underlying regex object used for X URL extraction."""
    return x_url_extract_regex_source_result(x_url_extract_regex_source_value())


def x_url_extract_regex_source_result(regex: Any) -> Any:
    """Return regex result value for X URL extraction source helper."""
    return regex


def x_url_extract_regex_source_value() -> Any:
    """Return value used by X URL extraction regex source helper."""
    return x_url_extract_regex_source_input()


def x_url_extract_regex_source_input() -> Any:
    """Return upstream regex source used by X regex source value helper."""
    return x_url_extract_regex_for_source_input()


def x_url_extract_regex_for_source_input() -> Any:
    """Return X URL extraction regex used by source-input helper."""
    return x_url_extract_regex()


def x_url_extract_regex() -> Any:
    """Return compiled URL extraction regex used for broad URL harvesting."""
    return x_url_extract_compiled_regex(
        x_url_extract_regex_pattern(),
        flags=x_url_extract_compile_flags(),
    )


def x_url_extract_compiled_regex(pattern: str, *, flags: int) -> Any:
    """Return compiled X URL extraction regex from pattern and flags."""
    return compile_url_extract_regex(pattern, flags=flags)


def x_url_extract_regex_pattern() -> str:
    """Return pattern string used by compiled X URL extraction regex."""
    return x_url_extract_regex_pattern_source()


def x_url_extract_regex_pattern_source() -> str:
    """Return source pattern string used for X URL extraction regex."""
    return x_url_extract_regex_pattern_input()


def x_url_extract_regex_pattern_input() -> str:
    """Return upstream pattern string used by X regex pattern source helper."""
    return x_url_extract_pattern()


def x_url_extract_compile_flags() -> int:
    """Return compile-time flags used by X URL extraction regex."""
    return x_url_extract_compile_flags_source()


def x_url_extract_compile_flags_source() -> int:
    """Return source compile flags used by X URL extraction regex."""
    return x_url_extract_compile_flags_input()


def x_url_extract_compile_flags_input() -> int:
    """Return upstream compile flags used by X compile-flags source helper."""
    return x_url_extract_flags()


def compile_url_extract_regex(pattern: str, *, flags: int) -> Any:
    """Compile and return URL extraction regex from pattern and flags."""
    return compile_regex(
        compile_url_extract_pattern_argument(pattern),
        flags=compile_url_extract_flags_argument(flags),
    )


def compile_url_extract_pattern_argument(pattern: str) -> str:
    """Return normalized pattern argument for URL regex compilation."""
    return compile_url_extract_pattern_for_argument(pattern)


def compile_url_extract_pattern_for_argument(pattern: str) -> str:
    """Return URL-extract compile pattern via argument handoff helper."""
    return compile_url_extract_pattern_value(pattern)


def compile_url_extract_pattern_value(pattern: str) -> str:
    """Return value used by URL extract pattern argument helper."""
    return pattern


def compile_url_extract_flags_argument(flags: int) -> int:
    """Return normalized flags argument for URL regex compilation."""
    return compile_url_extract_flags_for_argument(flags)


def compile_url_extract_flags_for_argument(flags: int) -> int:
    """Return URL-extract compile flags via argument handoff helper."""
    return compile_url_extract_flags_value(flags)


def compile_url_extract_flags_value(flags: int) -> int:
    """Return value used by URL extract flags argument helper."""
    return flags


def compile_regex(pattern: str, *, flags: int) -> Any:
    """Compile regex from pattern and flags for shared extraction helpers."""
    return re.compile(
        compile_regex_pattern_argument(pattern),
        compile_regex_flags_argument(flags),
    )


def compile_regex_pattern_argument(pattern: str) -> str:
    """Return normalized pattern argument for generic regex compilation."""
    return compile_regex_pattern_for_argument(pattern)


def compile_regex_pattern_for_argument(pattern: str) -> str:
    """Return generic compile pattern via argument handoff helper."""
    return compile_regex_pattern_value(pattern)


def compile_regex_pattern_value(pattern: str) -> str:
    """Return value used by generic regex pattern argument helper."""
    return pattern


def compile_regex_flags_argument(flags: int) -> int:
    """Return normalized flags argument for generic regex compilation."""
    return compile_regex_flags_for_argument(flags)


def compile_regex_flags_for_argument(flags: int) -> int:
    """Return generic compile flags via argument handoff helper."""
    return compile_regex_flags_value(flags)


def compile_regex_flags_value(flags: int) -> int:
    """Return value used by generic regex flags argument helper."""
    return flags


def x_url_extract_flags() -> int:
    """Return regex flags used by URL extraction pattern compiler."""
    return x_url_extract_flags_source()


def x_url_extract_flags_source() -> int:
    """Return source regex flags used by URL extraction helpers."""
    return x_url_extract_flags_value()


def x_url_extract_flags_value() -> int:
    """Return literal regex flags value used for URL extraction."""
    return x_url_extract_flags_literal()


def x_url_extract_flags_literal() -> int:
    """Return literal regex flag constant for URL extraction."""
    return re.IGNORECASE


def extract_raw_urls_from_texts(texts: Iterable[str]) -> List[str]:
    """Extract raw URLs from multiple text blobs in-order with de-duplication."""
    raw_urls = raw_url_items_buffer()
    collect_raw_urls_fail_open(items=raw_urls, texts=texts)
    return raw_url_items_result(raw_urls)


def collect_raw_urls_fail_open(*, items: List[str], texts: Iterable[str]) -> None:
    """Collect raw URLs while preserving fail-open behavior on extraction errors."""
    try:
        url_re = raw_url_extract_regex()
        collect_raw_urls_into_items(items, texts, url_re=url_re)
    except Exception:
        pass


def collect_raw_urls_into_items(
    items: List[str],
    texts: Iterable[str],
    *,
    url_re: Any,
) -> None:
    """Collect raw URLs into provided items list using compiled regex."""
    collect_raw_urls_from_texts(items, texts, url_re=url_re)


def raw_url_items_result(items: List[str]) -> List[str]:
    """Return collected raw URL items for extraction call sites."""
    return items


def raw_url_extract_regex() -> Any:
    """Return compiled regex used for raw URL extraction flows."""
    return raw_url_extract_regex_source()


def raw_url_extract_regex_source() -> Any:
    """Return source regex used by raw URL extraction helper."""
    return raw_url_extract_regex_value()


def raw_url_extract_regex_value() -> Any:
    """Return upstream regex value used by raw URL extraction source helper."""
    return x_url_extract_regex()


def raw_url_items_buffer() -> List[str]:
    """Build mutable list buffer for extracted raw URL collection."""
    return raw_url_items_buffer_source()


def raw_url_items_buffer_source() -> List[str]:
    """Return source list buffer used by raw URL items helper."""
    return raw_url_items_buffer_value()


def raw_url_items_buffer_value() -> List[str]:
    """Return value used by raw URL items buffer source helper."""
    return []


def collect_raw_urls_from_texts(
    items: List[str],
    texts: Iterable[str],
    *,
    url_re: Any,
) -> None:
    """Collect de-duplicated raw URLs from multiple text blobs."""
    for t in raw_url_source_texts(texts):
        for u in raw_url_candidate_values(t, url_re=url_re):
            append_raw_url_if_present(items, raw_url_candidate_value(u))


def raw_url_candidate_values(text: str, *, url_re: Any) -> Iterable[str]:
    """Yield raw URL candidate values for collector loops."""
    yield from raw_url_candidate_values_source(text, url_re=url_re)


def raw_url_candidate_values_source(text: str, *, url_re: Any) -> Iterable[str]:
    """Yield source iterator for raw URL candidate values helper."""
    yield from raw_url_candidate_values_iter(text, url_re=url_re)


def raw_url_candidate_values_iter(text: str, *, url_re: Any) -> Iterable[str]:
    """Yield iterator used by raw URL candidate values source helper."""
    yield from iter_text_urls(text, url_re=url_re)


def raw_url_candidate_value(raw_url: str) -> str:
    """Return raw URL candidate value used by raw URL collector loop."""
    return raw_url_candidate_value_source(raw_url)


def raw_url_candidate_value_source(raw_url: str) -> str:
    """Return source raw URL candidate value for collector helper."""
    return raw_url_candidate_value_result(raw_url)


def raw_url_candidate_value_result(raw_url: str) -> str:
    """Return value used by raw URL candidate source helper."""
    return raw_url


def raw_url_source_texts(texts: Iterable[str]) -> Iterable[str]:
    """Yield source text blobs consumed by raw URL collection."""
    yield from iter_raw_url_source_texts(texts)


def iter_raw_url_source_texts(texts: Iterable[str]) -> Iterable[str]:
    """Yield raw URL source texts via iterator handoff helper."""
    yield from raw_url_source_texts_iter(texts)


def raw_url_source_texts_iter(texts: Iterable[str]) -> Iterable[str]:
    """Yield iterator source used by raw URL source text helper."""
    yield from texts


def iter_text_urls(text: str, *, url_re: Any) -> Iterable[str]:
    """Yield raw URL matches from one text blob using provided compiled regex."""
    for m in iter_text_url_matches(text, url_re=url_re):
        yield url_match_group_value(m)


def iter_text_url_matches(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield source URL match objects used by iter_text_urls helper."""
    yield from url_matches(text, url_re=url_re)


def url_matches(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield URL regex match objects for text URL extraction loops."""
    yield from iter_url_matches_for_url_matches(text, url_re=url_re)


def iter_url_matches_for_url_matches(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield URL matches via url-matches handoff helper."""
    yield from url_matches_source(text, url_re=url_re)


def url_matches_source(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield source URL matches used by url_matches helper."""
    yield from iter_url_matches_for_source(text, url_re=url_re)


def iter_url_matches_for_source(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield URL matches via url-matches-source handoff helper."""
    yield from url_matches_iter(text, url_re=url_re)


def url_matches_iter(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield iterator used by url_matches source helper."""
    yield from iter_url_matches(text, url_re=url_re)


def url_match_group_value(match: Any) -> str:
    """Return URL string value from one regex match object."""
    return match.group(url_match_group_index())


def url_match_group_index() -> int:
    """Return regex group index used for URL match value extraction."""
    return 0


def iter_url_matches(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield raw regex match objects for URL pattern scans in one text blob."""
    yield from iter_url_matches_source(text, url_re=url_re)


def iter_url_matches_source(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield source regex matches used by iter_url_matches helper."""
    yield from iter_url_matches_iter(text, url_re=url_re)


def iter_url_matches_iter(text: str, *, url_re: Any) -> Iterable[Any]:
    """Yield iterator used by iter_url_matches source helper."""
    yield from url_re_finditer(url_re, text)


def url_re_finditer(url_re: Any, text: Any) -> Iterable[Any]:
    """Yield regex matches for normalized text using provided compiled regex."""
    yield from iter_url_re_finditer_matches(url_re, text)


def iter_url_re_finditer_matches(url_re: Any, text: Any) -> Iterable[Any]:
    """Yield URL-regex matches via url_re_finditer handoff helper."""
    yield from url_re_finditer_source(url_re, text)


def url_re_finditer_source(url_re: Any, text: Any) -> Iterable[Any]:
    """Yield source regex matches used by url_re_finditer helper."""
    yield from url_re_finditer_iter(url_re, text)


def url_re_finditer_iter(url_re: Any, text: Any) -> Iterable[Any]:
    """Yield iterator used by url_re_finditer source helper."""
    yield from url_re.finditer(url_scan_text_for_finditer(text))


def url_scan_text_for_finditer(text: Any) -> str:
    """Return normalized text used specifically by url_re.finditer calls."""
    return url_scan_text(text)


def url_scan_text(text: Any) -> str:
    """Return normalized text input used for URL regex scans."""
    return url_scan_text_source(text)


def url_scan_text_source(text: Any) -> str:
    """Return source normalized text used by url_scan_text helper."""
    return url_scan_text_value(text)


def url_scan_text_value(text: Any) -> str:
    """Return value used by url_scan_text source helper."""
    return text or url_scan_text_fallback()


def url_scan_text_fallback() -> str:
    """Return fallback text used when URL scan input is falsey."""
    return ""


def filter_canonical_x_urls(
    raw_urls: Iterable[str],
    *,
    is_x_url: Callable[[str], bool],
    canonicalize_x_url: Callable[[str], str],
) -> List[str]:
    """Filter URL list to X/Twitter URLs and canonicalize with de-duplication."""
    out = canonical_x_url_items_buffer()
    for u in raw_urls:
        append_x_url_if_match(
            out,
            u,
            is_x_url=is_x_url,
            canonicalize_x_url=canonicalize_x_url,
        )
    return out


def canonical_x_url_items_buffer() -> List[str]:
    """Build mutable list buffer for canonicalized X URL collection."""
    return canonical_x_url_items_buffer_source()


def canonical_x_url_items_buffer_source() -> List[str]:
    """Return source list buffer used by canonical X URL helper."""
    return canonical_x_url_items_buffer_for_source()


def canonical_x_url_items_buffer_for_source() -> List[str]:
    """Return canonical X URL list buffer via source handoff helper."""
    return canonical_x_url_items_buffer_value()


def canonical_x_url_items_buffer_value() -> List[str]:
    """Return value used by canonical X URL items buffer source helper."""
    return []


def is_x_url_candidate(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return True when raw URL is an X/Twitter URL candidate by predicate."""
    return is_x_url_candidate_source(raw_url, is_x_url=is_x_url)


def is_x_url_candidate_source(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return source candidate predicate result for X URL helper."""
    return is_x_url_candidate_result(raw_url, is_x_url=is_x_url)


def is_x_url_candidate_result(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return value used by X URL candidate source helper."""
    return is_x_url_candidate_for_result(raw_url, is_x_url=is_x_url)


def is_x_url_candidate_for_result(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return X URL predicate result via candidate-result handoff helper."""
    return is_x_url(raw_url)


def append_x_url_if_match(
    items: List[str],
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
    canonicalize_x_url: Callable[[str], str],
) -> None:
    """Append canonical X/Twitter URL only when raw URL matches predicate."""
    if x_url_matches_predicate(raw_url, is_x_url=is_x_url):
        append_matched_x_url(items, raw_url, canonicalize_x_url=canonicalize_x_url)


def x_url_matches_predicate(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return whether raw URL matches X/Twitter predicate for append gating."""
    return x_url_matches_predicate_source(raw_url, is_x_url=is_x_url)


def x_url_matches_predicate_source(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return source predicate result used by X URL match helper."""
    return x_url_matches_predicate_result(raw_url, is_x_url=is_x_url)


def x_url_matches_predicate_result(
    raw_url: str,
    *,
    is_x_url: Callable[[str], bool],
) -> bool:
    """Return value used by X URL match predicate source helper."""
    return is_x_url_candidate(raw_url, is_x_url=is_x_url)


def append_unique_str(items: List[str], value: str) -> None:
    """Append value to list only when it is not already present."""
    if unique_value_missing(items, value):
        items.append(value)


def unique_value_missing(items: List[str], value: str) -> bool:
    """Return whether value is missing from list used for unique append."""
    return unique_value_missing_source(items, value)


def unique_value_missing_source(items: List[str], value: str) -> bool:
    """Return source membership check result for unique-value helper."""
    return unique_value_missing_result(items, value)


def unique_value_missing_result(items: List[str], value: str) -> bool:
    """Return value used by unique-value-missing source helper."""
    return value not in items


def append_raw_url_if_present(items: List[str], raw_url: str) -> None:
    """Append extracted raw URL only when non-empty and not yet present."""
    if raw_url_should_append(raw_url):
        append_unique_str(items, raw_url)


def raw_url_should_append(raw_url: str) -> bool:
    """Return whether raw URL should be appended by presence gating."""
    return raw_url_should_append_source(raw_url)


def raw_url_should_append_source(raw_url: str) -> bool:
    """Return source presence-gating result for raw URL append helper."""
    return raw_url_should_append_result(raw_url)


def raw_url_should_append_result(raw_url: str) -> bool:
    """Return value used by raw URL should-append source helper."""
    return raw_url_should_append_for_result(raw_url)


def raw_url_should_append_for_result(raw_url: str) -> bool:
    """Return raw URL append-gating result via result handoff helper."""
    return raw_url_is_present(raw_url)


def raw_url_is_present(raw_url: str) -> bool:
    """Return True when extracted raw URL is non-empty."""
    return raw_url_is_present_source(raw_url)


def raw_url_is_present_source(raw_url: str) -> bool:
    """Return source non-empty check result for raw URL presence helper."""
    return raw_url_is_present_result(raw_url)


def raw_url_is_present_result(raw_url: str) -> bool:
    """Return value used by raw URL is-present source helper."""
    return raw_url_is_present_for_result(raw_url)


def raw_url_is_present_for_result(raw_url: str) -> bool:
    """Return raw URL presence via is-present result handoff helper."""
    return bool(raw_url)


def append_canonical_x_url(
    items: List[str],
    url: str,
    *,
    canonicalize_x_url: Callable[[str], str],
) -> None:
    """Canonicalize URL then append uniquely to the target list."""
    append_canonicalized_value(
        items,
        canonical_x_raw_value(url),
        canonicalize=canonicalize_x_url,
    )


def canonical_x_raw_value(url: str) -> str:
    """Return raw X URL value used by canonical X append path."""
    return canonical_x_raw_value_source(url)


def canonical_x_raw_value_source(url: str) -> str:
    """Return source raw X URL value for canonical append helper."""
    return canonical_x_raw_value_result(url)


def canonical_x_raw_value_result(url: str) -> str:
    """Return value used by canonical X raw-value source helper."""
    return url


def append_canonical_status_url(
    items: List[str],
    raw_url: str,
    *,
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Canonicalize status URL then append uniquely to target list."""
    append_canonicalized_value(
        items,
        canonical_status_raw_value(raw_url),
        canonicalize=canonicalize_status_url,
    )


def canonical_status_raw_value(raw_url: str) -> str:
    """Return raw status URL value used by canonical status append path."""
    return canonical_status_raw_value_source(raw_url)


def canonical_status_raw_value_source(raw_url: str) -> str:
    """Return source raw status URL value for canonical append helper."""
    return canonical_status_raw_value_result(raw_url)


def canonical_status_raw_value_result(raw_url: str) -> str:
    """Return value used by canonical status raw-value source helper."""
    return raw_url


def append_canonicalized_value(
    items: List[str],
    raw_value: str,
    *,
    canonicalize: Callable[[str], str],
) -> None:
    """Canonicalize raw value then append uniquely to target list."""
    append_unique_str(
        items,
        canonicalized_value(raw_value, canonicalize=canonicalize),
    )


def canonicalized_value(
    raw_value: str,
    *,
    canonicalize: Callable[[str], str],
) -> str:
    """Return canonicalized form of one raw value."""
    return canonicalized_value_source(raw_value, canonicalize=canonicalize)


def canonicalized_value_source(
    raw_value: str,
    *,
    canonicalize: Callable[[str], str],
) -> str:
    """Return source canonicalized value for canonicalization helper."""
    return canonicalized_value_result(raw_value, canonicalize=canonicalize)


def canonicalized_value_result(
    raw_value: str,
    *,
    canonicalize: Callable[[str], str],
) -> str:
    """Return value used by canonicalized-value source helper."""
    return canonicalized_value_for_result(raw_value, canonicalize=canonicalize)


def canonicalized_value_for_result(
    raw_value: str,
    *,
    canonicalize: Callable[[str], str],
) -> str:
    """Return canonicalized value via canonicalized-result handoff helper."""
    return canonicalize(raw_value)


def append_status_url_if_match(
    items: List[str],
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Append canonical status URL only when raw URL matches status predicate."""
    if status_url_matches_predicate(raw_url, is_status_url=is_status_url):
        append_matched_status_url(
            items,
            raw_url,
            canonicalize_status_url=canonicalize_status_url,
        )


def status_url_matches_predicate(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return whether raw URL matches status predicate for append gating."""
    return status_url_matches_predicate_source(raw_url, is_status_url=is_status_url)


def status_url_matches_predicate_source(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return source predicate result used by status URL match helper."""
    return status_url_matches_predicate_result(raw_url, is_status_url=is_status_url)


def status_url_matches_predicate_result(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return value used by status URL match predicate source helper."""
    return is_status_url_candidate(raw_url, is_status_url=is_status_url)


def is_status_url_candidate(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return True when raw URL is a candidate status URL by predicate."""
    return is_status_url_candidate_source(raw_url, is_status_url=is_status_url)


def is_status_url_candidate_source(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return source candidate predicate result for status URL helper."""
    return is_status_url_candidate_result(raw_url, is_status_url=is_status_url)


def is_status_url_candidate_result(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return value used by status URL candidate source helper."""
    return is_status_url_candidate_for_result(
        raw_url,
        is_status_url=is_status_url,
    )


def is_status_url_candidate_for_result(
    raw_url: str,
    *,
    is_status_url: Callable[[str], bool],
) -> bool:
    """Return status URL predicate result via candidate-result handoff helper."""
    return is_status_url(raw_url)


def append_matched_status_url(
    items: List[str],
    raw_url: str,
    *,
    canonicalize_status_url: Callable[[str], str],
) -> None:
    """Append canonicalized status URL for a URL already known to match."""
    append_canonical_status_url(
        items,
        matched_status_raw_value(raw_url),
        canonicalize_status_url=canonicalize_status_url,
    )


def matched_status_raw_value(raw_url: str) -> str:
    """Return matched status raw URL value before canonical append."""
    return matched_status_raw_value_source(raw_url)


def matched_status_raw_value_source(raw_url: str) -> str:
    """Return source matched status raw URL value for append helper."""
    return matched_status_raw_value_result(raw_url)


def matched_status_raw_value_result(raw_url: str) -> str:
    """Return value used by matched-status raw-value source helper."""
    return matched_status_raw_value_for_result(raw_url)


def matched_status_raw_value_for_result(raw_url: str) -> str:
    """Return matched status raw URL via result handoff helper."""
    return raw_url


def append_matched_x_url(
    items: List[str],
    raw_url: str,
    *,
    canonicalize_x_url: Callable[[str], str],
) -> None:
    """Append canonicalized X/Twitter URL for a URL already known to match."""
    append_canonical_x_url(
        items,
        matched_x_raw_value(raw_url),
        canonicalize_x_url=canonicalize_x_url,
    )


def matched_x_raw_value(raw_url: str) -> str:
    """Return matched X raw URL value before canonical append."""
    return matched_x_raw_value_source(raw_url)


def matched_x_raw_value_source(raw_url: str) -> str:
    """Return source matched X raw URL value for append helper."""
    return matched_x_raw_value_result(raw_url)


def matched_x_raw_value_result(raw_url: str) -> str:
    """Return value used by matched-X raw-value source helper."""
    return matched_x_raw_value_for_result(raw_url)


def matched_x_raw_value_for_result(raw_url: str) -> str:
    """Return matched X raw URL via result handoff helper."""
    return raw_url
