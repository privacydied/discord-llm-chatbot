import re
from types import SimpleNamespace

import pytest

from bot.router_components.x_routing import (
    canonicalize_twitter_status_url,
    collect_x_candidate_urls,
    extract_fxtwitter_tweet_node,
    extract_x_api_primary_text,
    normalize_x_api_text,
    extract_x_api_first_item,
    extract_x_api_primary_tweet,
    extract_sparse_media_resolution,
    normalize_sparse_kind_value,
    normalize_sparse_url_value,
    normalize_sparse_images_value,
    extract_raw_urls_from_texts,
    extract_x_status_urls_from_text,
    filter_canonical_x_urls,
    canonical_x_url_items_buffer,
    canonical_x_url_items_buffer_source,
    canonical_x_url_items_buffer_for_source,
    canonical_x_url_items_buffer_value,
    x_url_matches_predicate,
    x_url_matches_predicate_source,
    x_url_matches_predicate_result,
    is_x_url_candidate,
    is_x_url_candidate_source,
    is_x_url_candidate_result,
    is_x_url_candidate_for_result,
    append_x_url_if_match,
    append_unique_str,
    unique_value_missing,
    unique_value_missing_source,
    unique_value_missing_result,
    append_raw_url_if_present,
    raw_url_should_append,
    raw_url_should_append_source,
    raw_url_should_append_result,
    raw_url_should_append_for_result,
    raw_url_is_present,
    raw_url_is_present_source,
    raw_url_is_present_result,
    raw_url_is_present_for_result,
    append_canonicalized_value,
    canonicalized_value,
    canonicalized_value_source,
    canonicalized_value_result,
    canonicalized_value_for_result,
    append_canonical_x_url,
    canonical_x_raw_value,
    canonical_x_raw_value_source,
    canonical_x_raw_value_result,
    append_canonical_status_url,
    canonical_status_raw_value,
    canonical_status_raw_value_source,
    canonical_status_raw_value_result,
    append_status_url_if_match,
    status_url_matches_predicate,
    status_url_matches_predicate_source,
    status_url_matches_predicate_result,
    is_status_url_candidate,
    is_status_url_candidate_source,
    is_status_url_candidate_result,
    is_status_url_candidate_for_result,
    append_matched_status_url,
    matched_status_raw_value,
    matched_status_raw_value_source,
    matched_status_raw_value_result,
    matched_status_raw_value_for_result,
    append_matched_x_url,
    matched_x_raw_value,
    matched_x_raw_value_source,
    matched_x_raw_value_result,
    matched_x_raw_value_for_result,
    is_tweet_media_url,
    is_blocked_tweet_media_path,
    is_poster_tweet_media_path,
    is_twitter_media_cdn,
    is_twitter_media_cdn_host,
    is_twitter_thumbnail_host,
    is_twitter_thumbnail_url,
    is_twitter_url,
    normalize_x_host,
    normalize_x_path,
    normalize_x_url,
    parse_twitter_status_id,
    is_unwrap_x_media_proxy_host,
    is_unwrap_x_media_candidate_url,
    unwrap_x_media_param_keys,
    classify_stt_error_reason,
    is_stt_hard_error,
    build_stt_fail_log_payload,
    build_stt_fail_detail,
    build_caption_only_fallback_log_payload,
    build_caption_only_fallback_detail,
    build_x_video_stt_error_result_payload,
    normalize_stt_error_value,
    resolve_caption_only_base_text,
    resolve_video_stt_error_base_text,
    normalize_base_text_value,
    syndication_article_has_blocks,
    has_non_empty_block_text,
    normalize_article_block_text,
    extract_x_article_text,
    truncate_x_article_text,
    syndication_needs_article_hydration,
    resolve_syndication_pointer_text,
    article_has_metadata_hints,
    has_news_action_type,
    is_tco_pointer_text,
    extract_syndication_base_text,
    extract_note_tweet_text,
    merge_syndication_base_with_article,
    base_text_contains_tco_link,
    extract_syndication_text,
    extract_syndication_article_text,
    build_x_text_miss_log_payload,
    build_x_text_miss_payload,
    build_x_text_resolve_payload,
    build_syndication_non_200_log_payload,
    build_syndication_non_200_metric_payload,
    build_syndication_fetch_failed_payload,
    build_x_text_canon_payload,
    build_oembed_text_payload,
    extract_oembed_payload_from_response,
    build_syndication_oembed_url,
    build_syndication_oembed_endpoint_url,
    build_syndication_oembed_key,
    build_syndication_oembed_host,
    build_syndication_cdn_host,
    build_syndication_base_url,
    build_syndication_user_agent_platform,
    build_syndication_fetch_user_agent,
    build_syndication_fetch_accept_language,
    build_syndication_region_locale,
    build_syndication_accept_language_primary_entry,
    build_syndication_accept_language_pair,
    build_syndication_lang_quality,
    build_syndication_accept_language_secondary_entry,
    build_syndication_fetch_referer,
    build_syndication_platform_host,
    build_syndication_fetch_accept,
    build_syndication_accept_primary_mimes,
    build_syndication_accept_json_mime,
    build_syndication_accept_text_mime,
    build_syndication_accept_text_quality,
    build_syndication_accept_text_entry,
    build_syndication_accept_any_mime,
    build_syndication_accept_any_quality,
    build_syndication_accept_any_entry,
    build_syndication_lang,
    build_syndication_dnt_value,
    build_syndication_omit_script_value,
    build_syndication_hide_thread_value,
    build_syndication_bool_true_value,
    build_syndication_bool_false_value,
    build_syndication_fetch_headers,
    build_syndication_fetch_headers_base,
    build_syndication_fetch_header_map,
    build_syndication_fetch_header_keys,
    build_syndication_fetch_header_values,
    build_syndication_dnt_key,
    build_syndication_id_key,
    build_syndication_lang_key,
    build_syndication_fetch_params_core,
    build_syndication_fetch_params_core_map,
    build_syndication_fetch_params_with_optional_dnt,
    maybe_add_syndication_dnt_param,
    build_syndication_fetch_params,
    build_syndication_fetch_params_variants_list,
    build_syndication_fetch_params_variants,
    build_syndication_widgets_params_variant,
    build_syndication_widgets_params_variant_with_dnt,
    build_syndication_tweet_result_params_variant,
    build_syndication_widgets_endpoint,
    build_syndication_tweet_result_endpoint,
    build_syndication_widgets_tweet_path,
    build_syndication_tweet_result_path,
    build_syndication_cache_ts_key,
    build_syndication_negative_cache_key,
    build_syndication_cache_data_key,
    build_syndication_negative_cache_hit_label,
    build_syndication_data_cache_hit_label,
    build_syndication_oembed_params,
    build_syndication_oembed_params_bundle,
    build_syndication_oembed_params_core,
    build_syndication_oembed_params_core_map,
    build_syndication_oembed_url_key,
    build_syndication_oembed_host_for_flag,
    build_syndication_oembed_status_url,
    build_syndication_status_url,
    build_syndication_status_url_prefix,
    build_syndication_status_path,
    build_syndication_oembed_options,
    build_syndication_oembed_options_map,
    build_syndication_oembed_options_map_from_pairs,
    build_syndication_oembed_option_keys,
    build_syndication_oembed_option_values,
    build_syndication_oembed_dnt_key,
    build_syndication_oembed_omit_script_key,
    build_syndication_oembed_hide_thread_key,
    build_syndication_oembed_hosts,
    build_syndication_oembed_hosts_tuple,
    build_syndication_twitter_host,
    build_syndication_x_host,
    is_syndication_x_host,
    is_syndication_twitter_host,
    build_syndication_oembed_metric_endpoint,
    build_syndication_oembed_metric_default_endpoint,
    build_syndication_oembed_x_metric_endpoint,
    build_syndication_oembed_fallback_item,
    build_syndication_oembed_fallback_items_list,
    build_syndication_oembed_fallback_params,
    build_syndication_oembed_fallback_plan,
    build_syndication_oembed_fallback_plan_components,
    build_syndication_oembed_fallback_plan_tuple,
    build_syndication_fetch_plan_components,
    build_syndication_fetch_plan_values,
    build_syndication_fetch_plan,
    build_syndication_fetch_plan_tuple,
    build_syndication_metric_endpoint_key,
    build_syndication_metric_payload_map,
    build_syndication_fetch_metric_payload,
    syndication_cache_ttl_s,
    syndication_negative_cache_ttl_value,
    syndication_cache_timestamp_value,
    syndication_cache_is_fresh,
    build_syndication_cache_hit_label,
    classify_syndication_cache_hit,
    build_syndication_negative_cache_entry,
    build_syndication_cache_entry,
    build_syndication_cache_timestamp_field,
    build_syndication_negative_cache_flag_field,
    build_syndication_cache_data_field,
    build_syndication_endpoint_url,
    build_syndication_endpoint_suffix,
    build_syndication_negative_cache_ttl_cap_s,
    extract_oembed_html_text,
    syndication_has_usable_payload,
    syndication_node_has_media_hints,
    syndication_media_hint_keys,
    format_syndication_body_text,
    format_syndication_truncated_text,
    format_syndication_missing_text_fallback,
    format_syndication_header_line,
    format_syndication_header_username,
    format_syndication_header_media_hint,
    format_syndication_header_prefix,
    format_syndication_header_stamp,
    format_syndication_header_compose,
    format_syndication_error_fallback,
    format_syndication_error_payload_repr,
    format_syndication_error_payload_max_chars,
    extract_syndication_photo_urls,
    append_syndication_photo_item_urls,
    extract_syndication_photo_url_from_dict,
    extract_syndication_photo_urls_from_item,
    syndication_photo_url_is_usable,
    resolve_twitter_status_id,
    is_twitter_status_url,
    resolve_twitter_status_parser,
    stt_result_has_transcription,
    stt_transcription_value_is_present,
    unwrap_x_media_url,
    x_syn_probe_budget_timeout_s,
    x_syn_connect_read_timeout_s,
    x_syn_timeout_cap,
    x_syn_timeout_with_offset_and_cap,
    x_syn_quick_request_timeouts,
    build_syndication_photo_payload,
    build_syndication_photo_items,
    format_twitter_syndication_images_log_line,
    format_twitter_syndication_images_detail,
    format_twitter_syndication_msg_suffix,
    format_twitter_syndication_host_label,
    format_twitter_syndication_image_count,
    resolve_first_image_host,
    parse_image_host,
    resolve_first_image_url,
    first_list_item_or_empty,
    probed_image_urls_or_empty,
    normalize_probed_image_urls,
    build_twitter_image_probe_result,
    resolve_and_probe_twitter_images,
    status_url_items_buffer,
    status_url_items_result,
    collect_status_urls_into_items,
    collect_status_urls_fail_open,
    collect_status_urls_from_candidates,
    append_status_url_candidate,
    status_url_raw_candidates,
    iter_status_url_candidate_values,
    status_url_candidate_values,
    iter_status_url_candidates_source,
    status_url_candidate_raw_value,
    status_url_candidates,
    status_url_candidates_source,
    status_url_candidates_iter,
    x_url_extract_pattern,
    x_url_extract_pattern_source,
    x_url_extract_pattern_value,
    x_url_extract_flags,
    x_url_extract_flags_source,
    x_url_extract_flags_value,
    x_url_extract_flags_literal,
    compile_url_extract_regex,
    status_url_extract_regex_source,
    status_url_extract_regex_source_value,
    status_url_extract_regex_source_input,
    x_url_extract_regex_source,
    x_url_extract_regex_source_result,
    x_url_extract_regex_source_value,
    x_url_extract_regex_source_input,
    x_url_extract_regex_for_source_input,
    status_url_extract_regex_result,
    status_url_extract_regex_result_value,
    status_url_extract_regex_source_call,
    status_url_extract_regex_source_for_call,
    status_url_extract_regex,
    status_url_extract_regex_source_result,
    status_url_candidates_regex_value,
    status_url_candidates_regex_for_extraction,
    status_url_candidates_regex_value_source,
    status_url_candidates_regex_source,
    status_url_candidates_regex,
    status_url_candidates_from_text,
    iter_status_url_candidates,
    iter_status_url_candidates_from_text,
    x_url_extract_regex,
    x_url_extract_compiled_regex,
    x_url_extract_regex_pattern_input,
    x_url_extract_regex_pattern_source,
    x_url_extract_regex_pattern,
    x_url_extract_compile_flags_input,
    x_url_extract_compile_flags_source,
    x_url_extract_compile_flags,
    raw_url_extract_regex,
    raw_url_extract_regex_source,
    raw_url_extract_regex_value,
    compile_url_extract_pattern_argument,
    compile_url_extract_pattern_for_argument,
    compile_url_extract_pattern_value,
    compile_url_extract_flags_argument,
    compile_url_extract_flags_for_argument,
    compile_url_extract_flags_value,
    compile_regex,
    compile_regex_pattern_argument,
    compile_regex_pattern_for_argument,
    compile_regex_pattern_value,
    compile_regex_flags_argument,
    compile_regex_flags_for_argument,
    compile_regex_flags_value,
    collect_raw_urls_into_items,
    collect_raw_urls_fail_open,
    raw_url_items_result,
    raw_url_items_buffer,
    raw_url_items_buffer_source,
    raw_url_items_buffer_value,
    raw_url_candidate_values,
    raw_url_candidate_values_source,
    raw_url_candidate_values_iter,
    raw_url_candidate_value,
    raw_url_candidate_value_source,
    raw_url_candidate_value_result,
    raw_url_source_texts,
    iter_raw_url_source_texts,
    raw_url_source_texts_iter,
    collect_raw_urls_from_texts,
    iter_url_matches_for_source,
    url_matches_source,
    url_matches_iter,
    url_matches,
    iter_url_matches_for_url_matches,
    iter_url_matches_source,
    iter_url_matches_iter,
    iter_url_matches,
    url_re_finditer_source,
    url_re_finditer_iter,
    url_re_finditer,
    url_scan_text_for_finditer,
    iter_url_re_finditer_matches,
    url_scan_text_source,
    url_scan_text_value,
    url_scan_text,
    url_scan_text_fallback,
    url_match_group_index,
    url_match_group_value,
    iter_text_url_matches,
    iter_text_urls,
)


def test_is_twitter_url_and_status_id_parsing() -> None:
    url = "https://x.com/user/status/2022790791047823773?s=20"
    assert is_twitter_url(url) is True
    assert parse_twitter_status_id(url) == "2022790791047823773"
    assert is_twitter_url("https://example.com/page") is False


def test_canonicalize_and_normalize_x_urls() -> None:
    src = "https://twitter.com/user/status/2022790791047823773?s=20"
    assert (
        canonicalize_twitter_status_url(src)
        == "https://x.com/i/status/2022790791047823773"
    )
    assert normalize_x_url("https://mobile.twitter.com/user/status/1?s=20#frag") == (
        "https://x.com/user/status/1"
    )


def test_normalize_x_path() -> None:
    assert normalize_x_path("/user/status/1/") == "/user/status/1"
    assert normalize_x_path("/user/status/1") == "/user/status/1"


def test_normalize_x_host() -> None:
    assert normalize_x_host("mobile.twitter.com") == "x.com"
    assert normalize_x_host("www.fxtwitter.com") == "x.com"
    assert normalize_x_host("example.com") == "example.com"


def test_is_twitter_thumbnail_host() -> None:
    assert is_twitter_thumbnail_host("pbs.twimg.com")
    assert is_twitter_thumbnail_host("pbs-2.twimg.com")
    assert not is_twitter_thumbnail_host("video.twimg.com")
    assert not is_twitter_thumbnail_host("example.com")


def test_is_twitter_thumbnail_url() -> None:
    assert is_twitter_thumbnail_url("https://pbs.twimg.com/media/abc.jpg")
    assert is_twitter_thumbnail_url("https://pbs-1.twimg.com/media/abc.jpg")
    assert not is_twitter_thumbnail_url("https://video.twimg.com/ext_tw_video/abc.mp4")
    assert not is_twitter_thumbnail_url("not-a-url")


def test_is_twitter_media_cdn_host() -> None:
    assert is_twitter_media_cdn_host("video.twimg.com")
    assert is_twitter_media_cdn_host("pbs.twimg.com")
    assert not is_twitter_media_cdn_host("example.com")


def test_collect_x_candidate_urls_for_source_types() -> None:
    url_item = SimpleNamespace(source_type="url", payload="https://x.com/u/status/1")
    assert collect_x_candidate_urls(url_item) == ["https://x.com/u/status/1"]

    embed_item = SimpleNamespace(
        source_type="embed",
        payload=SimpleNamespace(
            url="https://x.com/u/status/2",
            video=SimpleNamespace(url="https://video.twimg.com/ext_tw_video/abc"),
            image=SimpleNamespace(url="https://pbs.twimg.com/media/xyz.jpg"),
            thumbnail=None,
        ),
    )
    embed_urls = collect_x_candidate_urls(embed_item)
    assert "https://x.com/u/status/2" in embed_urls
    assert "https://video.twimg.com/ext_tw_video/abc" in embed_urls
    assert "https://pbs.twimg.com/media/xyz.jpg" in embed_urls

    att_item = SimpleNamespace(
        source_type="attachment",
        payload=SimpleNamespace(
            url="https://video.twimg.com/ext_tw_video/att.mp4",
            proxy_url="https://cdn.discordapp.com/proxy",
        ),
    )
    att_urls = collect_x_candidate_urls(att_item)
    assert "https://video.twimg.com/ext_tw_video/att.mp4" in att_urls
    assert "https://cdn.discordapp.com/proxy" in att_urls


def test_twitter_host_and_media_path_helpers() -> None:
    assert is_twitter_thumbnail_url("https://pbs.twimg.com/media/a.jpg") is True
    assert is_twitter_media_cdn("https://video.twimg.com/ext_tw_video/abc") is True
    assert is_tweet_media_url("https://pbs.twimg.com/media/abc123.jpg") is True
    assert (
        is_tweet_media_url("https://pbs.twimg.com/profile_images/123/avatar.jpg")
        is False
    )
    assert (
        is_tweet_media_url("https://pbs.twimg.com/ext_tw_video_thumb/123/pu/img.jpg")
        is False
    )


def test_is_blocked_tweet_media_path() -> None:
    assert is_blocked_tweet_media_path("/profile_images/123/avatar.jpg")
    assert is_blocked_tweet_media_path("/emoji/v2/72x72/1f525.png")
    assert not is_blocked_tweet_media_path("/media/abc123.jpg")


def test_is_poster_tweet_media_path() -> None:
    assert is_poster_tweet_media_path("/ext_tw_video_thumb/123/pu/img.jpg")
    assert is_poster_tweet_media_path("/tweet_video_thumb/123/img.jpg")
    assert not is_poster_tweet_media_path("/media/abc123.jpg")


def test_extract_x_status_urls_from_text() -> None:
    text = (
        "a https://x.com/u/status/1 and b "
        "https://twitter.com/u/status/2?s=20 and duplicate https://x.com/u/status/1"
    )
    urls = extract_x_status_urls_from_text(
        text,
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert urls == ["https://x.com/u/status/1", "https://x.com/u/status/2"]


def test_collect_status_urls_from_candidates() -> None:
    items = []
    collect_status_urls_from_candidates(
        items,
        "a https://x.com/u/status/1 and https://example.com b https://twitter.com/u/status/2?s=20",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1", "https://x.com/u/status/2"]


def test_append_status_url_candidate() -> None:
    items: List[str] = []
    append_status_url_candidate(
        items=items,
        raw="https://x.com/u/status/1",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_status_url_candidate(
        items=items,
        raw="https://example.com",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1"]


def test_collect_status_urls_into_items_delegates_collection() -> None:
    items = []
    collect_status_urls_into_items(
        items,
        "a https://x.com/u/status/1 and https://twitter.com/u/status/2?s=20",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1", "https://x.com/u/status/2"]


def test_status_url_items_buffer_starts_empty() -> None:
    assert status_url_items_buffer() == []


def test_status_url_items_result_identity() -> None:
    assert status_url_items_result(["u1", "u2"]) == ["u1", "u2"]


def test_status_url_candidate_raw_value_identity() -> None:
    assert status_url_candidate_raw_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_status_url_candidate_values_delegates_candidates() -> None:
    urls = list(status_url_candidate_values("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_iter_status_url_candidates_source_delegates_source() -> None:
    urls = list(iter_status_url_candidates_source("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_status_url_raw_candidates_normalizes_raw_values() -> None:
    urls = list(status_url_raw_candidates("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_iter_status_url_candidate_values_delegates_values() -> None:
    urls = list(iter_status_url_candidate_values("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_x_url_extract_pattern_constant() -> None:
    assert x_url_extract_pattern() == r"https?://[^\s<>\"'\[\]{}|\\^`]+"


def test_x_url_extract_pattern_source_constant() -> None:
    assert x_url_extract_pattern_source() == r"https?://[^\s<>\"'\[\]{}|\\^`]+"


def test_x_url_extract_pattern_value_constant() -> None:
    assert x_url_extract_pattern_value() == r"https?://[^\s<>\"'\[\]{}|\\^`]+"


def test_x_url_extract_flags_constant() -> None:
    assert x_url_extract_flags() == re.IGNORECASE


def test_x_url_extract_flags_source_constant() -> None:
    assert x_url_extract_flags_source() == re.IGNORECASE


def test_x_url_extract_flags_value_constant() -> None:
    assert x_url_extract_flags_value() == re.IGNORECASE


def test_x_url_extract_flags_literal_constant() -> None:
    assert x_url_extract_flags_literal() == re.IGNORECASE


def test_compile_url_extract_regex_uses_pattern_and_flags() -> None:
    compiled = compile_url_extract_regex(r"https?://x\\.com", flags=re.IGNORECASE)
    assert compiled.pattern == r"https?://x\\.com"
    assert compiled.flags & re.IGNORECASE


def test_status_url_extract_regex_matches_x_url_extract_regex() -> None:
    assert status_url_extract_regex().pattern == x_url_extract_regex().pattern
    assert status_url_extract_regex().flags == x_url_extract_regex().flags


def test_status_url_extract_regex_source_matches_x_url_extract_regex() -> None:
    assert status_url_extract_regex_source().pattern == x_url_extract_regex().pattern
    assert status_url_extract_regex_source().flags == x_url_extract_regex().flags


def test_status_url_extract_regex_source_value_matches_x_url_extract_regex() -> None:
    assert status_url_extract_regex_source_value().pattern == x_url_extract_regex().pattern
    assert status_url_extract_regex_source_value().flags == x_url_extract_regex().flags


def test_status_url_extract_regex_source_input_matches_x_url_extract_regex() -> None:
    assert status_url_extract_regex_source_input().pattern == x_url_extract_regex().pattern
    assert status_url_extract_regex_source_input().flags == x_url_extract_regex().flags


def test_status_url_extract_regex_result_identity() -> None:
    regex = status_url_extract_regex_source()
    assert status_url_extract_regex_result(regex) is regex


def test_status_url_extract_regex_result_value_identity() -> None:
    regex = status_url_extract_regex_source()
    assert status_url_extract_regex_result_value(regex) is regex


def test_status_url_extract_regex_source_call_matches_source() -> None:
    assert status_url_extract_regex_source_call().pattern == status_url_extract_regex_source().pattern
    assert status_url_extract_regex_source_call().flags == status_url_extract_regex_source().flags


def test_status_url_extract_regex_source_for_call_matches_source() -> None:
    assert (
        status_url_extract_regex_source_for_call().pattern
        == status_url_extract_regex_source().pattern
    )
    assert (
        status_url_extract_regex_source_for_call().flags
        == status_url_extract_regex_source().flags
    )


def test_status_url_extract_regex_source_result_identity() -> None:
    regex = status_url_extract_regex_source()
    assert status_url_extract_regex_source_result(regex) is regex


def test_x_url_extract_regex_source_matches_x_url_extract_regex() -> None:
    assert x_url_extract_regex_source().pattern == x_url_extract_regex().pattern
    assert x_url_extract_regex_source().flags == x_url_extract_regex().flags


def test_x_url_extract_regex_source_result_identity() -> None:
    regex = x_url_extract_regex_source()
    assert x_url_extract_regex_source_result(regex) is regex


def test_x_url_extract_regex_source_value_matches_x_url_extract_regex() -> None:
    assert x_url_extract_regex_source_value().pattern == x_url_extract_regex().pattern
    assert x_url_extract_regex_source_value().flags == x_url_extract_regex().flags


def test_x_url_extract_regex_source_input_matches_x_url_extract_regex() -> None:
    assert x_url_extract_regex_source_input().pattern == x_url_extract_regex().pattern
    assert x_url_extract_regex_source_input().flags == x_url_extract_regex().flags


def test_x_url_extract_regex_for_source_input_matches_x_url_extract_regex() -> None:
    assert (
        x_url_extract_regex_for_source_input().pattern
        == x_url_extract_regex().pattern
    )
    assert (
        x_url_extract_regex_for_source_input().flags
        == x_url_extract_regex().flags
    )


def test_status_url_candidates_regex_matches_status_url_extract_regex() -> None:
    assert status_url_candidates_regex().pattern == status_url_extract_regex().pattern
    assert status_url_candidates_regex().flags == status_url_extract_regex().flags


def test_status_url_candidates_regex_value_matches_status_url_candidates_regex() -> None:
    assert status_url_candidates_regex_value().pattern == status_url_candidates_regex().pattern
    assert status_url_candidates_regex_value().flags == status_url_candidates_regex().flags


def test_status_url_candidates_regex_for_extraction_matches_status_candidates_regex() -> None:
    assert (
        status_url_candidates_regex_for_extraction().pattern
        == status_url_candidates_regex().pattern
    )
    assert (
        status_url_candidates_regex_for_extraction().flags
        == status_url_candidates_regex().flags
    )


def test_status_url_candidates_regex_value_source_matches_status_url_extract_regex() -> None:
    assert status_url_candidates_regex_value_source().pattern == status_url_extract_regex().pattern
    assert status_url_candidates_regex_value_source().flags == status_url_extract_regex().flags


def test_status_url_candidates_regex_source_matches_status_url_extract_regex() -> None:
    assert status_url_candidates_regex_source().pattern == status_url_extract_regex().pattern
    assert status_url_candidates_regex_source().flags == status_url_extract_regex().flags


def test_raw_url_extract_regex_source_matches_x_url_extract_regex() -> None:
    assert raw_url_extract_regex_source().pattern == x_url_extract_regex().pattern
    assert raw_url_extract_regex_source().flags == x_url_extract_regex().flags


def test_raw_url_extract_regex_value_matches_x_url_extract_regex() -> None:
    assert raw_url_extract_regex_value().pattern == x_url_extract_regex().pattern
    assert raw_url_extract_regex_value().flags == x_url_extract_regex().flags


def test_iter_status_url_candidates_yields_raw_matches() -> None:
    urls = list(iter_status_url_candidates("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_iter_status_url_candidates_from_text_yields_raw_matches() -> None:
    urls = list(iter_status_url_candidates_from_text("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_status_url_candidates_from_text_yields_raw_matches() -> None:
    urls = list(status_url_candidates_from_text("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_status_url_candidates_delegates_iterator() -> None:
    urls = list(status_url_candidates("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_status_url_candidates_source_delegates_iterator() -> None:
    urls = list(status_url_candidates_source("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_status_url_candidates_iter_delegates_iterator() -> None:
    urls = list(status_url_candidates_iter("a https://x.com/u/status/1 b"))
    assert urls == ["https://x.com/u/status/1"]


def test_x_url_extract_regex_matches_expected_urls() -> None:
    matches = [m.group(0) for m in x_url_extract_regex().finditer("x https://x.com/u/status/1 y")]
    assert matches == ["https://x.com/u/status/1"]


def test_x_url_extract_compiled_regex_uses_pattern_and_flags() -> None:
    compiled = x_url_extract_compiled_regex(r"https?://x\\.com", flags=re.IGNORECASE)
    assert compiled.pattern == r"https?://x\\.com"
    assert compiled.flags & re.IGNORECASE


def test_x_url_extract_regex_pattern_matches_x_url_extract_pattern() -> None:
    assert x_url_extract_regex_pattern() == x_url_extract_pattern()


def test_x_url_extract_regex_pattern_source_matches_x_url_extract_pattern() -> None:
    assert x_url_extract_regex_pattern_source() == x_url_extract_pattern()


def test_x_url_extract_regex_pattern_input_matches_x_url_extract_pattern() -> None:
    assert x_url_extract_regex_pattern_input() == x_url_extract_pattern()


def test_x_url_extract_compile_flags_matches_x_url_extract_flags() -> None:
    assert x_url_extract_compile_flags() == x_url_extract_flags()


def test_x_url_extract_compile_flags_source_matches_x_url_extract_flags() -> None:
    assert x_url_extract_compile_flags_source() == x_url_extract_flags()


def test_x_url_extract_compile_flags_input_matches_x_url_extract_flags() -> None:
    assert x_url_extract_compile_flags_input() == x_url_extract_flags()


def test_raw_url_extract_regex_matches_x_url_extract_regex() -> None:
    assert raw_url_extract_regex().pattern == x_url_extract_regex().pattern
    assert raw_url_extract_regex().flags == x_url_extract_regex().flags


def test_compile_regex_uses_pattern_and_flags() -> None:
    compiled = compile_regex(r"https?://x\\.com", flags=re.IGNORECASE)
    assert compiled.pattern == r"https?://x\\.com"
    assert compiled.flags & re.IGNORECASE


def test_compile_regex_pattern_argument_identity() -> None:
    assert compile_regex_pattern_argument(r"https?://x\\.com") == r"https?://x\\.com"


def test_compile_regex_pattern_for_argument_identity() -> None:
    assert compile_regex_pattern_for_argument(r"https?://x\\.com") == r"https?://x\\.com"


def test_compile_regex_pattern_value_identity() -> None:
    assert compile_regex_pattern_value(r"https?://x\\.com") == r"https?://x\\.com"


def test_compile_regex_flags_argument_identity() -> None:
    assert compile_regex_flags_argument(re.IGNORECASE) == re.IGNORECASE


def test_compile_regex_flags_for_argument_identity() -> None:
    assert compile_regex_flags_for_argument(re.IGNORECASE) == re.IGNORECASE


def test_compile_regex_flags_value_identity() -> None:
    assert compile_regex_flags_value(re.IGNORECASE) == re.IGNORECASE


def test_compile_url_extract_flags_argument_identity() -> None:
    assert compile_url_extract_flags_argument(re.IGNORECASE) == re.IGNORECASE


def test_compile_url_extract_flags_for_argument_identity() -> None:
    assert compile_url_extract_flags_for_argument(re.IGNORECASE) == re.IGNORECASE


def test_compile_url_extract_flags_value_identity() -> None:
    assert compile_url_extract_flags_value(re.IGNORECASE) == re.IGNORECASE


def test_compile_url_extract_pattern_argument_identity() -> None:
    assert compile_url_extract_pattern_argument(r"https?://x\\.com") == r"https?://x\\.com"


def test_compile_url_extract_pattern_for_argument_identity() -> None:
    assert compile_url_extract_pattern_for_argument(r"https?://x\\.com") == r"https?://x\\.com"


def test_compile_url_extract_pattern_value_identity() -> None:
    assert compile_url_extract_pattern_value(r"https?://x\\.com") == r"https?://x\\.com"


def test_collect_raw_urls_into_items_delegates_collection() -> None:
    items = []
    collect_raw_urls_into_items(
        items,
        ["a https://x.com/u/status/1 b"],
        url_re=x_url_extract_regex(),
    )
    assert items == ["https://x.com/u/status/1"]


def test_collect_raw_urls_fail_open_variants() -> None:
    items: List[str] = []
    collect_raw_urls_fail_open(
        items=items,
        texts=["a https://x.com/u/status/1 b"],
    )
    assert items == ["https://x.com/u/status/1"]

    class _BadTexts:
        def __iter__(self):
            raise RuntimeError("boom")

    items_err: List[str] = []
    collect_raw_urls_fail_open(items=items_err, texts=_BadTexts())
    assert items_err == []


def test_collect_status_urls_fail_open_variants() -> None:
    items: List[str] = []
    collect_status_urls_fail_open(
        items=items,
        text="https://x.com/u/status/1",
        is_status_url=lambda _url: True,
        canonicalize_status_url=lambda url: url,
    )
    assert items == ["https://x.com/u/status/1"]

    # Fail-open behavior must swallow collector exceptions.
    items_err: List[str] = []
    collect_status_urls_fail_open(
        items=items_err,
        text="x",
        is_status_url=lambda _url: (_ for _ in ()).throw(RuntimeError("boom")),
        canonicalize_status_url=lambda url: url,
    )
    assert items_err == []


def test_raw_url_items_result_identity() -> None:
    assert raw_url_items_result(["u1", "u2"]) == ["u1", "u2"]


def test_raw_url_candidate_value_identity() -> None:
    assert raw_url_candidate_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_raw_url_candidate_value_source_identity() -> None:
    assert raw_url_candidate_value_source("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_raw_url_candidate_value_result_identity() -> None:
    assert raw_url_candidate_value_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_raw_url_candidate_values_delegates_iter_text_urls() -> None:
    urls = list(raw_url_candidate_values("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert urls == ["https://x.com/u/status/1"]


def test_raw_url_candidate_values_source_delegates_iter_text_urls() -> None:
    urls = list(raw_url_candidate_values_source("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert urls == ["https://x.com/u/status/1"]


def test_raw_url_candidate_values_iter_delegates_iter_text_urls() -> None:
    urls = list(raw_url_candidate_values_iter("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert urls == ["https://x.com/u/status/1"]


def test_raw_url_items_buffer_starts_empty() -> None:
    assert raw_url_items_buffer() == []


def test_raw_url_items_buffer_source_starts_empty() -> None:
    assert raw_url_items_buffer_source() == []


def test_raw_url_items_buffer_value_starts_empty() -> None:
    assert raw_url_items_buffer_value() == []


def test_raw_url_source_texts_delegates_iterable() -> None:
    assert list(raw_url_source_texts(["a", "b"])) == ["a", "b"]


def test_iter_raw_url_source_texts_delegates_iterable() -> None:
    assert list(iter_raw_url_source_texts(["a", "b"])) == ["a", "b"]


def test_raw_url_source_texts_iter_delegates_iterable() -> None:
    assert list(raw_url_source_texts_iter(["a", "b"])) == ["a", "b"]


def test_iter_text_urls_yields_raw_matches() -> None:
    urls = list(iter_text_urls("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert urls == ["https://x.com/u/status/1"]


def test_iter_text_url_matches_yields_match_objects() -> None:
    matches = list(iter_text_url_matches("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_collect_raw_urls_from_texts() -> None:
    items = []
    collect_raw_urls_from_texts(
        items,
        ["a https://x.com/u/status/1", "b https://x.com/u/status/1 c https://example.com"],
        url_re=x_url_extract_regex(),
    )
    assert items == ["https://x.com/u/status/1", "https://example.com"]


def test_iter_url_matches_yields_match_objects() -> None:
    matches = list(iter_url_matches("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_iter_url_matches_source_yields_match_objects() -> None:
    matches = list(iter_url_matches_source("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_iter_url_matches_iter_yields_match_objects() -> None:
    matches = list(iter_url_matches_iter("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_matches_source_yields_match_objects() -> None:
    matches = list(url_matches_source("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_iter_url_matches_for_source_yields_match_objects() -> None:
    matches = list(
        iter_url_matches_for_source("a https://x.com/u/status/1 b", url_re=x_url_extract_regex())
    )
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_matches_iter_yields_match_objects() -> None:
    matches = list(url_matches_iter("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_re_finditer_yields_match_objects() -> None:
    matches = list(url_re_finditer(x_url_extract_regex(), "a https://x.com/u/status/1 b"))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_iter_url_re_finditer_matches_yields_match_objects() -> None:
    matches = list(
        iter_url_re_finditer_matches(x_url_extract_regex(), "a https://x.com/u/status/1 b")
    )
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_re_finditer_source_yields_match_objects() -> None:
    matches = list(url_re_finditer_source(x_url_extract_regex(), "a https://x.com/u/status/1 b"))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_re_finditer_iter_yields_match_objects() -> None:
    matches = list(url_re_finditer_iter(x_url_extract_regex(), "a https://x.com/u/status/1 b"))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_matches_delegates_iter_url_matches() -> None:
    matches = list(url_matches("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_iter_url_matches_for_url_matches_delegates_source() -> None:
    matches = list(
        iter_url_matches_for_url_matches(
            "a https://x.com/u/status/1 b", url_re=x_url_extract_regex()
        )
    )
    assert [m.group(0) for m in matches] == ["https://x.com/u/status/1"]


def test_url_scan_text_normalizes_falsey_inputs() -> None:
    assert url_scan_text("abc") == "abc"
    assert url_scan_text("") == ""
    assert url_scan_text(None) == ""


def test_url_scan_text_for_finditer_normalizes_falsey_inputs() -> None:
    assert url_scan_text_for_finditer("abc") == "abc"
    assert url_scan_text_for_finditer("") == ""
    assert url_scan_text_for_finditer(None) == ""


def test_url_scan_text_source_normalizes_falsey_inputs() -> None:
    assert url_scan_text_source("abc") == "abc"
    assert url_scan_text_source("") == ""
    assert url_scan_text_source(None) == ""


def test_url_scan_text_value_normalizes_falsey_inputs() -> None:
    assert url_scan_text_value("abc") == "abc"
    assert url_scan_text_value("") == ""
    assert url_scan_text_value(None) == ""


def test_url_scan_text_fallback_empty_string() -> None:
    assert url_scan_text_fallback() == ""


def test_url_match_group_index_constant() -> None:
    assert url_match_group_index() == 0


def test_url_match_group_value_returns_group_zero() -> None:
    match = next(iter_url_matches("a https://x.com/u/status/1 b", url_re=x_url_extract_regex()))
    assert url_match_group_value(match) == "https://x.com/u/status/1"


def test_extract_raw_urls_and_filter_canonical_x_urls() -> None:
    raw = extract_raw_urls_from_texts(
        [
            "one https://x.com/a/status/1",
            "two https://example.com/z and https://twitter.com/a/status/1?s=20",
        ]
    )
    assert "https://x.com/a/status/1" in raw
    assert "https://example.com/z" in raw

    filtered = filter_canonical_x_urls(
        raw,
        is_x_url=lambda u: ("x.com/" in u or "twitter.com/" in u),
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert filtered == ["https://x.com/a/status/1"]


def test_canonical_x_url_items_buffer_starts_empty() -> None:
    assert canonical_x_url_items_buffer() == []


def test_canonical_x_url_items_buffer_source_starts_empty() -> None:
    assert canonical_x_url_items_buffer_source() == []


def test_canonical_x_url_items_buffer_for_source_starts_empty() -> None:
    assert canonical_x_url_items_buffer_for_source() == []


def test_canonical_x_url_items_buffer_value_starts_empty() -> None:
    assert canonical_x_url_items_buffer_value() == []


def test_append_unique_str_only_appends_new_values() -> None:
    items = ["a"]
    append_unique_str(items, "a")
    append_unique_str(items, "b")
    assert items == ["a", "b"]


def test_unique_value_missing_checks_membership() -> None:
    assert unique_value_missing(["a"], "b")
    assert not unique_value_missing(["a"], "a")


def test_unique_value_missing_source_checks_membership() -> None:
    assert unique_value_missing_source(["a"], "b")
    assert not unique_value_missing_source(["a"], "a")


def test_unique_value_missing_result_checks_membership() -> None:
    assert unique_value_missing_result(["a"], "b")
    assert not unique_value_missing_result(["a"], "a")


def test_is_x_url_candidate_source_delegates_predicate() -> None:
    assert is_x_url_candidate_source("https://x.com/u/status/1", is_x_url=lambda _: True)
    assert not is_x_url_candidate_source("https://example.com", is_x_url=lambda _: False)


def test_is_x_url_candidate_result_delegates_predicate() -> None:
    assert is_x_url_candidate_result("https://x.com/u/status/1", is_x_url=lambda _: True)
    assert not is_x_url_candidate_result("https://example.com", is_x_url=lambda _: False)


def test_is_x_url_candidate_for_result_delegates_predicate() -> None:
    assert is_x_url_candidate_for_result("https://x.com/u/status/1", is_x_url=lambda _: True)
    assert not is_x_url_candidate_for_result("https://example.com", is_x_url=lambda _: False)


def test_x_url_matches_predicate_source_delegates_predicate() -> None:
    assert x_url_matches_predicate_source("https://x.com/u/status/1", is_x_url=lambda _: True)
    assert not x_url_matches_predicate_source("https://example.com", is_x_url=lambda _: False)


def test_x_url_matches_predicate_result_delegates_predicate() -> None:
    assert x_url_matches_predicate_result("https://x.com/u/status/1", is_x_url=lambda _: True)
    assert not x_url_matches_predicate_result("https://example.com", is_x_url=lambda _: False)


def test_append_raw_url_if_present_only_appends_non_empty_unique() -> None:
    items = ["https://x.com/a/status/1"]
    append_raw_url_if_present(items, "")
    append_raw_url_if_present(items, "https://x.com/a/status/1")
    append_raw_url_if_present(items, "https://x.com/b/status/2")
    assert items == ["https://x.com/a/status/1", "https://x.com/b/status/2"]


def test_raw_url_is_present() -> None:
    assert raw_url_is_present("https://x.com/u/status/1")
    assert not raw_url_is_present("")


def test_raw_url_is_present_source() -> None:
    assert raw_url_is_present_source("https://x.com/u/status/1")
    assert not raw_url_is_present_source("")


def test_raw_url_is_present_result() -> None:
    assert raw_url_is_present_result("https://x.com/u/status/1")
    assert not raw_url_is_present_result("")


def test_raw_url_is_present_for_result() -> None:
    assert raw_url_is_present_for_result("https://x.com/u/status/1")
    assert not raw_url_is_present_for_result("")


def test_raw_url_should_append_delegates_presence_check() -> None:
    assert raw_url_should_append("https://x.com/u/status/1")
    assert not raw_url_should_append("")


def test_raw_url_should_append_source_delegates_presence_check() -> None:
    assert raw_url_should_append_source("https://x.com/u/status/1")
    assert not raw_url_should_append_source("")


def test_raw_url_should_append_result_delegates_presence_check() -> None:
    assert raw_url_should_append_result("https://x.com/u/status/1")
    assert not raw_url_should_append_result("")


def test_raw_url_should_append_for_result_delegates_presence_check() -> None:
    assert raw_url_should_append_for_result("https://x.com/u/status/1")
    assert not raw_url_should_append_for_result("")


def test_append_canonicalized_value_only_appends_unique_canonical() -> None:
    items = ["x:a"]
    append_canonicalized_value(items, "a", canonicalize=lambda s: f"x:{s}")
    append_canonicalized_value(items, "b", canonicalize=lambda s: f"x:{s}")
    assert items == ["x:a", "x:b"]


def test_canonicalized_value_delegates_transform() -> None:
    assert canonicalized_value("a", canonicalize=lambda s: f"x:{s}") == "x:a"


def test_canonicalized_value_source_delegates_transform() -> None:
    assert canonicalized_value_source("a", canonicalize=lambda s: f"x:{s}") == "x:a"


def test_canonicalized_value_result_delegates_transform() -> None:
    assert canonicalized_value_result("a", canonicalize=lambda s: f"x:{s}") == "x:a"


def test_canonicalized_value_for_result_delegates_transform() -> None:
    assert canonicalized_value_for_result("a", canonicalize=lambda s: f"x:{s}") == "x:a"


def test_append_canonical_x_url_only_appends_unique_canonical() -> None:
    items = ["https://x.com/a/status/1"]
    append_canonical_x_url(
        items,
        "https://twitter.com/a/status/1?s=20",
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_canonical_x_url(
        items,
        "https://twitter.com/b/status/2?s=20",
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/a/status/1", "https://x.com/b/status/2"]


def test_canonical_x_raw_value_identity() -> None:
    assert canonical_x_raw_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_canonical_x_raw_value_source_identity() -> None:
    assert canonical_x_raw_value_source("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_canonical_x_raw_value_result_identity() -> None:
    assert canonical_x_raw_value_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_append_canonical_status_url_only_appends_unique_canonical() -> None:
    items = ["https://x.com/u/status/1"]
    append_canonical_status_url(
        items,
        "https://twitter.com/u/status/1?s=20",
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_canonical_status_url(
        items,
        "https://twitter.com/v/status/2?s=20",
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1", "https://x.com/v/status/2"]


def test_canonical_status_raw_value_identity() -> None:
    assert canonical_status_raw_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_canonical_status_raw_value_source_identity() -> None:
    assert canonical_status_raw_value_source("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_canonical_status_raw_value_result_identity() -> None:
    assert canonical_status_raw_value_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_append_status_url_if_match() -> None:
    items = []
    append_status_url_if_match(
        items,
        "https://x.com/u/status/1?s=20",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0],
    )
    append_status_url_if_match(
        items,
        "https://example.com/page",
        is_status_url=lambda u: "/status/" in u,
        canonicalize_status_url=lambda u: u.split("?")[0],
    )
    assert items == ["https://x.com/u/status/1"]


def test_is_status_url_candidate_delegates_predicate() -> None:
    assert is_status_url_candidate("https://x.com/u/status/1", is_status_url=lambda u: "/status/" in u)
    assert not is_status_url_candidate("https://example.com", is_status_url=lambda u: "/status/" in u)


def test_is_status_url_candidate_source_delegates_predicate() -> None:
    assert is_status_url_candidate_source("https://x.com/u/status/1", is_status_url=lambda u: "/status/" in u)
    assert not is_status_url_candidate_source("https://example.com", is_status_url=lambda u: "/status/" in u)


def test_is_status_url_candidate_result_delegates_predicate() -> None:
    assert is_status_url_candidate_result("https://x.com/u/status/1", is_status_url=lambda u: "/status/" in u)
    assert not is_status_url_candidate_result("https://example.com", is_status_url=lambda u: "/status/" in u)


def test_is_status_url_candidate_for_result_delegates_predicate() -> None:
    assert is_status_url_candidate_for_result(
        "https://x.com/u/status/1", is_status_url=lambda u: "/status/" in u
    )
    assert not is_status_url_candidate_for_result(
        "https://example.com", is_status_url=lambda u: "/status/" in u
    )


def test_status_url_matches_predicate_delegates_candidate_check() -> None:
    assert status_url_matches_predicate(
        "https://x.com/u/status/1",
        is_status_url=lambda u: "/status/" in u,
    )
    assert not status_url_matches_predicate(
        "https://example.com",
        is_status_url=lambda u: "/status/" in u,
    )


def test_status_url_matches_predicate_source_delegates_candidate_check() -> None:
    assert status_url_matches_predicate_source(
        "https://x.com/u/status/1",
        is_status_url=lambda u: "/status/" in u,
    )
    assert not status_url_matches_predicate_source(
        "https://example.com",
        is_status_url=lambda u: "/status/" in u,
    )


def test_status_url_matches_predicate_result_delegates_candidate_check() -> None:
    assert status_url_matches_predicate_result(
        "https://x.com/u/status/1",
        is_status_url=lambda u: "/status/" in u,
    )
    assert not status_url_matches_predicate_result(
        "https://example.com",
        is_status_url=lambda u: "/status/" in u,
    )


def test_append_matched_status_url_only_appends_unique_canonical() -> None:
    items = ["https://x.com/u/status/1"]
    append_matched_status_url(
        items,
        "https://twitter.com/u/status/1?s=20",
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_matched_status_url(
        items,
        "https://twitter.com/v/status/2?s=20",
        canonicalize_status_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1", "https://x.com/v/status/2"]


def test_matched_status_raw_value_identity() -> None:
    assert matched_status_raw_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_status_raw_value_source_identity() -> None:
    assert matched_status_raw_value_source("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_status_raw_value_result_identity() -> None:
    assert matched_status_raw_value_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_status_raw_value_for_result_identity() -> None:
    assert matched_status_raw_value_for_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_append_matched_x_url_only_appends_unique_canonical() -> None:
    items = ["https://x.com/u/status/1"]
    append_matched_x_url(
        items,
        "https://twitter.com/u/status/1?s=20",
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_matched_x_url(
        items,
        "https://twitter.com/v/status/2?s=20",
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1", "https://x.com/v/status/2"]


def test_matched_x_raw_value_identity() -> None:
    assert matched_x_raw_value("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_x_raw_value_source_identity() -> None:
    assert matched_x_raw_value_source("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_x_raw_value_result_identity() -> None:
    assert matched_x_raw_value_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_matched_x_raw_value_for_result_identity() -> None:
    assert matched_x_raw_value_for_result("https://x.com/u/status/1") == "https://x.com/u/status/1"


def test_is_x_url_candidate_delegates_predicate() -> None:
    assert is_x_url_candidate("https://x.com/u/status/1", is_x_url=lambda u: "x.com" in u)
    assert not is_x_url_candidate("https://example.com", is_x_url=lambda u: "x.com" in u)


def test_x_url_matches_predicate_delegates_candidate_check() -> None:
    assert x_url_matches_predicate(
        "https://x.com/u/status/1",
        is_x_url=lambda u: "x.com" in u,
    )
    assert not x_url_matches_predicate(
        "https://example.com",
        is_x_url=lambda u: "x.com" in u,
    )


def test_append_x_url_if_match() -> None:
    items = []
    append_x_url_if_match(
        items,
        "https://twitter.com/u/status/1?s=20",
        is_x_url=lambda u: ("x.com/" in u or "twitter.com/" in u),
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    append_x_url_if_match(
        items,
        "https://example.com/page",
        is_x_url=lambda u: ("x.com/" in u or "twitter.com/" in u),
        canonicalize_x_url=lambda u: u.split("?")[0].replace("twitter.com", "x.com"),
    )
    assert items == ["https://x.com/u/status/1"]


def test_unwrap_x_media_url() -> None:
    wrapped = (
        "https://api.fxtwitter.com/dl?url="
        "https%3A%2F%2Fvideo.twimg.com%2Fext_tw_video%2Fabc%2Fvid%2F720x1280%2Fv.mp4"
    )
    assert (
        unwrap_x_media_url(wrapped)
        == "https://video.twimg.com/ext_tw_video/abc/vid/720x1280/v.mp4"
    )
    assert unwrap_x_media_url("https://x.com/user/status/1") == "https://x.com/user/status/1"


def test_unwrap_x_media_param_keys() -> None:
    assert unwrap_x_media_param_keys() == ("url", "media_url", "target", "u")


def test_is_unwrap_x_media_proxy_host() -> None:
    assert is_unwrap_x_media_proxy_host("api.fxtwitter.com")
    assert is_unwrap_x_media_proxy_host("api.vxtwitter.com")
    assert not is_unwrap_x_media_proxy_host("api.twitter.com")


def test_is_unwrap_x_media_candidate_url() -> None:
    assert is_unwrap_x_media_candidate_url("https://video.twimg.com/ext_tw_video/abc.mp4")
    assert not is_unwrap_x_media_candidate_url("/ext_tw_video/abc.mp4")


def test_extract_x_api_primary_tweet_variants() -> None:
    assert extract_x_api_primary_tweet({"data": {"id": "1"}}) == {"id": "1"}
    assert extract_x_api_primary_tweet({"data": [{"id": "2"}]}) == {"id": "2"}
    assert extract_x_api_primary_tweet({"data": []}) == {}
    assert extract_x_api_primary_tweet({"data": ["bad"]}) == {}
    assert extract_x_api_primary_tweet(None) == {}


def test_extract_x_api_first_item_variants() -> None:
    assert extract_x_api_first_item([{"id": "1"}]) == {"id": "1"}
    assert extract_x_api_first_item([]) == {}


def test_extract_x_api_primary_text_variants() -> None:
    assert extract_x_api_primary_text({"data": {"text": "dict text"}}) == "dict text"
    assert extract_x_api_primary_text({"data": [{"text": "list text"}]}) == "list text"
    assert extract_x_api_primary_text({"data": []}) == ""
    assert extract_x_api_primary_text(None) == ""


def test_normalize_x_api_text() -> None:
    assert normalize_x_api_text("  hello ") == "hello"
    assert normalize_x_api_text(None) == ""


def test_extract_sparse_media_resolution_defaults_and_sanitizes() -> None:
    assert extract_sparse_media_resolution(
        None,
        default_url="https://x.com/a",
    ) == ("unknown", [], "https://x.com/a")

    assert extract_sparse_media_resolution(
        {"kind": "video", "images": "bad", "url": ""},
        default_url="https://x.com/b",
    ) == ("video", [], "https://x.com/b")

    assert extract_sparse_media_resolution(
        {"kind": "", "images": ["i1"], "url": "https://x.com/c"},
        default_url="https://x.com/d",
    ) == ("unknown", ["i1"], "https://x.com/c")


def test_normalize_sparse_url_value() -> None:
    assert normalize_sparse_url_value("https://x.com/a", default_url="https://x.com/b") == "https://x.com/a"
    assert normalize_sparse_url_value("", default_url="https://x.com/b") == "https://x.com/b"
    assert normalize_sparse_url_value(None, default_url="https://x.com/b") == "https://x.com/b"


def test_normalize_sparse_kind_value() -> None:
    assert normalize_sparse_kind_value("video") == "video"
    assert normalize_sparse_kind_value("") == "unknown"
    assert normalize_sparse_kind_value(None) == "unknown"


def test_normalize_sparse_images_value() -> None:
    assert normalize_sparse_images_value(["i1"]) == ["i1"]
    assert normalize_sparse_images_value(None) == []
    assert normalize_sparse_images_value("bad") == []


def test_extract_fxtwitter_tweet_node_variants() -> None:
    assert extract_fxtwitter_tweet_node({"tweet": {"id": "1"}}) == {"id": "1"}
    assert extract_fxtwitter_tweet_node({"status": {"id": "2"}}) == {"id": "2"}
    assert extract_fxtwitter_tweet_node({"tweet": "bad"}) == {}
    assert extract_fxtwitter_tweet_node({"status": []}) == {}
    assert extract_fxtwitter_tweet_node(None) == {}


def test_stt_result_has_transcription_matches_router_semantics() -> None:
    assert stt_result_has_transcription({"transcription": "hello"}) is True
    # Preserve bool() semantics used in router call sites.
    assert stt_result_has_transcription({"transcription": "   "}) is True
    assert stt_result_has_transcription({"transcription": ""}) is False
    assert stt_result_has_transcription({"text": "fallback only"}) is False
    assert stt_result_has_transcription(None) is False


def test_stt_transcription_value_is_present() -> None:
    assert stt_transcription_value_is_present("hello") is True
    assert stt_transcription_value_is_present("   ") is True
    assert stt_transcription_value_is_present("") is False


def test_resolve_twitter_status_id_prefers_hint_then_parser() -> None:
    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            tweet_id="123",
            parse_status_id=lambda _url: (_ for _ in ()).throw(
                AssertionError("parser should not run when hint is provided")
            ),
        )
        == "123"
    )

    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: "456",
        )
        == "456"
    )

    assert (
        resolve_twitter_status_id(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: None,
        )
        == ""
    )


def test_is_twitter_status_url_uses_parser() -> None:
    assert (
        is_twitter_status_url(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: "111",
        )
        is True
    )
    assert (
        is_twitter_status_url(
            "https://x.com/user/status/111",
            parse_status_id=lambda _url: None,
        )
        is False
    )


def test_resolve_twitter_status_parser_default_and_override() -> None:
    default_parser = resolve_twitter_status_parser()
    assert default_parser is parse_twitter_status_id

    custom = lambda _u: "999"
    assert resolve_twitter_status_parser(custom) is custom


def test_classify_stt_error_reason_matches_router_semantics() -> None:
    assert classify_stt_error_reason("error") == "error"
    assert classify_stt_error_reason(None) == "no_speech"
    assert classify_stt_error_reason("timeout") == "no_speech"
    # Preserve legacy exact-match behavior (case-sensitive).
    assert classify_stt_error_reason("ERROR") == "no_speech"


def test_is_stt_hard_error_matches_router_semantics() -> None:
    assert is_stt_hard_error("error") is True
    assert is_stt_hard_error("ERROR") is False
    assert is_stt_hard_error(None) is False


def test_build_stt_fail_log_payload_includes_optional_fields() -> None:
    assert build_stt_fail_log_payload("no_speech") == {
        "event": "stt.fail",
        "detail": {"reason": "no_speech"},
    }

    assert build_stt_fail_log_payload(
        "error",
        media_kind="video",
        msg_id=123,
    ) == {
        "event": "stt.fail",
        "detail": {"reason": "error", "media_kind": "video"},
        "msg_id": 123,
    }


def test_build_stt_fail_detail_includes_optional_media_kind() -> None:
    assert build_stt_fail_detail("no_speech") == {"reason": "no_speech"}
    assert build_stt_fail_detail("error", media_kind="video") == {
        "reason": "error",
        "media_kind": "video",
    }


def test_build_caption_only_fallback_log_payload_shape() -> None:
    assert build_caption_only_fallback_log_payload() == {
        "event": "fallback",
        "detail": {"kind": "caption_only"},
    }


def test_build_caption_only_fallback_detail_shape() -> None:
    assert build_caption_only_fallback_detail() == {"kind": "caption_only"}


def test_build_x_video_stt_error_result_payload_defaults_and_shape() -> None:
    assert build_x_video_stt_error_result_payload(
        url="https://x.com/u/status/1",
        stt_error=None,
    ) == {
        "transcription": None,
        "error": "transcription_failed",
        "media_kind": "video",
        "url": "https://x.com/u/status/1",
    }

    assert build_x_video_stt_error_result_payload(
        url="https://x.com/u/status/2",
        stt_error="network_error",
    ) == {
        "transcription": None,
        "error": "network_error",
        "media_kind": "video",
        "url": "https://x.com/u/status/2",
    }


def test_normalize_stt_error_value() -> None:
    assert normalize_stt_error_value("network_error") == "network_error"
    assert normalize_stt_error_value("") == "transcription_failed"
    assert normalize_stt_error_value(None) == "transcription_failed"


def test_resolve_caption_only_base_text_preserves_router_precedence() -> None:
    assert (
        resolve_caption_only_base_text(
            api_text="api text",
            tweet_text="tweet text",
            base_text="base text",
        )
        == "api text"
    )
    assert (
        resolve_caption_only_base_text(
            api_text="",
            tweet_text="tweet text",
            base_text="base text",
        )
        == "tweet text"
    )
    # Preserve legacy behavior: truthy whitespace API text still wins before strip().
    assert (
        resolve_caption_only_base_text(
            api_text="   ",
            tweet_text="tweet text",
            base_text="base text",
        )
        == ""
    )


def test_resolve_video_stt_error_base_text_preserves_router_precedence() -> None:
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="tweet text",
            base_text="base text",
        )
        == "tweet text"
    )
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="",
            base_text="base text",
        )
        == "base text"
    )
    # Preserve legacy behavior: truthy whitespace tweet text still wins before strip().
    assert (
        resolve_video_stt_error_base_text(
            tweet_text="   ",
            base_text="base text",
        )
        == ""
    )


def test_normalize_base_text_value() -> None:
    assert normalize_base_text_value("  hello ") == "hello"
    assert normalize_base_text_value("") == ""
    assert normalize_base_text_value(None) == ""


def test_syndication_article_has_blocks_variants() -> None:
    assert syndication_article_has_blocks(None) is False
    assert syndication_article_has_blocks({}) is False
    assert syndication_article_has_blocks({"content": {"blocks": "bad"}}) is False
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"text": "   "}]}}) is False
    )
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"text": "hello"}]}})
        is True
    )
    assert (
        syndication_article_has_blocks({"content": {"blocks": [{"x": 1}, "bad"]}})
        is False
    )


def test_has_non_empty_block_text() -> None:
    assert has_non_empty_block_text({"text": "hello"}) is True
    assert has_non_empty_block_text({"text": "   "}) is False
    assert has_non_empty_block_text({"x": 1}) is False


def test_normalize_article_block_text() -> None:
    assert normalize_article_block_text({"text": "Body &amp; B"}) == "Body & B"
    assert normalize_article_block_text({"text": "   "}) == ""
    assert normalize_article_block_text({"x": 1}) == ""


def test_extract_x_article_text_dedupes_unescapes_and_caps() -> None:
    article = {
        "title": "Title &amp; Co",
        "preview_text": "Preview",
        "content": {
            "blocks": [
                {"text": "Body A"},
                {"text": "Body A"},
                {"text": "Body &amp; B"},
                {"x": 1},
                "bad",
            ]
        },
    }
    out = extract_x_article_text(article)
    assert out == "Title & Co\n\nPreview\n\nBody A\n\nBody & B"


def test_extract_x_article_text_truncates_at_12000_chars() -> None:
    huge = "a" * 12050
    out = extract_x_article_text({"title": huge})
    assert len(out) == 12000
    assert out.endswith("…")


def test_truncate_x_article_text_variants() -> None:
    assert truncate_x_article_text("abc") == "abc"
    assert truncate_x_article_text("a" * 12, max_chars=12) == ("a" * 12)
    out = truncate_x_article_text("a" * 13, max_chars=12)
    assert out == ("a" * 11) + "…"


def test_syndication_needs_article_hydration_variants() -> None:
    assert syndication_needs_article_hydration({}) is False
    assert syndication_needs_article_hydration({"news_action_type": "article"}) is True
    assert (
        syndication_needs_article_hydration(
            {"text": "https://t.co/abc123"},
            allow_tco_pointer=True,
        )
        is True
    )
    assert (
        syndication_needs_article_hydration(
            {
                "article": {"id": "1"},
            },
            article_has_blocks=lambda _article: False,
        )
        is True
    )
    assert (
        syndication_needs_article_hydration(
            {
                "article": {"id": "1"},
            },
            article_has_blocks=lambda _article: True,
        )
        is False
    )


def test_resolve_syndication_pointer_text_precedence() -> None:
    assert (
        resolve_syndication_pointer_text(
            {
                "text": "primary",
                "full_text": "secondary",
                "legacy": {"full_text": "tertiary"},
            }
        )
        == "primary"
    )
    assert (
        resolve_syndication_pointer_text(
            {
                "text": " ",
                "full_text": "secondary",
                "legacy": {"full_text": "tertiary"},
            }
        )
        == "secondary"
    )
    assert (
        resolve_syndication_pointer_text(
            {"text": " ", "full_text": " ", "legacy": {"full_text": "tertiary"}}
        )
        == "tertiary"
    )
    assert resolve_syndication_pointer_text({}) == ""


def test_article_has_metadata_hints_variants() -> None:
    assert article_has_metadata_hints({}) is False
    assert article_has_metadata_hints({"id": "1"}) is True
    assert article_has_metadata_hints({"rest_id": "2"}) is True
    assert article_has_metadata_hints({"title": "headline"}) is True
    assert article_has_metadata_hints({"preview_text": " "}) is False


def test_has_news_action_type() -> None:
    assert has_news_action_type({"news_action_type": "article"}) is True
    assert has_news_action_type({"news_action_type": "   "}) is False
    assert has_news_action_type({}) is False


def test_is_tco_pointer_text() -> None:
    assert is_tco_pointer_text("https://t.co/abc123") is True
    assert is_tco_pointer_text("http://t.co/abc123") is True
    assert is_tco_pointer_text("https://t.co/abc123 extra") is False
    assert is_tco_pointer_text("hello https://t.co/abc123") is False


def test_extract_syndication_base_text_precedence() -> None:
    assert (
        extract_syndication_base_text(
            {
                "note_tweet": {"text": "note"},
                "legacy": {"full_text": "legacy"},
                "full_text": "full",
                "text": "text",
            }
        )
        == "note"
    )
    assert (
        extract_syndication_base_text(
            {
                "legacy": {"full_text": "legacy"},
                "full_text": "full",
                "text": "text",
            }
        )
        == "legacy"
    )
    assert extract_syndication_base_text({"text": " text "}) == "text"
    assert extract_syndication_base_text(None) == ""


def test_extract_note_tweet_text() -> None:
    assert extract_note_tweet_text({"text": "note"}) == "note"
    assert extract_note_tweet_text({"x": 1}) is None
    assert extract_note_tweet_text("bad") is None


def test_merge_syndication_base_with_article_variants() -> None:
    assert (
        merge_syndication_base_with_article(
            base_text="base",
            article_text="",
        )
        == "base"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="base",
            article_text="article",
        )
        == "base\n\n[Linked X Article]\narticle"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="base article body",
            article_text="article body",
        )
        == "base article body"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="https://t.co/abc123",
            article_text="article",
        )
        == "article"
    )
    assert (
        merge_syndication_base_with_article(
            base_text="",
            article_text="article",
        )
        == "article"
    )


def test_base_text_contains_tco_link() -> None:
    assert base_text_contains_tco_link("https://t.co/abc123")
    assert base_text_contains_tco_link("hello https://t.co/abc123 world")
    assert not base_text_contains_tco_link("hello world")


def test_extract_syndication_text_variants() -> None:
    assert extract_syndication_text(None) == ""
    assert extract_syndication_text({"text": " hello "}) == "hello"

    merged = extract_syndication_text(
        {"text": "base", "article": {"id": "1"}},
        extract_article_text=lambda _article: "article",
    )
    assert merged == "base\n\n[Linked X Article]\narticle"

    # Preserve fail-open behavior when article extractor errors.
    assert (
        extract_syndication_text(
            {"text": "base", "article": {"id": "1"}},
            extract_article_text=lambda _article: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        == "base"
    )


def test_extract_syndication_article_text_variants() -> None:
    assert (
        extract_syndication_article_text(
            node={"article": {"id": "1"}},
            article_extractor=lambda article: str(article.get("id") or ""),
        )
        == "1"
    )
    assert (
        extract_syndication_article_text(
            node={"article": {"id": "1"}},
            article_extractor=lambda _article: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        == ""
    )


def test_build_x_text_miss_log_payload_shape_and_primary_id() -> None:
    assert build_x_text_miss_log_payload(
        "https://x.com/u/status/2022790791047823773"
    ) == {
        "event": "x.text.miss",
        "detail": {
            "primary": "2022790791047823773",
            "layer": "format",
            "reason": "empty_text",
        },
    }
    assert build_x_text_miss_log_payload("https://example.com/nope") == {
        "event": "x.text.miss",
        "detail": {
            "primary": "",
            "layer": "format",
            "reason": "empty_text",
        },
    }
    assert build_x_text_miss_log_payload(
        "https://x.com/u/status/123#ptid=2023475721184907773"
    ) == {
        "event": "x.text.miss",
        "detail": {
            "primary": "2023475721184907773",
            "layer": "format",
            "reason": "empty_text",
        },
    }


def test_build_x_text_miss_payload_accepts_explicit_fields() -> None:
    assert build_x_text_miss_payload(
        primary="2022790791047823773",
        layer="syndication",
        reason="no_text",
    ) == {
        "event": "x.text.miss",
        "detail": {
            "primary": "2022790791047823773",
            "layer": "syndication",
            "reason": "no_text",
        },
    }


def test_build_x_text_resolve_payload_shape() -> None:
    assert build_x_text_resolve_payload(
        primary="2022790791047823773",
        source="syndication",
        chars=42,
    ) == {
        "event": "x.text.resolve",
        "detail": {
            "primary": "2022790791047823773",
            "source": "syndication",
            "chars": 42,
        },
    }


def test_build_syndication_non_200_log_payload_shape() -> None:
    assert build_syndication_non_200_log_payload(
        tweet_id="2022790791047823773",
        status=403,
        endpoint="widgets",
    ) == {
        "detail": {
            "tweet_id": "2022790791047823773",
            "status": 403,
            "endpoint": "widgets",
        }
    }


def test_build_syndication_non_200_metric_payload_shape() -> None:
    assert build_syndication_non_200_metric_payload(
        status=403,
        endpoint="widgets",
    ) == {
        "status": "403",
        "endpoint": "widgets",
    }


def test_build_syndication_fetch_failed_payload_shape() -> None:
    assert build_syndication_fetch_failed_payload(
        tweet_id="2022790791047823773",
        error="timeout",
    ) == {
        "detail": {
            "tweet_id": "2022790791047823773",
            "error": "timeout",
        }
    }


def test_build_x_text_canon_payload_shape() -> None:
    assert build_x_text_canon_payload(
        url="https://x.com/u/status/2022790791047823773?ptid=2022790791047823773",
        primary="2022790791047823773",
    ) == {
        "event": "x.text.canon",
        "detail": {
            "url": "https://x.com/u/status/2022790791047823773?ptid=2022790791047823773",
            "primary": "2022790791047823773",
        },
    }


def test_extract_oembed_html_text_strips_tags_and_unescapes() -> None:
    assert (
        extract_oembed_html_text("<p>Hello <strong>world</strong> &amp; co</p>")
        == "Hello world & co"
    )


def test_extract_oembed_html_text_empty_input_returns_empty_string() -> None:
    assert extract_oembed_html_text("") == ""
    assert extract_oembed_html_text(None) == ""


def test_extract_oembed_html_text_non_string_truthy_input_raises() -> None:
    with pytest.raises(TypeError):
        extract_oembed_html_text({"html": "<p>bad</p>"})


def test_build_oembed_text_payload_happy_path_shape() -> None:
    assert build_oembed_text_payload(
        {"html": "<p>Hello &amp; co</p>", "author_name": "alice"}
    ) == {
        "text": "Hello & co",
        "user": {"name": "alice"},
    }


def test_build_oembed_text_payload_returns_none_for_non_usable_obj() -> None:
    assert build_oembed_text_payload(None) is None
    assert build_oembed_text_payload({"html": ""}) is None


def test_build_oembed_text_payload_non_string_html_raises() -> None:
    with pytest.raises(TypeError):
        build_oembed_text_payload({"html": {"bad": "value"}})


def test_build_syndication_oembed_params_default_and_x_host() -> None:
    assert build_syndication_oembed_params("2022790791047823773") == {
        "url": "https://twitter.com/i/status/2022790791047823773",
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
        "lang": "en",
    }
    assert build_syndication_oembed_params(
        "2022790791047823773", use_x_host=True
    ) == {
        "url": "https://x.com/i/status/2022790791047823773",
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
        "lang": "en",
    }


def test_build_syndication_oembed_params_core_shape() -> None:
    assert build_syndication_oembed_params_core(
        "twitter.com",
        "2022790791047823773",
    ) == {
        "url": "https://twitter.com/i/status/2022790791047823773",
        "lang": "en",
    }


def test_build_syndication_oembed_params_core_map_shape() -> None:
    assert build_syndication_oembed_params_core_map(
        "twitter.com",
        "2022790791047823773",
        "en",
    ) == {
        "url": "https://twitter.com/i/status/2022790791047823773",
        "lang": "en",
    }


def test_build_syndication_oembed_params_bundle_shape() -> None:
    assert build_syndication_oembed_params_bundle(
        "twitter.com",
        "2022790791047823773",
    ) == {
        "url": "https://twitter.com/i/status/2022790791047823773",
        "lang": "en",
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
    }


def test_build_syndication_oembed_hosts_order() -> None:
    assert build_syndication_oembed_hosts() == ("twitter.com", "x.com")


def test_build_syndication_oembed_hosts_tuple_order() -> None:
    assert build_syndication_oembed_hosts_tuple() == ("twitter.com", "x.com")


def test_build_syndication_oembed_host_for_flag() -> None:
    assert build_syndication_oembed_host_for_flag(False) == "twitter.com"
    assert build_syndication_oembed_host_for_flag(True) == "x.com"


def test_build_syndication_host_constants() -> None:
    assert build_syndication_twitter_host() == "twitter.com"
    assert build_syndication_x_host() == "x.com"


def test_is_syndication_x_host() -> None:
    assert is_syndication_x_host("x.com") is True
    assert is_syndication_x_host("twitter.com") is False
    assert is_syndication_x_host("X.COM") is False


def test_is_syndication_twitter_host() -> None:
    assert is_syndication_twitter_host("twitter.com") is True
    assert is_syndication_twitter_host("x.com") is False
    assert is_syndication_twitter_host("TWITTER.COM") is False


def test_build_syndication_oembed_status_url_shape() -> None:
    assert (
        build_syndication_oembed_status_url("twitter.com", "2022790791047823773")
        == "https://twitter.com/i/status/2022790791047823773"
    )


def test_build_syndication_status_url_shape() -> None:
    assert (
        build_syndication_status_url("x.com", "2022790791047823773")
        == "https://x.com/i/status/2022790791047823773"
    )


def test_build_syndication_oembed_url_key_constant() -> None:
    assert build_syndication_oembed_url_key() == "url"


def test_build_syndication_status_path_constant() -> None:
    assert build_syndication_status_path() == "i/status"


def test_build_syndication_status_url_prefix_shape() -> None:
    assert build_syndication_status_url_prefix("twitter.com") == (
        "https://twitter.com/i/status/"
    )


def test_build_syndication_oembed_options_constant() -> None:
    assert build_syndication_oembed_options() == {
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
    }


def test_build_syndication_oembed_options_map_constant() -> None:
    assert build_syndication_oembed_options_map() == {
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
    }


def test_build_syndication_oembed_options_map_from_pairs() -> None:
    assert build_syndication_oembed_options_map_from_pairs(
        ("dnt", "omit_script", "hide_thread"),
        ("false", "true", "true"),
    ) == {
        "dnt": "false",
        "omit_script": "true",
        "hide_thread": "true",
    }


def test_build_syndication_oembed_option_keys_constant() -> None:
    assert build_syndication_oembed_option_keys() == (
        "dnt",
        "omit_script",
        "hide_thread",
    )


def test_build_syndication_oembed_option_values_constant() -> None:
    assert build_syndication_oembed_option_values() == (
        "false",
        "true",
        "true",
    )


def test_build_syndication_oembed_dnt_key_constant() -> None:
    assert build_syndication_oembed_dnt_key() == "dnt"


def test_build_syndication_oembed_omit_script_key_constant() -> None:
    assert build_syndication_oembed_omit_script_key() == "omit_script"


def test_build_syndication_oembed_hide_thread_key_constant() -> None:
    assert build_syndication_oembed_hide_thread_key() == "hide_thread"


def test_build_syndication_oembed_metric_endpoint_mapping() -> None:
    assert build_syndication_oembed_metric_endpoint("twitter.com") == "oembed"
    assert build_syndication_oembed_metric_endpoint("x.com") == "oembed_x"
    assert build_syndication_oembed_metric_endpoint("unknown") == "oembed"


def test_build_syndication_oembed_metric_endpoint_constants() -> None:
    assert build_syndication_oembed_metric_default_endpoint() == "oembed"
    assert build_syndication_oembed_x_metric_endpoint() == "oembed_x"


def test_build_syndication_oembed_key_constant() -> None:
    assert build_syndication_oembed_key() == "oembed"


def test_build_syndication_oembed_host_constant() -> None:
    assert build_syndication_oembed_host() == "publish.twitter.com"


def test_build_syndication_oembed_fallback_params_ordered_variants() -> None:
    assert build_syndication_oembed_fallback_params("2022790791047823773") == [
        (
            "oembed",
            {
                "url": "https://twitter.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
        (
            "oembed_x",
            {
                "url": "https://x.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
    ]


def test_build_syndication_oembed_fallback_items_list_ordered_variants() -> None:
    assert build_syndication_oembed_fallback_items_list("2022790791047823773") == [
        (
            "oembed",
            {
                "url": "https://twitter.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
        (
            "oembed_x",
            {
                "url": "https://x.com/i/status/2022790791047823773",
                "dnt": "false",
                "omit_script": "true",
                "hide_thread": "true",
                "lang": "en",
            },
        ),
    ]


def test_build_syndication_oembed_fallback_item_shape() -> None:
    assert build_syndication_oembed_fallback_item(
        "twitter.com",
        "2022790791047823773",
    ) == (
        "oembed",
        {
            "url": "https://twitter.com/i/status/2022790791047823773",
            "dnt": "false",
            "omit_script": "true",
            "hide_thread": "true",
            "lang": "en",
        },
    )


def test_build_syndication_oembed_fallback_plan_shape() -> None:
    url, variants = build_syndication_oembed_fallback_plan("2022790791047823773")
    assert url == "https://publish.twitter.com/oembed"
    assert variants == build_syndication_oembed_fallback_params("2022790791047823773")


def test_build_syndication_oembed_fallback_plan_components_shape() -> None:
    url, variants = build_syndication_oembed_fallback_plan_components(
        "2022790791047823773"
    )
    assert url == "https://publish.twitter.com/oembed"
    assert variants == build_syndication_oembed_fallback_params("2022790791047823773")


def test_build_syndication_oembed_fallback_plan_tuple_passthrough() -> None:
    url = "https://publish.twitter.com/oembed"
    variants = [("oembed", {"url": "u"})]
    assert build_syndication_oembed_fallback_plan_tuple(url, variants) == (url, variants)


def test_build_syndication_oembed_url_constant() -> None:
    assert build_syndication_oembed_url() == "https://publish.twitter.com/oembed"


def test_build_syndication_oembed_endpoint_url_composes_host_and_key() -> None:
    assert (
        build_syndication_oembed_endpoint_url("publish.twitter.com", "oembed")
        == "https://publish.twitter.com/oembed"
    )


def test_build_syndication_cdn_host_constant() -> None:
    assert build_syndication_cdn_host() == "cdn.syndication.twimg.com"


def test_build_syndication_base_url_constant() -> None:
    assert build_syndication_base_url() == "https://cdn.syndication.twimg.com/"


def test_build_syndication_fetch_headers_shape() -> None:
    headers = build_syndication_fetch_headers()
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert "Mozilla/5.0" in headers["User-Agent"]


def test_build_syndication_fetch_headers_base_shape() -> None:
    headers = build_syndication_fetch_headers_base()
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert "Mozilla/5.0" in headers["User-Agent"]


def test_build_syndication_fetch_header_map_shape() -> None:
    headers = build_syndication_fetch_header_map(
        keys=("User-Agent", "Accept", "Accept-Language", "Referer"),
        values=("ua", "accept", "al", "ref"),
    )
    assert headers == {
        "User-Agent": "ua",
        "Accept": "accept",
        "Accept-Language": "al",
        "Referer": "ref",
    }


def test_build_syndication_fetch_header_keys_constant() -> None:
    assert build_syndication_fetch_header_keys() == (
        "User-Agent",
        "Accept",
        "Accept-Language",
        "Referer",
    )


def test_build_syndication_fetch_header_values_shape() -> None:
    user_agent, accept, accept_language, referer = (
        build_syndication_fetch_header_values()
    )
    assert "Mozilla/5.0" in user_agent
    assert accept == "application/json, text/javascript;q=0.9, */*;q=0.8"
    assert accept_language == "en-US,en;q=0.9"
    assert referer == "https://platform.twitter.com/"


def test_build_syndication_fetch_user_agent_shape() -> None:
    user_agent = build_syndication_fetch_user_agent()
    assert "Mozilla/5.0" in user_agent
    assert "Chrome/126.0.0.0" in user_agent


def test_build_syndication_user_agent_platform_constant() -> None:
    assert build_syndication_user_agent_platform() == "Windows NT 10.0; Win64; x64"


def test_build_syndication_fetch_accept_language_constant() -> None:
    assert build_syndication_fetch_accept_language() == "en-US,en;q=0.9"


def test_build_syndication_region_locale_constant() -> None:
    assert build_syndication_region_locale() == "en-US"


def test_build_syndication_accept_language_primary_entry_constant() -> None:
    assert build_syndication_accept_language_primary_entry() == "en-US"


def test_build_syndication_accept_language_pair_constant() -> None:
    assert build_syndication_accept_language_pair() == "en-US,en"


def test_build_syndication_lang_quality_constant() -> None:
    assert build_syndication_lang_quality() == "q=0.9"


def test_build_syndication_accept_language_secondary_entry_constant() -> None:
    assert build_syndication_accept_language_secondary_entry() == "en;q=0.9"


def test_build_syndication_fetch_referer_constant() -> None:
    assert build_syndication_fetch_referer() == "https://platform.twitter.com/"


def test_build_syndication_platform_host_constant() -> None:
    assert build_syndication_platform_host() == "platform.twitter.com"


def test_build_syndication_fetch_accept_constant() -> None:
    assert build_syndication_fetch_accept() == (
        "application/json, text/javascript;q=0.9, */*;q=0.8"
    )


def test_build_syndication_accept_primary_mimes_constant() -> None:
    assert (
        build_syndication_accept_primary_mimes()
        == "application/json, text/javascript"
    )


def test_build_syndication_accept_json_mime_constant() -> None:
    assert build_syndication_accept_json_mime() == "application/json"


def test_build_syndication_accept_text_mime_constant() -> None:
    assert build_syndication_accept_text_mime() == "text/javascript"


def test_build_syndication_accept_text_quality_constant() -> None:
    assert build_syndication_accept_text_quality() == "q=0.9"


def test_build_syndication_accept_text_entry_constant() -> None:
    assert build_syndication_accept_text_entry() == "text/javascript;q=0.9"


def test_build_syndication_accept_any_mime_constant() -> None:
    assert build_syndication_accept_any_mime() == "*/*"


def test_build_syndication_accept_any_quality_constant() -> None:
    assert build_syndication_accept_any_quality() == "q=0.8"


def test_build_syndication_accept_any_entry_constant() -> None:
    assert build_syndication_accept_any_entry() == "*/*;q=0.8"


def test_build_syndication_lang_constant() -> None:
    assert build_syndication_lang() == "en"


def test_build_syndication_dnt_value_constant() -> None:
    assert build_syndication_dnt_value() == "false"


def test_build_syndication_omit_script_value_constant() -> None:
    assert build_syndication_omit_script_value() == "true"


def test_build_syndication_hide_thread_value_constant() -> None:
    assert build_syndication_hide_thread_value() == "true"


def test_build_syndication_bool_true_value_constant() -> None:
    assert build_syndication_bool_true_value() == "true"


def test_build_syndication_bool_false_value_constant() -> None:
    assert build_syndication_bool_false_value() == "false"


def test_build_syndication_dnt_key_constant() -> None:
    assert build_syndication_dnt_key() == "dnt"


def test_build_syndication_id_key_constant() -> None:
    assert build_syndication_id_key() == "id"


def test_build_syndication_lang_key_constant() -> None:
    assert build_syndication_lang_key() == "lang"


def test_build_syndication_fetch_params_variants_with_and_without_dnt() -> None:
    assert build_syndication_fetch_params_core("2022790791047823773") == {
        "id": "2022790791047823773",
        "lang": "en",
    }
    assert build_syndication_fetch_params("2022790791047823773") == {
        "id": "2022790791047823773",
        "lang": "en",
    }
    assert build_syndication_fetch_params(
        "2022790791047823773",
        include_dnt=True,
    ) == {
        "id": "2022790791047823773",
        "lang": "en",
        "dnt": "false",
    }


def test_build_syndication_fetch_params_core_map() -> None:
    assert build_syndication_fetch_params_core_map("2022790791047823773", "en") == {
        "id": "2022790791047823773",
        "lang": "en",
    }


def test_build_syndication_fetch_params_with_optional_dnt() -> None:
    core = {"id": "2022790791047823773", "lang": "en"}
    assert build_syndication_fetch_params_with_optional_dnt(dict(core), False) == core
    assert build_syndication_fetch_params_with_optional_dnt(dict(core), True) == {
        "id": "2022790791047823773",
        "lang": "en",
        "dnt": "false",
    }


def test_maybe_add_syndication_dnt_param() -> None:
    core = {"id": "2022790791047823773", "lang": "en"}
    assert maybe_add_syndication_dnt_param(params=dict(core), include_dnt=False) == core
    assert maybe_add_syndication_dnt_param(params=dict(core), include_dnt=True) == {
        "id": "2022790791047823773",
        "lang": "en",
        "dnt": "false",
    }


def test_build_syndication_fetch_params_variants_shape() -> None:
    variants = build_syndication_fetch_params_variants("2022790791047823773")
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_fetch_params_variants_list_shape() -> None:
    variants = build_syndication_fetch_params_variants_list("2022790791047823773")
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_widgets_params_variant_shape() -> None:
    assert build_syndication_widgets_params_variant("2022790791047823773") == (
        "widgets",
        {"id": "2022790791047823773", "lang": "en"},
    )
    assert build_syndication_widgets_params_variant(
        "2022790791047823773",
        include_dnt=True,
    ) == (
        "widgets",
        {"id": "2022790791047823773", "lang": "en", "dnt": "false"},
    )


def test_build_syndication_widgets_params_variant_with_dnt_shape() -> None:
    assert build_syndication_widgets_params_variant_with_dnt(
        "2022790791047823773"
    ) == (
        "widgets",
        {"id": "2022790791047823773", "lang": "en", "dnt": "false"},
    )


def test_build_syndication_tweet_result_params_variant_shape() -> None:
    assert build_syndication_tweet_result_params_variant("2022790791047823773") == (
        "tweet-result",
        {"id": "2022790791047823773", "lang": "en"},
    )


def test_build_syndication_endpoint_name_constants() -> None:
    assert build_syndication_widgets_endpoint() == "widgets"
    assert build_syndication_tweet_result_endpoint() == "tweet-result"


def test_build_syndication_endpoint_path_constants() -> None:
    assert build_syndication_widgets_tweet_path() == "widgets/tweet"
    assert build_syndication_tweet_result_path() == "tweet-result"


def test_extract_oembed_payload_from_response_variants() -> None:
    good = SimpleNamespace(
        status_code=200,
        json=lambda: {"html": "<p>Hello &amp; world</p>", "author_name": "alice"},
    )
    assert extract_oembed_payload_from_response(good) == {
        "text": "Hello & world",
        "user": {"name": "alice"},
    }

    non_200 = SimpleNamespace(status_code=404, json=lambda: {"html": "<p>x</p>"})
    assert extract_oembed_payload_from_response(non_200) is None

    bad_json = SimpleNamespace(
        status_code=200,
        json=lambda: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert extract_oembed_payload_from_response(bad_json) is None


def test_extract_oembed_payload_from_response_missing_status_raises() -> None:
    with pytest.raises(AttributeError):
        extract_oembed_payload_from_response(object())


def test_build_syndication_fetch_plan_shape_and_values() -> None:
    base, headers, variants = build_syndication_fetch_plan("2022790791047823773")
    assert base == "https://cdn.syndication.twimg.com/"
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert headers["Accept-Language"] == "en-US,en;q=0.9"
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_fetch_plan_components_shape() -> None:
    base, headers, variants = build_syndication_fetch_plan_components(
        "2022790791047823773"
    )
    assert base == "https://cdn.syndication.twimg.com/"
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_fetch_plan_values_shape() -> None:
    base, headers, variants = build_syndication_fetch_plan_values(
        "2022790791047823773"
    )
    assert base == "https://cdn.syndication.twimg.com/"
    assert headers["Referer"] == "https://platform.twitter.com/"
    assert variants == [
        ("widgets", {"id": "2022790791047823773", "lang": "en"}),
        ("tweet-result", {"id": "2022790791047823773", "lang": "en"}),
        ("widgets", {"id": "2022790791047823773", "lang": "en", "dnt": "false"}),
    ]


def test_build_syndication_fetch_plan_tuple_passthrough() -> None:
    base = "https://cdn.syndication.twimg.com/"
    headers = {"User-Agent": "ua"}
    variants = [("widgets", {"id": "1", "lang": "en"})]
    assert build_syndication_fetch_plan_tuple(base, headers, variants) == (
        base,
        headers,
        variants,
    )


def test_build_syndication_fetch_metric_payload_shape() -> None:
    assert build_syndication_fetch_metric_payload("widgets") == {"endpoint": "widgets"}
    assert build_syndication_fetch_metric_payload("oembed_x") == {
        "endpoint": "oembed_x"
    }


def test_build_syndication_metric_payload_map_shape() -> None:
    assert build_syndication_metric_payload_map("endpoint", "widgets") == {
        "endpoint": "widgets"
    }


def test_build_syndication_metric_endpoint_key_constant() -> None:
    assert build_syndication_metric_endpoint_key() == "endpoint"


def test_syndication_cache_ttl_s_caps_negative_entries() -> None:
    assert syndication_cache_ttl_s(600.0, {"neg": True}) == 300.0
    assert syndication_cache_ttl_s(120.0, {"neg": True}) == 120.0
    assert syndication_cache_ttl_s(600.0, {"neg": False}) == 600.0


def test_syndication_negative_cache_ttl_value_caps_default_ttl() -> None:
    assert syndication_negative_cache_ttl_value(600.0) == 300.0
    assert syndication_negative_cache_ttl_value(120.0) == 120.0


def test_build_syndication_negative_cache_ttl_cap_s_constant() -> None:
    assert build_syndication_negative_cache_ttl_cap_s() == 300.0


def test_build_syndication_cache_ts_key_constant() -> None:
    assert build_syndication_cache_ts_key() == "ts"


def test_build_syndication_negative_cache_key_constant() -> None:
    assert build_syndication_negative_cache_key() == "neg"


def test_build_syndication_cache_data_key_constant() -> None:
    assert build_syndication_cache_data_key() == "data"


def test_build_syndication_cache_hit_label_constants() -> None:
    assert build_syndication_negative_cache_hit_label() == "neg"
    assert build_syndication_data_cache_hit_label() == "data"


def test_build_syndication_cache_hit_label_variants() -> None:
    assert build_syndication_cache_hit_label({"neg": True}) == "neg"
    assert build_syndication_cache_hit_label({"neg": False, "data": {}}) == "data"


def test_syndication_cache_ttl_s_preserves_attribute_error_for_bad_cache() -> None:
    with pytest.raises(AttributeError):
        syndication_cache_ttl_s(600.0, object())


def test_syndication_cache_is_fresh_respects_ttl_policy() -> None:
    assert (
        syndication_cache_is_fresh(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": False},
        )
        is True
    )
    assert (
        syndication_cache_is_fresh(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": True},
        )
        is False
    )


def test_syndication_cache_timestamp_value() -> None:
    assert syndication_cache_timestamp_value({"ts": 123.4}) == 123.4
    assert syndication_cache_timestamp_value({}) == 0.0


def test_syndication_cache_is_fresh_preserves_ts_parse_error() -> None:
    with pytest.raises(ValueError):
        syndication_cache_is_fresh(1_000.0, 600.0, {"ts": "bad", "neg": False})


def test_classify_syndication_cache_hit_variants() -> None:
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            600.0,
            {"ts": 500.0, "neg": False},
        )
        == "data"
    )
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            600.0,
            {"ts": 850.0, "neg": True},
        )
        == "neg"
    )
    assert (
        classify_syndication_cache_hit(
            1_000.0,
            120.0,
            {"ts": 850.0, "neg": False},
        )
        is None
    )


def test_classify_syndication_cache_hit_preserves_ts_parse_error() -> None:
    with pytest.raises(ValueError):
        classify_syndication_cache_hit(1_000.0, 600.0, {"ts": "bad", "neg": False})


def test_build_syndication_negative_cache_entry_shape() -> None:
    assert build_syndication_negative_cache_entry(123.45) == {
        "neg": True,
        "ts": 123.45,
    }


def test_build_syndication_cache_timestamp_field_shape() -> None:
    assert build_syndication_cache_timestamp_field(123.45) == {"ts": 123.45}


def test_build_syndication_negative_cache_flag_field_shape() -> None:
    assert build_syndication_negative_cache_flag_field() == {"neg": True}


def test_build_syndication_cache_data_field_shape() -> None:
    assert build_syndication_cache_data_field({"text": "hello"}) == {
        "data": {"text": "hello"}
    }


def test_build_syndication_cache_entry_shape() -> None:
    data = {"text": "hello"}
    assert build_syndication_cache_entry(data, 321.0) == {
        build_syndication_cache_data_key(): data,
        "ts": 321.0,
    }


def test_build_syndication_endpoint_url_mapping() -> None:
    base = "https://cdn.syndication.twimg.com/"
    assert build_syndication_endpoint_url(base, "widgets") == (
        "https://cdn.syndication.twimg.com/widgets/tweet"
    )
    assert build_syndication_endpoint_url(base, "tweet-result") == (
        "https://cdn.syndication.twimg.com/tweet-result"
    )
    # Preserve legacy fallback behavior for unknown endpoint values.
    assert build_syndication_endpoint_url(base, "unknown") == (
        "https://cdn.syndication.twimg.com/tweet-result"
    )


def test_build_syndication_endpoint_suffix_mapping() -> None:
    assert build_syndication_endpoint_suffix("widgets") == "widgets/tweet"
    assert build_syndication_endpoint_suffix("tweet-result") == "tweet-result"
    assert build_syndication_endpoint_suffix("unknown") == "tweet-result"


def test_syndication_has_usable_payload_with_text_or_media_hints() -> None:
    assert (
        syndication_has_usable_payload(
            {"text": "hello"},
            extract_text=lambda node: str(node.get("text") or "").strip(),
            media_hint_keys=("entities", "media"),
        )
        is True
    )
    assert (
        syndication_has_usable_payload(
            {"entities": {}},
            extract_text=lambda _node: "",
            media_hint_keys=("entities", "media"),
        )
        is True
    )
    assert (
        syndication_has_usable_payload(
            {"x": 1},
            extract_text=lambda _node: "",
            media_hint_keys=("entities", "media"),
        )
        is False
    )


def test_syndication_has_usable_payload_non_dict_returns_false() -> None:
    assert (
        syndication_has_usable_payload(
            None,
            extract_text=lambda _node: "text",
            media_hint_keys=("entities",),
        )
        is False
    )


def test_syndication_node_has_media_hints_variants() -> None:
    assert syndication_node_has_media_hints({"entities": {}}, ("entities", "media"))
    assert not syndication_node_has_media_hints({"text": "x"}, ("entities", "media"))


def test_syndication_media_hint_keys_matches_router_contract() -> None:
    assert syndication_media_hint_keys() == (
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


def test_format_syndication_body_text_variants() -> None:
    assert format_syndication_body_text("short") == "short"
    assert format_syndication_body_text("") == (
        "(Tweet text not available. If you want analysis, paste the text or add a screenshot.)"
    )
    long_text = "a" * 5000
    out = format_syndication_body_text(long_text)
    assert len(out) == 3991
    assert out.endswith("…")


def test_format_syndication_truncated_text_shape() -> None:
    out = format_syndication_truncated_text("x" * 5000)
    assert len(out) == 3991
    assert out.endswith("…")


def test_format_syndication_missing_text_fallback_shape() -> None:
    assert format_syndication_missing_text_fallback() == (
        "(Tweet text not available. If you want analysis, paste the text or add a screenshot.)"
    )


def test_format_syndication_header_line_variants() -> None:
    assert (
        format_syndication_header_line(
            user={"screen_name": "alice", "name": "Alice"},
            created_at="2026-02-17",
            photos=["p1", "p2"],
            url="https://x.com/u/status/1",
        )
        == "@alice • 2026-02-17 • media:2 → https://x.com/u/status/1"
    )
    assert (
        format_syndication_header_line(
            user={},
            created_at=None,
            photos=[],
            url="https://x.com/u/status/1",
        )
        == "Tweet → https://x.com/u/status/1"
    )


def test_format_syndication_header_username_prefers_screen_name() -> None:
    assert format_syndication_header_username({"screen_name": "alice", "name": "Alice"}) == "alice"
    assert format_syndication_header_username({"name": "Alice"}) == "Alice"


def test_format_syndication_header_media_hint_variants() -> None:
    assert format_syndication_header_media_hint(["p1", "p2"]) == " • media:2"
    assert format_syndication_header_media_hint([]) == ""


def test_format_syndication_header_prefix_variants() -> None:
    assert format_syndication_header_prefix("alice") == "@alice"
    assert format_syndication_header_prefix("") == "Tweet"


def test_format_syndication_header_stamp_variants() -> None:
    assert format_syndication_header_stamp("2026-02-17") == " • 2026-02-17"
    assert format_syndication_header_stamp(None) == ""


def test_format_syndication_header_compose_shape() -> None:
    assert (
        format_syndication_header_compose(
            prefix="@alice",
            stamp=" • 2026-02-17",
            media_hint=" • media:2",
            url="https://x.com/u/status/1",
        )
        == "@alice • 2026-02-17 • media:2 → https://x.com/u/status/1"
    )


def test_format_syndication_header_line_preserves_len_error_behavior() -> None:
    with pytest.raises(TypeError):
        format_syndication_header_line(
            user={"name": "Alice"},
            created_at="2026-02-17",
            photos=3,
            url="https://x.com/u/status/1",
        )


def test_format_syndication_error_fallback_truncates_payload_repr() -> None:
    syn_data = {"blob": "a" * 5000}
    out = format_syndication_error_fallback("https://x.com/u/status/1", syn_data)
    assert out.startswith("Tweet → https://x.com/u/status/1\n")
    payload_part = out.split("\n", 1)[1]
    assert len(payload_part) == 4000


def test_format_syndication_error_payload_repr_truncates() -> None:
    assert len(format_syndication_error_payload_repr({"blob": "a" * 5000})) == 4000


def test_format_syndication_error_payload_max_chars_constant() -> None:
    assert format_syndication_error_payload_max_chars() == 4000


def test_extract_syndication_photo_urls_variants() -> None:
    photos = [
        {"url": "u1"},
        {"media_url_https": "u2"},
        {"media_url": "u3"},
        {"url": None},
        "u4",
    ]
    assert extract_syndication_photo_urls(photos) == ["u1", "u2", "u3", "u4"]

    # Preserve current semantics for string payloads (iterates chars).
    assert extract_syndication_photo_urls("ab") == ["a", "b"]

    with pytest.raises(TypeError):
        extract_syndication_photo_urls(1)


def test_append_syndication_photo_item_urls_variants() -> None:
    urls: List[str] = []
    append_syndication_photo_item_urls(urls=urls, photo={"url": "u1"})
    append_syndication_photo_item_urls(urls=urls, photo="u2")
    append_syndication_photo_item_urls(urls=urls, photo={"url": None})
    assert urls == ["u1", "u2"]


def test_extract_syndication_photo_url_from_dict_priority() -> None:
    assert (
        extract_syndication_photo_url_from_dict(
            {"url": "u1", "media_url_https": "u2", "media_url": "u3"}
        )
        == "u1"
    )
    assert extract_syndication_photo_url_from_dict({"media_url_https": "u2"}) == "u2"
    assert extract_syndication_photo_url_from_dict({"media_url": "u3"}) == "u3"
    assert extract_syndication_photo_url_from_dict({}) is None


def test_extract_syndication_photo_urls_from_item_variants() -> None:
    assert extract_syndication_photo_urls_from_item({"url": "u1"}) == ["u1"]
    assert extract_syndication_photo_urls_from_item("u2") == ["u2"]
    assert extract_syndication_photo_urls_from_item({"url": None}) == []
    assert extract_syndication_photo_urls_from_item(1) == []


def test_syndication_photo_url_is_usable_variants() -> None:
    assert syndication_photo_url_is_usable("u1")
    assert not syndication_photo_url_is_usable("")
    assert not syndication_photo_url_is_usable(None)
    assert not syndication_photo_url_is_usable(1)


def test_x_syn_probe_budget_timeout_s_caps_and_offsets() -> None:
    assert x_syn_probe_budget_timeout_s(9.0) == 4.5
    assert x_syn_probe_budget_timeout_s(2.2) == 3.2


def test_x_syn_timeout_cap() -> None:
    assert x_syn_timeout_cap(9.0, 3.0) == 3.0
    assert x_syn_timeout_cap(1.2, 3.0) == 1.2


def test_x_syn_connect_read_timeout_s() -> None:
    assert x_syn_connect_read_timeout_s(9.0) == 3.0
    assert x_syn_connect_read_timeout_s(1.2) == 1.2


def test_x_syn_timeout_with_offset_and_cap() -> None:
    assert x_syn_timeout_with_offset_and_cap(2.2, 1.0, 4.5) == 3.2
    assert x_syn_timeout_with_offset_and_cap(9.0, 1.0, 4.5) == 4.5


def test_x_syn_quick_request_timeouts_caps_and_offsets() -> None:
    assert x_syn_quick_request_timeouts(9.0) == (3.0, 3.0, 3.5)
    assert x_syn_quick_request_timeouts(1.2) == (1.2, 1.2, 1.7)


def test_build_syndication_photo_payload_shape() -> None:
    payload = build_syndication_photo_payload("caption text", ["u1", "u2"])
    assert payload == {
        "text": "caption text",
        "photos": [{"url": "u1"}, {"url": "u2"}],
    }

    payload_none = build_syndication_photo_payload(None, [])
    assert payload_none == {"text": None, "photos": []}


def test_build_syndication_photo_items_shape() -> None:
    assert build_syndication_photo_items(["u1", "u2"]) == [{"url": "u1"}, {"url": "u2"}]
    assert build_syndication_photo_items([]) == []


def test_format_twitter_syndication_images_log_line_with_and_without_msg_id() -> None:
    assert (
        format_twitter_syndication_images_log_line(
            ["https://pbs.twimg.com/media/abc.jpg"],
            msg_id=123,
        )
        == "route.twitter.syndication | images=1 | pbs.twimg.com | msg_id=123"
    )
    assert (
        format_twitter_syndication_images_log_line(["not a url"])
        == "route.twitter.syndication | images=1 | n/a"
    )


def test_format_twitter_syndication_images_detail_shape() -> None:
    assert (
        format_twitter_syndication_images_detail(
            image_count=2,
            host_label="pbs.twimg.com",
            suffix=" | msg_id=123",
        )
        == "route.twitter.syndication | images=2 | pbs.twimg.com | msg_id=123"
    )


def test_format_twitter_syndication_msg_suffix_variants() -> None:
    assert format_twitter_syndication_msg_suffix(123) == " | msg_id=123"
    assert format_twitter_syndication_msg_suffix(None) == ""


def test_format_twitter_syndication_host_label_variants() -> None:
    assert format_twitter_syndication_host_label("pbs.twimg.com") == "pbs.twimg.com"
    assert format_twitter_syndication_host_label("") == "n/a"


def test_format_twitter_syndication_image_count_variants() -> None:
    assert format_twitter_syndication_image_count([]) == 0
    assert format_twitter_syndication_image_count(["u1", "u2"]) == 2


def test_resolve_first_image_host_variants() -> None:
    assert resolve_first_image_host(["https://pbs.twimg.com/media/abc.jpg"]) == "pbs.twimg.com"
    assert resolve_first_image_host(["not a url"]) == ""
    assert resolve_first_image_host([]) == ""


def test_parse_image_host_variants() -> None:
    assert parse_image_host("https://pbs.twimg.com/media/abc.jpg") == "pbs.twimg.com"
    assert parse_image_host("not a url") == ""


def test_resolve_first_image_url_variants() -> None:
    assert resolve_first_image_url(["u1", "u2"]) == "u1"
    assert resolve_first_image_url([]) == ""


def test_first_list_item_or_empty_variants() -> None:
    assert first_list_item_or_empty(["u1", "u2"]) == "u1"
    assert first_list_item_or_empty([]) == ""


def test_normalize_probed_image_urls_variants() -> None:
    assert normalize_probed_image_urls(["u1"]) == ["u1"]
    assert normalize_probed_image_urls(None) == []
    assert normalize_probed_image_urls([]) == []


def test_probed_image_urls_or_empty_variants() -> None:
    assert probed_image_urls_or_empty(["u1"]) == ["u1"]
    assert probed_image_urls_or_empty(None) == []
    assert probed_image_urls_or_empty([]) == []


def test_build_twitter_image_probe_result_normalizes_urls() -> None:
    assert build_twitter_image_probe_result("123", ["u1"]) == ("123", ["u1"])
    assert build_twitter_image_probe_result("123", None) == ("123", [])


@pytest.mark.asyncio
async def test_resolve_and_probe_twitter_images_delegates_and_normalizes() -> None:
    calls = {}

    def _resolve_status(url, tweet_id=None):
        calls["resolve"] = (url, tweet_id)
        return "123"

    async def _probe_images(url, status_id):
        calls["probe"] = (url, status_id)
        return ["u1", "u2"]

    status_id, image_urls = await resolve_and_probe_twitter_images(
        url="https://x.com/u/status/1",
        tweet_id="hint",
        resolve_status_id=_resolve_status,
        probe_images=_probe_images,
    )

    assert status_id == "123"
    assert image_urls == ["u1", "u2"]
    assert calls["resolve"] == ("https://x.com/u/status/1", "hint")
    assert calls["probe"] == ("https://x.com/u/status/1", "123")


@pytest.mark.asyncio
async def test_resolve_and_probe_twitter_images_normalizes_empty_probe() -> None:
    status_id, image_urls = await resolve_and_probe_twitter_images(
        url="https://x.com/u/status/1",
        resolve_status_id=lambda _url, tweet_id=None: "123",
        probe_images=lambda _url, _status_id: _async_none(),
    )
    assert status_id == "123"
    assert image_urls == []


async def _async_none():
    return None
