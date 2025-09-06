import os
import copy
from bot.syndication.extract import extract_text_and_images_from_syndication
from bot.syndication.url_utils import upgrade_pbs_to_orig


def _pbs(url_base: str, size: str = "small", fmt: str = "jpg") -> str:
    # Construct a representative pbs url
    return f"https://pbs.twimg.com/media/{url_base}?format={fmt}&name={size}"


def _pbs_legacy(url_base: str, size_suffix: str = ":large") -> str:
    return f"https://pbs.twimg.com/media/{url_base}{size_suffix}"


def test_native_over_card_prefers_photos_only(monkeypatch):
    tw = {
        "text": "A caption",
        "photos": [
            {"url": _pbs("AAA111")},
        ],
        "card": {
            "binding_values": {
                "photo_image_full_size_large": {
                    "image_value": {"url": _pbs("CARD999", size="large")}
                }
            }
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["source"] == "photos"
    assert res["image_urls"] == [upgrade_pbs_to_orig(_pbs("AAA111"))]
    # Ensure card present but ignored
    assert res["had_card"] is True


def test_multi_photo_order_and_highres():
    tw = {
        "text": "Multi",
        "photos": [
            {"url": _pbs("A1", size="small")},
            {"url": _pbs("B2", size="large")},
            {"url": _pbs("C3", size="medium")},
        ],
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["image_urls"] == [
        upgrade_pbs_to_orig(_pbs("A1", size="small")),
        upgrade_pbs_to_orig(_pbs("B2", size="large")),
        upgrade_pbs_to_orig(_pbs("C3", size="medium")),
    ]


def test_quoted_fallback_when_primary_empty(monkeypatch):
    monkeypatch.setenv("SYND_INCLUDE_QUOTED_MEDIA", "1")
    tw = {
        "text": "",
        "quoted_tweet": {
            "photos": [
                {"url": _pbs("QQ1")}
            ]
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["source"] == "quoted_photos"
    assert res["image_urls"] == [upgrade_pbs_to_orig(_pbs("QQ1"))]


def test_card_fallback_when_no_native():
    tw = {
        "text": "Link card only",
        "card": {
            "binding_values": {
                "thumbnail_image_large": {
                    "image_value": {"url": _pbs("CARDX", size="small")}
                }
            }
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["source"] == "card"
    assert res["image_urls"] == [upgrade_pbs_to_orig(_pbs("CARDX", size="small"))]


def test_dedup_when_card_and_native_same_asset():
    base_native = _pbs("SAME1", size="small")
    base_card = _pbs("SAME1", size="large")
    tw = {
        "text": "",
        "photos": [
            {"url": base_native}
        ],
        "card": {
            "binding_values": {
                "photo_image_full_size": {
                    "image_value": {"url": base_card}
                }
            }
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["image_urls"] == [upgrade_pbs_to_orig(base_native)]


def test_video_poster_thumbnail_used_from_entities():
    tw = {
        "text": "Video",
        "extended_entities": {
            "media": [
                {
                    "type": "video",
                    "thumbnail_url": _pbs("THUMBVID", size="small"),
                }
            ]
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["image_urls"] == [upgrade_pbs_to_orig(_pbs("THUMBVID", size="small"))]


def test_legacy_size_suffix_normalized_to_orig():
    url = _pbs_legacy("LEG111", ":large")
    assert upgrade_pbs_to_orig(url).endswith("name=orig")


def test_regression_wrong_card_previously_selected():
    # Previously picked card image; ensure native photo chosen now
    tw = {
        "text": "A nice picture",
        "photos": [
            {"url": _pbs("NATIVEZ", size="small")}
        ],
        "card": {
            "binding_values": {
                "thumbnail_image": {"string_value": _pbs("CARDZ", size="large")}
            }
        }
    }
    res = extract_text_and_images_from_syndication(tw)
    assert res["source"] == "photos"
    assert res["image_urls"] == [upgrade_pbs_to_orig(_pbs("NATIVEZ"))]
