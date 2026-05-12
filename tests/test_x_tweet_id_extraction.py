import pytest

from bot.x_api_client import XApiClient


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://x.com/user/status/1234567890123456789", "1234567890123456789"),
        ("https://twitter.com/user/status/1234567890123456789", "1234567890123456789"),
        (
            "https://mobile.twitter.com/user/status/1234567890123456789",
            "1234567890123456789",
        ),
        (
            "https://m.twitter.com/user/status/1234567890123456789",
            "1234567890123456789",
        ),
        ("https://x.com/i/status/1234567890123456789", "1234567890123456789"),
        ("https://x.com/i/web/status/1234567890123456789", "1234567890123456789"),
        (
            "https://fxtwitter.com/user/status/1234567890123456789",
            "1234567890123456789",
        ),
        (
            "https://vxtwitter.com/user/status/1234567890123456789",
            "1234567890123456789",
        ),
        ("https://fixupx.com/user/status/1234567890123456789", "1234567890123456789"),
        # Syndication/embed-ish query forms
        (
            "https://example.com/embed?tweet_id=1234567890123456789",
            "1234567890123456789",
        ),
        (
            "https://example.com/syndication?id=1234567890123456789",
            "1234567890123456789",
        ),
    ],
)
def test_extract_tweet_id_variants(url: str, expected: str):
    assert XApiClient.extract_tweet_id(url) == expected


def test_extract_tweet_id_none_cases():
    assert XApiClient.extract_tweet_id("") is None
    assert XApiClient.extract_tweet_id("https://x.com/user") is None
    assert XApiClient.extract_tweet_id("https://example.com/not-twitter") is None
