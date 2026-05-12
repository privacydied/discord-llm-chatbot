import types

from bot.modality import collect_image_urls_from_message


def _mk_embed(image_url=None, thumb_url=None, thumb_proxy=None, embed_type="rich"):
    img = types.SimpleNamespace(url=image_url) if image_url else None
    thumb = None
    if thumb_url or thumb_proxy:
        thumb = types.SimpleNamespace(url=thumb_url, proxy_url=thumb_proxy)
    return types.SimpleNamespace(type=embed_type, image=img, thumbnail=thumb)


def _mk_message(embeds=None):
    return types.SimpleNamespace(attachments=[], embeds=embeds or [])


def test_collect_image_urls_prefers_thumbnail_proxy_over_thumbnail_url_when_no_image() -> (
    None
):
    msg = _mk_message(
        embeds=[
            _mk_embed(
                image_url=None,
                thumb_url="https://cdn.discordapp.com/thumb.png",
                thumb_proxy="https://media.discordapp.net/thumb.png",
            )
        ]
    )

    refs = collect_image_urls_from_message(msg)

    assert len(refs) == 1
    assert refs[0].url == "https://media.discordapp.net/thumb.png"
    assert "https://cdn.discordapp.com/thumb.png" in refs[0].fallback_urls


def test_collect_image_urls_uses_embed_image_and_adds_thumbnail_candidates() -> None:
    msg = _mk_message(
        embeds=[
            _mk_embed(
                image_url="https://cdn.discordapp.com/image.png",
                thumb_url="https://cdn.discordapp.com/thumb.png",
                thumb_proxy="https://media.discordapp.net/thumb.png",
            )
        ]
    )

    refs = collect_image_urls_from_message(msg)

    assert len(refs) == 1
    assert refs[0].url == "https://cdn.discordapp.com/image.png"
    assert "https://media.discordapp.net/thumb.png" in refs[0].fallback_urls
    assert "https://cdn.discordapp.com/thumb.png" in refs[0].fallback_urls
