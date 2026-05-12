"""Tests for media capability URL allowlisting."""

from bot.media_capability import MediaCapabilityDetector


def test_instagram_alias_hosts_are_media_capable_domains(tmp_path, monkeypatch):
    """Instagram mirror/proxy hosts should pass the existing media-capable allowlist."""
    monkeypatch.setattr("bot.media_capability.CACHE_DIR", tmp_path / "probes")
    detector = MediaCapabilityDetector()

    assert detector._is_whitelisted_domain(
        "https://www.kkinstagram.com/reel/DWjQyv_Dt4k/"
    )
    assert detector._is_whitelisted_domain(
        "https://d.vxinstagram.com/reel/DWjQyv_Dt4k/"
    )
    assert detector._is_whitelisted_domain(
        "https://www.instagram.com/reel/DWjQyv_Dt4k/"
    )
    assert not detector._is_whitelisted_domain(
        "https://random.example.com/reel/DWjQyv_Dt4k/"
    )
