"""Tests for multimodal pipeline fixes:
- VL image capability filtering
- STT URL identity and cache keys
- URL classification for PDF/documents vs video.

Run with: python3 -m pytest tests/test_multimodal_fixes.py -v
"""

import pytest


class TestVLImageCapabilityFilter:
    """Tests for VL model image capability filtering."""

    def test_image_capable_model_kimi_vl(self) -> None:
        """Kimi VL models should be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        assert _is_image_capable_model("moonshotai/kimi-vl-a3b-thinking:free")
        assert _is_image_capable_model("moonshotai/kimi-vl-a3b:free")

    def test_image_capable_model_qwen_vl(self) -> None:
        """Qwen VL models should be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        assert _is_image_capable_model("qwen/qwen2.5-vl-32b-instruct:free")
        assert _is_image_capable_model("qwen/qwen2.5-vl-72b-instruct:free")

    def test_image_capable_model_google_gemma(self) -> None:
        """Google Gemma VL models should be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        assert _is_image_capable_model("google/gemma-3-27b-it:free")
        assert _is_image_capable_model("google/gemini-2.0-flash-exp:free")

    def test_text_only_model_mistral(self) -> None:
        """Mistral text-only models should NOT be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        # This model caused the VL regression - it doesn't support images
        assert not _is_image_capable_model("mistralai/mistral-small-3.2-24b-instruct:free")
        assert not _is_image_capable_model("mistralai/mistral-small-3.1-24b-instruct:free")

    def test_text_only_model_deepseek(self) -> None:
        """DeepSeek text models should NOT be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        assert not _is_image_capable_model("deepseek/deepseek-chat-v3-0324:free")
        assert not _is_image_capable_model("deepseek/deepseek-r1-0528:free")

    def test_image_capable_by_keyword(self) -> None:
        """Models with VL keywords should be detected as image-capable."""
        from bot.enhanced_retry import _is_image_capable_model

        # These should match by keyword heuristics
        assert _is_image_capable_model("some-vendor/new-vision-model:free")
        assert _is_image_capable_model("vendor/model-vl-7b:free")
        assert _is_image_capable_model("pixtral-12b-2409")

    def test_empty_and_none_models(self) -> None:
        """Empty and None model names should return False."""
        from bot.enhanced_retry import _is_image_capable_model

        assert not _is_image_capable_model("")
        assert not _is_image_capable_model(None)


class TestSTTUrlIdentity:
    """Tests for STT URL identity and cache key generation."""

    def test_youtube_watch_url_normalization(self) -> None:
        """YouTube watch URLs should normalize to canonical form."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url2 = "https://youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"

        norm1 = mgr._normalize_youtube_url(url1)
        norm2 = mgr._normalize_youtube_url(url2)

        assert norm1 == "youtube://video/dQw4w9WgXcQ"
        assert norm2 == "youtube://video/dQw4w9WgXcQ"

    def test_youtube_shorts_url_normalization(self) -> None:
        """YouTube Shorts URLs should normalize to same canonical form as watch URLs."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        shorts_url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        watch_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        norm_shorts = mgr._normalize_youtube_url(shorts_url)
        norm_watch = mgr._normalize_youtube_url(watch_url)

        # Both should normalize to the same canonical form
        assert norm_shorts == norm_watch
        assert norm_shorts == "youtube://video/dQw4w9WgXcQ"

    def test_youtube_different_videos_different_keys(self) -> None:
        """Different YouTube videos should produce different normalized URLs."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        url1 = "https://www.youtube.com/shorts/abc123def45"
        url2 = "https://www.youtube.com/shorts/xyz789ghi01"

        norm1 = mgr._normalize_youtube_url(url1)
        norm2 = mgr._normalize_youtube_url(url2)

        assert norm1 != norm2
        assert "abc123def45" in norm1
        assert "xyz789ghi01" in norm2

    def test_tiktok_url_normalization(self) -> None:
        """TikTok URLs should normalize to canonical form."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        url1 = "https://www.tiktok.com/@user/video/1234567890123456789"
        url2 = "https://vm.tiktok.com/ZMhXYZ123/"

        norm1 = mgr._normalize_tiktok_url(url1)
        norm2 = mgr._normalize_tiktok_url(url2)

        # First URL should extract video ID
        assert norm1 == "tiktok://video/1234567890123456789"
        # Short URL should normalize to path-based key
        assert norm2.startswith("tiktok://")

    def test_tiktok_player_url_detection(self) -> None:
        """TikTok player/embed URLs should be detected."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        player_url = "https://www.tiktok.com/player/v1/1234567890"
        normal_url = "https://www.tiktok.com/@user/video/1234567890"

        assert mgr._is_tiktok_player_url(player_url)
        assert not mgr._is_tiktok_player_url(normal_url)

    def test_video_identity_canonicalization(self) -> None:
        """Video identity should be canonical across URL variants."""
        from bot.video_ingest import VideoIngestionManager

        mgr = VideoIngestionManager

        # With metadata, should use extractor:id format
        metadata = {"extractor_key": "youtube", "id": "dQw4w9WgXcQ"}
        identity = mgr._canonicalize_video_identity("https://youtube.com/watch?v=dQw4w9WgXcQ", metadata)

        assert identity == "youtube:dQw4w9WgXcQ"


class TestURLClassification:
    """Tests for URL classification to correct modality."""

    @pytest.mark.asyncio
    async def test_pdf_url_classified_as_document(self) -> None:
        """PDF URLs should be classified as PDF_DOCUMENT."""
        from bot.modality import InputModality, _map_url_to_modality

        pdf_url = "https://example.com/document.pdf"
        modality = await _map_url_to_modality(pdf_url)

        assert modality == InputModality.PDF_DOCUMENT

    @pytest.mark.asyncio
    async def test_pdf_with_query_params_classified_as_document(self) -> None:
        """PDF URLs with query params should still be classified as PDF_DOCUMENT."""
        from bot.modality import InputModality, _map_url_to_modality

        pdf_url = "https://example.com/document.pdf?token=abc123"
        modality = await _map_url_to_modality(pdf_url)

        assert modality == InputModality.PDF_DOCUMENT

    @pytest.mark.asyncio
    async def test_image_url_classified_as_image(self) -> None:
        """Image URLs should be classified as SINGLE_IMAGE."""
        from bot.modality import InputModality, _map_url_to_modality

        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            url = f"https://example.com/image{ext}"
            modality = await _map_url_to_modality(url)
            assert modality == InputModality.SINGLE_IMAGE, f"Failed for {ext}"

    @pytest.mark.asyncio
    async def test_docx_url_classified_as_general(self) -> None:
        """Document URLs should be classified as GENERAL_URL for document processing."""
        from bot.modality import InputModality, _map_url_to_modality

        for ext in [".docx", ".doc", ".rtf", ".md", ".txt"]:
            url = f"https://example.com/document{ext}"
            modality = await _map_url_to_modality(url)
            assert modality == InputModality.GENERAL_URL, f"Failed for {ext}"

    @pytest.mark.asyncio
    async def test_youtube_watch_classified_as_video(self) -> None:
        """YouTube watch URLs should be classified as VIDEO_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        modality = await _map_url_to_modality(youtube_url)

        assert modality == InputModality.VIDEO_URL

    @pytest.mark.asyncio
    async def test_youtube_shorts_classified_as_video(self) -> None:
        """YouTube Shorts URLs should be classified as VIDEO_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        shorts_url = "https://www.youtube.com/shorts/abc123def45"
        modality = await _map_url_to_modality(shorts_url)

        assert modality == InputModality.VIDEO_URL

    @pytest.mark.asyncio
    async def test_tiktok_classified_as_video(self) -> None:
        """TikTok URLs should be classified as VIDEO_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        tiktok_url = "https://www.tiktok.com/@user/video/1234567890"
        modality = await _map_url_to_modality(tiktok_url)

        assert modality == InputModality.VIDEO_URL

    @pytest.mark.asyncio
    async def test_tiktok_player_classified_as_general(self) -> None:
        """TikTok player/embed URLs should be classified as GENERAL_URL (not VIDEO)."""
        from bot.modality import InputModality, _map_url_to_modality

        player_url = "https://www.tiktok.com/player/v1/1234567890"
        modality = await _map_url_to_modality(player_url)

        assert modality == InputModality.GENERAL_URL

    @pytest.mark.asyncio
    async def test_twitter_status_classified_as_general(self) -> None:
        """Twitter/X status URLs should be classified as GENERAL_URL for API-first."""
        from bot.modality import InputModality, _map_url_to_modality

        for domain in ["twitter.com", "x.com", "fxtwitter.com", "vxtwitter.com"]:
            url = f"https://{domain}/user/status/1234567890"
            modality = await _map_url_to_modality(url)
            assert modality == InputModality.GENERAL_URL, f"Failed for {domain}"

    @pytest.mark.asyncio
    async def test_nytimes_video_classified_as_video(self) -> None:
        """NYTimes video URLs should be classified as VIDEO_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        video_url = "https://www.nytimes.com/video/some-video"
        modality = await _map_url_to_modality(video_url)

        assert modality == InputModality.VIDEO_URL

    @pytest.mark.asyncio
    async def test_nytimes_article_classified_as_general(self) -> None:
        """NYTimes article URLs should be classified as GENERAL_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        article_url = "https://www.nytimes.com/2024/01/01/some-article"
        modality = await _map_url_to_modality(article_url)

        assert modality == InputModality.GENERAL_URL

    @pytest.mark.asyncio
    async def test_github_classified_as_general(self) -> None:
        """GitHub URLs should be classified as GENERAL_URL for web scraping."""
        from bot.modality import InputModality, _map_url_to_modality

        github_url = "https://github.com/user/repo"
        modality = await _map_url_to_modality(github_url)

        assert modality == InputModality.GENERAL_URL

    @pytest.mark.asyncio
    async def test_generic_article_classified_as_general(self) -> None:
        """Generic article URLs should be classified as GENERAL_URL."""
        from bot.modality import InputModality, _map_url_to_modality

        article_url = "https://example.com/blog/some-article"
        modality = await _map_url_to_modality(article_url)

        assert modality == InputModality.GENERAL_URL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
