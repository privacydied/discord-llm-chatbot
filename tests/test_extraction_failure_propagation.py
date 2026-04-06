"""Tests verifying that extraction failure propagates correctly through
the multimodal pipeline and does NOT become false-success."""
import pytest
from dataclasses import dataclass
from bot.exceptions import DispatchEmptyError
from bot.result_aggregator import ResultAggregator
from bot.modality import InputModality, InputItem
from bot.web_extraction_service import ExtractionResult


class DummyItem:
    """Minimal InputItem-like object for testing."""
    def __init__(self, payload="<payload>"):
        self.source_type = "url"
        self.payload = payload


# -- ResultAggregator failure-accounting tests --

def test_failed_extraction_not_counted_as_successful():
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://example.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content from URL: https://example.com",
                   success=False)
    stats = agg.get_summary_stats()
    assert stats["successful_items"] == 0
    assert stats["failed_items"] == 1


def test_failed_extraction_not_in_aggregated_prompt():
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://example.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content from URL: https://example.com",
                   success=False)
    prompt = agg.get_aggregated_prompt("")
    # Failed items should not appear in content sections, only summary headers
    assert "Could not extract content" not in prompt


def test_failed_item_does_not_trigger_implicit_ack_prompt():
    """A failed GENERAL_URL should not cause media-only ack injection."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://example.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content",
                   success=False)
    prompt = agg.get_aggregated_prompt("")
    assert "acknowledge the content" not in prompt.lower()


def test_mixed_success_and_failure_only_successful_in_prompt():
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://good.com"),
                   InputModality.GENERAL_URL,
                   result_text="Web content from https://good.com: Hello world",
                   success=True)
    agg.add_result(1, DummyItem("https://bad.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content from URL: https://bad.com",
                   success=False)
    stats = agg.get_summary_stats()
    assert stats["successful_items"] == 1
    assert stats["failed_items"] == 1
    prompt = agg.get_aggregated_prompt("")
    assert "Hello world" in prompt
    # The failed item should appear only in the failure-count summary line,
    # NOT in the actual content sections
    sections = prompt.split("### ")
    content_sections = [s for s in sections if "URL:" in s]
    assert len(content_sections) == 1
    assert "Hello world" in content_sections[0]


def test_all_failed_items_produce_empty_prompt_no_text_sources():
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://bad.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content",
                   success=False)
    prompt = agg.get_aggregated_prompt("")
    stats = agg.get_summary_stats()
    assert stats["successful_items"] == 0
    # has_text_sources should be False, so no implicit ack prompt
    assert "acknowledge" not in prompt.lower()


def test_has_scraped_text_requires_success_flag():
    """has_scraped_text must be False when the item has success=False."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://bad.com"),
                   InputModality.GENERAL_URL,
                   result_text="Error message text",
                   success=False)
    prompt = agg.get_aggregated_prompt("")
    # Failed GENERAL_URL should NOT count as scraped text source, so has_text_sources
    # is False -- this means the implicit ack prompt should NOT be present.
    # The prompt may still include a summary line like "I processed 1 input"
    assert "acknowledge the content" not in prompt.lower()


# -- DispatchEmptyError propagation tests --

def test_dispatch_empty_error_is_exception():
    with pytest.raises(DispatchEmptyError):
        raise DispatchEmptyError("test")


def test_dispatch_empty_error_message():
    try:
        raise DispatchEmptyError("Could not extract content from URL: https://example.com")
    except DispatchEmptyError as e:
        assert "Could not extract content from URL" in str(e)
        assert "https://example.com" in str(e)


# -- ExtractionResult contract tests --

def test_extraction_result_failure_message():
    er = ExtractionResult(success=False, tier_used="B",
                          error="version mismatch server=1.59 client=1.55")
    assert er.success is False
    msg = er.to_message()
    assert "failed" in msg.lower()


def test_extraction_result_success_message():
    er = ExtractionResult(success=True, tier_used="A",
                          canonical_url="https://example.com",
                          text="Hello")
    assert er.success is True
    msg = er.to_message()
    assert "Hello" in msg
    assert "failed" not in msg.lower()


# -- Router text-flow gate tests --

def test_all_failed_url_produces_empty_prompt_no_original_text():
    """When a single GENERAL_URL item fails and there is no original text,
    get_aggregated_prompt must return empty string so the router gate
    (not aggregated_prompt.strip()) will NOT invoke _invoke_text_flow."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://financetwitter.com/some-link"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content",
                   success=False)
    # No original text — just the mention-stripped message content
    prompt = agg.get_aggregated_prompt("")
    assert prompt == "", f"Expected empty prompt, got: {prompt!r}"

    # Verify the router gate condition would correctly skip text flow
    assert not prompt.strip(), "Prompt must not be routable"


def test_all_failed_item_with_only_whitespace_original_text():
    """When original_text is just whitespace, it should not count as routable content."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://example.com/document"),
                   InputModality.PDF_DOCUMENT,
                   result_text="Could not extract content",
                   success=False)
    prompt = agg.get_aggregated_prompt("   ")
    assert prompt == "", f"Expected empty prompt, got: {prompt!r}"


def test_failed_item_with_meaningful_original_text_still_routes():
    """When there IS meaningful original text alongside failed items,
    the prompt should be non-empty so text flow can still answer the user text."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://bad.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content",
                   success=False)
    prompt = agg.get_aggregated_prompt("what do you think about this?")
    # Should include the original question — user asked a real question
    assert "what do you think" in prompt
    # Should NOT include the failed extraction content
    assert "Could not extract" not in prompt


def test_mixed_failure_prompts_only_successful_content():
    """When some items succeeded and some failed, the prompt should only
    contain successful content — never failed extraction residue."""
    agg = ResultAggregator()
    agg.add_result(0, DummyItem("https://good.com"),
                   InputModality.GENERAL_URL,
                   result_text="Web content from https://good.com: Python is great",
                   success=True)
    agg.add_result(1, DummyItem("https://bad.com"),
                   InputModality.GENERAL_URL,
                   result_text="Could not extract content from URL: https://bad.com",
                   success=False)
    prompt = agg.get_aggregated_prompt("")
    assert "Python is great" in prompt
    assert "Could not extract content" not in prompt
    assert "https://bad.com" not in prompt
