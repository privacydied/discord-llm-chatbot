"""
Result aggregation for multimodal processing.
Combines per-item results into a single coherent response for the text flow.
"""

from dataclasses import dataclass
from typing import List, Optional
from .modality import InputModality, InputItem
from .util.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedResult:
    """Represents the result of processing a single input item."""

    item_index: int
    modality: InputModality
    item_name: str
    result_text: str
    success: bool
    duration: Optional[float] = None
    attempts: int = 1


class ResultAggregator:
    """Aggregates multiple item processing results into a single coherent response."""

    def __init__(self):
        self.results: List[ProcessedResult] = []

    def add_result(
        self,
        item_index: int,
        item: InputItem,
        modality: InputModality,
        result_text: str,
        success: bool = True,
        duration: Optional[float] = None,
        attempts: int = 1,
    ) -> None:
        """Add a processing result for an item."""
        item_name = self._get_item_display_name(item)

        result = ProcessedResult(
            item_index=item_index,
            modality=modality,
            item_name=item_name,
            result_text=result_text,
            success=success,
            duration=duration,
            attempts=attempts,
        )

        self.results.append(result)
        logger.debug(
            f"Added result {item_index}: {modality.name} - {item_name} ({'success' if success else 'failed'})"
        )

    def get_aggregated_prompt(self, original_text: str = "") -> str:
        """
        Generate a single aggregated prompt for the text flow.

        Args:
            original_text: Original message text content (after removing URLs)

        Returns:
            Single aggregated prompt containing all item results
        """
        if not self.results:
            return original_text

        # Build the aggregated prompt
        parts = []

        # Add header with summary
        total_items = len(self.results)
        successful_items = sum(1 for r in self.results if r.success)
        failed_items = total_items - successful_items

        if total_items == 1:
            parts.append("I processed 1 input from your message:")
        else:
            if failed_items == 0:
                parts.append(f"I processed {total_items} inputs from your message:")
            else:
                parts.append(
                    f"I processed {total_items} inputs from your message ({successful_items} successful, {failed_items} failed):"
                )

        parts.append("")  # Empty line

        # Add each item result with provenance
        for result in self.results:
            # Create section header
            status_emoji = "✅" if result.success else "❌"
            modality_display = self._get_modality_display_name(result.modality)

            header = f"### [{result.item_index + 1}/{total_items}] {status_emoji} {modality_display}: {result.item_name}"

            # Add timing info if available
            if result.duration is not None:
                header += f" ({result.duration:.1f}s"
                if result.attempts > 1:
                    header += f", {result.attempts} attempts"
                header += ")"
            elif result.attempts > 1:
                header += f" ({result.attempts} attempts)"

            parts.append(header)
            parts.append("")  # Empty line

            # Add the actual result content
            if result.result_text.strip():
                parts.append(result.result_text.strip())
            else:
                parts.append("(No content extracted)")

            parts.append("")  # Empty line between items

        # Add original text content if present
        if original_text and original_text.strip():
            parts.append("### Original Message Text:")
            parts.append("")
            parts.append(original_text.strip())
            parts.append("")

        # Join all parts
        aggregated = "\n".join(parts).strip()

        logger.info(
            f"📋 Aggregated {total_items} results into {len(aggregated)} character prompt"
        )
        return aggregated

    def get_summary_stats(self) -> dict:
        """Get summary statistics for logging/metrics."""
        if not self.results:
            return {}

        total_items = len(self.results)
        successful_items = sum(1 for r in self.results if r.success)
        failed_items = total_items - successful_items

        # Calculate average duration for successful items
        successful_durations = [
            r.duration for r in self.results if r.success and r.duration is not None
        ]
        avg_duration = (
            sum(successful_durations) / len(successful_durations)
            if successful_durations
            else None
        )

        # Count modalities
        modality_counts = {}
        for result in self.results:
            modality_name = result.modality.name
            modality_counts[modality_name] = modality_counts.get(modality_name, 0) + 1

        return {
            "total_items": total_items,
            "successful_items": successful_items,
            "failed_items": failed_items,
            "success_rate": successful_items / total_items if total_items > 0 else 0,
            "avg_duration": avg_duration,
            "modality_counts": modality_counts,
            "total_attempts": sum(r.attempts for r in self.results),
        }

    def _get_item_display_name(self, item: InputItem) -> str:
        """Get a human-readable display name for an item."""
        if item.source_type == "attachment":
            return item.payload.filename
        elif item.source_type == "url":
            # Extract meaningful part of URL
            url = item.payload
            if "/" in url:
                # Try to get filename or last path segment
                parts = url.rstrip("/").split("/")
                for part in reversed(parts):
                    if part and "." in part:  # Likely a filename
                        return part
                    elif part and len(part) > 3:  # Meaningful path segment
                        return part
                return parts[-1] if parts[-1] else url
            return url
        elif item.source_type == "embed":
            if hasattr(item.payload, "title") and item.payload.title:
                return item.payload.title
            elif hasattr(item.payload, "url") and item.payload.url:
                return self._get_item_display_name(
                    InputItem("url", item.payload.url, 0)
                )
            return "embed"

        return f"{item.source_type}_item"

    def _get_modality_display_name(self, modality: InputModality) -> str:
        """Get a human-readable display name for a modality."""
        display_names = {
            InputModality.TEXT_ONLY: "Text",
            InputModality.SINGLE_IMAGE: "Image",
            InputModality.MULTI_IMAGE: "Images",
            InputModality.VIDEO_URL: "Video URL",
            InputModality.AUDIO_VIDEO_FILE: "Audio/Video File",
            InputModality.PDF_DOCUMENT: "PDF Document",
            InputModality.PDF_OCR: "PDF (OCR)",
            InputModality.GENERAL_URL: "URL",
            InputModality.SCREENSHOT_URL: "URL Screenshot",
            InputModality.UNKNOWN: "Unknown",
        }
        return display_names.get(modality, modality.name)
