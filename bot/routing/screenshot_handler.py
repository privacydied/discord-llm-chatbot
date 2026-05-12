"""Screenshot URL handler extracted from bot/router.py.

Handles URLs that need screenshot capture + vision analysis.
Screenshots are explicitly command-gated (e.g., !ss).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from .base import RouteContext, RouteHandler
from ..see import see_infer
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..modality import InputItem

logger = get_logger(__name__)


class ScreenshotHandler(RouteHandler):
    """Extracted screenshot handler from the router.

    Handles:
    - Screenshot capture via external API
    - Vision analysis of the captured screenshot
    - Progress callbacks for streaming UX
    """

    def can_handle(self, ctx: RouteContext) -> bool:
        """Return True when this handler can process the context."""
        return ctx.source_type == "url"

    async def handle(self, ctx: RouteContext) -> str:
        """Handle screenshot URL capture and analysis.

        Returns the captured screenshot analysis as a string.
        Progress callbacks are invoked at key stages if provided.
        """
        try:
            url = ctx.payload
            progress_cb = ctx.progress_cb

            logger.info("Taking screenshot of URL: %s", url)

            if progress_cb:
                await progress_cb("validate", 1)

            # Lazy-import to avoid circular deps
            from ..utils.external_api import external_screenshot

            # Preparation phase
            if progress_cb:
                await progress_cb("prepare", 2)

            if progress_cb:
                await progress_cb("capture", 3)

            screenshot_path = await external_screenshot(url)
            if not screenshot_path:
                logger.warning("Screenshot API did not return an image for %s", url)
                return (
                    f"Could not capture a screenshot for: {url}. "
                    "Please try again later."
                )

            if progress_cb:
                await progress_cb("saved", 4)

            logger.info("Screenshot saved at: %s. Sending to VL.", screenshot_path)

            try:
                if progress_cb:
                    await progress_cb("analyze", 5)

                analysis = await see_infer(
                    image_path=screenshot_path,
                    prompt=(
                        f"Analyze this screenshot from {url}. "
                        "Summarize the main content, visible text, "
                        "and any important details. Be concise."
                    ),
                )

                if analysis:
                    if progress_cb:
                        await progress_cb("done", 6)
                    return f"Screenshot content from {url}: {analysis}"
                else:
                    if progress_cb:
                        await progress_cb("done", 6)
                    return (
                        f"Captured screenshot from {url}, but vision "
                        "analysis returned no content."
                    )

            except Exception as vl_err:
                logger.error(
                    "Vision analysis failed for %s: %s",
                    screenshot_path,
                    vl_err,
                    exc_info=True,
                )
                if progress_cb:
                    await progress_cb("done", 6)
                return (
                    f"Captured screenshot from {url}, but could not "
                    "analyze it right now."
                )

        except Exception as e:
            logger.error("Error taking screenshot of URL: %s", e, exc_info=True)
            return f"Failed to screenshot URL: {ctx.payload}"


# Module-level singleton for router delegation
screenshot_handler: ScreenshotHandler = ScreenshotHandler()


async def handle_screenshot_url(
    item: Any,
    progress_cb: Optional[Callable[[str, int], Any]] = None,
) -> str:
    """Compatibility function for router delegation.

    Maps InputItem-like objects to RouteContext, mirroring the original
    router._handle_screenshot_url signature.
    """
    ctx = RouteContext(
        source_type=item.source_type,
        payload=item.payload,
        item=item,
        progress_cb=progress_cb,
    )
    return await screenshot_handler.handle(ctx)
