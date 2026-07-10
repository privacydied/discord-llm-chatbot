"""Regression test for /imgedit: the safety filter and every provider plugin
in bot/vision/unified_adapter.py read `VisionRequest.input_image_data` (raw
bytes), never `.input_image` (a Path). Before this fix, /imgedit only set
`.input_image`, so every edit request was silently blocked by the safety
filter's "missing_input_image" check and never reached a provider. [SFT][REH]
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.commands.vision_commands import VisionCommands
from bot.vision.types import VisionJob, VisionJobState, VisionTask


def _mock_bot():
    bot = MagicMock()
    return bot


def _mock_interaction():
    interaction = MagicMock()
    interaction.user.id = 111
    interaction.guild.id = 222
    interaction.channel.id = 333
    interaction.id = 444
    interaction.response.send_message = AsyncMock()
    return interaction


def _mock_image_attachment():
    attachment = MagicMock()
    attachment.filename = "photo.png"
    attachment.size = 1024
    attachment.url = "https://cdn.discordapp.com/attachments/1/2/photo.png"
    return attachment


@pytest.mark.asyncio
async def test_imgedit_populates_input_image_data_bytes(tmp_path) -> None:
    with patch("bot.commands.vision_commands.load_config", return_value={"VISION_ENABLED": True, "VISION_EPHEMERAL_RESPONSES": True}):
        cog = VisionCommands(_mock_bot())

    fake_path = tmp_path / "downloaded.png"
    fake_path.write_bytes(b"raw-png-bytes")
    cog._download_attachment = AsyncMock(return_value=fake_path)

    submitted_job = VisionJob(job_id="job-1", request=MagicMock(estimated_cost=0.05), state=VisionJobState.QUEUED)
    submitted_job.provider_assigned = None
    cog._orchestrator = MagicMock()
    cog._orchestrator.submit_job = AsyncMock(return_value=submitted_job)
    cog._monitor_job_progress = AsyncMock()

    await cog.imgedit_command.callback(
        cog,
        _mock_interaction(),
        image=_mock_image_attachment(),
        prompt="give this man a beard",
    )
    await asyncio.sleep(0)  # let the background monitor task run to completion

    cog._orchestrator.submit_job.assert_awaited_once()
    submitted_request = cog._orchestrator.submit_job.await_args.args[0]

    assert submitted_request.task == VisionTask.IMAGE_TO_IMAGE
    assert submitted_request.input_image == fake_path
    # The actual regression: providers/safety-filter need bytes, not a Path.
    assert submitted_request.input_image_data == b"raw-png-bytes"


@pytest.mark.asyncio
async def test_imgedit_request_passes_real_safety_filter(tmp_path) -> None:
    """End-to-end-ish: the exact VisionRequest /imgedit builds must be
    approved by the real (unmocked) safety filter, since before this fix it
    was unconditionally BLOCKED by "missing_input_image"."""
    from bot.vision.safety_filter import VisionSafetyFilter
    from bot.vision.types import VisionRequest

    fake_path = tmp_path / "downloaded.png"
    fake_path.write_bytes(b"raw-png-bytes")

    request = VisionRequest(
        task=VisionTask.IMAGE_TO_IMAGE,
        prompt="give this man a beard",
        user_id="111",
        input_image=fake_path,
        input_image_data=fake_path.read_bytes(),
    )

    safety_filter = VisionSafetyFilter({})
    result = await safety_filter.validate_request(request)

    assert result.approved is True
    assert "missing_input_image" not in result.detected_issues
