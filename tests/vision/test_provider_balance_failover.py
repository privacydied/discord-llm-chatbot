"""Provider-side regression tests for the 2026-08-27 image_to_image failure.

Novita answered 403 NOT_ENOUGH_BALANCE; the adapter filed it as a generic
PROVIDER_ERROR, the gateway rewrapped it as "please try again", nothing was
benched (so every later job repeated the same failure), and the reason a
provider had silently vanished from the ladder was never logged. [REH][SFT]
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.vision.gateway import VisionGateway
from bot.vision.types import VisionError, VisionErrorType, VisionRequest, VisionTask
from bot.vision.unified_adapter import UnifiedVisionAdapter, _is_balance_error

NOVITA_403 = '{"code":403, "reason":"NOT_ENOUGH_BALANCE", "message":"insufficient balance", "metadata":{}}'


@pytest.fixture
def adapter():
    config = {
        "VISION_ALLOWED_PROVIDERS": ["nvidia", "novita", "together"],
        "VISION_DEFAULT_PROVIDER": "nvidia",
        "VISION_API_KEY": "test-vision-api-key-0123456789",
        "NVIDIA_NIM_API_KEY": "test-nvidia-api-key-0123456789",
    }
    return UnifiedVisionAdapter(config)


# --- Balance-error classification -----------------------------------------


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, NOVITA_403),
        (403, '{"error":"not_enough_balance"}'),
        (403, "Insufficient funds on this account"),
        (402, "anything at all"),
        (403, "quota exhausted for this billing period"),
    ],
)
def test_balance_errors_are_detected(status, body):
    assert _is_balance_error(status, body) is True


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, "Forbidden: content policy violation"),
        (403, "invalid api key"),
        (401, "unauthorized"),
        (429, "rate limited"),
        (500, "internal error, credit"),  # 5xx is an outage, not a payment problem
        (200, ""),
    ],
)
def test_non_balance_errors_are_not_misfiled(status, body):
    assert _is_balance_error(status, body) is False


# --- Quota bench -----------------------------------------------------------


def test_quota_bench_removes_provider_until_cooldown_expires(adapter, monkeypatch):
    assert adapter._is_provider_healthy("novita") is True

    adapter._bench_provider_for_quota("novita")
    assert adapter._is_provider_healthy("novita") is False
    assert adapter._quota_cooldown_remaining("novita") > 0

    # Fast-forward past the cooldown.
    import bot.vision.unified_adapter as ua

    later = ua.time.monotonic() + adapter._quota_cooldown_s() + 1
    monkeypatch.setattr(ua.time, "monotonic", lambda: later)
    assert adapter._quota_cooldown_remaining("novita") == 0
    assert adapter._is_provider_healthy("novita") is True


def test_quota_bench_is_per_provider(adapter):
    adapter._bench_provider_for_quota("novita:qwen-image")
    assert adapter._is_provider_healthy("novita") is False
    assert adapter._is_provider_healthy("nvidia") is True


def test_zero_cooldown_disables_the_bench(adapter):
    adapter.config["VISION_PROVIDER_QUOTA_COOLDOWN_S"] = 0
    adapter._bench_provider_for_quota("novita")
    assert adapter._is_provider_healthy("novita") is True


def test_config_reload_clears_the_bench(adapter):
    adapter._bench_provider_for_quota("novita")
    assert adapter._is_provider_healthy("novita") is False
    adapter.update_config(dict(adapter.config))
    assert adapter._is_provider_healthy("novita") is True


# --- Provider filtering is diagnosable ------------------------------------


def test_filter_reports_why_each_provider_was_dropped(adapter):
    adapter._bench_provider_for_quota("novita")
    kept, dropped = adapter._filter_provider_order(["novita", "nvidia", "ghostprovider"])

    assert "novita" not in kept
    assert dropped["novita"].startswith("quota_benched_")
    assert dropped["ghostprovider"] in {"not_in_allowlist", "not_initialized"}
    assert "nvidia" in kept


def test_filter_flags_missing_credentials(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_has_valid_credentials", lambda name: name != "together")
    _kept, dropped = adapter._filter_provider_order(["together", "novita"])
    assert dropped["together"] == "no_credentials"


def test_filter_dedupes_provider_variants(adapter):
    kept, _dropped = adapter._filter_provider_order(["novita:qwen-image", "novita:txt2img", "novita"])
    assert kept == ["novita"]


@pytest.mark.asyncio
async def test_no_capable_provider_raises_a_truthful_error(adapter, monkeypatch):
    """With every provider benched, the user must not be told to 'try again'."""
    for name in list(adapter.providers):
        adapter._bench_provider_for_quota(name)

    request = VisionRequest(task=VisionTask.IMAGE_TO_IMAGE, prompt="edit it", user_id="1", input_image_data=b"x")
    with pytest.raises(VisionError) as exc:
        await adapter.submit(request)

    assert exc.value.error_type == VisionErrorType.SYSTEM_ERROR
    assert "try again" not in (exc.value.user_message or "").lower()


@pytest.mark.asyncio
async def test_quota_failure_benches_the_provider_for_the_next_job(adapter):
    """The production loop: every image job re-paid a round trip to an empty account."""
    novita = adapter.providers["novita"]
    novita.submit = AsyncMock(
        side_effect=VisionError(
            message="Novita.ai balance/quota exhausted (403)",
            error_type=VisionErrorType.QUOTA_EXCEEDED,
            user_message="out of credit",
        ),
    )
    request = VisionRequest(task=VisionTask.IMAGE_TO_IMAGE, prompt="edit it", user_id="1", input_image_data=b"x")

    with pytest.raises(VisionError):
        await adapter.submit(request)

    assert adapter._is_provider_healthy("novita") is False


# --- Gateway keeps the diagnosis ------------------------------------------


@pytest.mark.asyncio
async def test_gateway_preserves_vision_error_type_and_message():
    gateway = VisionGateway.__new__(VisionGateway)
    gateway.logger = MagicMock()
    gateway.active_jobs = {}

    import asyncio

    gateway._active_jobs_lock = asyncio.Lock()
    original = VisionError(
        message="Novita.ai balance/quota exhausted (403)",
        error_type=VisionErrorType.QUOTA_EXCEEDED,
        user_message="The image provider's account is out of credit, so image generation is unavailable right now.",
    )
    gateway.adapter = MagicMock()
    gateway.adapter.submit = AsyncMock(side_effect=original)

    request = VisionRequest(task=VisionTask.IMAGE_TO_IMAGE, prompt="edit it", user_id="1", input_image_data=b"x")
    with pytest.raises(VisionError) as exc:
        await gateway.submit_job(request)

    assert exc.value is original
    assert exc.value.error_type == VisionErrorType.QUOTA_EXCEEDED
    assert "out of credit" in exc.value.user_message


@pytest.mark.asyncio
async def test_gateway_still_wraps_unexpected_exceptions():
    gateway = VisionGateway.__new__(VisionGateway)
    gateway.logger = MagicMock()
    gateway.active_jobs = {}

    import asyncio

    gateway._active_jobs_lock = asyncio.Lock()
    gateway.adapter = MagicMock()
    gateway.adapter.submit = AsyncMock(side_effect=RuntimeError("boom"))

    request = VisionRequest(task=VisionTask.TEXT_TO_IMAGE, prompt="a cat", user_id="1")
    with pytest.raises(VisionError) as exc:
        await gateway.submit_job(request)

    assert exc.value.error_type == VisionErrorType.PROVIDER_ERROR


# --- Startup capability audit ---------------------------------------------


def test_task_coverage_reports_who_can_serve_each_task(adapter):
    coverage = adapter.audit_task_coverage()

    # nvidia/openrouter are text-to-image only; novita is the only i2i provider.
    assert "nvidia" in coverage["text_to_image"]
    assert "nvidia" not in coverage["image_to_image"]
    assert "novita" in coverage["image_to_image"]


def test_task_coverage_excludes_providers_without_credentials(adapter, monkeypatch):
    """A provider with no key must not look like image-edit coverage."""
    monkeypatch.setattr(adapter, "_has_valid_credentials", lambda name: name not in ("novita", "together"))
    coverage = adapter.audit_task_coverage()
    assert coverage["image_to_image"] == []
    assert "nvidia" in coverage["text_to_image"]


def test_task_coverage_respects_the_allowlist(adapter):
    adapter.allowed_providers = ["nvidia"]
    coverage = adapter.audit_task_coverage()
    assert coverage["image_to_image"] == []
    assert coverage["text_to_image"] == ["nvidia"]


def test_startup_warns_when_default_provider_cannot_serve_a_task(adapter):
    """VISION_DEFAULT_PROVIDER=nvidia + an image-edit request must be visible at boot."""
    from unittest.mock import MagicMock as _MagicMock

    adapter.logger = _MagicMock()
    adapter._log_task_coverage()

    warned = " ".join(str(call) for call in adapter.logger.warning.call_args_list)
    assert "cannot_serve" in warned
    assert "image_to_image" in warned
