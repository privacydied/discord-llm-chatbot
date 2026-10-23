"""Free image-edit providers: Cloudflare Workers AI + Hugging Face Space [CA][REH][SFT]."""

from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from bot.vision import free_edit_providers as fep
from bot.vision.free_edit_providers import (
    CloudflareWorkersAIPlugin,
    HuggingFaceSpacePlugin,
    output_dims,
    space_base_url,
)
from bot.vision.types import VisionError, VisionErrorType, VisionTask
from bot.vision.unified_adapter import HFSPACE_ANONYMOUS_KEY, NormalizedRequest, UnifiedStatus, UnifiedVisionAdapter


def _png(w=800, h=600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, "PNG")
    return buf.getvalue()


def _request(**kw) -> NormalizedRequest:
    base = dict(task=VisionTask.IMAGE_TO_IMAGE, prompt="make it blue", input_image_data=_png(), seed=None)
    base.update(kw)
    return NormalizedRequest(**base)


class _Resp:
    def __init__(self, status=200, text="", body=b"", lines=(), headers=None):
        self.status, self._text, self._body = status, text, body
        self.headers = headers or {}
        self.content = self._lines(lines)

    @staticmethod
    async def _gen(lines):
        for line in lines:
            yield line.encode()

    def _lines(self, lines):
        return self._gen(lines)

    async def text(self):
        return self._text

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, url, kw):
        self.calls.append((method, url, kw))
        return self.responses.pop(0)

    def post(self, url, **kw):
        return self._next("POST", url, kw)

    def get(self, url, **kw):
        return self._next("GET", url, kw)


def _cf(session) -> CloudflareWorkersAIPlugin:
    plugin = CloudflareWorkersAIPlugin("cloudflare", {"account_id": "acct123"}, "cf-token-abcdefghij")
    plugin.session = session
    return plugin


def _form_fields(form) -> dict:
    return {f[0]["name"]: f[2] for f in form._fields}


# -- helpers ---------------------------------------------------------------


def test_output_dims_keep_aspect_and_limits():
    assert output_dims(800, 600) == (1024, 768)
    assert output_dims(600, 800) == (768, 1024)
    assert output_dims(4000, 100) == (1024, 256)  # min side clamp
    w, h = output_dims(1000, 999)
    assert w % 16 == 0 and h % 16 == 0


def test_space_base_url():
    assert space_base_url("black-forest-labs/FLUX.1-Kontext-dev") == "https://black-forest-labs-flux-1-kontext-dev.hf.space"


# -- cloudflare ---------------------------------------------------------------


async def test_cloudflare_edit_sends_small_input_and_returns_data_url():
    session = _Session([_Resp(200, json.dumps({"result": {"image": "QUJD"}, "success": True}))])
    plugin = _cf(session)
    job = await plugin.submit(_request(seed=7))
    method, url, kw = session.calls[0]
    assert url.endswith("/accounts/acct123/ai/run/@cf/black-forest-labs/flux-2-klein-4b")
    assert kw["headers"]["Authorization"] == "Bearer cf-token-abcdefghij"
    fields = _form_fields(kw["data"])
    assert fields["prompt"] == "make it blue" and fields["width"] == "1024" and fields["height"] == "768" and fields["seed"] == "7"
    with Image.open(io.BytesIO(fields["input_image_0"])) as sent:
        assert max(sent.size) < 512  # Cloudflare: inputs must be smaller than 512x512
    assert (await plugin.poll(job)).status == UnifiedStatus.COMPLETED
    assert (await plugin.fetch_result(job)).assets == ["data:image/png;base64,QUJD"]


@pytest.mark.parametrize(
    ("status", "text", "kind"),
    [
        (429, '{"errors":[{"code":4006,"message":"you have used up your daily free allocation of 10,000 neurons"}]}', VisionErrorType.QUOTA_EXCEEDED),
        (429, '{"errors":[{"message":"rate limited"}]}', VisionErrorType.RATE_LIMITED),
        (401, "bad token", VisionErrorType.AUTHENTICATION_ERROR),
        (400, '{"errors":[{"message":"input flagged as NSFW"}]}', VisionErrorType.CONTENT_FILTERED),
        (503, "down", VisionErrorType.SERVER_ERROR),
        (200, '{"result":{}}', VisionErrorType.PROVIDER_ERROR),
    ],
)
async def test_cloudflare_error_mapping(status, text, kind):
    with pytest.raises(VisionError) as err:
        await _cf(_Session([_Resp(status, text)])).submit(_request())
    assert err.value.error_type == kind


async def test_cloudflare_rejects_non_edit_and_bad_image():
    plugin = _cf(_Session([]))
    with pytest.raises(VisionError) as err:
        await plugin.submit(_request(task=VisionTask.TEXT_TO_IMAGE))
    assert err.value.error_type == VisionErrorType.UNSUPPORTED_TASK
    with pytest.raises(VisionError) as err:
        await plugin.submit(_request(input_image_data=b"not an image"))
    assert err.value.error_type == VisionErrorType.VALIDATION_ERROR


# -- hugging face space ------------------------------------------------------------


def _hf(session, key=HFSPACE_ANONYMOUS_KEY) -> HuggingFaceSpacePlugin:
    plugin = HuggingFaceSpacePlugin("hfspace", {}, key)
    plugin.session = session
    return plugin


def _hf_ok(final_lines):
    return [
        _Resp(200, json.dumps(["/tmp/gradio/x/input.png"])),
        _Resp(200, json.dumps({"event_id": "ev1"})),
        _Resp(200, lines=final_lines),
        _Resp(200, body=b"WEBP", headers={"Content-Type": "image/webp"}),
    ]


async def test_hfspace_edit_flow_anonymous():
    url = "https://black-forest-labs-flux-1-kontext-dev.hf.space/gradio_api/file=/tmp/out.webp"
    session = _Session(_hf_ok(["event: generating", "data: null", "event: complete", f'data: [{{"url": "{url}"}}, 42]']))
    plugin = _hf(session)
    job = await plugin.submit(_request())
    upload, call, stream, download = session.calls
    assert upload[1].endswith("/gradio_api/upload") and call[1].endswith("/gradio_api/call/infer")
    data = call[2]["json"]["data"]
    assert data[0]["path"] == "/tmp/gradio/x/input.png" and data[1] == "make it blue" and data[3] is True  # random seed
    assert stream[1].endswith("/gradio_api/call/infer/ev1") and download[1] == url
    assert all("Authorization" not in c[2].get("headers", {}) for c in session.calls)  # placeholder key never sent
    assert (await plugin.fetch_result(job)).assets == ["data:image/webp;base64," + base64.b64encode(b"WEBP").decode()]


async def test_hfspace_sends_real_token_and_fixed_seed():
    session = _Session(_hf_ok(["event: complete", 'data: [{"url": "https://x.hf.space/f.webp"}]']))
    await _hf(session, key="hf_realtoken123").submit(_request(seed=5))
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer hf_realtoken123"
    data = session.calls[1][2]["json"]["data"]
    assert data[2] == 5 and data[3] is False


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ('"You have exceeded your free GPU quota (60s requested vs. 30s left)."', VisionErrorType.QUOTA_EXCEEDED),
        ("null", VisionErrorType.PROVIDER_ERROR),
    ],
)
async def test_hfspace_space_errors(payload, kind):
    session = _Session(_hf_ok(["event: error", f"data: {payload}"])[:3])
    with pytest.raises(VisionError) as err:
        await _hf(session).submit(_request())
    assert err.value.error_type == kind


async def test_hfspace_busy_and_stream_without_result():
    with pytest.raises(VisionError) as err:
        await _hf(_Session([_Resp(429, "busy")])).submit(_request())
    assert err.value.error_type == VisionErrorType.RATE_LIMITED
    with pytest.raises(VisionError) as err:
        await _hf(_Session(_hf_ok(["event: heartbeat"])[:3])).submit(_request())
    assert err.value.error_type == VisionErrorType.PROVIDER_ERROR


def test_hfspace_timeout_ignores_short_global_timeout():
    plugin = HuggingFaceSpacePlugin("hfspace", {"timeout_ms": 45000, "timeout_s": 120}, HFSPACE_ANONYMOUS_KEY)
    assert plugin.session_timeout_s() == 120


# -- adapter wiring ---------------------------------------------------------------


def _adapter(**cfg) -> UnifiedVisionAdapter:
    base = {"VISION_ALLOWED_PROVIDERS": "cloudflare,hfspace,novita", "VISION_API_KEY": "novita-key-1234567890", "HF_SPACE_EDIT_ENABLED": True}
    base.update(cfg)
    return UnifiedVisionAdapter(base)


def test_adapter_edit_coverage_and_order():
    adapter = _adapter(CLOUDFLARE_ACCOUNT_ID="acct123", CLOUDFLARE_API_TOKEN="cf-token-abcdefghij")
    assert adapter.audit_task_coverage()["image_to_image"][:2] == ["cloudflare", "hfspace"]
    order, _ = adapter._filter_provider_order(["cloudflare", "hfspace", "novita"])
    assert order == ["cloudflare", "hfspace", "novita"]
    assert adapter.providers["cloudflare"].account_id == "acct123"
    assert adapter.audit_task_coverage()["text_to_image"] == ["novita"]  # free edit providers do not take generation


def test_adapter_without_cloudflare_keys_still_has_hfspace():
    adapter = _adapter()
    assert "cloudflare" not in adapter.providers
    assert adapter.audit_task_coverage()["image_to_image"][0] == "hfspace"


def test_hfspace_can_be_disabled():
    assert "hfspace" not in _adapter(HF_SPACE_EDIT_ENABLED=False).audit_task_coverage()["image_to_image"]


def test_hot_reload_never_hands_novita_key_to_free_providers():
    cfg = {"VISION_ALLOWED_PROVIDERS": "cloudflare,hfspace,novita", "VISION_API_KEY": "novita-key-1234567890", "CLOUDFLARE_ACCOUNT_ID": "acct123", "CLOUDFLARE_API_TOKEN": "cf-token-abcdefghij"}
    adapter = UnifiedVisionAdapter(dict(cfg))
    adapter.update_config(dict(cfg))
    assert adapter.providers["cloudflare"].api_key == "cf-token-abcdefghij"
    assert adapter.providers["hfspace"].api_key == HFSPACE_ANONYMOUS_KEY
    assert adapter.providers["novita"].api_key == "novita-key-1234567890"


async def test_adapter_falls_through_cloudflare_quota_to_hfspace(monkeypatch):
    from bot.vision.types import VisionRequest

    adapter = _adapter(CLOUDFLARE_ACCOUNT_ID="acct123", CLOUDFLARE_API_TOKEN="cf-token-abcdefghij")

    async def cf_quota(request):
        raise VisionError(message="4006", error_type=VisionErrorType.QUOTA_EXCEEDED, user_message="used up")

    async def hf_ok(request):
        return "hfspace_1"

    monkeypatch.setattr(adapter.providers["cloudflare"], "submit", cf_quota)
    monkeypatch.setattr(adapter.providers["hfspace"], "submit", hf_ok)
    response = await adapter.submit(VisionRequest(task=VisionTask.IMAGE_TO_IMAGE, prompt="make it blue", user_id="1", input_image_data=_png()))
    assert response.job_id == "hfspace:hfspace_1"
    assert adapter._quota_cooldown_remaining("cloudflare") > 0  # benched for the rest of the cooldown


def test_quota_cooldown_reaches_config():
    from bot.config import load_config

    assert "VISION_PROVIDER_QUOTA_COOLDOWN_S" in load_config()
    assert fep.CLOUDFLARE_INPUT_MAX_SIDE < 512
