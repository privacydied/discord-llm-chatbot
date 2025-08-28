import asyncio
import json
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from bot.voice.publisher import VoiceMessagePublisher, IS_VOICE_MESSAGE_FLAG


class _FakeResp:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload or {}
        self.history = None
        self.request_info = None

    async def text(self):
        return json.dumps(self._payload)

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self):
        self.captured = None

    def post(self, url, headers=None, json=None, timeout=None):
        # Capture the payload for assertions
        self.captured = {"url": url, "headers": headers, "json": json, "timeout": timeout}
        return _FakeResp(status=200, payload={"id": "100"})


@pytest.mark.asyncio
async def test__post_voice_message_uses_8192_flag(monkeypatch):
    pub = VoiceMessagePublisher()
    session = _FakeSession()

    # Execute
    res = await pub._post_voice_message(
        session=session,
        channel_id=123456,
        token="X",
        uploaded_filename="upload_abc",
        duration_secs=2.34567,
        waveform_b64="AAECAwQ=",
        reply_to_id=789,
    )

    assert isinstance(res, dict)
    assert session.captured is not None
    payload = session.captured["json"]
    assert payload["flags"] == IS_VOICE_MESSAGE_FLAG
    assert isinstance(payload["attachments"], list) and len(payload["attachments"]) == 1
    att = payload["attachments"][0]
    assert att["filename"] == "voice-message.ogg"
    assert att["uploaded_filename"] == "upload_abc"
    assert att["waveform"] == "AAECAwQ="
    # Duration rounded to 3 decimals
    assert abs(att["duration_secs"] - 2.346) < 1e-6
    # Ensure reply reference present
    assert payload.get("message_reference", {}).get("message_id") == str(789)


class _DummyChannel:
    def __init__(self, cid: int, fetch_msg: MagicMock | AsyncMock | None = None):
        self.id = cid
        # Allow overriding fetch behavior
        self._fetch = fetch_msg or AsyncMock(return_value=MagicMock())

    async def fetch_message(self, mid):
        return await self._fetch(mid)


class _DummyMessage:
    def __init__(self, cid: int, mid: int):
        self.channel = _DummyChannel(cid)
        self.id = mid


@pytest.mark.asyncio
async def test_publish_success_flow(monkeypatch, tmp_path: Path):
    # Config
    monkeypatch.setattr(
        "bot.voice.publisher.load_config",
        lambda: {"VOICE_ENABLE_NATIVE": True, "DISCORD_TOKEN": "token"},
    )

    # Prepare a minimal WAV
    wav_path = tmp_path / "in.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 24000)  # 1 sec silence

    # Stub helpers to avoid external tools
    out_ogg = tmp_path / "out.ogg"
    out_ogg.write_bytes(b"OggS\x00\x02test")
    monkeypatch.setattr(
        "bot.voice.publisher.transcode_to_ogg_opus",
        AsyncMock(return_value=out_ogg),
    )
    monkeypatch.setattr(
        "bot.voice.publisher.compute_waveform_b64",
        lambda p: "AA==",
    )

    pub = VoiceMessagePublisher()
    monkeypatch.setattr(pub, "_probe_duration", AsyncMock(return_value=1.0))
    monkeypatch.setattr(
        pub,
        "_attachments_create",
        AsyncMock(return_value={
            "attachments": [{"upload_url": "https://upload", "upload_filename": "u_fn"}]
        }),
    )
    monkeypatch.setattr(pub, "_upload_file", AsyncMock(return_value=None))
    monkeypatch.setattr(pub, "_post_voice_message", AsyncMock(return_value={"id": "4242"}))

    msg = _DummyMessage(cid=98765, mid=321)
    result = await pub.publish(message=msg, wav_path=wav_path)

    assert result.ok is True
    assert result.ogg_path and result.ogg_path.suffix == ".ogg"
    # message may be fetched or None depending on stub; our stub returns a MagicMock
    assert result.message is not None


@pytest.mark.asyncio
async def test_publish_blocks_on_50173(monkeypatch, tmp_path: Path):
    # Config
    monkeypatch.setattr(
        "bot.voice.publisher.load_config",
        lambda: {"VOICE_ENABLE_NATIVE": True, "DISCORD_TOKEN": "token"},
    )

    # Minimal WAV for first call (audio prep runs before upload flow)
    wav_path = tmp_path / "in.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(b"\x00\x00" * 24000)

    # Stubs
    out_ogg = tmp_path / "out.ogg"
    out_ogg.write_bytes(b"OggS\x00\x02test")
    monkeypatch.setattr(
        "bot.voice.publisher.transcode_to_ogg_opus",
        AsyncMock(return_value=out_ogg),
    )
    monkeypatch.setattr(
        "bot.voice.publisher.compute_waveform_b64",
        lambda p: "AA==",
    )

    pub = VoiceMessagePublisher()
    monkeypatch.setattr(pub, "_probe_duration", AsyncMock(return_value=1.0))

    # attachments.create returns 50173 error -> should block channel
    err = Exception("bad request")
    err.status = 400
    err.message = json.dumps({"code": 50173})
    attachments_mock = AsyncMock(side_effect=err)
    monkeypatch.setattr(pub, "_attachments_create", attachments_mock)

    msg = _DummyMessage(cid=12345, mid=999)
    res1 = await pub.publish(message=msg, wav_path=wav_path)
    assert res1.ok is False

    # Second call should early skip due to block; does not need a valid wav file
    res2 = await pub.publish(message=msg, wav_path=wav_path)
    assert res2.ok is False
    # Ensure upload flow didn't fire again
    assert attachments_mock.call_count == 1
