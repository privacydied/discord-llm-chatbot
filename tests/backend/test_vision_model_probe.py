"""Tests for free vision-model liveness probing + quarantine. [REH][SFT]"""

import json
import time

import httpx
import pytest

from bot.vision import free_model_discovery as disc
from bot.vision import free_model_probe as probe

OK_BODY = {"choices": [{"message": {"content": "blue"}}]}
DEAD_403 = {"error": {"message": "model:free is only available on agentic harnesses.", "code": 403}}
DEAD_404 = {"error": {"message": "No endpoints found for vendor/model:free.", "code": 404}}
RATE_LIMITED = {"error": {"message": "temporarily rate-limited upstream", "code": 429}}


@pytest.fixture(autouse=True)
def _isolated_probe(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_AUTO_DISCOVERY", "1")  # overrides the suite-wide off switch
    monkeypatch.setenv("VISION_DISCOVERY_PROBE", "1")
    monkeypatch.setenv("VISION_DISCOVERY_QUARANTINE_PATH", str(tmp_path / "quarantine.json"))
    monkeypatch.setenv("VISION_DISCOVERY_CACHE_PATH", str(tmp_path / "models.json"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    # The real .env must not leak its ladder into these assertions.
    monkeypatch.delenv("VL_MODEL", raising=False)
    monkeypatch.delenv("VISION_FALLBACK_MODELS", raising=False)
    disc._reset_for_tests()
    yield
    disc._reset_for_tests()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- Classification --------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (200, "good"),
        (400, "dead"),
        (403, "dead"),
        (404, "dead"),
        (410, "dead"),
        (422, "dead"),
        (401, "account"),
        (402, "account"),
        (429, "transient"),
        (500, "transient"),
        (503, "transient"),
    ],
)
def test_status_classification(status, expected):
    verdict, _reason = probe._classify(status, "body")
    assert verdict == expected


async def test_probe_one_marks_live_model_good():
    async with _client(lambda _r: httpx.Response(200, json=OK_BODY)) as client:
        assert await probe._probe_one(client, "vendor/good:free", "k") == ("vendor/good:free", "good", "")


async def test_probe_one_marks_404_dead():
    async with _client(lambda _r: httpx.Response(404, json=DEAD_404)) as client:
        model, verdict, reason = await probe._probe_one(client, "vendor/gone:free", "k")
    assert (model, verdict) == ("vendor/gone:free", "dead")
    assert "No endpoints" in reason


async def test_probe_one_marks_403_agentic_only_dead():
    async with _client(lambda _r: httpx.Response(403, json=DEAD_403)) as client:
        _model, verdict, _reason = await probe._probe_one(client, "vendor/harness:free", "k")
    assert verdict == "dead"


async def test_probe_one_marks_429_transient():
    async with _client(lambda _r: httpx.Response(429, json=RATE_LIMITED)) as client:
        _model, verdict, _reason = await probe._probe_one(client, "vendor/busy:free", "k")
    assert verdict == "transient"


async def test_probe_one_treats_network_error_as_transient():
    def boom(_request):
        raise httpx.ConnectError("dns failure")

    async with _client(boom) as client:
        _model, verdict, _reason = await probe._probe_one(client, "vendor/x:free", "k")
    assert verdict == "transient"


async def test_probe_one_treats_200_without_choices_as_transient():
    async with _client(lambda _r: httpx.Response(200, json={"choices": []})) as client:
        _model, verdict, _reason = await probe._probe_one(client, "vendor/x:free", "k")
    assert verdict == "transient"


def test_probe_body_sends_an_image_part():
    parts = probe._probe_body("m")["messages"][0]["content"]
    assert any(p["type"] == "image_url" for p in parts)
    assert probe._probe_body("m")["max_tokens"] == probe.PROBE_MAX_TOKENS


# --- probe_models ----------------------------------------------------------


async def _fake_probe_factory(verdicts: dict[str, str]):
    async def fake(_client, model, _key):
        verdict = verdicts.get(model, "good")
        return (model, verdict, "" if verdict == "good" else f"reason:{verdict}")

    return fake


async def test_probe_models_splits_good_transient_dead(monkeypatch):
    monkeypatch.setattr(
        probe,
        "_probe_one",
        await _fake_probe_factory({"b:free": "transient", "c:free": "dead"}),
    )
    report = await probe.probe_models(["a:free", "b:free", "c:free"])
    assert report.good == ["a:free"]
    assert report.transient == ["b:free"]
    assert list(report.dead) == ["c:free"]
    # Live models lead; blipped ones keep a place behind them.
    assert report.usable == ["a:free", "b:free"]


async def test_account_error_keeps_every_model(monkeypatch):
    """A bad key must not quarantine innocent models."""
    monkeypatch.setattr(probe, "_probe_one", await _fake_probe_factory({"a:free": "account", "b:free": "account"}))
    report = await probe.probe_models(["a:free", "b:free"])
    assert report.dead == {}
    assert report.transient == ["a:free", "b:free"]
    assert report.skipped is True


async def test_probe_skipped_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(probe, "resolve_openrouter_key", lambda: "")
    report = await probe.probe_models(["a:free"])
    assert report.skipped is True
    assert report.usable == ["a:free"]


async def test_probe_disabled_passes_models_through(monkeypatch):
    monkeypatch.setenv("VISION_DISCOVERY_PROBE", "0")
    report = await probe.probe_models(["a:free"])
    assert report.skipped is True
    assert report.usable == ["a:free"]


# --- Quarantine ------------------------------------------------------------


async def test_quarantine_round_trip_and_filtering():
    await probe.quarantine_models({"vendor/gone:free": "HTTP 404"})
    assert "vendor/gone:free" in probe.load_quarantine()
    assert probe.filter_quarantined(["vendor/gone:free", "vendor/ok:free"]) == ["vendor/ok:free"]


async def test_expired_quarantine_entries_are_ignored(tmp_path):
    path = probe.quarantine_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": {"vendor/old:free": {"reason": "x", "until": time.time() - 1}}}))
    assert probe.load_quarantine() == {}
    assert probe.filter_quarantined(["vendor/old:free"]) == ["vendor/old:free"]


async def test_corrupt_quarantine_file_is_tolerated():
    probe.quarantine_path().parent.mkdir(parents=True, exist_ok=True)
    probe.quarantine_path().write_text("{not json")
    assert probe.load_quarantine() == {}


# --- Discovery integration -------------------------------------------------


CATALOGUE = {
    "data": [
        {
            "id": f"vendor/m{i}-vl:free",
            "created": 100 - i,
            "context_length": 128000,
            "architecture": {"input_modalities": ["image"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0", "completion": "0"},
        }
        for i in range(4)
    ],
}


async def test_discovery_drops_dead_models_and_quarantines_them(monkeypatch):
    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setenv("VISION_DISCOVERY_MAX_MODELS", "2")
    monkeypatch.setattr(
        probe,
        "_probe_one",
        await _fake_probe_factory({"vendor/m0-vl:free": "dead", "vendor/m1-vl:free": "dead"}),
    )

    models = await disc.discover_free_vision_models(force=True)
    assert models == ["vendor/m2-vl:free", "vendor/m3-vl:free"]
    assert set(probe.load_quarantine()) == {"vendor/m0-vl:free", "vendor/m1-vl:free"}


async def test_quarantined_model_is_not_reselected_next_round(monkeypatch):
    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setenv("VISION_DISCOVERY_MAX_MODELS", "4")
    await probe.quarantine_models({"vendor/m0-vl:free": "HTTP 404"})
    monkeypatch.setattr(probe, "_probe_one", await _fake_probe_factory({}))

    models = await disc.discover_free_vision_models(force=True)
    assert "vendor/m0-vl:free" not in models


async def test_probe_failure_falls_back_to_unprobed_candidates(monkeypatch):
    async def fake_fetch(_timeout):
        return CATALOGUE

    async def exploding_probe(_models):
        raise RuntimeError("probe subsystem down")

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setattr(probe, "probe_models", exploding_probe)
    monkeypatch.setenv("VISION_DISCOVERY_MAX_MODELS", "2")

    models = await disc.discover_free_vision_models(force=True)
    assert models == ["vendor/m0-vl:free", "vendor/m1-vl:free"]


async def test_all_models_dead_keeps_previous_ladder(monkeypatch):
    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setattr(disc, "_memory_cache", (["vendor/previous:free"], 0.0))
    monkeypatch.setattr(
        probe,
        "_probe_one",
        await _fake_probe_factory({f"vendor/m{i}-vl:free": "dead" for i in range(4)}),
    )

    models = await disc.discover_free_vision_models(force=True)
    assert models == ["vendor/previous:free"]


# --- Operator-configured (.env) rungs are vetted too -----------------------


def test_env_ladder_candidates_parses_prefixes_and_skips_paid(monkeypatch):
    monkeypatch.setenv("VL_MODEL", "openrouter|vendor/head:free")
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "vendor/tail:free, openai/gpt-4o-mini ,vendor/head:free")
    assert probe.env_ladder_candidates() == ["vendor/head:free", "vendor/tail:free"]


def test_env_ladder_candidates_empty_when_unset(monkeypatch):
    monkeypatch.delenv("VL_MODEL", raising=False)
    monkeypatch.delenv("VISION_FALLBACK_MODELS", raising=False)
    assert probe.env_ladder_candidates() == []


async def test_dead_env_model_is_quarantined_and_leaves_the_ladder(monkeypatch):
    """The exact production failure: .env points at a 404 slug."""
    from bot.enhanced_retry import EnhancedRetryManager

    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setenv("VISION_DISCOVERY_MAX_MODELS", "2")
    monkeypatch.setenv("VL_MODEL", "vendor/retired:free")
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "vendor/retired:free")
    monkeypatch.setattr(probe, "_probe_one", await _fake_probe_factory({"vendor/retired:free": "dead"}))

    models = await disc.discover_free_vision_models(force=True)
    assert "vendor/retired:free" not in models
    assert "vendor/retired:free" in probe.load_quarantine()

    ladder = [pc.model for pc in EnhancedRetryManager().provider_configs["vision"]]
    assert "vendor/retired:free" not in ladder
    assert ladder[0] == "vendor/m0-vl:free"


async def test_live_env_model_survives_probing(monkeypatch):
    async def fake_fetch(_timeout):
        return CATALOGUE

    monkeypatch.setattr(disc, "_fetch_catalogue", fake_fetch)
    monkeypatch.setenv("VISION_DISCOVERY_MAX_MODELS", "4")
    monkeypatch.setenv("VL_MODEL", "vendor/mine:free")
    monkeypatch.setenv("VISION_FALLBACK_MODELS", "")
    monkeypatch.setattr(probe, "_probe_one", await _fake_probe_factory({}))

    await disc.discover_free_vision_models(force=True)
    assert probe.load_quarantine() == {}
