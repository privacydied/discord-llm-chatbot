# Text-to-Speech Setup Guide

The bot speaks with [Kyutai Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
(MIT, ~100M parameters, real-time on CPU, built-in voices and voice cloning).
It replaced the old Kokoro ONNX stack (custom G2P/IPA pipeline), which ran at
3–4x slower than real time and 2.5 GB RSS on the NAS.

## How it is wired

- `bot/tts/interface.py` — `TTSManager`: text cleaning, cache, WAV/OGG output,
  timeouts. Used by `!tts`/`!say`, voice-reply actions and voice messages.
- `bot/tts/engines/pocket.py` — `PocketTTSEngine`: HTTP client for a Pocket
  server (`POST /tts`, `GET /health`).
- `bot/tts/engines/stub.py` — beep engine for tests (`TTS_ENGINE=stub`).

Pocket runs as its **own server process in its own venv**, so torch never loads
into the bot and the bot's venv is untouched. With `POCKET_TTS_AUTOSTART=true`
the bot launches the server on the first TTS request, waits for `/health`, and
stops it again after `MODEL_IDLE_UNLOAD_SECONDS` without use (frees ~1.2 GB).
A server already running on `POCKET_TTS_URL` is used as-is and never stopped.

## One-time install (on the host that runs the bot)

```bash
mkdir -p /volume1/py/pocket-tts && cd /volume1/py/pocket-tts
export TMPDIR=$PWD/tmp && mkdir -p tmp   # Synology mounts /tmp noexec
uv venv --python 3.11 venv
VIRTUAL_ENV=venv uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch pocket-tts
```

Model weights download on first start into `POCKET_TTS_HF_HOME`.

## Configuration (.env)

```
TTS_ENGINE=pocket                          # pocket | stub
TTS_VOICE=alba
POCKET_TTS_LANGUAGE=english               # english (100M) | english_18m_2026-10 (2x faster)
POCKET_TTS_URL=http://127.0.0.1:8767      # 8765 is taken on the NAS
POCKET_TTS_AUTOSTART=true
POCKET_TTS_PYTHON=/volume1/py/pocket-tts/venv/bin/python
POCKET_TTS_HF_HOME=/volume1/py/pocket-tts/hf
POCKET_TTS_THREADS=2
POCKET_TTS_START_TIMEOUT_S=180
```

Built-in voices: `alba`, `anna`, `azelma`, `bill_boerst`, `caro_davy`,
`charles`, `cosette`, `eponine`, `eve`, `fantine`, `george`, `jane`, `javert`,
`jean`, `marius`, `mary`, `michael`, `paul`, `peter_yearsley`, `stuart_bell`,
`vera` (English) plus `daan`, `estelle`, `giovanni`, `juergen`, `lola`,
`rafael`. Samples: https://huggingface.co/kyutai/tts-voices.
`TTS_VOICE` also accepts an `hf://` or `https://` URL to a short reference clip
(voice cloning — only with the speaker's consent).

## Measured on the NAS (AMD Ryzen R1600, 2 threads)

| model | real-time factor | RAM |
|---|---|---|
| `english` (100M) | 0.88–0.91 | 1.2 GB |
| `english_18m_2026-10` | 0.39–0.46 | 655 MB |

Request timeouts scale with text length (10 s + 0.12 s/char, plus the start
timeout while the server is cold), so long replies are not cut off.

## TTS commands

1. `!tts on` / `!tts off` — voice replies for you
2. `!tts all on` / `!tts all off` — (admin) voice replies for everyone
3. `!speak <prompt>` — the bot answers the prompt (same context, tools and
   persona as an @mention) and replies as a voice note; `!speak` alone makes
   your next reply spoken
4. `!say <text>` / `!tts <text>` — speak the exact text as a voice note

## Troubleshooting

- Server log: `logs/pocket_tts.log`. Bot log events: `tts.pocket.spawned`,
  `tts.pocket.ready`, `tts.pocket.synth`, `tts.pocket.stopped`.
- By hand: `curl -s http://127.0.0.1:8767/health` → `{"status":"healthy"}`.
- `POCKET_TTS_PYTHON is not a valid interpreter` → the venv path is wrong.
- `Pocket TTS server exited (...)` → last line of `logs/pocket_tts.log` is in
  the error; usually a bad `POCKET_TTS_LANGUAGE` or voice name.

## Testing

```bash
uv run -m pytest tests/tts -q
```
