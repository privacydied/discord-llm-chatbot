# Repository Guidelines

## Project Structure & Module Organization
- Core bot code lives in `bot/` (routing, backends, memory, vision, TTS/STT, commands).
- Command cogs are under `bot/commands/`; startup/Discord wiring is in `bot/core/` and `bot/main.py`.
- Tests live in `tests/` and should mirror `bot/` layout where practical.
- Utility or diagnostic scripts belong in `utils/` (not repo root).
- Config/docs: `.env.example`, `bot/config.py`, `docs/`, and prompts in `prompts/`.

## Build, Test, and Development Commands
- `uv run python -m bot.main` — run the bot locally.
- `uv run -m pytest tests/test_router.py -v` — run one file.
- `uv run -m pytest tests/test_router.py::test_name -v` — run one test.
- `uv run -m pytest -k "ambient or router" -q` — run tests matching a keyword.
- `uv run ruff check .` and `uv run ruff format .` — lint/format.
- `uv run bandit -q -r bot` — security scan.
- `uv run playwright install chromium` — required once for screenshot/web flows.

Always use `uv run` for reproducible interpreter/environment behavior.

**Never run `uv run -m pytest -q` (full suite, no filter).** The suite is large (2000+ tests, ~5 min) — always scope to the files/tests relevant to the change. Only run it unfiltered if explicitly asked.

## Coding Style & Naming Conventions
- Python 3.11+ with async-first patterns; avoid blocking calls in event loop code.
- Follow existing module patterns and naming (`snake_case` functions/files, clear boundary-specific exceptions).
- Prefer constants/enums over magic values.
- Keep functions small and flat where possible (target: <=30 lines, nesting <=3 when practical).
- Preserve dual-sink logging behavior (Rich console + JSONL file).

## Testing Guidelines
- Framework: `pytest` with `asyncio_mode = auto` (`pytest.ini`).
- Put tests in `tests/`, named `test_*.py`.
- Cover routing, retries/timeouts, and failure paths for new behavior.
- Run targeted tests for the files/areas touched; do not run the full unfiltered suite (see Build/Test commands above) unless explicitly asked.

## Commit & Pull Request Guidelines
- Use imperative commit subjects, optionally with scope/tags, e.g.:
  - `fix(router): handle X timeout [REH][PA]`
- Common tags: `[CA] [REH] [CSD] [IV] [RM] [CMV] [SFT] [PA]`.
- PRs should include: problem statement, root cause, patch summary, test evidence, and env/config changes.

## Security & Operations Notes
- Never commit secrets; keep `.env` local and update `.env.example` when adding settings.
- Enforce bounded timeouts/retries for external I/O and log retry attempts.
- For production incidents, provide root cause, minimal repro, and a ready patch.
