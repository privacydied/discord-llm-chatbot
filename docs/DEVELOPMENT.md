# Development

> **Fill me in**
> - [ ] Confirm lint/test toolchain.
> - [ ] Add sandbox guild IDs or instructions.

## Local Workflow
```bash
git clone https://github.com/example/discord-llm-chatbot.git
cd discord-llm-chatbot
uv venv --python 3.11
source .venv/bin/activate
uv pip sync requirements.txt
cp .env.example .env
python run.py
```

## Hot Reload
Use the `--reload` flag (if implemented) or restart the bot after code changes.

## Lint & Test
```bash
pytest
ruff check .
```

## Testing Strategy
- Unit tests for utility functions and command handlers.
- Integration tests using mocks of Discord API responses.

## Sandbox Guild
- Create a private Discord server for testing.
- Use dummy accounts and limit access to trusted developers.
