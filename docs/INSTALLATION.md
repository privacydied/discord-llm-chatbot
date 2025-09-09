# Installation

> **Fill me in**
> - [ ] Confirm supported platforms and versions.
> - [ ] Add real Docker image name and deployment targets.

## Prerequisites
- Unix-like OS or Windows with WSL
- Python 3.11+
- `pip` or `uv` package manager
- [Docker](https://www.docker.com/) 20+

## Local Installation
```bash
git clone https://github.com/example/discord-llm-chatbot.git
cd discord-llm-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
cp .env.example .env
python run.py
```

## Docker
### compose.yaml
```yaml
services:
  bot:
    image: ghcr.io/example/discord-llm-chatbot:latest
    env_file: .env
    restart: unless-stopped
```

### One-shot
```bash
docker run --rm --env-file .env ghcr.io/example/discord-llm-chatbot:latest
```

## Platform Deploy Guides
### Docker
- Use the compose snippet above or run the image directly.

### Render
- TODO: describe deployment via Render.

### Fly.io
- TODO: describe fly.toml and secrets.

### Heroku
- TODO: container build and release steps.

### VPS
- Provision Python 3.11, install requirements, run as a systemd service.

### Kubernetes
- TODO: Deployment manifest and secret mounts.

## Post-install Verification
- Bot logs "Ready" after connecting to Discord.
- Run `/help` (or `!help`) in your guild to confirm command registration.
