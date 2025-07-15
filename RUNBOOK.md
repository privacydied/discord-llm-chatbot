# Discord LLM Chatbot Runbook

This document provides clear, step-by-step instructions for setting up, testing, and running the Discord LLM Chatbot.

## 1. Project Setup

These steps will guide you through setting up the project environment from scratch.

### Prerequisites

- Python 3.11
- `uv` (the project's package manager)

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd discord-llm-chatbot
    ```

2.  **Create Virtual Environment**

    This project uses a `.venv` virtual environment managed by `uv`.

    ```bash
    # Create the virtual environment
    python3.11 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate
    ```

3.  **Install Dependencies**

    Install all required packages from `requirements.txt` using `uv`.

    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**

    Copy the sample environment file and fill in your specific API keys and configuration details.

    ```bash
    cp .env-sample .env
    ```

    Now, edit the `.env` file with your credentials:
    - `DISCORD_TOKEN`: Your Discord bot token.
    - `TEXT_BACKEND`: The text generation backend (`openai` or `ollama`).
    - `KOKORO_MODEL_PATH`: Path to the Kokoro TTS model.
    - `KOKORO_VOICE_PACK_PATH`: Path to the Kokoro voice pack.
    - Any other required API keys (e.g., `OPENAI_API_KEY`).

## 2. Running the Tests

To ensure all components are working correctly, run the full test suite using `pytest`.

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Run all tests
uv run pytest
```

All tests should pass before running the bot.

## 3. Running the Bot

Once setup and testing are complete, you can start the bot using the provided shell script.

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Start the bot
./discord-llm-chatbot.sh start
```

The bot should now be online and responding to commands in your Discord server.

To stop the bot, you can run:

```bash
./discord-llm-chatbot.sh stop
```
