# Discord LLM ChatBot

An advanced Discord chatbot with memory, web search, file processing, and AI-powered responses using Ollama as the backend.

## ✨ Features

- **AI-Powered Chat**: Natural conversations powered by Ollama's language models
- **Persistent Memory**: Remembers context and user preferences across conversations
- **Text-to-Speech**: Optional TTS functionality with DIA TTS
- **Web Search**: Integrated web search for up-to-date information
- **File Processing**: Read and process text files and PDFs
- **Modular Architecture**: Clean, organized codebase for easy extension
- **User Profiles**: Per-user settings and memory storage
- **Server Profiles**: Server-specific configuration and memory
- **Rate Limiting**: Built-in protection against API abuse
- **Comprehensive Logging**: Detailed logs for debugging and moderation

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Discord Bot Token ([Get one here](https://discord.com/developers/applications))
- Ollama installed and running ([Installation Guide](https://ollama.com/))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/discord-llm-chatbot.git
   cd discord-llm-chatbot
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```env
   DISCORD_TOKEN=your_discord_token_here
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3  # or your preferred model
   ```

5. Run the bot:
   ```bash
   python -m bot
   ```

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `DISCORD_TOKEN` | Your Discord bot token | ✅ | - |
| `OLLAMA_BASE_URL` | URL to your Ollama server | ❌ | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default Ollama model to use | ❌ | `llama3` |
| `COMMAND_PREFIX` | Bot command prefix | ❌ | `!` |
| `USER_PROFILE_DIR` | Directory for user profiles | ❌ | `user_profiles` |
| `SERVER_PROFILE_DIR` | Directory for server profiles | ❌ | `server_profiles` |
| `USER_LOGS_DIR` | Directory for user logs | ❌ | `user_logs` |
| `TEMP_DIR` | Directory for temporary files | ❌ | `temp` |
| `MAX_MEMORIES` | Maximum memories per user | ❌ | `100` |
| `MAX_SERVER_MEMORY` | Maximum memories per server | ❌ | `100` |

## 🤖 Commands

### User Commands
- `!help` - Show help information
- `!ping` - Check if the bot is alive
- `!tts [on/off]` - Toggle TTS for your messages
- `!memory add <text>` - Add a memory
- `!memory list` - List your memories
- `!memory clear` - Clear your memories
- `!search <query>` - Search the web
- `!say <text>` - Convert text to speech

### Admin Commands
- `!tts-all [on/off]` - Toggle TTS for all users (admin only)
- `!servermemories` - View server memories (admin only)
- `!clearservermemories` - Clear server memories (admin only)

## 🏗️ Project Structure

```
bot/
  ├── __init__.py         # Package initialization
  ├── main.py            # Bot entry point
  ├── config.py          # Configuration loading
  ├── memory.py          # User/server profile management
  ├── context.py         # Conversation context
  ├── tts.py             # Text-to-speech functionality
  ├── search.py          # Web search functionality
  ├── web.py             # Web content extraction
  ├── pdf_utils.py       # PDF processing
  ├── ollama.py          # AI model interactions
  ├── utils.py           # Utility functions
  ├── logs.py            # Logging setup
  └── commands/          # Command handlers
       ├── __init__.py   # Command registration
       ├── memory_cmds.py
       └── tts_cmds.py
```

## 📚 Documentation

### Memory System
The bot maintains two types of memory:
1. **User Memory**: Personal memories and preferences for each user
2. **Server Memory**: Shared memories within a server

Memories are stored as JSON files in the respective profile directories and loaded on demand.

### TTS System
The bot supports text-to-speech using DIA TTS. Users can enable/disable TTS for their messages, and server admins can control global TTS settings.

### Search System
Integrated web search allows the bot to provide up-to-date information by querying search engines and extracting relevant content.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.com/) for the amazing local LLM framework
- [Discord.py](https://github.com/Rapptz/discord.py) for the Discord API wrapper
- [DIA TTS](https://github.com/diart-team/dia-tts) for text-to-speech functionality

```sh
git clone https://github.com/Amine-LG/Discord-Ollama-ChatBot.git
cd Discord-Ollama-ChatBot
```

### 2. Create & Activate a Virtual Environment

```sh
python -m venv bot-env
source bot-env/bin/activate     # On Windows: bot-env\Scripts\activate
```

### 3. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root.
**Example:**

```env
# ===== DISCORD BOT SETTINGS =====
DISCORD_TOKEN=your_discord_bot_token_here
TEXT_BACKEND=openai   # openai or ollama

# ===== OPENAI / OPENROUTER SETTINGS =====
OPENAI_API_KEY=your_openrouter_or_openai_api_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENAI_TEXT_MODEL=qwen/qwen3-235b-a22b:free        # Chat model
VL_MODEL=qwen/qwen2.5-vl-72b-instruct:free          # Vision model for images

# ===== OLLAMA SETTINGS (uncomment if using local models) =====
# OLLAMA_BASE_URL=http://localhost:11434
# TEXT_MODEL=qwen3-235b-a22b

# ===== BOT BEHAVIOR / MEMORY =====
TEMPERATURE=0.7
TIMEOUT=120.0
CHANGE_NICKNAME=True
MAX_CONVERSATION_LENGTH=50
MAX_TEXT_ATTACHMENT_SIZE=20000
MAX_FILE_SIZE=2097152
PROMPT_FILE=prompts/prompt-pry-super-chill-v2.txt
MAX_USER_MEMORY=1000
MEMORY_SAVE_INTERVAL=30
```

> **Tip:**
>
> * Set `TEXT_BACKEND=openai` to use OpenAI/OpenRouter for text
> * Set `TEXT_BACKEND=ollama` and uncomment Ollama lines to use local inference

---

### 5. (Optional) Ensure Ollama is Running

If you want to use Ollama:

* Install [Ollama](https://ollama.com/) and run a model, e.g.:

  ```sh
  ollama run qwen3-235b-a22b
  ```

---

### 6. (Optional) Download Vision-Language Model

If using Ollama for VL (not default, since OpenRouter/OpenAI is recommended for vision), run:

```sh
ollama run qwen2.5-vl
```

*(Not required if using OpenRouter or OpenAI for vision tasks)*

---

### 7. Run the Bot

```sh
python main.py
```

---

## 🧑‍💻 Usage and Commands

### **Text vs Image**

* **Text message?** Bot uses your selected backend for chat.
* **Image attached?** Bot automatically uses the vision model (e.g., Qwen2.5-VL, GPT-4V, Llama3-Vision) via OpenAI-compatible API, with your message as the prompt.

### **Key Commands**

| Command                     | Description                                               | Permission            |
| --------------------------- | --------------------------------------------------------- | --------------------- |
| `!reset`                    | Reset the conversation context/log for your channel or DM | Anyone                |
| `!show-memories`            | View what the bot remembers about you                     | Anyone                |
| `!remember <text>`          | Manually add something to your memory                     | Anyone                |
| `!preference <key> <value>` | Set a personal preference (e.g., humor style, topics)     | Anyone                |
| `!forget [@user]`           | Forget all memories for yourself or another user          | Self/Admin            |
| `!search <query>`           | Search the web using DuckDuckGo for up-to-date info       | Anyone                |
| `!extract-memories [limit]` | Extract and save new memories from recent messages        | Anyone (rate-limited) |

#### **Image Inference**

* **Attach an image** and type a prompt (e.g. "What's happening here?").
* Bot replies with the vision-language model (VL\_MODEL).
* No special command required—**just upload an image**.

#### **File Attachments**

* Attach text files to have them summarized or analyzed (max size configurable).
* Large text files are split into chunks for long outputs.

#### **Automatic Web Search**

* If your question looks like a "fact" query (`who is...`, `when was...`, etc.), the bot automatically performs a DuckDuckGo search and uses the result as extra context for the AI model.

---

### **Memory Management**

* The bot **automatically extracts facts and personal info** from your chat (name, interests, location, dislikes, etc.).
* View your memories with `!show-memories`
* Add with `!remember something`
* Wipe your memory with `!forget`
* Stores up to `MAX_USER_MEMORY` facts per user (set in `.env`)
* All memories are stored **persistently in per-user files** (never lost on bot restart)

---

### **User Profiles & Logging**

* Each user gets their own persistent JSON file in `user_profiles/`
* All DMs are logged by user in `dm_logs/`
* All server messages are logged by user in `user_logs/`

---

### **Permissions & Rate Limiting**

* Some commands (e.g., wiping another user's memory) require admin permissions
* Most commands are rate-limited for abuse prevention (e.g., `!extract-memories` once per hour per user)

---

### **Bot Context & Prompt**

* Each server channel or DM keeps its own context window
* Context auto-prunes to save memory and avoid confusion
* **Prompt file hotloads** at every reset or bot start—just edit the file to update the system personality

---

### **Error Handling**

* The bot **never dumps raw API or JSON errors**—all error messages are friendly and "in character"
* E.g., if your image is rejected for inappropriate content, the bot responds with a helpful, non-technical message

---

### **Switching Backend (Ollama vs OpenAI/OpenRouter)**

* **To use local models**:

  * Set `TEXT_BACKEND=ollama` and set/uncomment the Ollama config variables
* **To use OpenAI/OpenRouter**:

  * Set `TEXT_BACKEND=openai` and fill in your API key and model names
* **Images** always use the VL model defined in `VL_MODEL` via OpenAI-compatible API

---

## 🔧 Advanced Configuration

* **Change system prompt/personality:**
  Edit the file at `PROMPT_FILE` and restart or use `!reset`
* **Log files:**

  * User memories and stats: `user_profiles/`
  * Server message logs: `user_logs/`
  * DM message logs: `dm_logs/`
* **Timeouts, file size limits, conversation length:**
  All adjustable in `.env`

---

## 📝 License

MIT License. See [LICENSE](LICENSE).

---

## 🤝 Contributing

Contributions are welcome!
Open a Pull Request or file an issue.

---

## 📚 Resources

* [Discord Developer Portal](https://discord.com/developers/docs/intro)
* [Ollama Documentation](https://ollama.com/)
* [OpenAI API Docs](https://platform.openai.com/docs/)
* [OpenRouter Docs](https://openrouter.ai/docs)
* [DuckDuckGo Instant Answer API](https://duckduckgo.com/api)

---

## Video Tutorials

[![Watch the video](https://img.youtube.com/vi/S7Dztn9qPSw/0.jpg)](https://youtu.be/S7Dztn9qPSw)

---

**Tip:** For best results, grant the bot all "Read Message", "Send Message", "Attach Files", "Embed Links", "Add Reactions", and "Manage Nicknames" permissions in your Discord server.
