# Discord Multimodal AI ChatBot (Ollama + OpenRouter + OpenAI)

This is an **advanced, multimodal Discord chatbot** that can use **Ollama (local), OpenRouter, or OpenAI** as a backend for text conversation and image (vision-language) understanding.
It features **automatic memory extraction**, per-user persistent memory, rich logging, smart context, permission & rate limiting, file & image processing, and auto web search.

> **Supports**:
>
> * Ollama (local models: Qwen, Llama, Mistral, etc.)
> * OpenRouter (huge model hub: Qwen, Llama, Claude, etc.)
> * OpenAI API (GPT-4o, GPT-4V, etc.)
> * Vision-Language for images: Qwen2.5-VL, Llama3-Vision, GPT-4V, etc.

---

## ✨ Features

* **Hybrid backend:** Switch text backend between Ollama (local) or OpenAI/OpenRouter (cloud)
* **Multimodal:** Automatically uses vision-language models for images, text models for chat
* **Automatic internet search:** Uses DuckDuckGo to fetch up-to-date answers for questions that require it
* **Per-user persistent memory:** Remembers facts, preferences, context, and "memories" about each user in their own file
* **Memory extraction:** Extracts facts/traits from regular conversation, not just manual commands
* **Manual memory management:** Users can add, view, or wipe their own memories with commands
* **Personalized conversation:** Maintains contextual tone, stats, and user notes per server or DM
* **User and DM logging:** All user and DM messages logged per-user (JSONL), DM logs in separate folder
* **File & image attachment support:** Reads and summarizes text files, can analyze attached images using vision models
* **Smart context:** Separate context/memory for each user per channel/DM, auto-prunes old messages
* **Permission system:** Commands like memory wipe require admin unless self-initiated
* **Rate limiting:** Prevents abuse and API spam
* **Robust error handling:** Friendly and in-character error messages (never dumps ugly API JSON)
* **Dynamic prompt loading:** Reloads your system prompt file on each reset
* **Bot nickname auto-update:** Can change its nickname to match the model (optional)
* **Web search command:** `!search` uses DuckDuckGo for facts/news
* **Full slash and prefix command support**
* **Batch processing:** Efficiently handles large sets of messages for memory extraction

---

## Prerequisites

* **Python 3.8+**
* **A Discord bot token** (see [Discord Developer Portal](https://discord.com/developers/applications))
* **Ollama installed locally** (for local LLM inference, [see here](https://ollama.com/))
* **OpenAI or OpenRouter API Key** (for cloud models, vision-language, and text)
* **Discord server with proper bot permissions**

---

## 🚀 Setup Instructions

### 1. Clone the Repository

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
