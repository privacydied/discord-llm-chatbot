# Discord LLM Chatbot with RAG & Vision Support

This is a highly extensible, **local-first Discord chatbot** that can:

- Respond using your own LLMs (Ollama, OpenAI, etc)
- Perform **RAG** (Retrieval-Augmented Generation) via simple **keyword search** over your `.txt` knowledge base
- Analyze and describe images (via OpenAI Vision models)
- Save and recall per-user memories, preferences, and message logs
- Auto-search DuckDuckGo for current events or factual lookups
- Hot-reload configuration and system prompts

---

## Features

- **Supports Ollama and OpenAI models**
- **Keyword RAG**: Pull in relevant context from local text files with fast keyword matching  
- **Image-to-Text (Vision-Language)**: Uses OpenAI Vision (`gpt-4-vision-preview`) for images
- **Full Discord Bot Functionality:** Commands, user memories, preferences, logs
- **Easy Extensibility:** Plug in other text or vision backends easily
