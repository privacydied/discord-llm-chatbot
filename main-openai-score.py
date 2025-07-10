import os
import re
import json
import time
import random
import asyncio
import logging
import base64
import pathlib
from typing import Any, Callable, Coroutine, Dict, List, Optional, TypeVar
from functools import wraps
from dotenv import load_dotenv
from discord import Intents, Message
from discord.ext import commands
import discord
from datetime import datetime
import httpx
from collections import defaultdict

T = TypeVar('T', bound=Callable[..., Coroutine[Any, Any, Any]])

# ---- Configuration and Loading ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompt_file():
    prompt_path = os.getenv('PROMPT_FILE', 'prompt.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"Prompt file '{prompt_path}' not found, using fallback prompt.")
        return "You are an AI assistant on Discord."

def load_config():
    load_dotenv(override=True)
    return {
        "DISCORD_TOKEN": os.getenv('DISCORD_TOKEN'),
        "TEXT_BACKEND": os.getenv('TEXT_BACKEND', 'ollama'),  # 'ollama' or 'openai'
        "OLLAMA_BASE_URL": os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        "TEXT_MODEL": os.getenv('TEXT_MODEL', 'qwen3-235b-a22b'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "OPENAI_API_BASE": os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        "OPENAI_TEXT_MODEL": os.getenv('OPENAI_TEXT_MODEL', 'gpt-4o'),
        "VL_MODEL": os.getenv('VL_MODEL', 'gpt-4-vision-preview'),  # Could be DashScope or OpenAI
        "TEMPERATURE": float(os.getenv('TEMPERATURE', '0.7')),
        "TIMEOUT": float(os.getenv('TIMEOUT', '120.0')),
        "CHANGE_NICKNAME": os.getenv('CHANGE_NICKNAME', 'True').lower() == 'true',
        "MAX_CONVERSATION_LENGTH": int(os.getenv('MAX_CONVERSATION_LENGTH', '30')),
        "MAX_TEXT_ATTACHMENT_SIZE": int(os.getenv('MAX_TEXT_ATTACHMENT_SIZE', '20000')),
        "MAX_FILE_SIZE": int(os.getenv('MAX_FILE_SIZE', str(2 * 1024 * 1024))),
        "SYSTEM_PROMPT": load_prompt_file(),
        "MAX_USER_MEMORY": int(os.getenv('MAX_USER_MEMORY', '10')),
        "MEMORY_SAVE_INTERVAL": int(os.getenv('MEMORY_SAVE_INTERVAL', '30')),
    }

USER_PROFILE_DIR = pathlib.Path("user_profiles"); USER_PROFILE_DIR.mkdir(exist_ok=True)
USER_LOGS_DIR = pathlib.Path("user_logs"); USER_LOGS_DIR.mkdir(exist_ok=True)
DM_LOGS_DIR = pathlib.Path("dm_logs"); DM_LOGS_DIR.mkdir(exist_ok=True)
user_cache: Dict[str, dict] = {}

def default_profile(user_id=None, username=None):
    return {
        "discord_id": user_id if user_id else "",
        "username": username if username else "",
        "history": [],
        "last_active": 0,
        "tone": "neutral",
        "memories": [],
        "total_messages": 0,
        "first_seen": datetime.now().isoformat(),
        "preferences": {},
        "context_notes": [],
        "last_memory_extraction": 0,
        "message_batches_processed": 0
    }

def get_profile(user_id: str, username: Optional[str] = None) -> dict:
    file_path = USER_PROFILE_DIR / f"{user_id}.json"
    if user_id in user_cache:
        profile = user_cache[user_id]
        if username and profile.get("username") != username:
            profile["username"] = username
            save_profile(user_id)
        return profile
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default_profile(user_id, username).items():
            data.setdefault(k, v)
        if username and data.get("username") != username:
            data["username"] = username
        if data.get("discord_id") != user_id:
            data["discord_id"] = user_id
        user_cache[user_id] = data
        save_profile(user_id)
        return data
    else:
        data = default_profile(user_id, username)
        user_cache[user_id] = data
        save_profile(user_id)
        return data

def save_profile(user_id: str):
    file_path = USER_PROFILE_DIR / f"{user_id}.json"
    profile = user_cache.get(user_id)
    if not profile:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def save_all_profiles():
    for user_id in user_cache:
        save_profile(user_id)

# ---- Memory Extraction ----
def extract_memory_from_message(message_content, user_profile):
    content = message_content.lower()
    memories = []
    name_patterns = [r"my name is (\w+)", r"call me (\w+)", r"i'm (\w+)", r"i am (\w+)"]
    for pattern in name_patterns:
        match = re.search(pattern, content)
        if match:
            name = match.group(1).capitalize()
            memories.append(f"Preferred name: {name}")
    interest_patterns = [
        r"i love (\w+)", r"i like (\w+)", r"i enjoy (\w+)", r"i'm into (\w+)",
        r"i work as (?:a |an )?(\w+)", r"i'm (?:a |an )?(\w+) by profession",
        r"i study (\w+)", r"i'm studying (\w+)"
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, content)
        if match:
            interest = match.group(1)
            memories.append(f"Interest/occupation: {interest}")
    if any(word in content for word in ["hate", "dislike", "don't like"]):
        for word in ["hate", "dislike", "don't like"]:
            if word in content:
                dislike_match = re.search(f"{word} (\\w+)", content)
                if dislike_match:
                    memories.append(f"Dislikes: {dislike_match.group(1)}")
    location_patterns = [r"i'm from (\w+)", r"i live in (\w+)", r"i'm in (\w+)"]
    for pattern in location_patterns:
        match = re.search(pattern, content)
        if match:
            location = match.group(1).capitalize()
            memories.append(f"Location: {location}")
    return memories

def add_memory(user_id, memory_text):
    config = load_config()
    profile = get_profile(user_id)
    for existing_memory in profile["memories"]:
        if memory_text.lower() in existing_memory.lower() or existing_memory.lower() in memory_text.lower():
            return
    timestamped_memory = f"[{datetime.now().strftime('%Y-%m-%d')}] {memory_text}"
    profile["memories"].append(timestamped_memory)
    if len(profile["memories"]) > config["MAX_USER_MEMORY"]:
        profile["memories"].pop(0)
    save_profile(user_id)
    logging.info(f"Added memory for user {user_id}: {memory_text}")

def get_user_context(user_id):
    profile = get_profile(user_id)
    context_parts = []
    if profile["memories"]:
        context_parts.append(f"Important memories: {' | '.join(profile['memories'])}")
    if profile["preferences"]:
        prefs = [f"{k}: {v}" for k, v in profile["preferences"].items()]
        context_parts.append(f"User preferences: {' | '.join(prefs)}")
    if profile["context_notes"]:
        context_parts.append(f"Recent context: {' | '.join(profile['context_notes'])}")
    context_parts.append(f"Total messages: {profile['total_messages']}")
    days_known = (datetime.now() - datetime.fromisoformat(profile["first_seen"])).days
    if days_known > 0:
        context_parts.append(f"Known user for {days_known} days")
    return " | ".join(context_parts) if context_parts else "New user"

def detect_tone(msg):
    msg = msg.lower()
    if any(word in msg for word in ["fuck", "idiot", "dumb", "stupid"]): return "hostile"
    elif any(word in msg for word in ["thank", "pls", "please", "appreciate"]): return "friendly"
    elif any(word in msg for word in ["lol", "lmao", "rofl", "bro", "mate"]): return "banter"
    return "neutral"

def log_dm_message(message: Message):
    if not isinstance(message.channel, discord.DMChannel):
        return
    username = str(message.author).replace("#", "_")
    filename = DM_LOGS_DIR / f"{message.author.id}_{username}.jsonl"
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": str(message.author.id),
        "username": str(message.author),
        "content": message.content,
        "attachments": [
            {
                "filename": att.filename,
                "size": att.size,
                "url": att.url
            } for att in message.attachments
        ]
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def log_user_message(message: Message):
    user_id = str(message.author.id)
    log_path = USER_LOGS_DIR / f"{user_id}.jsonl"
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "channel_id": str(message.channel.id),
        "guild_id": str(message.guild.id) if message.guild else None,
        "content": message.content,
        "attachments": [
            {
                "filename": att.filename,
                "size": att.size,
                "url": att.url
            } for att in message.attachments
        ]
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ---- DuckDuckGo Search ----
async def ddg_search(query: str) -> str:
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url, params=params)
            data = r.json()
            if data.get("AbstractText"):
                return data["AbstractText"] + (f"\nSource: {data.get('AbstractURL','')}" if data.get("AbstractURL") else "")
            elif data.get("Answer"):
                return data["Answer"]
            elif data.get("Definition"):
                return data["Definition"]
            elif data.get("RelatedTopics"):
                topics = data["RelatedTopics"]
                if topics and "Text" in topics[0]:
                    return topics[0]["Text"]
            return "No relevant DuckDuckGo result found."
        except Exception as e:
            return f"Search failed: {e}"

async def should_auto_search(content: str) -> Optional[str]:
    content = content.lower().strip()
    patterns = [
        r"\bwho is\b", r"\bwho was\b", r"\bwhat is\b", r"\bwhat was\b",
        r"\bwhen did\b", r"\bhow many\b", r"\bhow much\b", r"\bhow long\b",
        r"\bcurrent\b.*\b(time|date|president|prime minister|exchange rate|price)\b",
        r"\b(latest|current|recent)\b", r"\bnews\b", r"\bmeaning of\b", r"\bdefine\b",
        r"\bcapital of\b", r"\bpopulation of\b", r"\bwho invented\b", r"\bwho wrote\b",
        r"\bhow do i\b",
    ]
    ignore_patterns = [r"\b(joke|opinion|should i|recommend)\b"]
    for pat in ignore_patterns:
        if re.search(pat, content):
            return None
    for pat in patterns:
        if re.search(pat, content):
            q = re.sub(r"^<@!?[0-9]+>", "", content).strip(" ,")
            qm = q.find("?")
            if qm != -1:
                q = q[:qm+1]
            return q
    return None

# ---- Context Management ----
conversation_store = defaultdict(list)
CONTEXT_TTL = 900
CONTEXT_MAXLEN = 30
last_active = {}

def context_key(message: Message):
    if isinstance(message.channel, discord.DMChannel):
        return (str(message.author.id), "DM")
    else:
        return (str(message.guild.id), str(message.channel.id), str(message.author.id))

def reset_context(message: Message, config=None):
    key = context_key(message)
    conversation_store[key].clear()
    if not config:
        config = load_config()
    conversation_store[key].append({'role': 'system', 'content': config["SYSTEM_PROMPT"]})

def get_context(message: Message):
    key = context_key(message)
    return conversation_store[key]

def update_last_active(message: Message):
    key = context_key(message)
    last_active[key] = time.time()

def should_reset_context(message: Message):
    key = context_key(message)
    now = time.time()
    if key not in last_active:
        return True
    if now - last_active[key] > CONTEXT_TTL:
        return True
    if len(conversation_store[key]) > CONTEXT_MAXLEN:
        return True
    return False

intents = Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

def clean_response(text):
    return re.sub(r"</?[\w_]+>", "", text)

def is_text_file(file_content):
    try:
        file_content.decode('utf-8')
        return True
    except (UnicodeDecodeError, AttributeError):
        return False

async def send_in_chunks(channel, text, reference=None, chunk_size=2000):
    text = text.strip()
    for start in range(0, len(text), chunk_size):
        await channel.send(text[start:start + chunk_size], reference=reference if start == 0 else None)

def build_system_prompt(message, config):
    user_id = str(message.author.id)
    username = str(message.author)
    profile = get_profile(user_id, username)
    last_tone = profile['tone']
    history_count = len(profile['history'])
    user_context = get_user_context(user_id)
    guild_note = f"(Server: {message.guild.name})" if message.guild else "(Direct Message)"
    custom_blurb = (
        f"{guild_note} | Current user: {username} (id: {user_id}). "
        f"Messages: {history_count}, Last detected tone: {last_tone}.\n"
        f"User context: {user_context}\n"
        f"Adjust your attitude, banter, or respect as appropriate for this user's history and what you know about them."
    )
    return f"{config['SYSTEM_PROMPT']}\n{custom_blurb}"

def should_include_mention(message: discord.Message, response: str, context_len=1) -> bool:
    if isinstance(message.channel, discord.DMChannel): return False
    if context_len > 3: return False
    if len(response.split()) < 5: return False
    bot_mention = f"<@{bot.user.id}>"
    if bot_mention in message.content.lower(): return True
    if message.content.lower().startswith(bot.user.name.lower()): return True
    if any(marker in message.content.lower() for marker in ['?', 'what', 'when', 'where', 'why', 'how', 'who', 'can you']): return True
    return False

# ---- TEXT BACKEND ROUTING ----
async def ollama_chat(messages, config, model=None):
    base_url = config["OLLAMA_BASE_URL"]
    payload = {
        "model": model or config["TEXT_MODEL"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": config["TEMPERATURE"]}
    }
    try:
        async with httpx.AsyncClient(timeout=config["TIMEOUT"]) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=config["TIMEOUT"]
            )
            data = resp.json()
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"]
            if "messages" in data and isinstance(data["messages"], list):
                return " ".join([m["content"] for m in data["messages"]])
            return data.get("response", "No response from Ollama model.")
    except Exception as e:
        return f"Model error: {e}"

async def openai_chat(messages, config, model=None):
    api_key = config["OPENAI_API_KEY"]
    api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = model or config.get("OPENAI_TEXT_MODEL", "gpt-4o")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": config.get("TEMPERATURE", 0.7)
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=config["TIMEOUT"]) as client:
        resp = await client.post(f"{api_base}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            return f"OpenAI error: {resp.status_code} {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ---- Vision-Language Inference via OpenAI API ----
async def openai_vl_inference(image_path, prompt, config):
    import base64
    api_key = config["OPENAI_API_KEY"]
    api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = config.get("VL_MODEL", "gpt-4-vision-preview")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": config["SYSTEM_PROMPT"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe this image."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": config.get("TEMPERATURE", 0.7)
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{api_base}/chat/completions", json=payload, headers=headers)
        # Handle error codes and sanitize output
        if resp.status_code != 200:
            try:
                data = resp.json()
                # Alibaba OpenRouter/vision error handler
                if data.get("error", {}).get("code") == 400 or data.get("error", {}).get("code") == "data_inspection_failed":
                    err_message = (
                        "nice attempt, BUT THEY ZOGGED ME VRO, sadly that content’s not allowed. keep it legal and respectful, ok? "
                        "try another image or check if it violates the content rules."
                    )
                    return err_message
                # General OpenAI error
                if data.get("error", {}).get("message"):
                    err_message = data["error"]["message"]
                    # Keep it short and in character
                    return f"Sorry, I couldn't process that image: {err_message}."
            except Exception:
                pass
            # Fallback
            return "Sorry, I couldn't process that image."
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---- Main Hybrid Handler ----
async def process_message(message: Message, is_dm: bool):
    config = load_config()
    key = context_key(message)
    if should_reset_context(message):
        reset_context(message, config)
    update_last_active(message)
    user_id = str(message.author.id)
    username = str(message.author)
    user_mention = message.author.mention
    profile = get_profile(user_id, username)
    profile['last_active'] = time.time()
    profile['history'].append(message.content)
    profile['tone'] = detect_tone(message.content)
    profile['total_messages'] = profile.get('total_messages', 0) + 1
    save_profile(user_id)
    context = get_context(message)

    # Web search
    search_match = re.search(r"\[search:(.*?)\]", message.content, re.I)
    auto_search_query = None
    if search_match:
        query = search_match.group(1).strip()
        search_result = await ddg_search(query)
        context.append({
            "role": "system",
            "content": f"DuckDuckGo search for '{query}':\n{search_result}"
        })
    else:
        auto_search_query = await should_auto_search(message.content)
        if auto_search_query:
            search_result = await ddg_search(auto_search_query)
            if search_result and not search_result.lower().startswith("no relevant"):
                context.append({
                    "role": "system",
                    "content": f"DuckDuckGo search for '{auto_search_query}':\n{search_result}"
                })
    # Memory extraction
    new_memories = extract_memory_from_message(message.content, profile)
    for memory in new_memories:
        add_memory(user_id, memory)
    user_context = get_user_context(user_id)
    system_prompt_with_user = build_system_prompt(message, config)
    if user_context:
        context.append({"role": "system", "content": user_context.strip()})
    context[0]['content'] = system_prompt_with_user.strip()

    # ---- VL: Image message handling via OpenAI API ----
    image_attachments = [
        att for att in message.attachments if att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    if image_attachments:
        img_bytes = await image_attachments[0].read()
        temp_img_path = f"temp_{user_id}_{int(time.time())}_{image_attachments[0].filename}"
        with open(temp_img_path, 'wb') as f:
            f.write(img_bytes)
        prompt = message.content.strip() if message.content.strip() else "Describe this image."
        reply = await openai_vl_inference(temp_img_path, prompt, config)
        os.remove(temp_img_path)
        reply = clean_response(reply).strip()
        await send_in_chunks(message.channel, reply, reference=message)
        return

    # ---- TEXT: Use selected backend ----
    total_text_content = ""
    if message.attachments:
        for attachment in message.attachments:
            if attachment.size > config["MAX_FILE_SIZE"]:
                await message.channel.send(
                    f"File {attachment.filename} is too big. Maximum size is {config['MAX_FILE_SIZE'] // (1024 * 1024)} MB."
                )
                return
            try:
                file_content = await attachment.read()
                if not is_text_file(file_content):
                    await message.channel.send(f"{attachment.filename} isn't a text file.")
                    return
                file_text = file_content.decode('utf-8')
                total_text_content += f"\n\n{attachment.filename}\n{file_text}\n"
                if len(total_text_content) > config["MAX_TEXT_ATTACHMENT_SIZE"]:
                    await message.channel.send(
                        f"Files are too big. Maximum total size is {config['MAX_TEXT_ATTACHMENT_SIZE']} characters."
                    )
                    return
            except Exception as e:
                logging.error(f"Error reading attachment {attachment.filename}: {e}")
                await message.channel.send(f"Error reading file {attachment.filename}")
                return
        content = message.content + "\n\n" + total_text_content[:config["MAX_TEXT_ATTACHMENT_SIZE"]]
    else:
        content = message.content
    context.append({'role': 'user', 'content': content.strip()})
    async with message.channel.typing():
        messages = [
            {"role": msg['role'], "content": msg['content']}
            for msg in context
            if msg['role'] in ('system', 'user', 'assistant')
        ]
        if config["TEXT_BACKEND"] == "openai":
            response = await openai_chat(messages, config)
        else:
            response = await ollama_chat(messages, config, model=config["TEXT_MODEL"])
    if response:
        response = clean_response(response).strip()
        if not is_dm and should_include_mention(message, response, len(context)):
            response = f"{user_mention} {response}"
        await send_in_chunks(message.channel, response, reference=message)
        context.append({"role": "assistant", "content": response.strip()})
        if len(context) > CONTEXT_MAXLEN:
            system_prompt = context[0]
            context.clear()
            context.append(system_prompt)
        if random.random() < 0.1:
            save_all_profiles()

# ---- Commands ----

@bot.command(name='show-memories')
async def show_memories(ctx):
    user_id = str(ctx.author.id)
    profile = get_profile(user_id, str(ctx.author))
    memories = profile["memories"]
    if not memories:
        await ctx.send("I don't have any memories about you yet!")
        return
    embed = discord.Embed(
        title=f"What I remember about {ctx.author.display_name}",
        color=discord.Color.blue()
    )
    memories_text = "\n".join(f"• {memory}" for memory in memories)
    embed.add_field(name="Memories", value=memories_text or "No memories yet", inline=False)
    embed.add_field(name="Total Messages", value=profile["total_messages"], inline=True)
    embed.add_field(name="First Seen", value=profile["first_seen"][:10], inline=True)
    embed.add_field(name="Current Tone", value=profile["tone"].capitalize(), inline=True)
    if profile["preferences"]:
        prefs = "\n".join(f"• {k}: {v}" for k, v in profile["preferences"].items())
        embed.add_field(name="Preferences", value=prefs, inline=False)
    await ctx.send(embed=embed)

@bot.command(name='remember')
async def manual_memory(ctx, *, memory_text):
    user_id = str(ctx.author.id)
    add_memory(user_id, memory_text)
    save_profile(user_id)
    await ctx.send(f"✅ I'll remember: {memory_text}")

@bot.command(name='preference')
async def set_preference(ctx, key, *, value):
    user_id = str(ctx.author.id)
    profile = get_profile(user_id, str(ctx.author))
    profile["preferences"][key] = value
    save_profile(user_id)
    await ctx.send(f"✅ Set preference `{key}` to `{value}`")

@bot.command(name='forget')
async def forget_user(ctx, user_mention=None):
    if user_mention:
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("Only administrators can forget other users' memories.")
            return
        user_id = user_mention.strip('<@!>').strip('<@>')
    else:
        user_id = str(ctx.author.id)
    profile = get_profile(user_id)
    profile["memories"] = []
    profile["preferences"] = {}
    profile["context_notes"] = []
    save_profile(user_id)
    await ctx.send(f"Forgotten all memories for user <@{user_id}>")

@bot.command(name='reset')
async def reset(ctx):
    reset_context(ctx.message, load_config())
    await ctx.send("Conversation context has been reset (hot config).")

@bot.command(name='search')
async def cmd_search(ctx, *, query):
    search_result = await ddg_search(query)
    await send_in_chunks(ctx.channel, search_result)

async def change_nickname(guild):
    config = load_config()
    nickname = config["TEXT_MODEL"].split('/')[-1].split(':')[0].capitalize()
    try:
        await guild.me.edit(nick=nickname)
        logging.info(f"Nickname changed to {nickname} in guild {guild.name}")
    except discord.Forbidden:
        logging.warning(f"No permission to change nickname in guild {guild.name}")
    except Exception as e:
        logging.error(f"Failed to change nickname in guild {guild.name}: {str(e)}")

@bot.event
async def on_ready():
    config = load_config()
    logging.info(f'{bot.user.name} is now running! (hot config)')
    asyncio.create_task(periodic_save())
    if config["CHANGE_NICKNAME"]:
        for guild in bot.guilds:
            await change_nickname(guild)

async def periodic_save():
    config = load_config()
    while True:
        await asyncio.sleep(config["MEMORY_SAVE_INTERVAL"])
        save_all_profiles()

@bot.event
async def on_message(message: Message):
    if message.author.bot:
        return
    log_user_message(message)
    if isinstance(message.channel, discord.DMChannel):
        log_dm_message(message)
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_guild_mention = (not is_dm and hasattr(message, 'mentions') and bot.user in message.mentions)
    if not is_dm and (message.content.startswith('!') or message.is_system()):
        return
    if is_dm or is_guild_mention:
        await process_message(message, is_dm)
    await bot.process_commands(message)

@bot.event
async def on_error(event, *args, **kwargs):
    logging.error(f'Discord event error in {event}: {args}')

def main():
    config = load_config()
    if not config["DISCORD_TOKEN"]:
        raise ValueError("DISCORD_TOKEN not set in .env!")
    try:
        bot.run(config["DISCORD_TOKEN"])
    except discord.LoginFailure:
        logging.error("Invalid Discord token!")
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
    finally:
        save_all_profiles()
        logging.info("Bot shutdown complete.")

if __name__ == '__main__':
    main()
