import os
import re
import json
import time
import random
import asyncio
import logging
import base64
import pathlib
import glob
import tempfile
import mimetypes
import subprocess
import shutil
from typing import Optional, Tuple
from typing import Any, Callable, Coroutine, Dict, List, TypeVar, Union
from functools import wraps
from dataclasses import dataclass
from dotenv import load_dotenv
from discord import Intents, Message, File
from discord.ext import commands
import discord
from datetime import datetime
import httpx
from collections import defaultdict, Counter

# ---- RAG (Retrieval Augmented Generation) ----
@dataclass
class KnowledgeSnippet:
    """Represents a snippet of text from the knowledge base."""
    text: str
    source_file: str
    line_number: int

class KnowledgeBase:
    def __init__(self, kb_dir: str = "kb"):
        self.kb_dir = kb_dir
        self.knowledge: List[KnowledgeSnippet] = []
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load all .txt and .md files from the knowledge base directory."""
        self.knowledge = []
        if not os.path.exists(self.kb_dir):
            logging.warning(f"Knowledge base directory '{self.kb_dir}' not found.")
            return
            
        for ext in ('*.txt', '*.md'):
            for file_path in glob.glob(os.path.join(self.kb_dir, '**', ext), recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            line = line.strip()
                            if line:  # Skip empty lines
                                self.knowledge.append(
                                    KnowledgeSnippet(
                                        text=line,
                                        source_file=os.path.basename(file_path),
                                        line_number=i
                                    )
                                )
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
    
    def get_relevant_snippets(self, query: str, top_k: int = 3) -> List[KnowledgeSnippet]:
        """
        Retrieve the most relevant snippets from the knowledge base.
        
        Args:
            query: The user's query
            top_k: Maximum number of snippets to return
            
        Returns:
            List of relevant KnowledgeSnippet objects
        """
        if not query or not self.knowledge:
            return []
            
        # Simple keyword matching - can be replaced with more sophisticated retrieval
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored_snippets = []
        for snippet in self.knowledge:
            snippet_terms = set(re.findall(r'\w+', snippet.text.lower()))
            common_terms = query_terms & snippet_terms
            if common_terms:
                # Simple scoring: number of matching terms divided by snippet length
                score = len(common_terms) / (len(snippet_terms) + 1)
                scored_snippets.append((score, snippet))
        
        # Sort by score (highest first) and take top_k
        scored_snippets.sort(key=lambda x: x[0], reverse=True)
        return [snippet for _, snippet in scored_snippets[:top_k]]

# Initialize the knowledge base
knowledge_base = KnowledgeBase()

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

def load_vl_prompt():
    vl_prompt_path = os.getenv('VL_PROMPT_FILE', 'prompts/vl-prompt.txt')
    try:
        with open(vl_prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"VL prompt file '{vl_prompt_path}' not found, using default VL prompt.")
        return "Describe this image in detail."

async def transcribe_audio(
    audio_path: str, 
    config: dict, 
    original_filename: Optional[str] = None,
    keep_converted: bool = False
) -> tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio using the Whisper API with robust error handling and validation.
    
    Args:
        audio_path: Path to the audio file to transcribe
        config: Configuration dictionary containing Whisper settings
        original_filename: Original filename (for logging and model selection)
        keep_converted: If True, keep the converted WAV file for debugging
        
    Returns:
        Tuple of (transcript, error_message) where:
        - transcript is the transcribed text if successful, None otherwise
        - error_message is None if successful, or contains an error message
    """
    # Validate configuration
    whisper_api_key = config.get("WHISPER_API_KEY")
    whisper_api_base = config.get("WHISPER_API_BASE")
    whisper_model = str(config.get("WHISPER_MODEL", "whisper-1"))  # Ensure model is a string
    
    if not all([whisper_api_key, whisper_api_base, whisper_model]):
        error_msg = "Missing required Whisper API configuration"
        logging.error(error_msg)
        return None, error_msg
    
    # Validate input file
    if not os.path.exists(audio_path):
        error_msg = f"Audio file not found: {audio_path}"
        logging.error(error_msg)
        return None, error_msg
    
    converted_path = None
    try:
        # Always convert to WAV for consistency, even if the file already has a .wav extension
        logging.info(f"Processing audio file: {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
        converted_path, conversion_error = convert_audio_to_wav(audio_path, keep_failed=keep_converted)
        
        if not converted_path or conversion_error:
            error_msg = f"Failed to convert audio file: {conversion_error or 'Unknown error'}"
            logging.error(error_msg)
            return None, error_msg
        
        # Verify the converted file
        if not os.path.exists(converted_path):
            error_msg = "Converted file not found after successful conversion"
            logging.error(error_msg)
            return None, error_msg
            
        converted_size = os.path.getsize(converted_path)
        logging.info(f"Converted file: {converted_path} (size: {converted_size} bytes)")
        
        # Prepare the API request
        url = f"{whisper_api_base}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {whisper_api_key}",
        }
        
        # Use the original filename if available, otherwise use the WAV filename
        filename = original_filename if original_filename else os.path.basename(audio_path)
        
        try:
            with open(converted_path, "rb") as audio_file:
                files = {
                    "file": (filename, audio_file, "audio/wav"),
                    "model": (None, whisper_model),
                    "response_format": (None, "text"),
                }
                
                # Log the request details (without sensitive data)
                log_files = {k: (v[0], f"<{len(v[1])} bytes>" if k == "file" else v[1]) 
                            for k, v in files.items()}
                logging.info(f"Sending request to Whisper API: {url} with files: {log_files}")
                
                # Make the API request with timeout
                timeout = config.get("TIMEOUT", 30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    start_time = time.time()
                    response = await client.post(url, headers=headers, files=files)
                    process_time = time.time() - start_time
                    
                    logging.info(f"Whisper API response in {process_time:.2f}s: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_detail = response.text[:500]  # Limit error detail length
                        error_msg = (
                            f"Whisper API error {response.status_code}: {error_detail}"
                        )
                        logging.error(f"{error_msg}. Full response: {response.text}")
                        return None, error_msg
                    
                    # Success!
                    transcript = response.text.strip()
                    logging.info(f"Transcription successful. Length: {len(transcript)} characters")
                    return transcript, None
                    
        except httpx.TimeoutException:
            error_msg = f"Whisper API request timed out after {timeout} seconds"
            logging.error(error_msg)
            return None, error_msg
            
        except httpx.RequestError as e:
            error_msg = f"Request to Whisper API failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during Whisper API request: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error during audio transcription: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg
        
    finally:
        # Clean up the converted file if it was created and we're not keeping it
        if converted_path and os.path.exists(converted_path) and not keep_converted:
            try:
                os.unlink(converted_path)
                logging.debug(f"Cleaned up temporary file: {converted_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file {converted_path}: {e}")

def convert_audio_to_wav(input_path: str, keep_failed: bool = False) -> Optional[Tuple[str, str]]:
    """
    Convert any audio file to a Whisper-compatible WAV file using ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        keep_failed: If True, keep failed conversion files for debugging
        
    Returns:
        Tuple of (output_path, ffmpeg_log) if successful, or (None, error_message) on failure
    """
    # Verify input file exists and has content
    try:
        input_size = os.path.getsize(input_path)
        if input_size < 1024:  # 1KB minimum size
            error_msg = f"Input file too small ({input_size} bytes), likely corrupted or empty"
            logging.warning(error_msg)
            return None, error_msg
    except OSError as e:
        error_msg = f"Failed to access input file: {e}"
        logging.error(error_msg)
        return None, error_msg

    if not shutil.which('ffmpeg'):
        error_msg = "ffmpeg is not installed. Please install ffmpeg to enable audio conversion."
        logging.error(error_msg)
        return None, error_msg
    
    output_path = None
    try:
        # Create a temporary file for the output with a predictable name for debugging
        temp_dir = tempfile.gettempdir()
        temp_prefix = f"discord_audio_{int(time.time())}_"
        temp_fd, output_path = tempfile.mkstemp(prefix=temp_prefix, suffix='.wav')
        os.close(temp_fd)  # We'll let ffmpeg create the file
        
        # Build the ffmpeg command with explicit settings for Whisper compatibility
        cmd = [
            'ffmpeg',
            '-hide_banner',  # Less verbose output
            '-loglevel', 'info',  # But still log important info
            '-y',  # Overwrite output file if it exists
            '-i', input_path,  # Input file
            '-ar', '16000',  # Sample rate: 16kHz (Whisper's expected rate)
            '-ac', '1',  # Mono audio
            '-c:a', 'pcm_s16le',  # 16-bit little-endian PCM
            '-f', 'wav',  # Force WAV format
            '-fflags', '+bitexact',  # Ensure consistent output
            output_path
        ]
        
        logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Run ffmpeg and capture both stdout and stderr
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,
            check=False  # We'll handle non-zero exit codes ourselves
        )
        
        ffmpeg_log = result.stdout
        
        # Check if output file was created and has content
        if not os.path.exists(output_path):
            error_msg = f"FFmpeg failed to create output file. Command: {' '.join(cmd)}"
            logging.error(f"{error_msg}\nFFmpeg output:\n{ffmpeg_log}")
            return None, error_msg
            
        output_size = os.path.getsize(output_path)
        logging.info(f"Output file created: {output_path} ({output_size} bytes)")
        
        if output_size < 1024:  # 1KB minimum size for WAV header + some audio data
            error_msg = f"Output file too small ({output_size} bytes), conversion likely failed"
            logging.error(f"{error_msg}\nFFmpeg output:\n{ffmpeg_log}")
            if not keep_failed:
                os.unlink(output_path)
                output_path = None
            return None, error_msg
        
        if result.returncode != 0:
            error_msg = f"FFmpeg exited with code {result.returncode}"
            logging.error(f"{error_msg}\nFFmpeg output:\n{ffmpeg_log}")
            if not keep_failed:
                os.unlink(output_path)
                output_path = None
            return None, error_msg
            
        return output_path, ffmpeg_log
        
    except Exception as e:
        error_msg = f"Error during audio conversion: {str(e)}"
        logging.error(error_msg, exc_info=True)
        if output_path and os.path.exists(output_path) and not keep_failed:
            try:
                os.unlink(output_path)
            except Exception as cleanup_err:
                logging.error(f"Failed to clean up output file: {cleanup_err}")
        return None, error_msg

def is_audio_file(filename: str, content_type: Optional[str] = None) -> bool:
    """Check if a file is an audio file based on extension and/or content type."""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.opus', '.m4b', '.aac', '.wma', '.webm'}
    audio_content_types = {
        'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/ogg', 'audio/webm',
        'audio/flac', 'audio/opus', 'audio/x-wav', 'audio/x-m4a', 'audio/aac',
        'audio/x-m4b', 'audio/x-ms-wma', 'audio/aacp', 'audio/mp4a-latm',
        'audio/mpeg3', 'audio/x-mpeg-3', 'audio/x-m4p', 'audio/x-m4b', 'audio/x-m4r'
    }
    
    # Check by extension
    ext = os.path.splitext(filename.lower())[1]
    if ext in audio_extensions:
        return True
        
    # Check by content type if provided
    if content_type and any(ct in content_type.lower() for ct in audio_content_types):
        return True
        
    return False

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
        "VL_MODEL": os.getenv('VL_MODEL', 'gpt-4-vision-preview'),
        "VL_PROMPT": load_vl_prompt(),
        "WHISPER_API_KEY": os.getenv('WHISPER_API_KEY'),
        "WHISPER_API_BASE": os.getenv('WHISPER_API_BASE'),
        "WHISPER_MODEL": os.getenv('WHISPER_MODEL', 'whisper-1'),
        "TEMPERATURE": float(os.getenv('TEMPERATURE', '0.7')),
        "TIMEOUT": float(os.getenv('TIMEOUT', '120.0')),
        "CHANGE_NICKNAME": os.getenv('CHANGE_NICKNAME', 'True').lower() == 'true',
        "MAX_CONVERSATION_LENGTH": int(os.getenv('MAX_CONVERSATION_LENGTH', '30')),
        "MAX_FILE_SIZE": int(os.getenv('MAX_FILE_SIZE', '20')) * 1024 * 1024,  # 20MB default for general files
        "MAX_AUDIO_SIZE": int(os.getenv('MAX_AUDIO_SIZE', '50')) * 1024 * 1024,  # 50MB default for audio files
        "MAX_TEXT_ATTACHMENT_SIZE": int(os.getenv('MAX_TEXT_ATTACHMENT_SIZE', '20000')),
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
async def get_vision_caption(image_path, prompt, config):
    """
    Get a caption/description of the image using the vision-language model.
    
    Args:
        image_path: Path to the image file
        prompt: Optional user prompt, falls back to VL_PROMPT from config
        config: Configuration dictionary
        
    Returns:
        str: The generated caption/description or error message
    """
    import base64
    api_key = config["OPENAI_API_KEY"]
    api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = config.get("VL_MODEL", "gpt-4-vision-preview")
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logging.error(f"Error reading image file {image_path}: {e}")
        return None
        
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Use the VL-specific prompt if no user prompt is provided
    user_prompt = prompt or config.get("VL_PROMPT", "Describe this image in detail.")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1  # Lower temperature for more factual descriptions
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{api_base}/chat/completions", 
                json=payload, 
                headers=headers
            )
            
            if resp.status_code != 200:
                try:
                    data = resp.json()
                    if data.get("error", {}).get("code") in (400, "data_inspection_failed"):
                        return "[Image contains content that cannot be processed]"
                    if data.get("error", {}).get("message"):
                        logging.error(f"VL API error: {data['error']['message']}")
                except Exception as e:
                    logging.error(f"Error parsing VL API error response: {e}")
                return None
                
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        logging.error(f"Error in VL API call: {e}")
        return None

# Kept for backward compatibility
async def openai_vl_inference(image_path, prompt, config):
    return await get_vision_caption(image_path, prompt, config)


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
    
    # Add RAG-retrieved knowledge if available
    if message.content.strip():
        relevant_snippets = knowledge_base.get_relevant_snippets(message.content)
        if relevant_snippets:
            knowledge_text = "\n".join(
                f"- {snippet.text} (from {snippet.source_file}:{snippet.line_number})"
                for snippet in relevant_snippets
            )
            system_prompt_with_user += "\n\nRetrieved knowledge:\n" + knowledge_text
    
    # Update context with user context and system prompt
    if user_context:
        context.append({"role": "system", "content": user_context.strip()})
    context[0]['content'] = system_prompt_with_user.strip()

    # Process all attachments
    total_text_content = ""
    
    # Separate attachments by type
    image_attachments = []
    audio_attachments = []
    text_attachments = []
    other_attachments = []
    
    for att in message.attachments:
        # Get content type and filename
        content_type = getattr(att, 'content_type', '')
        filename = getattr(att, 'filename', '')
        
        # Check if it's an audio file (using our enhanced detection)
        if is_audio_file(filename, content_type):
            audio_attachments.append(att)
        # Check if it's an image
        is_image = (content_type and content_type.startswith('image/')) or \
                  (filename and any(ext in filename for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']))
        
        # Check if it's an audio file
        is_audio = is_audio_file(filename, content_type)
        
        if len(audio_attachments) > 1:
            await message.channel.send("I'll process the first audio file. Please send other audio files one at a time.")
        
        # Process the first audio attachment if present
        if audio_attachments:
            audio_attachment = audio_attachments[0]
            temp_path = None
            try:
                # Notify user we're processing the audio
                processing_msg = await message.channel.send("🎤 Processing audio message...")
                
                # Create a temporary file to save the audio with original extension
                file_ext = os.path.splitext(audio_attachment.filename)[1] or '.bin'
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
                    temp_path = temp_file.name
                
                # Download the audio file with progress feedback
                await message.channel.typing()
                await audio_attachment.save(temp_path)
                file_size = os.path.getsize(temp_path)
                logging.info(f"Downloaded audio file: {temp_path} ({file_size} bytes)")
                
                # Validate file size
                max_audio_size = config.get("MAX_AUDIO_SIZE", 50 * 1024 * 1024)  # Default 50MB
                if file_size > max_audio_size:
                    error_msg = f"Audio file is too large ({file_size/1024/1024:.1f}MB > {max_audio_size/1024/1024}MB limit)"
                    logging.warning(error_msg)
                    await processing_msg.edit(content=f"❌ {error_msg}")
                    return
                
                if file_size < 1024:  # 1KB
                    error_msg = "Audio file is too small, it might be corrupted"
                    logging.warning(f"{error_msg} ({file_size} bytes)")
                    await processing_msg.edit(content=f"❌ {error_msg}")
                    return
                
                # Transcribe the audio with progress updates
                await processing_msg.edit(content="🔊 Transcribing audio (this may take a moment)...")
                
                # Keep the converted file for debugging if in debug mode
                keep_converted = config.get("DEBUG", False)
                transcript, error = await transcribe_audio(
                    temp_path, 
                    config, 
                    original_filename=audio_attachment.filename,
                    keep_converted=keep_converted
                )
                
                # Handle transcription result
                if transcript:
                    # Truncate very long transcripts in the message
                    display_transcript = (transcript[:500] + '...') if len(transcript) > 500 else transcript
                    
                    # Add the transcription to the message content
                    if message.content:
                        new_content = f"{message.content}\n\n🎤 **Audio transcription:** {display_transcript}"
                    else:
                        new_content = f"🎤 **Audio transcription:** {display_transcript}"
                    
                    # Try to edit the original message to include the transcription
                    try:
                        await message.edit(content=new_content)
                        await processing_msg.delete()  # Remove the processing message
                    except Exception as e:
                        logging.error(f"Failed to edit message: {e}")
                        # If we can't edit, update the processing message with the transcript
                        await processing_msg.edit(content=f"🎤 **Audio transcription:** {display_transcript}")
                    
                    # If transcript was truncated, send the full version in a follow-up message
                    if len(transcript) > 500:
                        await message.channel.send(f"Full transcription:\n{transcript}")
                    
                    logging.info(f"Successfully transcribed audio message (length: {len(transcript)} chars)")
                else:
                    error_msg = error or "Failed to transcribe audio"
                    logging.error(f"Transcription failed: {error_msg}")
                    await processing_msg.edit(content=f"❌ Failed to transcribe audio: {error_msg}")
            
            except Exception as e:
                error_msg = str(e) or "Unknown error"
                logging.error(f"Error processing audio attachment: {error_msg}", exc_info=True)
                
                # Try to provide a more user-friendly error message
                if "No space left on device" in error_msg:
                    user_error = "Not enough disk space to process audio"
                elif "timeout" in error_msg.lower():
                    user_error = "Audio processing timed out"
                else:
                    user_error = "An error occurred while processing the audio"
                
                try:
                    await processing_msg.edit(content=f"❌ {user_error}. Please try again later.")
                except:
                    await message.channel.send(f"❌ {user_error}. Please try again later.")
            
            finally:
                # Clean up the temporary file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                        logging.debug(f"Cleaned up temporary file: {temp_path}")
                    except Exception as e:
                        logging.error(f"Error cleaning up temporary file {temp_path}: {e}")
    
    # Log warning for unprocessed attachments
    for att in other_attachments:
        logging.warning(f"Unprocessed attachment type: {att.filename} (content_type: {getattr(att, 'content_type', 'unknown')})")
    
    # Process image attachments if any
    temp_img_path = None
    if image_attachments:
        # Notify if multiple images were attached
        if len(image_attachments) > 1:
            await message.channel.send("I'll analyze the first image you sent. Please send other images one at a time for analysis.")
        
        # Process the first image
        img_attachment = image_attachments[0]
        
        # Create a temporary file with proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_attachment.filename)[1]) as temp_img:
            temp_img_path = temp_img.name
            
            try:
                # Download the image with timeout
                img_bytes = await asyncio.wait_for(img_attachment.read(), timeout=30.0)
                temp_img.write(img_bytes)
                
                # Check file size after downloading
                if img_attachment.size > config["MAX_FILE_SIZE"]:
                    await message.channel.send(f"The image is too large (max {config['MAX_FILE_SIZE'] // (1024*1024)}MB). Please send a smaller image.")
                    return
                    
                # Get MIME type for the downloaded file
                mime_type, _ = mimetypes.guess_type(img_attachment.filename)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'application/octet-stream'
                
                # Always use the VL model for image analysis
                try:
                    # Get the VL prompt from config (loaded from file via VL_PROMPT_FILE)
                    vl_prompt = config.get("VL_PROMPT", "Describe this image in detail.")
                    vision_caption = await get_vision_caption(temp_img_path, vl_prompt, config)
                    
                    if not vision_caption:
                        logging.error("VL model returned empty caption")
                        # Continue with original message content if VL fails
                        vision_caption = "[Image processing failed]"
                    
                    # Format the message content based on whether there was original text
                    original_content = message.content.strip()
                    if original_content:
                        message.content = f"{original_content}\n[Image description: {vision_caption}]"
                    else:
                        message.content = f"[Image description: {vision_caption}]"
                    
                    # Add to context with MIME type information
                    context.append({
                        "role": "user",
                        "content": message.content,  # Use the formatted message content
                        "mime_type": mime_type,
                        "size_bytes": img_attachment.size
                    })
                    
                except Exception as e:
                    # Log the error but don't show it to the user
                    logging.error(f"Error in vision-language model: {e}", exc_info=True)
                    # Continue with original message content if VL processing fails
                    if not message.content.strip():
                        message.content = "[Image processing failed]"
                
            except asyncio.TimeoutError:
                await message.channel.send("Sorry, the image took too long to download. Please try again.")
                return
            except Exception as e:
                logging.error(f"Error downloading image: {e}")
                await message.channel.send("Sorry, I couldn't process that image. Please try another one.")
                return
            
    # Clean up temporary files after all processing
    def safe_remove(filepath):
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logging.error(f"Error removing temporary file {filepath}: {e}")
    
    # Clean up image file if it exists
    if 'temp_img_path' in locals() and temp_img_path:
        safe_remove(temp_img_path)
    
    # Process text attachments if any
    if text_attachments:
        for attachment in text_attachments:
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
                total_text_content += f"\n\n{attachment.filename}\n{file_text}"
                
                if len(total_text_content) > config["MAX_TEXT_ATTACHMENT_SIZE"]:
                    await message.channel.send(
                        f"Files are too big. Maximum total size is {config['MAX_TEXT_ATTACHMENT_SIZE']} characters."
                    )
                    return
                    
            except Exception as e:
                logging.error(f"Error reading attachment {attachment.filename}: {e}")
                await message.channel.send(f"Error reading file {attachment.filename}")
                return
    
    # Combine message content with any text from attachments
    if total_text_content:
        message.content = f"{message.content}\n\n[Attached files content:]{total_text_content}"
    
    # Add the final message content to the context
    context.append({'role': 'user', 'content': message.content.strip()})
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
    try:
        if message.author.bot:
            return
        log_user_message(message)
        if isinstance(message.channel, discord.DMChannel):
            log_dm_message(message)
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_guild_mention = (not is_dm and hasattr(message, 'mentions') and bot.user in message.mentions)
        # If command: process commands (even in DMs!)
        if message.content.startswith('!'):
            await bot.process_commands(message)
            return
        # If DM or mention, process as a chat message (but also process commands)
        if is_dm or is_guild_mention:
            await process_message(message, is_dm)
        # Always call process_commands (in case a command is hidden in normal text)
        await bot.process_commands(message)
    except Exception as e:
        logging.error(f"Discord event error in on_message: {e}", exc_info=True)


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
