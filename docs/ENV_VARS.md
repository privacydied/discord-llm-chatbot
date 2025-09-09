# Environment Variables (auto-generated)

| Name | Default | Description |
| --- | --- | --- |
| `ALERT_ADMIN_USER_IDS` | `your_discord_user_id_here` | Comma-separated Discord user IDs authorized to use !alert |
| `ALERT_ENABLE` | `false` | Enable/disable the admin DM alert system |
| `ALERT_SESSION_TIMEOUT_S` | `1800` | Alert session timeout in seconds (30 minutes) |
| `ANTHROPIC_API_KEY` | `—` | API setting for anthropic key. |
| `BOT_PREFIX` | `!` | Configuration for bot prefix. |
| `CACHE_BACKEND` | `memory` | Configuration for cache backend. |
| `CACHE_DIR` | `.cache/router` | Directory for cache. |
| `CACHE_READABILITY_TTL_S` | `14400` | Configuration for cache readability ttl s. |
| `CACHE_SINGLE_FLIGHT_ENABLE` | `true` | Enable or disable cache single flight. |
| `CHANGE_NICKNAME` | `False` | Should the bot auto-update its nickname? |
| `COMMAND_PREFIX` | `!` | Configuration for command prefix. |
| `CONTEXT_FILE_PATH` | `context.json` | For production, restrict permissions on this file (e.g., chmod 600) to protect contents. |
| `CUSTOM_SEARCH_API_ENDPOINT` | `—` | API setting for custom search endpoint. |
| `CUSTOM_SEARCH_API_KEY` | `—` | API setting for custom search key. |
| `CUSTOM_SEARCH_HEADERS` | `—` | Configuration for custom search headers. |
| `CUSTOM_SEARCH_RESULT_PATHS` | `—` | Configuration for custom search result paths. |
| `CUSTOM_SEARCH_TIMEOUT_MS` | `8000` | Timeout for custom search in ms. |
| `DDG_API_ENDPOINT` | `https://duckduckgo.com/html/` | API setting for ddg endpoint. |
| `DDG_API_KEY` | `—` | API setting for ddg key. |
| `DDG_FORCE_HTML` | `true` | Configuration for ddg force html. |
| `DDG_TIMEOUT_MS` | `5000` | Timeout for ddg in ms. |
| `DEBUG` | `False` | Configuration for debug. |
| `DEFAULT_TIMEOUT` | `—` | Timeout for default in seconds. |
| `DISCORD_TOKEN` | `your_discord_bot_token_here` | Configuration for discord token. |
| `DM_LOGS_DIR` | `dm_logs` | Directory for dm logs. |
| `ENABLE_MEDIA_INGESTION` | `true` | Enable or disable media ingestion. |
| `ENABLE_RAG` | `true` | Enable/disable RAG system |
| `ENABLE_TWITTER_VIDEO_DETECTION` | `true` | Enable or disable twitter video detection. |
| `FREQUENCY_PENALTY` | `—` | Configuration for frequency penalty. |
| `HF_HOME` | `—` | Configuration for hf home. |
| `HISTORY_WINDOW` | `10` | Configuration for history window. |
| `HTTP2_ENABLE` | `true` | Enable or disable http2. |
| `HTTP_CONNECT_TIMEOUT_MS` | `1500` | Timeout for http connect in ms. |
| `HTTP_DNS_CACHE_TTL_S` | `300` | Configuration for http dns cache ttl s. |
| `HTTP_MAX_CONNECTIONS` | `64` | Configuration for http max connections. |
| `HTTP_MAX_KEEPALIVE_CONNECTIONS` | `32` | Configuration for http max keepalive connections. |
| `HTTP_READ_TIMEOUT_MS` | `5000` | Timeout for http read in ms. |
| `HTTP_TOTAL_DEADLINE_MS` | `6000` | Configuration for http total deadline ms. |
| `HYBRID_FORCE_PERCEPTION_ON_REPLY` | `—` | Configuration for hybrid force perception on reply. |
| `IMAGEDL_DEBUG` | `0` | Configuration for imagedl debug. |
| `IMAGEDL_REFERER` | `https://x.com/` | Configuration for imagedl referer. |
| `IMAGEDL_TIMEOUT_PER_ATTEMPT_MS` | `800` | Timeout for imagedl per attempt in ms. |
| `IMG_ATTACHMENT_MAX_BYTES` | `262144` | Configuration for img attachment max bytes. |
| `IN_MEMORY_CONTEXT_ONLY` | `false` | When true, the bot stores conversation state in memory only and does not write to disk. |
| `KOKORO_FORCE_IPA` | `1` | Configuration for kokoro force ipa. |
| `KOKORO_GRAPHEME_FALLBACK` | `0` | Configuration for kokoro grapheme fallback. |
| `KOKORO_MODEL_PATH` | `—` | Path to kokoro model. |
| `KOKORO_TTS_TIMEOUT_COLD` | `60` | Timeout for kokoro tts cold in seconds. |
| `KOKORO_TTS_TIMEOUT_WARM` | `20` | Timeout for kokoro tts warm in seconds. |
| `KOKORO_VOICES_PATH` | `—` | Path to kokoro voices. |
| `LOGS_DIR` | `logs` | Directory for logs. |
| `LOG_FILE` | `logs/bot.jsonl` | Configuration for log file. |
| `LOG_JSONL_PATH` | `logs/bot.jsonl` | Path to log jsonl. |
| `LOG_LEVEL` | `—` | Configuration for log level. |
| `MAX_ATTACHMENT_SIZE_MB` | `—` | Maximum attachment size mb. |
| `MAX_CONTEXT_LENGTH` | `—` | Maximum context length. |
| `MAX_CONTEXT_MESSAGES` | `—` | Maximum context messages. |
| `MAX_CONVERSATION_LENGTH` | `50` | Maximum conversation length. |
| `MAX_CONVERSATION_LOG_SIZE` | `—` | Maximum conversation log size. |
| `MAX_FILE_SIZE` | `—` | Maximum file size. |
| `MAX_MEMORIES` | `—` | Maximum memories. |
| `MAX_RESPONSE_TOKENS` | `—` | Maximum response tokens. |
| `MAX_SERVER_MEMORY` | `—` | Maximum server memory. |
| `MAX_TEXT_ATTACHMENT_SIZE` | `—` | Maximum text attachment size. |
| `MAX_USER_MEMORY` | `1000` | Maximum user memory. |
| `MEDIA_DOWNLOAD_TIMEOUT` | `60` | Timeout for media download in seconds. |
| `MEDIA_FALLBACK_MODELS` | `—` | Configuration for media fallback models. |
| `MEDIA_FALLBACK_TIMEOUTS` | `—` | Configuration for media fallback timeouts. |
| `MEDIA_MAX_CONCURRENT` | `2` | Configuration for media max concurrent. |
| `MEDIA_MAX_TITLE_LENGTH` | `200` | Configuration for media max title length. |
| `MEDIA_MAX_UPLOADER_LENGTH` | `100` | Configuration for media max uploader length. |
| `MEDIA_MAX_URL_LENGTH` | `500` | URL for media max length. |
| `MEDIA_PER_ITEM_BUDGET` | `—` | Configuration for media per item budget. |
| `MEDIA_PROBE_CACHE_DIR` | `cache/media_probes` | Directory for media probe cache. |
| `MEDIA_PROBE_CACHE_TTL` | `300` | Configuration for media probe cache ttl. |
| `MEDIA_PROBE_TIMEOUT` | `10` | Timeout for media probe in seconds. |
| `MEDIA_PROVIDER_TIMEOUT` | `—` | Timeout for media provider in seconds. |
| `MEDIA_RETRY_BASE_DELAY` | `2.0` | Configuration for media retry base delay. |
| `MEDIA_RETRY_MAX_ATTEMPTS` | `3` | Configuration for media retry max attempts. |
| `MEDIA_SPEEDUP_FACTOR` | `1.5` | Configuration for media speedup factor. |
| `MEMORY_SAVE_INTERVAL` | `30` | Configuration for memory save interval. |
| `MULTIMODAL_PER_ITEM_BUDGET` | `45.0` | Configuration for multimodal per item budget. |
| `OBS_ENABLE_PROMETHEUS` | `false` | Enable or disable obs prometheus. |
| `OBS_ENABLE_RESOURCE_METRICS` | `true` | Enable or disable obs resource metrics. |
| `OBS_PARALLEL_STARTUP` | `false` | Configuration for obs parallel startup. |
| `OCR_BATCH_DEADLINE_MS` | `20000` | Configuration for ocr batch deadline ms. |
| `OCR_GLOBAL_DEADLINE_MS` | `240000` | Configuration for ocr global deadline ms. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL for ollama base. |
| `OLLAMA_HOST` | `—` | Configuration for ollama host. |
| `OLLAMA_MODEL` | `llama3` | Configuration for ollama model. |
| `OPENAI_API_BASE` | `https://openrouter.ai/api/v1` | API setting for openai base. |
| `OPENAI_API_KEY` | `your_openai_api_key_here` | API setting for openai key. |
| `OPENAI_MODEL` | `—` | Configuration for openai model. |
| `OPENAI_TEXT_MODEL` | `deepseek/deepseek-chat-v3-0324:free` | Configuration for openai text model. |
| `OPUS_BITRATE` | `64k` | Configuration for opus bitrate. |
| `OPUS_COMPRESSION_LEVEL` | `10` | Configuration for opus compression level. |
| `OPUS_VBR` | `on` | Configuration for opus vbr. |
| `OWNER_IDS` | `—` | Configuration for owner ids. |
| `PATH` | `—` | Path to . |
| `PRESENCE_PENALTY` | `—` | Configuration for presence penalty. |
| `PROMETHEUS_ENABLED` | `true` | Enable or disable prometheus. |
| `PROMETHEUS_HTTP_SERVER` | `true` | Configuration for prometheus http server. |
| `PROMETHEUS_PORT` | `8000` | Port for prometheus. |
| `PROMPT_FILE` | `prompts/prompt-example.txt` | Path to system prompt |
| `RAG_BACKGROUND_INDEXING` | `true` | Process new documents asynchronously (true) or synchronously (false) |
| `RAG_CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RAG_CHUNK_SIZE` | `512` | Text chunk size in tokens |
| `RAG_COMBINE_RESULTS` | `true` | Combine vector and keyword results |
| `RAG_DB_PATH` | `./chroma_db` | ChromaDB storage path |
| `RAG_DEDUPLICATION_THRESHOLD` | `0.9` | Similarity threshold for deduplication |
| `RAG_EAGER_VECTOR_LOAD` | `true` | Load vector index at startup (true) or lazily on first search (false) |
| `RAG_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Configuration for rag embedding model name. |
| `RAG_EMBEDDING_MODEL_TYPE` | `sentence-transformers` | Configuration for rag embedding model type. |
| `RAG_ENFORCE_GUILD_SCOPING` | `true` | Enforce guild-based access control |
| `RAG_ENFORCE_USER_SCOPING` | `true` | Enforce user-based access control |
| `RAG_FALLBACK_ON_FAILURE` | `true` | Fallback to keyword search on vector failure |
| `RAG_FALLBACK_ON_LOW_CONFIDENCE` | `true` | Fallback to keyword search on low confidence |
| `RAG_INDEXING_BATCH_SIZE` | `32` | Documents to process per batch/flush |
| `RAG_INDEXING_QUEUE_SIZE` | `256` | Maximum items in indexing queue (backpressure) |
| `RAG_INDEXING_WORKERS` | `2` | Number of background indexing workers |
| `RAG_KB_PATH` | `kb` | Knowledge base directory |
| `RAG_KEYWORD_WEIGHT` | `0.3` | Weight for keyword search (0.0-1.0) |
| `RAG_LAZY_LOAD_TIMEOUT` | `0.0` | Seconds to wait in search path for lazy load (0.0 = never block) |
| `RAG_LOG_CONFIDENCE_SCORES` | `true` | Log confidence scores |
| `RAG_LOG_RETRIEVAL_PATHS` | `true` | Log retrieval decision paths |
| `RAG_MAX_COMBINED_RESULTS` | `5` | Maximum combined results |
| `RAG_MAX_KEYWORD_RESULTS` | `3` | Maximum keyword search results |
| `RAG_MAX_VECTOR_RESULTS` | `5` | Maximum vector search results |
| `RAG_MIN_CHUNK_SIZE` | `100` | Minimum chunk size |
| `RAG_MIN_RESULTS_THRESHOLD` | `1` | Minimum results before fallback |
| `RAG_VECTOR_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for vector results |
| `RAG_VECTOR_WEIGHT` | `0.7` | Weight for vector search (0.0-1.0) |
| `RESOURCE_CHECK_INTERVAL` | `DEFAULT_RESOURCE_CHECK_INTERVAL` | Configuration for resource check interval. |
| `RESOURCE_CPU_CRITICAL_PERCENT` | `DEFAULT_CPU_CRITICAL_PERCENT` | Configuration for resource cpu critical percent. |
| `RESOURCE_CPU_WARNING_PERCENT` | `DEFAULT_CPU_WARNING_PERCENT` | Configuration for resource cpu warning percent. |
| `RESOURCE_RSS_CRITICAL_MB` | `DEFAULT_RSS_CRITICAL_MB` | Configuration for resource rss critical mb. |
| `RESOURCE_RSS_WARNING_MB` | `DEFAULT_RSS_WARNING_MB` | Configuration for resource rss warning mb. |
| `SCREENSHOT_API_COOKIES` | `—` | API setting for screenshot cookies. |
| `SCREENSHOT_API_DELAY` | `2000` | API setting for screenshot delay. |
| `SCREENSHOT_API_DEVICE` | `desktop` | API setting for screenshot device. |
| `SCREENSHOT_API_DIMENSION` | `1024x768` | API setting for screenshot dimension. |
| `SCREENSHOT_API_FORMAT` | `jpg` | API setting for screenshot format. |
| `SCREENSHOT_API_KEY` | `your_screenshot_api_key_here` | API setting for screenshot key. |
| `SCREENSHOT_API_URL` | `https://api.screenshotmachine.com` | URL for screenshot api. |
| `SCREENSHOT_FALLBACK_PLAYWRIGHT` | `true` | Configuration for screenshot fallback playwright. |
| `SCREENSHOT_PW_TIMEOUT_MS` | `15000` | Timeout for screenshot pw in ms. |
| `SCREENSHOT_PW_USER_AGENT` | `—` | Configuration for screenshot pw user agent. |
| `SCREENSHOT_PW_VIEWPORT` | `1280x1024` | Configuration for screenshot pw viewport. |
| `SEARCH_BREAKER_FAILURE_WINDOW` | `5` | Configuration for search breaker failure window. |
| `SEARCH_BREAKER_HALFOPEN_PROB` | `0.25` | Configuration for search breaker halfopen prob. |
| `SEARCH_BREAKER_OPEN_MS` | `15000` | Configuration for search breaker open ms. |
| `SEARCH_INLINE_MAX_CONCURRENCY` | `3` | Configuration for search inline max concurrency. |
| `SEARCH_LOCALE` | `—` | Configuration for search locale. |
| `SEARCH_MAX_RESULTS` | `5` | Configuration for search max results. |
| `SEARCH_POOL_MAX_CONNECTIONS` | `10` | Configuration for search pool max connections. |
| `SEARCH_PROVIDER` | `ddg` | ddg | custom |
| `SEARCH_SAFE` | `moderate` | off|moderate|strict |
| `SENTENCE_TRANSFORMERS_HOME` | `—` | Configuration for sentence transformers home. |
| `SERVER_PROFILE_DIR` | `server_profiles` | Directory for server profile. |
| `STREAMING_EMBED_STYLE` | `compact` | compact | detailed |
| `STREAMING_ENABLE` | `false` | Enable or disable streaming. |
| `STREAMING_ENABLE_MEDIA` | `true` | true keeps cards for heavy media (images/video/audio/PDF/screenshots) |
| `STREAMING_ENABLE_RAG` | `false` | true enables cards for RAG retrieval/ingest flows |
| `STREAMING_ENABLE_SEARCH` | `false` | true enables cards for online search flows |
| `STREAMING_ENABLE_TEXT` | `false` | true enables cards for pure text flows (default false) |
| `STREAMING_MAX_STEPS` | `8` | Configuration for streaming max steps. |
| `STREAMING_TICK_MS` | `750` | Configuration for streaming tick ms. |
| `STT_ACTIVE_PROVIDERS` | `local_whisper` | Comma-separated list of active providers. Currently supported: local_whisper |
| `STT_CACHE_DIR` | `stt/cache` | Directory for stt cache. |
| `STT_CACHE_TTL` | `600` | Configuration for stt cache ttl. |
| `STT_CACHE_TTL_S` | `604800` | Configuration for stt cache ttl s. |
| `STT_COMPUTE_TYPE` | `int8` | Configuration for stt compute type. |
| `STT_CONFIDENCE_MIN` | `0.0` | Acceptance threshold for providers that emit confidence (0.0-1.0) |
| `STT_ENABLE` | `true` | Enable orchestrated STT pipeline (falls back to legacy STT when false) |
| `STT_ENGINE` | `faster-whisper` | Configuration for stt engine. |
| `STT_FALLBACK` | `whispercpp` | Configuration for stt fallback. |
| `STT_GLOBAL_MAX_CONCURRENCY` | `3` | Configuration for stt global max concurrency. |
| `STT_INIT_TIMEOUT` | `8` | Timeout for stt init in seconds. |
| `STT_LOCAL_CONCURRENCY` | `2` | Local provider concurrency limit |
| `STT_LOCAL_ONLY` | `0` | Configuration for stt local only. |
| `STT_MODE` | `single` | single | cascade_primary_then_fallbacks | parallel_first_acceptable | parallel_best_of | hybrid_draft_then_finalize |
| `STT_TOTAL_DEADLINE_MS` | `300000` | Configuration for stt total deadline ms. |
| `SYND_DEBUG_MEDIA_PICK` | `0` | Configuration for synd debug media pick. |
| `SYND_INCLUDE_QUOTED_MEDIA` | `true` | Configuration for synd include quoted media. |
| `TEMPERATURE` | `0.7` | Configuration for temperature. |
| `TEMP_DIR` | `temp` | Directory for temp. |
| `TEXT_BACKEND` | `openai` | Backend for text chat: 'openai' (OpenRouter/OpenAI) or 'ollama' (local Ollama) |
| `TEXT_FALLBACK_MODELS` | `deepseek/deepseek-r1-0528:free,deepseek/deepseek-chat-v3-0324:free,z-ai/glm-4.5-air:free` | TEXT MODEL FALLBACK LADDER |
| `TEXT_FALLBACK_TIMEOUTS` | `20.0,25.0,30.0` | TEXT MODEL TIMEOUTS (seconds for each model above) |
| `TEXT_FINAL_MAX_CHARS` | `—` | Configuration for text final max chars. |
| `TEXT_MODEL` | `—` | Configuration for text model. |
| `TEXT_PROMPT_PATH` | `—` | Path to text prompt. |
| `TIMEOUT` | `120.0` | Timeout for operation in seconds. |
| `TOP_P` | `—` | Configuration for top p. |
| `TRANSFORMERS_CACHE` | `—` | Configuration for transformers cache. |
| `TTS_BACKEND` | `kokoro-onnx` | Configuration for tts backend. |
| `TTS_CACHE_MAX_ITEMS` | `100` | Configuration for tts cache max items. |
| `TTS_ENABLED` | `false` | Enable or disable tts. |
| `TTS_ENGINE` | `kokoro-onnx` | Configuration for tts engine. |
| `TTS_LANGUAGE` | `en` | Configuration for tts language. |
| `TTS_LEXICON` | `—` | Configuration for tts lexicon. |
| `TTS_MAX_CHARS` | `800` | Configuration for tts max chars. |
| `TTS_MODEL_FILE` | `tts/onnx/kokoro-v1.0.onnx` | Configuration for tts model file. |
| `TTS_MODEL_PATH` | `tts/onnx/kokoro-v1.0.onnx` | Path to tts model. |
| `TTS_PHONEMISER` | `—` | Configuration for tts phonemiser. |
| `TTS_PREFS_FILE` | `—` | Configuration for tts prefs file. |
| `TTS_TIMEOUT_COLD_S` | `—` | Timeout for tts cold in seconds. |
| `TTS_TIMEOUT_S` | `—` | Timeout for tts in seconds. |
| `TTS_TIMEOUT_WARM_S` | `—` | Timeout for tts warm in seconds. |
| `TTS_TOKENISER` | `—` | Configuration for tts tokeniser. |
| `TTS_TOKENIZER` | `—` | Configuration for tts tokenizer. |
| `TTS_VOICE` | `am_michael` | Configuration for tts voice. |
| `TTS_VOICES_PATH` | `tts/voices/voices-v1.0.bin` | Path to tts voices. |
| `TTS_VOICE_FILE` | `—` | Configuration for tts voice file. |
| `TWEET_CACHE_TTL_S` | `86400` | Configuration for tweet cache ttl s. |
| `TWEET_FLOW_ENABLED` | `true` | Enable or disable tweet flow. |
| `TWEET_NEGATIVE_TTL_S` | `900` | Configuration for tweet negative ttl s. |
| `TWEET_SYNDICATION_TOTAL_DEADLINE_MS` | `3500` | Configuration for tweet syndication total deadline ms. |
| `TWEET_WEB_TIER_A_MS` | `2500` | Configuration for tweet web tier a ms. |
| `TWEET_WEB_TIER_B_MS` | `8000` | Configuration for tweet web tier b ms. |
| `TWEET_WEB_TIER_C_MS` | `8000` | Configuration for tweet web tier c ms. |
| `TWITTER_ROUTE_DEFAULT` | `api_first` | Configuration for twitter route default. |
| `USER_LOGS_DIR` | `user_logs` | Directory for user logs. |
| `USER_PROFILE_DIR` | `user_profiles` | Directory for user profile. |
| `USE_ENHANCED_CONTEXT` | `true` | Configuration for use enhanced context. |
| `VIDEO_CACHE_DIR` | `cache/video_audio` | Directory for video cache. |
| `VIDEO_CACHE_EXPIRY_DAYS` | `7` | Configuration for video cache expiry days. |
| `VIDEO_MAX_CONCURRENT` | `3` | Configuration for video max concurrent. |
| `VIDEO_MAX_DURATION` | `600` | Configuration for video max duration. |
| `VIDEO_SPEEDUP` | `1.5` | Configuration for video speedup. |
| `VISION_ALLOWED_PROVIDERS` | `together,novita` | Comma-separated list of allowed providers: together,novita |
| `VISION_API_KEY` | `your_vision_api_key_here` | API key for Vision providers (Together.ai, Novita.ai) |
| `VISION_ARTIFACTS_DIR` | `vision_data/artifacts` | Directory for generated artifacts (images, videos) |
| `VISION_ARTIFACT_TTL_DAYS` | `7` | Artifact cache TTL in days |
| `VISION_AUDIT_ENABLED` | `true` | Enable detailed audit logging for all Vision operations |
| `VISION_BUDGET_DAILY_USD` | `5.00` | Daily budget in USD |
| `VISION_BUDGET_PER_JOB_USD` | `0.25` | Budget per job in USD |
| `VISION_DATA_DIR` | `vision_data` | Base directory for Vision system data storage |
| `VISION_DEFAULT_PROVIDER` | `together` | Default provider to use: together or novita |
| `VISION_DRY_RUN_MODE` | `false` | Enable dry run mode (no actual API calls, testing only) |
| `VISION_ENABLED` | `false` | Enable Vision generation features (text-to-image, image editing, video generation) |
| `VISION_EPHEMERAL_RESPONSES` | `false` | Use ephemeral responses for slash commands (true/false) |
| `VISION_FALLBACK_MODELS` | `moonshotai/kimi-vl-a3b-thinking:free,mistralai/mistral-small-3.2-24b-instruct:free,mistralai/mistral-small-3.2-24b-instruct:free` | Models are tried in order - if first fails, try second, then third, etc. |
| `VISION_FALLBACK_TIMEOUTS` | `12.0,15.0,18.0` | VISION MODEL TIMEOUTS (seconds for each model above) |
| `VISION_FORCE_OPENROUTER_THRESHOLD` | `—` | Configuration for vision force openrouter threshold. |
| `VISION_INTENT_THRESHOLD` | `—` | Configuration for vision intent threshold. |
| `VISION_JOBS_DIR` | `vision_data/jobs` | Directory for job state persistence |
| `VISION_JOB_TIMEOUT_SECONDS` | `300` | Job timeout in seconds (5 minutes default) |
| `VISION_LEDGER_PATH` | `vision_data/ledger.jsonl` | Path to budget and cost ledger file |
| `VISION_LOG_LEVEL` | `INFO` | Log level for Vision system: DEBUG, INFO, WARNING, ERROR |
| `VISION_MAX_ARTIFACT_SIZE_MB` | `50` | Maximum size per artifact in MB |
| `VISION_MAX_CONCURRENT_JOBS` | `5` | Maximum concurrent Vision jobs across all users |
| `VISION_MAX_TOTAL_ARTIFACTS_GB` | `5` | Maximum total artifacts storage in GB |
| `VISION_MAX_USER_CONCURRENT_JOBS` | `2` | Maximum concurrent jobs per user |
| `VISION_MODEL` | `""` | Examples: novita:qwen-image, novita:sdxl, together:flux.1-pro, qwen-image, flux.1-pro |
| `VISION_ORCH_DEBUG` | `0` | Configuration for vision orch debug. |
| `VISION_POLICY_PATH` | `configs/vision_policy.json` | Path to Vision safety policy configuration file |
| `VISION_PROGRESS_UPDATE_INTERVAL_S` | `10` | Progress update interval for Discord messages in seconds |
| `VISION_PROVIDER_MAX_RETRIES` | `2` | Maximum provider retry attempts |
| `VISION_PROVIDER_RETRY_DELAY_MS` | `1000` | Base delay between retries in milliseconds |
| `VISION_PROVIDER_TIMEOUT_MS` | `30000` | Provider API timeout in milliseconds |
| `VISION_REPLY_IMAGE_FORCE_VL` | `—` | Configuration for vision reply image force vl. |
| `VISION_REPLY_IMAGE_SILENT` | `—` | Configuration for vision reply image silent. |
| `VISION_T2I_ENABLED` | `—` | Enable or disable vision t2i. |
| `VISION_TRIGGER_DEBUG` | `0` | Configuration for vision trigger debug. |
| `VL_CONCURRENCY_LIMIT` | `4` | Configuration for vl concurrency limit. |
| `VL_DEBUG_FLOW` | `0` | Configuration for vl debug flow. |
| `VL_MAX_IMAGES` | `4` | Configuration for vl max images. |
| `VL_MODEL` | `moonshotai/kimi-vl-a3b-thinking:free` | Configuration for vl model. |
| `VL_NOTES_MAX_CHARS` | `—` | Configuration for vl notes max chars. |
| `VL_PROMPT_FILE` | `—` | Configuration for vl prompt file. |
| `VL_PROMPT_PATH` | `—` | Path to vl prompt. |
| `VL_REPLY_MAX_CHARS` | `—` | Configuration for vl reply max chars. |
| `VL_STRIP_REASONING` | `—` | Configuration for vl strip reasoning. |
| `VOICE_ENABLE_NATIVE` | `false` | Enable or disable voice native. |
| `VOICE_PUBLISHER_ATTACHMENTS_CREATE_TIMEOUT_S` | `—` | Timeout for voice publisher attachments create in seconds. |
| `VOICE_PUBLISHER_MESSAGE_POST_TIMEOUT_S` | `—` | Timeout for voice publisher message post in seconds. |
| `VOICE_PUBLISHER_OPUS_COMP_LEVEL` | `—` | Configuration for voice publisher opus comp level. |
| `VOICE_PUBLISHER_OPUS_VBR` | `on` | Configuration for voice publisher opus vbr. |
| `VOICE_PUBLISHER_TIMEOUT_S` | `—` | Timeout for voice publisher in seconds. |
| `VOICE_PUBLISHER_UPLOAD_TIMEOUT_S` | `—` | Timeout for voice publisher upload in seconds. |
| `WEBEX_ACCEPT_LANGUAGE` | `en-US,en;q=0.9` | Configuration for webex accept language. |
| `WEBEX_ENABLE_TIER_B` | `1` | Enable or disable webex tier b. |
| `WEBEX_TIER_A_TIMEOUT_S` | `6.0` | Timeout for webex tier a in seconds. |
| `WEBEX_TIER_B_TIMEOUT_S` | `12.0` | Timeout for webex tier b in seconds. |
| `WEB_FAST_FAIL_SPA_HOSTS` | `medium.com,heavy.com` | Configuration for web fast fail spa hosts. |
| `WEB_TIER_A_MS` | `2000` | Configuration for web tier a ms. |
| `WEB_TIER_B_MS` | `8000` | Configuration for web tier b ms. |
| `WEB_TIER_C_MS` | `8000` | Configuration for web tier c ms. |
| `WHISPER_API_BASE` | `—` | API setting for whisper base. |
| `WHISPER_API_KEY` | `—` | API setting for whisper key. |
| `WHISPER_CPP_MODEL` | `models/ggml-medium.bin` | Configuration for whisper cpp model. |
| `WHISPER_MODEL` | `whisper-1` | Configuration for whisper model. |
| `WHISPER_MODEL_SIZE` | `base` | Configuration for whisper model size. |
| `X_API_ALLOW_FALLBACK_ON_5XX` | `true` | Allow generic extraction fallback on 5xx/429 |
| `X_API_AUTH_MODE` | `oauth2_app` | API setting for x auth mode. |
| `X_API_BEARER_TOKEN` | `—` | API setting for x bearer token. |
| `X_API_BREAKER_FAILURE_WINDOW` | `5` | API setting for x breaker failure window. |
| `X_API_BREAKER_HALFOPEN_PROB` | `0.25` | API setting for x breaker halfopen prob. |
| `X_API_BREAKER_OPEN_MS` | `15000` | API setting for x breaker open ms. |
| `X_API_ENABLED` | `false` | Enable X API integration (requires bearer token) |
| `X_API_REQUIRE_API_FOR_TWITTER` | `false` | If true, do not scrape/fallback when API fails (401/403/404/410) |
| `X_API_RETRY_MAX_ATTEMPTS` | `5` | API setting for x retry max attempts. |
| `X_API_ROUTE_PHOTOS_TO_VL` | `false` | Photo media routing to Vision-Language analysis (disabled by default) |
| `X_API_TIMEOUT_MS` | `8000` | Timeout for x api in ms. |
| `X_API_TOTAL_DEADLINE_MS` | `6000` | API setting for x total deadline ms. |
| `X_EXPANSIONS` | `author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id` | Configuration for x expansions. |
| `X_MEDIA_FIELDS` | `media_key,type,url,preview_image_url,variants,width,height,alt_text,public_metrics` | Configuration for x media fields. |
| `X_POLL_FIELDS` | `—` | Configuration for x poll fields. |
| `X_SYNDICATION_ENABLED` | `true` | Enable or disable x syndication. |
| `X_TWEET_FIELDS` | `id,text,created_at,author_id,public_metrics,possibly_sensitive,lang,attachments,entities,referenced_tweets,conversation_id` | Configuration for x tweet fields. |
| `X_TWITTER_STT_PROBE_FIRST` | `true` | Fast probe: attempt STT on X URLs before API/syndication (helps video tweets route to STT) |
| `X_USER_FIELDS` | `id,name,username,profile_image_url,verified,protected` | Configuration for x user fields. |