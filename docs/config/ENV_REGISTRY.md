# Canonical Environment Registry

This file is auto-generated from `utils/env_inventory.json` via `utils/build_env_registry.py`.

Fields:
- key: environment variable name
- description: to be curated (placeholder)
- required: whether startup validation requires this key
- sensitive: contains secrets (never log values)
- type: inferred from code usage (int, float, bool, list, set, unknown)
- default: canonical default when present; multiple conflicting defaults are logged
- valid_values: enumerated allowed values, if known (TBD)

[IV][CMV][SFT][CA][CDiP]


## CORE

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| ALERT_ADMIN_USER_IDS |  |  | unknown | '' |  |
| ALERT_ENABLE |  |  | unknown | 'false' |  |
| ALERT_SESSION_TIMEOUT_S |  |  | unknown | '1800' |  |
| ANTHROPIC_API_KEY |  | ✔ | unknown |  |  |
| BOT_PREFIX |  |  | unknown | '!' |  |
| BOT_SPEAKS_ONLY_WHEN_SPOKEN_TO |  |  | unknown | 'True' |  |
| CHANGE_NICKNAME |  |  | unknown | 'False' |  |
| COMMAND_PREFIX |  |  | unknown | '!' |  |
| CONTEXT_FILE_PATH |  |  | unknown | 'runtime/context.json' |  |
| CUSTOM_SEARCH_API_ENDPOINT |  |  | unknown | '' |  |
| CUSTOM_SEARCH_API_KEY |  | ✔ | unknown | '' |  |
| CUSTOM_SEARCH_HEADERS |  |  | unknown | '' |  |
| CUSTOM_SEARCH_RESULT_PATHS |  |  | unknown | '' |  |
| CUSTOM_SEARCH_TIMEOUT_MS |  |  | unknown |  |  |
| DDG_API_ENDPOINT |  |  | unknown | 'https://html.duckduckgo.com/html/' |  |
| DDG_API_KEY |  | ✔ | unknown |  |  |
| DDG_FORCE_HTML |  |  | unknown | 'true' |  |
| DDG_TIMEOUT_MS |  |  | int | '5000' |  |
| DEBUG |  |  | unknown | 'False' |  |
| DEFAULT_TIMEOUT |  |  | unknown |  |  |
| DISCORD_TOKEN | ✔ | ✔ | unknown |  |  |
| DM_LOGS_DIR |  |  | unknown | 'dm_logs' |  |
| ENABLE_MEDIA_INGESTION |  |  | unknown | 'true' |  |
| ENABLE_RAG |  |  | unknown | 'true' |  |
| ENABLE_TWITTER_VIDEO_DETECTION |  |  | unknown | 'true' |  |
| FREQUENCY_PENALTY |  |  | unknown |  |  |
| HF_HOME |  |  | unknown |  |  |
| HISTORY_WINDOW |  |  | int | '10' |  |
| IN_MEMORY_CONTEXT_ONLY |  |  | unknown | 'false' |  |
| LOGS_DIR |  |  | unknown | 'logs' |  |
| LOG_FILE |  |  | unknown | 'logs/bot.jsonl' |  |
| LOG_JSONL_PATH |  |  | unknown | './logs/bot.jsonl' |  |
| LOG_LEVEL |  |  | unknown | 'INFO' |  |
| MAX_ATTACHMENT_SIZE_MB |  |  | unknown |  |  |
| MAX_AUDIO_SIZE |  |  | int | '50' |  |
| MAX_CONTEXT_LENGTH |  |  | unknown |  |  |
| MAX_CONTEXT_MESSAGES |  |  | int | max_messages |  |
| MAX_CONVERSATION_LENGTH |  |  | int | '30' |  |
| MAX_CONVERSATION_LOG_SIZE |  |  | unknown |  |  |
| MAX_FILE_SIZE |  |  | int | '20' |  |
| MAX_MEMORIES |  |  | unknown |  |  |
| MAX_RESPONSE_TOKENS |  | ✔ | unknown |  |  |
| MAX_SERVER_MEMORY |  |  | int | 100 |  |
| MAX_TEXT_ATTACHMENT_SIZE |  |  | int | '20000' |  |
| MAX_USER_MEMORY |  |  | int | '10' |  |
| MEMORY_SAVE_INTERVAL |  |  | int | '30' |  |
| MULTIMODAL_PER_ITEM_BUDGET |  |  | float | '30.0' |  |
| OBS_ENABLE_PROMETHEUS |  |  | unknown | 'false' |  |
| OBS_ENABLE_RESOURCE_METRICS |  |  | unknown | 'true' |  |
| OBS_PARALLEL_STARTUP |  |  | unknown | 'false' |  |
| OLLAMA_BASE_URL |  |  | unknown | 'http://localhost:11434' |  |
| OLLAMA_HOST |  |  | unknown |  |  |
| OLLAMA_MODEL |  |  | unknown | 'llama3' |  |
| OPENAI_API_BASE |  |  | unknown | 'https://api.openai.com/v1' |  |
| OPENAI_API_KEY |  | ✔ | unknown |  |  |
| OPENAI_MODEL |  |  | unknown | 'gpt-4' |  |
| OPENAI_TEXT_MODEL |  |  | unknown | 'gpt-4o' |  |
| OWNER_IDS |  |  | unknown | '' |  |
| PATH |  |  | unknown | '' |  |
| PRESENCE_PENALTY |  |  | unknown |  |  |
| PROMPT_FILE | ✔ |  | unknown | 'prompt.txt' |  |
| REPLY_TRIGGERS |  |  | unknown | 'dm,mention,reply,bot_threads,owner,command_prefix' |  |
| RESOURCE_CHECK_INTERVAL |  |  | float | DEFAULT_RESOURCE_CHECK_INTERVAL |  |
| RESOURCE_CPU_CRITICAL_PERCENT |  |  | float | DEFAULT_CPU_CRITICAL_PERCENT |  |
| RESOURCE_CPU_WARNING_PERCENT |  |  | float | DEFAULT_CPU_WARNING_PERCENT |  |
| RESOURCE_EVENT_LOOP_LAG_CRITICAL_MS |  |  | float | DEFAULT_EVENT_LOOP_LAG_CRITICAL_MS |  |
| RESOURCE_EVENT_LOOP_LAG_WARNING_MS |  |  | float | DEFAULT_EVENT_LOOP_LAG_WARNING_MS |  |
| RESOURCE_RSS_CRITICAL_MB |  |  | float | DEFAULT_RSS_CRITICAL_MB |  |
| RESOURCE_RSS_WARNING_MB |  |  | float | DEFAULT_RSS_WARNING_MB |  |
| SENTENCE_TRANSFORMERS_HOME |  |  | unknown |  |  |
| SERVER_PROFILE_DIR |  |  | unknown | 'server_profiles' |  |
| TEMPERATURE |  |  | float | '0.7' |  |
| TEMP_DIR |  |  | unknown | 'temp' |  |
| TEXT_BACKEND |  |  | unknown | '' |  |
| TEXT_FALLBACK_MODELS |  |  | unknown |  |  |
| TEXT_FALLBACK_TIMEOUTS |  |  | unknown |  |  |
| TEXT_MODEL |  |  | unknown | 'qwen3-235b-a22b' |  |
| TIMEOUT |  |  | float | '120.0' |  |
| TOP_P |  |  | unknown |  |  |
| TRANSFORMERS_CACHE |  |  | unknown |  |  |
| TWITTER_ROUTE_DEFAULT |  |  | unknown | 'api_first' |  |
| USER_LOGS_DIR |  |  | unknown | 'user_logs' |  |
| USER_PROFILE_DIR |  |  | unknown | 'user_profiles' |  |
| USE_ENHANCED_CONTEXT |  |  | unknown | 'true' |  |
| VISION_FALLBACK_MODELS |  |  | unknown |  |  |
| VISION_FALLBACK_TIMEOUTS |  |  | unknown |  |  |
| VL_MODEL |  |  | unknown | 'gpt-4-vision-preview' |  |
| VL_PROMPT_FILE | ✔ |  | unknown | 'prompts/vl-prompt.txt' |  |
| WHISPER_API_BASE |  |  | unknown |  |  |
| WHISPER_API_KEY |  | ✔ | unknown |  |  |
| WHISPER_CPP_MODEL |  |  | unknown | 'ggml-medium.bin' |  |
| WHISPER_MODEL |  |  | unknown | 'whisper-1' |  |
| WHISPER_MODEL_SIZE |  |  | unknown | 'base-int8' |  |
| XDG_CACHE_HOME |  |  | unknown | Path('tts/cache') |  |

## MEDIA

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| MEDIA_DOWNLOAD_TIMEOUT |  |  | int | '60' |  |
| MEDIA_FALLBACK_MODELS |  |  | unknown |  |  |
| MEDIA_FALLBACK_TIMEOUTS |  |  | unknown |  |  |
| MEDIA_MAX_CONCURRENT |  |  | int | '2' |  |
| MEDIA_MAX_TITLE_LENGTH |  |  | int | '200' |  |
| MEDIA_MAX_UPLOADER_LENGTH |  |  | int | '100' |  |
| MEDIA_MAX_URL_LENGTH |  |  | int | '500' |  |
| MEDIA_PER_ITEM_BUDGET |  |  | float | '120.0' |  |
| MEDIA_PROBE_CACHE_DIR |  |  | unknown | 'cache/media_probes' |  |
| MEDIA_PROBE_CACHE_TTL |  |  | int | '300' |  |
| MEDIA_PROBE_TIMEOUT |  |  | int | '10' |  |
| MEDIA_PROVIDER_TIMEOUT |  |  | unknown |  |  |
| MEDIA_RETRY_BASE_DELAY |  |  | float | '2.0' |  |
| MEDIA_RETRY_MAX_ATTEMPTS |  |  | int | '3' |  |
| MEDIA_SPEEDUP_FACTOR |  |  | float | '1.5' |  |

## PROMETHEUS

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| PROMETHEUS_ENABLED |  |  | unknown | 'true' |  |
| PROMETHEUS_HTTP_SERVER |  |  | unknown | 'true' |  |
| PROMETHEUS_PORT |  |  | int | '8000' |  |

## RAG

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| RAG_BACKGROUND_INDEXING |  |  | unknown | 'true' |  |
| RAG_CHUNK_OVERLAP |  |  | int | '50' |  |
| RAG_CHUNK_SIZE |  |  | int | '512' |  |
| RAG_COMBINE_RESULTS |  |  | unknown | 'true' |  |
| RAG_DB_PATH |  |  | unknown | './chroma_db' |  |
| RAG_DEDUPLICATION_THRESHOLD |  |  | float | '0.9' |  |
| RAG_EAGER_VECTOR_LOAD |  |  | unknown | 'false' |  |
| RAG_EMBEDDING_MODEL_NAME |  |  | unknown | 'sentence-transformers/all-MiniLM-L6-v2' |  |
| RAG_EMBEDDING_MODEL_TYPE |  |  | unknown | 'sentence-transformers' |  |
| RAG_ENFORCE_GUILD_SCOPING |  |  | unknown | 'true' |  |
| RAG_ENFORCE_USER_SCOPING |  |  | unknown | 'true' |  |
| RAG_FALLBACK_ON_FAILURE |  |  | unknown | 'true' |  |
| RAG_FALLBACK_ON_LOW_CONFIDENCE |  |  | unknown | 'true' |  |
| RAG_INDEXING_BATCH_SIZE |  |  | int | '10' |  |
| RAG_INDEXING_QUEUE_SIZE |  |  | int | '1000' |  |
| RAG_INDEXING_WORKERS |  |  | int | '2' |  |
| RAG_KB_PATH |  |  | unknown | 'kb' |  |
| RAG_KEYWORD_WEIGHT |  | ✔ | float | '0.3' |  |
| RAG_LAZY_LOAD_TIMEOUT |  |  | float | '30.0' |  |
| RAG_LOG_CONFIDENCE_SCORES |  |  | unknown | 'true' |  |
| RAG_LOG_RETRIEVAL_PATHS |  |  | unknown | 'true' |  |
| RAG_MAX_COMBINED_RESULTS |  |  | int | '5' |  |
| RAG_MAX_KEYWORD_RESULTS |  | ✔ | int | '3' |  |
| RAG_MAX_VECTOR_RESULTS |  |  | int | '5' |  |
| RAG_MIN_CHUNK_SIZE |  |  | int | '100' |  |
| RAG_MIN_RESULTS_THRESHOLD |  |  | int | '1' |  |
| RAG_VECTOR_CONFIDENCE_THRESHOLD |  |  | float | '0.7' |  |
| RAG_VECTOR_WEIGHT |  |  | float | '0.7' |  |

## SCREENSHOT

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| SCREENSHOT_API_COOKIES |  |  | unknown | '' |  |
| SCREENSHOT_API_DELAY |  |  | unknown | '' |  |
| SCREENSHOT_API_DEVICE |  |  | unknown | '' |  |
| SCREENSHOT_API_DIMENSION |  |  | unknown | '' |  |
| SCREENSHOT_API_FORMAT |  |  | unknown | '' |  |
| SCREENSHOT_API_KEY |  | ✔ | unknown | '' |  |
| SCREENSHOT_API_URL |  |  | unknown | '' |  |
| SCREENSHOT_FALLBACK_PLAYWRIGHT |  |  | unknown | 'true' |  |
| SCREENSHOT_PW_TIMEOUT_MS |  |  | int | '15000' |  |
| SCREENSHOT_PW_USER_AGENT |  |  | unknown | '' |  |
| SCREENSHOT_PW_VIEWPORT |  |  | unknown | '1280x1024' |  |

## SEARCH

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| SEARCH_BREAKER_FAILURE_WINDOW |  |  | unknown |  |  |
| SEARCH_BREAKER_HALFOPEN_PROB |  |  | unknown |  |  |
| SEARCH_BREAKER_OPEN_MS |  |  | unknown |  |  |
| SEARCH_INLINE_MAX_CONCURRENCY |  |  | int | '' |  |
| SEARCH_LOCALE |  |  | unknown | '' |  |
| SEARCH_MAX_RESULTS |  |  | unknown |  |  |
| SEARCH_POOL_MAX_CONNECTIONS |  |  | unknown |  |  |
| SEARCH_PROVIDER |  |  | unknown | 'ddg' |  |
| SEARCH_SAFE |  |  | unknown | 'moderate' |  |

## STREAMING

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| STREAMING_EMBED_STYLE |  |  | unknown | 'compact' |  |
| STREAMING_ENABLE |  |  | unknown | 'true' |  |
| STREAMING_ENABLE_MEDIA |  |  | unknown | 'true' |  |
| STREAMING_ENABLE_RAG |  |  | unknown | 'false' |  |
| STREAMING_ENABLE_SEARCH |  |  | unknown | 'false' |  |
| STREAMING_ENABLE_TEXT |  |  | unknown | 'false' |  |
| STREAMING_MAX_STEPS |  |  | unknown |  |  |
| STREAMING_TICK_MS |  |  | unknown |  |  |

## STT

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| STT_ACTIVE_PROVIDERS |  |  | unknown | 'local_whisper' |  |
| STT_CACHE_TTL |  |  | unknown |  |  |
| STT_CONFIDENCE_MIN |  |  | unknown |  |  |
| STT_ENABLE |  |  | unknown | 'true' |  |
| STT_ENGINE |  |  | unknown | 'faster-whisper' |  |
| STT_FALLBACK |  |  | unknown | 'whispercpp' |  |
| STT_LOCAL_CONCURRENCY |  |  | unknown |  |  |
| STT_MODE |  |  | unknown | 'single' |  |

## TTS

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| TTS_BACKEND |  |  | unknown | 'kokoro-onnx' |  |
| TTS_CACHE_DIR |  |  | unknown | 'cache/tts' |  |
| TTS_ENABLED |  |  | unknown | 'false' |  |
| TTS_ENGINE |  |  | unknown | 'stub' |  |
| TTS_LANGUAGE |  |  | unknown | 'en' |  |
| TTS_MODEL_FILE |  |  | unknown | 'tts/kokoro-v1.0.onnx' |  |
| TTS_MODEL_PATH |  |  | unknown | 'tts/onnx/kokoro-v1.0.onnx' |  |
| TTS_PHONEMISER |  |  | unknown | '' |  |
| TTS_PREFS_FILE |  |  | unknown |  |  |
| TTS_TOKENISER |  | ✔ | unknown | '' |  |
| TTS_VOICE |  |  | unknown | 'af' |  |
| TTS_VOICES_PATH |  |  | unknown | 'tts/onnx/voices/voices-v1.0.bin' |  |
| TTS_VOICE_FILE |  |  | unknown | 'tts/voices.json' |  |

## VIDEO

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| VIDEO_CACHE_DIR |  |  | unknown | 'cache/video_audio' |  |
| VIDEO_CACHE_EXPIRY_DAYS |  |  | int | '7' |  |
| VIDEO_MAX_CONCURRENT |  |  | int | '3' |  |
| VIDEO_MAX_DURATION |  |  | int | '600' |  |
| VIDEO_SPEEDUP |  |  | float | '1.5' |  |

## WEBEX

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| WEBEX_ACCEPT_LANGUAGE |  |  | unknown | 'en-US,en;q=0.9' |  |
| WEBEX_ENABLE_TIER_B |  |  | unknown | '1' |  |
| WEBEX_TIER_A_TIMEOUT_S |  |  | float | '6.0' |  |
| WEBEX_TIER_B_TIMEOUT_S |  |  | float | '12.0' |  |
| WEBEX_UA_DESKTOP |  |  | unknown | 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36' |  |

## X

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| X_EXPANSIONS |  |  | unknown | 'author_id,attachments.media_keys,referenced_tweets.id,referenced_tweets.id.author_id' |  |
| X_MEDIA_FIELDS |  |  | unknown | 'media_key,type,url,preview_image_url,variants,width,height,alt_text,public_metrics' |  |
| X_PLACE_FIELDS |  |  | unknown | 'full_name,id,country_code,geo,name,place_type' |  |
| X_POLL_FIELDS |  |  | unknown | 'id,options,duration_minutes,end_datetime,voting_status' |  |
| X_SYNDICATION_ENABLED |  |  | unknown | 'true' |  |
| X_SYNDICATION_FIRST |  |  | unknown | 'false' |  |
| X_SYNDICATION_TIMEOUT_MS |  |  | int | '4000' |  |
| X_SYNDICATION_TTL_S |  |  | float | '900' |  |
| X_TWEET_FIELDS |  |  | unknown | 'id,text,created_at,author_id,public_metrics,possibly_sensitive,lang,attachments,entities,referenced_tweets,conversation_id' |  |
| X_TWITTER_STT_PROBE_FIRST |  |  | unknown | 'true' |  |
| X_USER_FIELDS |  |  | unknown | 'id,name,username,profile_image_url,verified,protected' |  |

## X_API

| key | required | sensitive | type | default | description |
|---|:---:|:---:|:---:|---|---|
| X_API_ALLOW_FALLBACK_ON_5XX |  |  | unknown | 'true' |  |
| X_API_AUTH_MODE |  | ✔ | unknown | 'oauth2_app' |  |
| X_API_BEARER_TOKEN |  | ✔ | unknown |  |  |
| X_API_BREAKER_FAILURE_WINDOW |  |  | unknown |  |  |
| X_API_BREAKER_HALFOPEN_PROB |  |  | unknown |  |  |
| X_API_BREAKER_OPEN_MS |  |  | unknown |  |  |
| X_API_ENABLED |  |  | unknown | 'false' |  |
| X_API_REQUIRE_API_FOR_TWITTER |  |  | unknown | 'false' |  |
| X_API_RETRY_MAX_ATTEMPTS |  |  | unknown |  |  |
| X_API_ROUTE_PHOTOS_TO_VL |  |  | unknown | 'true' |  |
| X_API_TIMEOUT_MS |  |  | unknown |  |  |