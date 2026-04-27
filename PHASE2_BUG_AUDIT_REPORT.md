# Phase 2 Bug Audit Report - Discord LLM Chatbot
## Date: 2026-04-27
## Auditor: Hermes Agent

---

## Table 1: API Endpoints/Key Functions Verification

### 1. Router.dispatch_message() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/router.py:3097`
**Status**: 🟡 Medium

**Findings**:
- Uses `_processed_recent_set` with maxlen=512 for message dedupe (line 378-379)
- Has `_processing_locks` dict for per-message locking (line 381)
- Missing lock cleanup for completed messages - potential memory leak
- The `_dispatch_metadata` dict grows without bound (line 383)

**Fix Recommendation**: 
```python
# Add cleanup in dispatch completion:
if message_id in self._processing_locks:
    del self._processing_locks[message_id]
```

---

### 2. brain_infer() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/brain.py:15`
**Status**: ✅ OK

**Findings**:
- Proper error handling with user-friendly messages
- Sanitization applied before measuring content length
- Graceful fallback for empty responses

---

### 3. generate_response() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/ai_backend.py:15`
**Status**: 🟢 Minor

**Findings**:
- Routes correctly to openai/nvidia/ollama backends
- Missing return type annotation consistency (returns Union but function defined as async)
- Error message suppression logic is solid

---

### 4. generate_openai_response() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/openai_backend.py:127`
**Status**: 🟢 Minor

**Findings**:
- Retry logic present but no explicit timeout handling at this level
- Relies on underlying HTTP client timeouts

---

### 5. ContextManager.append() / get_context() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/memory/context_manager.py`
**Status**: ✅ OK

**Findings**:
- Proper DM filtering (line 91-93: filters out 'dm_' keys)
- File permission hardening present (line 104-105)
- Graceful fallback on JSON decode errors
- Uses MAX_CONTEXT_MESSAGES from env or default (line 33)

---

### 6. memory_cmds.add_memory_cmd() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/commands/memory_cmds.py:50`
**Status**: 🔴 Critical

**Findings**:
- **CRITICAL**: Prohibited pattern check is insufficient (line 74-77)
  - Only checks for basic script patterns but missing XSS vectors
  - No rate limiting on memory addition
  - MAX_MEMORY_LENGTH is hardcoded to 2000 (line 66) instead of using config

**Fix Recommendation**:
```python
# Add rate limiting and XSS protection
from html import escape
content = escape(content.strip())
config_max = self.config.get("MAX_MEMORY_LENGTH", 2000)
MAX_MEMORY_LENGTH = min(config_max, 2000)
```

---

### 7. img_commands.img_command() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/commands/img_commands.py:170`
**Status**: 🟡 Medium

**Findings**:
- MockIntentParams/MockIntentResult pattern used (lines 315-330)
  - Creates mock objects inline - should use proper factory or DI
- No validation that parsed attachment content is safe
- IMG_ATTACHMENT_MAX_BYTES = 256KB uses env var (line 30-32) ✓

**Fix Recommendation**: Inject vision task factory instead of inline mocks.

---

### 8. TTSManager.generate_tts() - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/tts/manager.py:7`
**Status**: 🟡 Medium

**Findings**:
- manager.py is a stub with NotImplementedError raised (line 13)
- Real implementation is in `/bot/tts/interface.py:200` (TTSManager class)
- manager_fixed.py exists but unclear which is used

**Fix Recommendation**: Remove dead code or consolidate TTS implementations.

---

## Table 2: State Machines Verification

### 1. Router._processed_recent_set (message dedupe) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/router.py:378-379`
**Status**: 🟡 Medium

**Findings**:
- Uses collections.deque with maxlen=512
- Stored both as deque and set - redundant, wastes memory
- No TTL on entries - message could be re-processed after restart

---

### 2. Router._processing_locks (concurrency) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/router.py:381`
**Status**: 🔴 Critical

**Findings**:
- **CRITICAL**: Dict grows without bound, never cleaned up
- Memory leak on long-running bot instances
- Keys: message_id -> asyncio.Lock objects

**Fix Recommendation**:
```python
# Add cleanup after message processing completes
try:
    # ... process message ...
finally:
    self._processing_locks.pop(message_id, None)
```

---

### 3. CircuitBreaker.state (CLOSED/OPEN/HALF_OPEN) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:75-115`
**Status**: ✅ OK

**Findings**:
- Proper state transitions
- Recovery timeout logic correct (15s default in HostLimits line 57)
- Thread-safe with asyncio

---

### 4. _config_cache with TTL - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/config.py`
**Status**: 🟢 Minor

**Findings**:
- CACHE_TTL = 5 minutes (300 seconds)
- `_cache_timestamp` tracked alongside `_config_cache`
- Race condition possible if config reloaded during read (line 340-350 in bot.py has reload callback)

---

### 5. LLMBot._boot_completed flag - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/core/bot.py:149`
**Status**: ✅ OK

**Findings**:
- DRY idempotency guard present
- Reset on failure (line 718: `self._boot_completed = False`)
- Check in setup_hook (line 329-333)

---

## Table 3: Timers Verification

### 1. _cache_timestamp + CACHE_TTL (5 minutes) - VERIFIED
**Status**: ✅ OK
- Standard 300 second TTL implemented

### 2. CircuitBreaker.recovery_timeout (15s default) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:57`
**Status**: ✅ OK
- Configurable via HostLimits.circuit_breaker_cooldown

### 3. HTTP retry delays with exponential backoff - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:372-376`
**Status**: 🟢 Minor
- Uses jitter (line 234-236)
- Configurable base/max/exponential parameters

### 4. TTS timeouts (cold/warm variants) - NOT FULLY VERIFIED
**Status**: 🟡 Medium
- Cold/warm variants mentioned but implementation unclear
- TTS timeout definitions scattered across files
- Source in `bot/core/bot.py` lines 1058-1064 references TTS_TIMEOUT_COLD/WARM

### 5. Discord typing context timeout - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/core/bot.py:238-296`
**Status**: ✅ OK
- 3 retry attempts with proper error handling
- Enter/exit context managed properly

---

## Table 4: Numeric Values Verification

### 1. MAX_CONTEXT_MESSAGES (default 10) - VERIFIED
**Status**: ✅ OK
- From env var MAX_CONTEXT_MESSAGES, falls back to 10
- Used in ContextManager (line 33 in context_manager.py)

### 2. MAX_USER_MEMORY (default 20) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/memory/profiles.py:190-206`
**Status**: ✅ OK
- Env var MAX_USER_MEMORY with fallback to 20
- Documented in profiles.py

### 3. _processed_recent maxlen=512 - VERIFIED
**Status**: ✅ OK
- Deque maxlen enforced

### 4. IMG_ATTACHMENT_MAX_BYTES (256KB) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/commands/img_commands.py:30-32`
**Status**: ✅ OK
- 262144 bytes = 256KB
- Overridable via env var

### 5. TEXT/TTS timeout values - PARTIALLY VERIFIED
**Status**: 🟡 Medium
- Text timeouts in enhanced_retry.py (default 15-30s per provider)
- TTS cold timeout: 180s, warm: 60s (from bot.py lines 1058-1064)

---

## Table 5: Data Flows Verification

### 1. Message → Router → Brain → AI Backend → Response - VERIFIED
**Status**: ✅ OK
- Message enters via `on_message` → `_process_single_message`
- Router dispatches via `dispatch_message()`
- Brain called via `brain_infer()`
- Backend selected in `generate_response()`
- Response flows back through router to Discord

### 2. Attachment → modality detection → handler → evidence - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/modality.py`
**Status**: ✅ OK
- collect_input_items() gathers attachments
- AttachmentClassifier for bucket sorting
- EvidenceBundle for collecting results

### 3. URL → extraction → syndication/web → text - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/router.py:858-1260+`
**Status**: 🟡 Medium
- Complex syndication logic with caching
- X/Twitter specific handling
- Potential for infinite loops if URL redirects to itself

---

## Table 6: Resource Ledger Verification

### 1. HTTP client connection pool (max 64) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:145`
**Status**: ✅ OK
- Configurable via HTTP_MAX_CONNECTIONS
- HTTP2 support with fallback

### 2. Memory profile storage (JSON files) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/memory/profiles.py`
**Status**: 🟡 Medium
- Thread-safe with user_cache_lock/server_lock
- Backup before write pattern present (line 216-217)
- **ISSUE**: Save operations not atomic - could corrupt on crash

### 3. Context history (in-memory + file) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/memory/context_manager.py`
**Status**: ✅ OK
- In-memory fallback on file errors
- DM conversations not saved to disk (privacy ✓)

### 4. TTS audio files - NOT FULLY VERIFIED
**Status**: 🟢 Minor
- tts_cache directory present in repo
- File cleanup not verified

---

## Table 7: Concurrency Hotspots Verification

### 1. user_cache_lock / server_lock (threading) - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/memory/profiles.py:15-16`
**Status**: 🟡 Medium
- Threading locks used for sync profile operations
- **ISSUE**: Mixing threading.Lock with asyncio - could block event loop

**Fix Recommendation**: Use asyncio.Lock for async operations.

### 2. _processing_locks per message - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/router.py:381`
**Status**: 🔴 Critical
- Already identified as memory leak
- No cleanup of completed message locks

### 3. HTTP semaphore per host - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:127-128`
**Status**: ✅ OK
- Proper per-host semaphore management
- Circuit breaker per host correct

### 4. Circuit breaker state checks - VERIFIED
**Location**: `/volume1/py/discord-llm-chatbot/bot/http_client.py:250-254`
**Status**: ✅ OK
- Thread-safe state checks
- Metrics tracking present

---

## Summary

### Critical Issues (🔴): 3
1. memory_cmds.add_memory_cmd() - Insufficient XSS protection
2. Router._processing_locks - Memory leak (grows unbounded)
3. Threading locks in async path - Event loop blocking risk

### Medium Issues (🟡): 6
1. Router._processed_recent_set - Redundant storage
2. Mock objects in img_commands - Should use DI
3. TTS implementation scattered/dead code
4. Profile save not atomic
5. URL extraction potential infinite redirect loop
6. Lock cleanup missing

### Minor Issues (🟢): 4
1. Hardcoded values in some places
2. Config race condition during reload
3. TTS timeout values scattered
4. TTS file cleanup unverified

### Verified OK (✅): 15

---

## Files with Issues Needing Fixes

1. `/volume1/py/discord-llm-chatbot/bot/commands/memory_cmds.py` - Add XSS protection
2. `/volume1/py/discord-llm-chatbot/bot/router.py` - Add lock cleanup
3. `/volume1/py/discord-llm-chatbot/bot/memory/profiles.py` - Use asyncio.Lock
4. `/volume1/py/discord-llm-chatbot/bot/tts/manager.py` - Consolidate or remove stub
5. `/volume1/py/discord-llm-chatbot/bot/commands/img_commands.py` - Use DI for mocks
