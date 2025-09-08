# Router Speed Overhaul Documentation

## Overview

This document describes the comprehensive router speed overhaul implemented to reduce end-to-end latency by 40–65% p95 while maintaining existing behavior, safety, and user experience.

## Key Performance Goals

- **Target**: Reduce router latency by 40–65% p95 with zero regressions
- **SSOT Gate**: No pre-gate work; enforce speak-when-spoken-to discipline
- **One-Message Discipline**: Only one Discord message per flow; edits coalesced
- **Zero-I/O Planning**: Fast classification and planning in ≤30ms
- **Bounded Concurrency**: Separate pools for LIGHT, NETWORK, HEAVY workloads
- **Single-Flight Deduplication**: Prevent duplicate external calls
- **Adaptive Budgets**: Per-family soft budgets and hard deadlines with route switching
- **Edit Coalescing**: ≥ EDIT_COALESCE_MIN_MS between edits to reduce chat noise

## Architecture Components

### 1. Fast Classification System (`router_classifier.py`)

**Purpose**: Zero-I/O classification and planning within 30ms budget

**Features**:
- Pre-compiled regex tables for URL pattern matching
- O(1) host-to-modality mapping for instant classification  
- Streaming eligibility computation before posting
- Tweet URL detection to force Tweet flow routing
- Direct image detection for bypass optimizations

**Key Classes**:
- `FastClassifier`: Main classification engine
- `ClassificationResult`: Per-item classification with confidence
- `PlanResult`: Complete message plan with timing metrics

**Configuration**:
```env
ROUTER_FAST_CLASSIFY_ENABLE=true
```

### 2. Shared HTTP/2 Client (`http_client.py`)

**Purpose**: Eliminate per-call client overhead, enable connection reuse

**Features**:
- HTTP/2 connection pooling with DNS caching
- Per-host concurrency limits and circuit breakers
- Retry logic with exponential backoff and jitter
- Connection timeout and deadline enforcement
- Comprehensive metrics and health monitoring

**Key Classes**:
- `SharedHttpClient`: Main async HTTP client with pooling
- `RequestConfig`: Per-request configuration options
- `HostLimits`: Per-host rate limiting and circuit breaking
- `HttpMetrics`: Connection and request metrics

**Configuration**:
```env
HTTP2_ENABLE=true
HTTP_DNS_CACHE_TTL_S=300
HTTP_MAX_CONNECTIONS=100
HTTP_MAX_KEEPALIVE_CONNECTIONS=20
HTTP_CONNECT_TIMEOUT_MS=5000
HTTP_READ_TIMEOUT_MS=30000
HTTP_TOTAL_DEADLINE_MS=45000
```

### 3. Concurrency Management (`concurrency_manager.py`)

**Purpose**: Bounded execution pools with hierarchical cancellation

**Features**:
- Separate thread pools for LIGHT (planning), NETWORK (HTTP), HEAVY (OCR/STT/ffmpeg)
- Hierarchical task cancellation with parent-child relationships  
- Backpressure handling when pools are saturated
- Task tracking and completion metrics
- Graceful shutdown with cleanup

**Key Classes**:
- `ConcurrencyManager`: Pool manager with bounded execution
- `PoolType`: Enum for LIGHT, NETWORK, HEAVY pool types
- `TaskSubmitter`: Context manager for pool submission
- `ConcurrencyMetrics`: Pool utilization and task metrics

**Configuration**:
```env
ROUTER_MAX_CONCURRENCY_LIGHT=10
ROUTER_MAX_CONCURRENCY_NETWORK=8  
ROUTER_MAX_CONCURRENCY_HEAVY=2
```

### 4. Single-Flight Cache (`single_flight_cache.py`)

**Purpose**: Deduplication and caching to prevent duplicate external calls

**Features**:
- In-memory LRU cache with TTL expiration
- Single-flight deduplication for concurrent identical requests
- Per-family TTL configuration (tweet, readability, STT, etc.)
- Negative caching for failed requests
- Cache hit/miss metrics with detailed logging

**Key Classes**:
- `SingleFlightCache`: Main cache with deduplication
- `CacheFamily`: Enum for different cache categories
- `CacheMetrics`: Hit rates, sizes, and performance metrics

**Configuration**:
```env
CACHE_SINGLE_FLIGHT_ENABLE=true
TWEET_CACHE_TTL_S=86400  # 24 hours
TWEET_NEGATIVE_TTL_S=3600  # 1 hour
STT_CACHE_TTL_S=604800  # 7 days  
CACHE_READABILITY_TTL_S=3600  # 1 hour
CACHE_BACKEND=memory  # or redis, disk
CACHE_DIR=./cache
```

### 5. Budget and Deadline Management (`budget_manager.py`)

**Purpose**: Adaptive timeouts with route switching and cancellation

**Features**:
- Per-family soft budgets and hard deadlines
- Adaptive budget adjustment based on p95 latencies
- Route switching on soft budget exceeded
- Task cancellation on hard deadline exceeded
- Comprehensive timing and switching metrics

**Key Classes**:
- `BudgetManager`: Main budget enforcement system
- `BudgetFamily`: Enum for different processing families
- `SoftBudgetExceeded`: Exception for route switching
- `HardDeadlineExceeded`: Exception for task cancellation
- `BudgetMetrics`: Timing, switches, and violation tracking

**Configuration**:
```env
TWEET_SYNDICATION_TOTAL_DEADLINE_MS=2000
TWEET_WEB_TIER_A_MS=1000
TWEET_WEB_TIER_B_MS=2000  
TWEET_WEB_TIER_C_MS=5000
X_API_TOTAL_DEADLINE_MS=3000
STT_TOTAL_DEADLINE_MS=30000
WEB_TIER_A_MS=1000
WEB_TIER_B_MS=3000
WEB_TIER_C_MS=8000
OCR_GLOBAL_DEADLINE_MS=45000
OCR_BATCH_DEADLINE_MS=15000
```

### 6. Edit Coalescing (`edit_coalescer.py`)

**Purpose**: Reduce chat noise with intelligent message editing

**Features**:
- Minimum interval enforcement between Discord message edits
- Text-only flow silencing (no streaming status cards)
- Streaming preservation for heavy work (media, OCR, STT)
- Edit batching and coalescing for rapid updates
- Session management with cleanup and finalization

**Key Classes**:
- `EditCoalescer`: Main edit management system
- `StreamingEligibility`: Enum for streaming eligibility reasons
- `EditCoalesceState`: Per-message session state tracking

**Configuration**:
```env
EDIT_COALESCE_MIN_MS=700
EDIT_COALESCE_MAX_DELAY_MS=2000
```

### 7. Dual Sink Logging (`logging_enforcer.py`)

**Purpose**: Rich console + structured JSON logging with startup enforcement

**Features**:
- Pretty console logging with colors, emojis, and level symbols (✔, ⚠, ✖, ℹ)
- Structured JSONL logging with frozen key set preservation
- Startup assertion ensuring both handlers are active
- Auto-truncation of overly long fields with ...+N indicators
- Millisecond precision timestamps in local time

**Key Classes**:
- `LoggingEnforcer`: Startup assertion and configuration
- `PrettyConsoleHandler`: Enhanced RichHandler with symbols
- `StructuredJsonFormatter`: JSON formatter with frozen keys

**Frozen Key Set**: `ts, level, name, subsys, guild_id, user_id, msg_id, event, detail`

### 8. Optimized Router (`optimized_router.py`)

**Purpose**: Main integration point for all speed optimizations

**Features**:
- SSOT gate with hard early return (no pre-gate work)
- Fast classification and planning within 30ms
- Tweet flow optimization (Cache → Syndication → Web → API-last)
- Bounded concurrency pool assignment
- Single-flight caching integration
- Budget management with route switching
- Edit coalescing for streaming responses
- Comprehensive instrumentation and metrics

**Key Classes**:
- `OptimizedRouter`: Main router with all optimizations integrated
- `OptimizedExecution`: Execution context with metrics tracking  
- `RouterMetrics`: Performance counters and timing statistics

## Router Flow Changes

### Before Optimization
```
Message → Basic Classification → Sequential Processing → Multiple Responses
```

### After Optimization  
```
Message → SSOT Gate → Fast Classification (≤30ms) → Plan → Bounded Pools → 
Single-Flight Cache Check → Budget-Managed Execution → Edit Coalescing → Single Response
```

## Tweet Flow Optimization

**New Flow**: Cache → Syndication → Web → API-last

1. **Cache Check**: Single-flight cache lookup (instant if hit)
2. **Syndication Tier**: Fast syndication with soft budget
3. **Web Extraction**: Tiered web extraction with fallback budgets  
4. **API Fallback**: X API as last resort with hard deadline

**Budget Management**:
- Syndication: 2000ms soft budget  
- Web Tier A: 1000ms (fast sites)
- Web Tier B: 2000ms (medium sites)
- Web Tier C: 5000ms (slow sites)
- X API: 3000ms hard deadline

**Route Switching**: Automatic fallback on soft budget exceeded

## STT Orchestrator Instrumentation

**Added Instrumentation**:
- Cache hit/miss logging with provider details
- Provider latency tracking in TranscriptResult
- Comprehensive processing metrics with timing
- Cache key truncation for privacy
- Success/failure rate tracking

**New `cache_hit` Flag**: Added to `TranscriptResult` for downstream metrics

## Performance Metrics

**Router Metrics**:
- Processing latency (p50, p95, p99)
- Cache hit rates by family
- Route switch counts and reasons
- Pool utilization and saturation
- Budget violations and deadline exceeded
- Edit coalescing efficiency

**HTTP Client Metrics**:
- Connection pool utilization
- Request success/failure rates
- Circuit breaker trips
- DNS cache hit rates
- Per-host latency tracking

**Concurrency Metrics**:
- Pool queue depths and wait times
- Task completion rates
- Cancellation and timeout counts
- Thread pool efficiency

## Configuration Summary

All new configuration keys are **additive only** (no renames). See `.env-sample` for complete list with documentation.

### Critical Settings
```env
# Core Router Optimizations
ROUTER_FAST_CLASSIFY_ENABLE=true
ROUTER_MAX_CONCURRENCY_LIGHT=10
ROUTER_MAX_CONCURRENCY_NETWORK=8
ROUTER_MAX_CONCURRENCY_HEAVY=2

# HTTP/2 Client Optimization  
HTTP2_ENABLE=true
HTTP_MAX_CONNECTIONS=100
HTTP_DNS_CACHE_TTL_S=300

# Tweet Flow Optimization
TWEET_FLOW_ENABLED=true
TWEET_CACHE_TTL_S=86400
TWEET_SYNDICATION_TOTAL_DEADLINE_MS=2000

# Edit Coalescing
EDIT_COALESCE_MIN_MS=700

# Single-Flight Caching
CACHE_SINGLE_FLIGHT_ENABLE=true
```

## Safety and Compatibility

**Preserved Behaviors**:
- All existing command semantics unchanged
- Input validation and safety checks maintained  
- Privacy protections intact
- Error handling and user messaging preserved
- Output quality and accuracy maintained

**Backward Compatibility**:
- All optimizations can be disabled via config
- Graceful fallbacks to original behavior
- No breaking changes to external interfaces
- Existing logging and monitoring preserved

## Deployment Considerations

**Rollout Strategy**:
1. Deploy with optimizations disabled
2. Enable fast classification first
3. Gradually enable HTTP/2 client and caching
4. Enable concurrency pools and budgets
5. Monitor p95 latencies and adjust budgets
6. Enable edit coalescing last

**Monitoring Requirements**:
- P95 latency monitoring across all flows
- Cache hit rate tracking per family
- Route switch frequency monitoring
- Budget violation alerting
- Pool saturation monitoring

**Tuning Parameters**:
- Adjust budgets based on actual p95 measurements
- Tune pool sizes based on workload patterns
- Adjust cache TTLs based on hit rates
- Configure circuit breaker thresholds per environment

## Testing Coverage

**Unit Tests**: Individual component testing with mocks
**Integration Tests**: End-to-end flow testing
**Performance Tests**: Latency measurement and regression detection
**Fault Injection Tests**: Error handling and graceful degradation
**Load Tests**: Concurrent request handling and pool saturation

See `tests/test_router_speed_optimization.py` for comprehensive test suite.

## Expected Performance Improvements

**Conservative Estimates**:
- 40% reduction in p95 latency for Tweet URLs
- 50% reduction in duplicate external calls via single-flight
- 30% reduction in Discord message noise via edit coalescing  
- 60% improvement in concurrent request handling

**Optimistic Scenarios**:
- 65% reduction in p95 latency for cache hits
- 80% reduction in duplicate calls for popular content
- 90% reduction in streaming message noise
- 200% improvement in peak concurrent handling

**Key Success Metrics**:
- Router p95 latency < 2000ms (down from 3500ms+)
- Tweet flow p95 < 1500ms (down from 2500ms+)
- Cache hit rate > 60% for tweets within 24h
- Edit noise reduction > 50% for streaming flows
