"""
Text Flow Performance Constants - All timing budgets and thresholds.
Implementation of CMV (Constants over Magic Values) rule.
"""

from typing import Dict, Final


# Phase timing constants [CMV]
class PhaseConstants:
    """Constants for text flow performance optimization."""

    # OpenRouter/LLM Configuration [PA][REH]
    OR_CONNECT_TIMEOUT_MS: Final[int] = 5000  # 5s connect timeout
    OR_READ_TIMEOUT_MS: Final[int] = 8000  # 8s read timeout (reduced from 12s)
    OR_TOTAL_DEADLINE_MS: Final[int] = 30000  # 30s total deadline (reduced from 120s)
    OR_MAX_RETRIES: Final[int] = 2  # 2 max attempts (reduced from 3)
    OR_RETRY_JITTER_MS: Final[int] = 1000  # 1s base retry jitter (reduced from 5s)
    OR_POOL_MAX_CONNECTIONS: Final[int] = 20  # Connection pool size
    OR_POOL_KEEPALIVE_SECS: Final[int] = 60  # Keep-alive duration
    OR_MAX_TOKENS_SIMPLE: Final[int] = 500  # Token cap for simple queries
    OR_WARN_SLOW_MS: Final[int] = 2000  # Warn if LLM call > 2s

    # Circuit Breaker Configuration [REH]
    OR_BREAKER_FAILURE_WINDOW: Final[int] = 2  # 2 failures to open (reduced from 3)
    OR_BREAKER_OPEN_MS: Final[int] = 15000  # 15s cooldown (reduced from 60s)
    OR_BREAKER_HALFOPEN_PROB: Final[float] = 0.5  # 50% chance to test recovery

    # Pipeline Parallelism [PA]
    PIPELINE_MAX_PARALLEL_TASKS: Final[int] = 3  # Max concurrent tasks
    ROUTER_DECISION_BUDGET_MS: Final[int] = 100  # Router timeout budget
    CONTEXT_CACHE_TTL_SECS: Final[int] = 120  # 2min context cache TTL

    # History Token Budgets [CMV]
    HISTORY_MAX_TOKENS_DM: Final[int] = 800  # DM history token limit
    HISTORY_MAX_TOKENS_GUILD: Final[int] = 1200  # Guild history token limit
    PROMPT_TEMPLATE_CACHE_TTL_SECS: Final[int] = 300  # 5min template cache

    # Queue Management [PA]
    QUEUE_PRIORITY_DM: Final[int] = 1  # High priority for DMs
    QUEUE_PRIORITY_GUILD: Final[int] = 2  # Lower priority for guild
    QUEUE_POLL_INTERVAL_MS: Final[int] = 50  # Faster polling (reduced from default)
    QUEUE_DIRECT_PROCESS_DEPTH: Final[int] = 2  # Process directly if queue < 2

    # SLO Targets [REH][PA]
    SLO_P95_PIPELINE_MS: Final[int] = 2500  # 2.5s total pipeline p95
    SLO_P95_LLM_MS: Final[int] = 1800  # 1.8s LLM call p95
    SLO_P95_DISCORD_MS: Final[int] = 250  # 250ms Discord send p95
    ALERT_CONSECUTIVE_WINDOWS: Final[int] = 3  # Alert after 3 consecutive breaches

    # Phase Names for Consistent Logging [CA]
    PHASE_QUEUE_WAIT: Final[str] = "QUEUE_WAIT"
    PHASE_ROUTER_DISPATCH: Final[str] = "ROUTER_DISPATCH"
    PHASE_CONTEXT_GATHER: Final[str] = "CONTEXT_GATHER"
    PHASE_RAG_QUERY: Final[str] = "RAG_QUERY"
    PHASE_PREP_GEN: Final[str] = "PREP_GEN"
    PHASE_LLM_CALL: Final[str] = "LLM_CALL"
    PHASE_POSTPROC: Final[str] = "POSTPROC"
    PHASE_DISCORD_DISPATCH: Final[str] = "DISCORD_DISPATCH"

    @classmethod
    def get_all_phases(cls) -> list[str]:
        """Get all phase names in processing order."""
        return [
            cls.PHASE_QUEUE_WAIT,
            cls.PHASE_ROUTER_DISPATCH,
            cls.PHASE_CONTEXT_GATHER,
            cls.PHASE_RAG_QUERY,
            cls.PHASE_PREP_GEN,
            cls.PHASE_LLM_CALL,
            cls.PHASE_POSTPROC,
            cls.PHASE_DISCORD_DISPATCH,
        ]

    @classmethod
    def get_slo_targets(cls) -> Dict[str, int]:
        """Get SLO targets by phase."""
        return {
            cls.PHASE_ROUTER_DISPATCH: 80,  # 80ms target
            cls.PHASE_CONTEXT_GATHER: 50,  # 50ms target
            cls.PHASE_RAG_QUERY: 30,  # 30ms target
            cls.PHASE_PREP_GEN: 150,  # 150ms target
            cls.PHASE_LLM_CALL: cls.SLO_P95_LLM_MS,
            cls.PHASE_DISCORD_DISPATCH: cls.SLO_P95_DISCORD_MS,
            "pipeline_total": cls.SLO_P95_PIPELINE_MS,
        }


# Export for easy access
PC = PhaseConstants
